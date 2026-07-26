import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from einops import rearrange
import random

def symlog(x):
    """실수 값을 SymLog 공간으로 압축합니다."""
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)

def symexp(x):
    """SymLog 공간의 값을 원래 스케일로 복원합니다."""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)

class FastHashBucket:
    def __init__(self):
        self.index_pos = {}
        self.items = []

    def add(self, index):
        if index in self.index_pos:
            return
        self.index_pos[index] = len(self.items)
        self.items.append(index)

    def remove(self, index):
        if index not in self.index_pos:
            return
        pos = self.index_pos[index]
        last_index = self.items[-1]
        
        self.items[pos] = last_index
        self.index_pos[last_index] = pos
        
        self.items.pop()
        del self.index_pos[index]

    def sample_and_remove(self, k):
        if not self.items:
            return []
        k = min(k, len(self.items))
        indices = random.sample(self.items, k)
        for idx in indices:
            self.remove(idx)
        return indices

    def __len__(self):
        return len(self.items)

class MacroLoss(nn.Module):
    def __init__(self, latent_dim, config, dyn_config=None, full_latent_dim=None, buffer_max_length=1000000):
        super().__init__()
        cfg = config or {}
        dyn_cfg = dyn_config or {}
        self.enabled = bool(cfg.get("Enable", False))
        
        self.aux_gamma = cfg.get("AuxGamma", None)
        if self.aux_gamma is not None:
            self.aux_gamma = float(self.aux_gamma)
        
        self.trigger_type = str(cfg.get("TriggerType", "reward")).lower()
        self.diff_type = str(cfg.get("DiffType", "reward")).lower()
        if self.diff_type not in ["reward", "td_error", "value", "aux_value"]:
            raise ValueError(f"DiffType must be 'reward', 'td_error', 'value', or 'aux_value', got {self.diff_type}")
            
        # AuxValueNet has been extracted to WorldModel.
        
        self.sigma_threshold = float(cfg.get("SigmaThreshold", 2.0))
        
        self.enable_max_relative_threshold = bool(cfg.get("EnableMaxRelativeThreshold", False))
        self.max_relative_threshold_ratio = float(cfg.get("MaxRelativeThresholdRatio", 0.1))
        self.max_relative_threshold_decay = float(cfg.get("MaxRelativeThresholdDecay", 0.999))
        # 훈련 재개 시에도 역대 최대치를 잊지 않도록 persistent=True로 등록합니다.
        self.register_buffer("running_max_trigger_metric", torch.tensor(1e-8, dtype=torch.float32), persistent=True)
        self.register_buffer("running_max_metric", torch.tensor(1e-8, dtype=torch.float32), persistent=True)
        
        # [수정] Tensor-based Hash Table을 위한 전역 버퍼 할당
        self.buffer_max_length = buffer_max_length
        self.hash_bits = int(cfg.get("HashBits", 12))
        self.num_buckets = 2 ** self.hash_bits
        self.max_queue_per_key = int(cfg.get("MaxQueuePerKey", 100000))
        self.max_past_samples = int(cfg.get("MaxPastSamples", 4))
        self.max_pairs_per_batch = int(cfg.get("MaxPairsPerBatch", 256))
        
        # Global storage (CPU)
        self.register_buffer("item_metrics", torch.zeros(buffer_max_length, dtype=torch.float32), persistent=False)
        self.register_buffer("item_latents", torch.zeros((buffer_max_length, latent_dim), dtype=torch.float32), persistent=False)
        self.register_buffer("item_bucket", torch.full((buffer_max_length,), -1, dtype=torch.int32), persistent=False)
        
        # O(1) FastHashBuckets (Python List of objects for CPU-side fast matching)
        self.buckets = [FastHashBucket() for _ in range(self.num_buckets)]
        self.register_buffer("index_to_bucket", torch.full((buffer_max_length,), -1, dtype=torch.int32), persistent=False)
        # ------------------------------------------
        
        # --- [수정된 부분] 특정 스텝 이후 비활성화 설정 및 Soft Margin 추가 ---
        self.loss_scale = float(cfg.get("LossScale", 1.0))
        self.disable_after_step = int(cfg.get("DisableAfterStep", -1))
        
        self.distance_metric = str(cfg.get("DistanceMetric", "margin")).lower()
        self.dbc_alpha = float(cfg.get("DBCAlpha", 0.5))
        self.margin = float(cfg.get("Margin", 0.9)) # 절벽 현상을 방지하는 Soft Margin (기본값 0.8)
        self.contrastive_sigma_ratio = float(cfg.get("ContrastiveSigmaRatio", 0.5))
        
        # --- [추가] Global Rebuild 설정 ---
        self.enable_global_rebuild = bool(cfg.get("EnableGlobalRebuild", True))
        self.rebuild_threshold = float(cfg.get("RebuildThreshold", 0.1))
        self.rebuild_cooldown = int(cfg.get("RebuildCooldown", 1000))
        self.rebuild_batch_size = int(cfg.get("RebuildBatchSize", 256))
        self.last_rebuild_step = -self.rebuild_cooldown # 초기 설정
        self.global_rebuild_count = 0 # [추가] Rebuild가 일어난 총 횟수를 추적 (wandb 로깅용)
        
        # --- [추가] Lazy Rebuild 설정 ---
        lazy_cfg = cfg.get("LazyRebuild", {})
        self.lazy_rebuild_enable = bool(lazy_cfg.get("Enable", True))
        self.lazy_rebuild_multiplier = int(lazy_cfg.get("SampleMultiplier", 10))
        self.lazy_rebuild_use_similar = bool(lazy_cfg.get("UseSimilarBuckets", True))
        # ------------------------------------------
        
        self.store_only_triggered = bool(cfg.get("StoreOnlyTriggered", False))
        # [추가] 직전 프레임 포함 여부 설정 로드
        self.include_prev_frame = bool(cfg.get("IncludePrevFrameInMacroLoss", False))

        if self.hash_bits < 1 or self.hash_bits > 62:
            raise ValueError(f"HashBits must be in [1, 62], got {self.hash_bits}.")
        self.max_queue_per_key = max(1, self.max_queue_per_key)
        self.max_past_samples = max(1, self.max_past_samples)
        self.max_pairs_per_batch = max(1, self.max_pairs_per_batch)
        
        # Cutout Params
        cutout_cfg = cfg.get('Cutout', {})
        self.use_cutout = bool(cutout_cfg.get('Enable', False))
        self.num_masks = int(cutout_cfg.get('NumMasks', 0))
        self.cutout_p = float(cutout_cfg.get('Probability', 0.5))
        self.cutout_scale = cutout_cfg.get('Scale', [0.02, 0.33])
        self.cutout_ratio = cutout_cfg.get('Ratio', [0.3, 3.3])
        self.cutout_value = float(cutout_cfg.get('Value', 0.0))
        self.cutout_disable_after = int(cutout_cfg.get('DisableAfterStep', -1))

        proj = torch.randn(latent_dim, self.hash_bits, dtype=torch.float32)
        self.register_buffer("hash_proj", proj, persistent=False)
        bit_values = 2 ** torch.arange(self.hash_bits, dtype=torch.int64)
        self.register_buffer("hash_bit_values", bit_values, persistent=False)

        # self.hash_memory = {} (삭제됨)
        
        if self.trigger_type == "td_error":
            self.register_buffer("td_error_mean", torch.tensor(0.0, dtype=torch.float64))
            self.register_buffer("td_error_var", torch.tensor(1.0, dtype=torch.float64))
            self.register_buffer("td_error_count", torch.tensor(1e-4, dtype=torch.float64))
            
        # Dynamics Weighting has been extracted to WorldModel.

    def _update_welford(self, td_error, valid_mask):
        valid_td = td_error[valid_mask].to(torch.float64)
        
        if valid_td.numel() == 0:
            return self.td_error_mean.to(torch.float32), self.td_error_var.to(torch.float32)

        batch_mean = torch.mean(valid_td)
        batch_var = torch.var(valid_td, unbiased=False)
        batch_count = torch.tensor(valid_td.numel(), dtype=torch.float64, device=td_error.device)

        tot_count = self.td_error_count + batch_count
        self.td_error_count.copy_(tot_count)

        # RL의 Non-stationary 특성을 고려한 EMA 업데이트
        # 초기에는 Welford(누적 평균)로 작동하여 안정성을 확보하고,
        # 데이터가 충분히 쌓이면(alpha <= 1-decay) EMA로 부드럽게 전환됩니다.
        decay = 0.999
        alpha = torch.clamp(batch_count / tot_count, min=1.0 - decay, max=1.0)

        old_mean = self.td_error_mean.clone()
        new_mean = (1.0 - alpha) * old_mean + alpha * batch_mean
        
        # Variance EMA 업데이트 수식
        new_var = (1.0 - alpha) * self.td_error_var + \
                  alpha * batch_var + \
                  alpha * (1.0 - alpha) * ((batch_mean - old_mean) ** 2)

        self.td_error_mean.copy_(new_mean)
        self.td_error_var.copy_(new_var)

        return self.td_error_mean.to(torch.float32), self.td_error_var.to(torch.float32)



    def _hash_keys(self, latent):
        if latent.numel() == 0:
            return torch.empty((0,), dtype=torch.int64, device=latent.device)
        scores = latent.float() @ self.hash_proj.float()
        bits = scores > 0
        bit_values = self.hash_bit_values.to(bits.device)
        keys = (bits.to(torch.int64) * bit_values).sum(dim=-1)
        return keys.detach()

    def _update_memory(self, obs, reward, latent, reward_mean, reward_std, td_error=None, value=None, aux_value=None, indexes=None, aux_mean=None, aux_std=None):
        if not self.enabled or indexes is None:
            return
        if obs.numel() == 0:
            return
            
        if reward.dim() == 3:
            reward = reward.squeeze(-1)
        if td_error is not None and td_error.dim() == 3:
            td_error = td_error.squeeze(-1)
        if value is not None and value.dim() == 3:
            value = value.squeeze(-1)

        align_mask = torch.ones_like(reward, dtype=torch.bool)
        if align_mask.dim() >= 2:
            align_mask[..., 0] = False
            align_mask[..., -1] = False

        if self.store_only_triggered:
            if self.trigger_type == "td_error":
                if td_error is None:
                    raise ValueError("TriggerType이 'td_error'일 경우 _update_memory에도 td_error를 전달해야 합니다.")
                td_error_float = td_error.to(torch.float32)
                td_std = torch.sqrt(self.td_error_var.to(torch.float32) + 1e-8)
                store_mask = (torch.abs(td_error_float - self.td_error_mean.to(torch.float32)) >= self.sigma_threshold * td_std) & align_mask
                
                if self.enable_max_relative_threshold:
                    linear_td_full = symexp(td_error_float + (value.detach() if value is not None else 0)) - symexp(value.detach() if value is not None else 0)
                    store_mask = store_mask & (torch.abs(linear_td_full) >= self.max_relative_threshold_ratio * self.running_max_trigger_metric)
                    
            elif self.trigger_type == "aux_value":
                if aux_value is None:
                    raise ValueError("aux_value must be provided when trigger_type is 'aux_value'")
                
                aux_val_linear = symexp(aux_value.detach()).to(torch.float32)
                # 글로벌 통계를 사용하여 마스크 생성
                if aux_mean is None or aux_std is None:
                    raise ValueError("aux_mean and aux_std must be provided when trigger_type is 'aux_value'")
                
                store_mask = (torch.abs(aux_val_linear - aux_mean) >= self.sigma_threshold * aux_std) & align_mask
                
                if self.enable_max_relative_threshold:
                    store_mask = store_mask & (torch.abs(aux_val_linear) >= self.max_relative_threshold_ratio * self.running_max_trigger_metric)

            # --- [추가된 부분] Value의 순간 변화량(차이)을 트리거로 사용하는 로직 ---
            elif self.trigger_type == "aux_value_diff":
                if aux_value is None:
                    raise ValueError("aux_value must be provided when trigger_type is 'aux_value_diff'")
                    
                # aux_value의 형태가 [B, L] 임을 활용하여 선형 스케일로 복원
                aux_val_linear = symexp(aux_value.detach()).to(torch.float32)
                
                # 시간 축(L)을 따라 프레임 간의 Value 절대 변화량 |V_t - V_{t-1}| 계산
                val_diff = torch.zeros_like(aux_val_linear)
                val_diff[..., 1:] = torch.abs(aux_val_linear[..., 1:] - aux_val_linear[..., :-1])
                
                # aux_value의 Welford 통계를 가져와서 마스크 생성
                if aux_std is None:
                    raise ValueError("aux_std must be provided when trigger_type is 'aux_value_diff'")
                
                # 변화량이 전체 가치의 표준편차 대비 비정상적으로 확 튈 때만 트리거
                store_mask = (val_diff >= self.sigma_threshold * aux_std) & align_mask
                
                if self.enable_max_relative_threshold:
                    store_mask = store_mask & (torch.abs(val_diff) >= self.max_relative_threshold_ratio * self.running_max_trigger_metric)
            # -----------------------------------------------------------------
                    
            else:
                store_mask = (torch.abs(reward - reward_mean) >= self.sigma_threshold * reward_std) & align_mask
                
                if self.enable_max_relative_threshold:
                    store_mask = store_mask & (torch.abs(symexp(reward)) >= self.max_relative_threshold_ratio * self.running_max_trigger_metric)
        else:
            store_mask = align_mask

        if not torch.any(store_mask):
            return

        if self.diff_type == "td_error":
            metric_to_store = td_error
        elif self.diff_type == "aux_value":
            if aux_value is None:
                raise ValueError("aux_value must be provided when diff_type is 'aux_value'")
            metric_to_store = aux_value
        elif self.diff_type == "value":
            metric_to_store = value if value is not None else torch.zeros_like(reward)
        else:
            metric_to_store = reward


        
        latent_masked = latent.detach()[store_mask]
        keys = self._hash_keys(latent_masked)
        if len(keys) == 0:
            return

        index_items = indexes[store_mask]
        metric_items = metric_to_store[store_mask]

        idx_ints = index_items.long()
        keys_int = keys.to(torch.int32)
        
        # Global storage 업데이트 (GPU 병렬화)
        self.item_metrics[idx_ints] = metric_items.to(self.item_metrics.dtype)
        self.item_latents[idx_ints] = latent_masked.detach().to(self.item_latents.dtype)
        self.item_bucket[idx_ints] = keys_int
        
        # FIFO Queue 순차 삽입
        # FIFO Queue Bulk Update (CPU-GPU Sync 최소화)
        self._bulk_update_buckets(keys_int, idx_ints)

    def _bulk_update_buckets(self, keys_int, idx_ints):
        if len(keys_int) == 0:
            return
            
        keys_cpu = keys_int.cpu().numpy()
        idx_cpu = idx_ints.cpu().numpy()
        
        index_to_bucket_cpu = self.index_to_bucket.cpu().numpy()
        
        for k, idx in zip(keys_cpu, idx_cpu):
            old_k = index_to_bucket_cpu[idx]
            
            # 이전 버킷과 다르다면, 이전 버킷에서 물리적으로 삭제 (유령 인덱스 방지)
            if old_k != -1 and old_k != k:
                self.buckets[old_k].remove(idx)
                
            # 새 버킷에 추가
            if old_k != k:
                self.buckets[k].add(idx)
                index_to_bucket_cpu[idx] = k
                
        # 변경된 매핑 테이블만 다시 GPU로 복사
        self.index_to_bucket.copy_(torch.from_numpy(index_to_bucket_cpu))

    def _rebuild_all_hash_buckets(self, replay_buffer, encode_fn, latent_dtype=torch.float32, extra_items=None):
        self.global_rebuild_count += 1
        
        valid_mask = self.item_bucket != -1
        valid_indices = torch.where(valid_mask)[0]
        all_indices = valid_indices.tolist()
        
        if extra_items:
            all_indices.extend([x[0] for x in extra_items])
            
        if not all_indices:
            return
            
        # Reset buckets
        for bucket in self.buckets:
            bucket.index_pos.clear()
            bucket.items.clear()
        self.index_to_bucket.fill_(-1)
        self.item_bucket.fill_(-1)
        
        indices_tensor = torch.tensor(all_indices, dtype=torch.int64, device=self.hash_proj.device)
        
        for i in range(0, len(all_indices), self.rebuild_batch_size):
            indices_tensor_batch = indices_tensor[i:i+self.rebuild_batch_size]
            
            if replay_buffer.store_on_gpu:
                obs_raw = replay_buffer.obs_buffer[indices_tensor_batch]
            else:
                idx_tensor = indices_tensor_batch.cpu().long()
                obs_raw = torch.from_numpy(replay_buffer.obs_buffer[idx_tensor])
                
            obs = obs_raw.to(device=self.hash_proj.device, dtype=latent_dtype) / 255.0
            obs = rearrange(obs, "N H W C -> N 1 C H W")
            
            with torch.no_grad():
                encoded = encode_fn(obs)
                if len(encoded) == 3:
                    latent, _, _ = encoded
                else:
                    latent, _ = encoded
                latent = latent.squeeze(1).detach()
                
            keys = self._hash_keys(latent)
            
            idx_ints = indices_tensor_batch.long()
            keys_int = keys.to(torch.int32)
            
            self.item_latents[idx_ints] = latent.detach().to(self.item_latents.dtype)
            self.item_bucket[idx_ints] = keys_int
            
            # FIFO Queue Bulk Update
            self._bulk_update_buckets(keys_int, idx_ints)

    def log_detailed_distribution(self, logger, global_step, replay_buffer=None):
        if logger is None:
            return
            
        demo_size = getattr(replay_buffer, 'protect_size', 0) if replay_buffer else 0
        
        hist_data = []
        data_for_bar = []
        
        for k, bucket in enumerate(self.buckets):
            size = len(bucket)
            if size > 0:
                demo_count = 0
                agent_count = 0
                if demo_size > 0:
                    for idx in bucket.items:
                        val = idx.item() if hasattr(idx, 'item') else idx
                        if val < demo_size:
                            demo_count += 1
                        else:
                            agent_count += 1
                else:
                    agent_count = size
                    
                if agent_count > 0:
                    hist_data.extend([[k, "Agent"]] * agent_count)
                if demo_count > 0:
                    hist_data.extend([[k, "Demo"]] * demo_count)
                    
                data_for_bar.append([str(k), size])
                
        if hist_data:
            logger.log("MacroLoss/bucket_distribution_split_hist", (
                hist_data, 
                ["Bucket ID", "Type"], 
                "Bucket ID", 
                "Type", 
                "Bucket Distribution (Agent vs Demo)"
            ), global_step=global_step)
            
        if data_for_bar:
            logger.log("MacroLoss/bucket_sizes_bar", (
                data_for_bar, 
                ["Bucket ID", "Item Count"], 
                "Bucket ID", 
                "Item Count", 
                "Current Bucket Distribution"
            ), global_step=global_step)

    # --- [수정된 부분] global_step, aux_mean, aux_std, aux_value_fn 인자 추가 ---
    def forward(self, obs, latent, logits, reward, encode_fn, reward_mean, reward_std, td_error=None, value=None, aux_value=None, termination=None, indexes=None, replay_buffer=None, global_step=None, latent_full=None, aux_gamma=None, aux_lam=None, aux_mean=None, aux_std=None, aux_value_fn=None):
        self.debug_metrics = {'rebuild_triggered': 0.0}
        
        # --- [추가] 버킷 분포 로깅 지표 ---
        sizes = [len(b) for b in self.buckets if len(b) > 0]
        if sizes:
            mean_val = sum(sizes) / len(sizes)
            self.debug_metrics['bucket_size_mean'] = mean_val
            self.debug_metrics['bucket_size_max'] = max(sizes)
            self.debug_metrics['bucket_size_min'] = min(sizes)
            if len(sizes) > 1:
                self.debug_metrics['bucket_size_var'] = sum((x - mean_val) ** 2 for x in sizes) / (len(sizes) - 1)
            else:
                self.debug_metrics['bucket_size_var'] = 0.0
            self.debug_metrics['bucket_active_count'] = len(sizes)
        # 구체적인 분포(히스토그램/바)는 evaluation 시에 log_detailed_distribution 메서드를 통해 따로 로깅합니다.
        # -----------------------------

        if not self.enabled and not (self.dyn_weighting_enable and self.dyn_use_aux_value_net):
            return obs.new_tensor(0.0), obs.new_tensor(0.0), obs.new_tensor(0.0), None, None, None
        
        if global_step is not None and self.disable_after_step > 0:
            if global_step >= self.disable_after_step:
                self.enabled = False
                
        if not self.enabled and not (self.dyn_weighting_enable and self.dyn_use_aux_value_net):
            return obs.new_tensor(0.0), obs.new_tensor(0.0), obs.new_tensor(0.0), None, None, None
        # -----------------------------------------------------------------

        if obs.numel() == 0:
            return latent.new_tensor(0.0), latent.new_tensor(0.0), latent.new_tensor(0.0), None, None, None
            
        if self.diff_type == "value" and value is None:
            raise ValueError("DiffType이 'value'일 경우 forward에 value를 반드시 전달해야 합니다.")
        if self.diff_type == "td_error" and td_error is None:
            raise ValueError("DiffType이 'td_error'일 경우 forward에 td_error를 반드시 전달해야 합니다.")
        # The check for diff_type == "aux_value" and value is None has been removed 
        # since distillation is now handled externally if value is not provided.
            
        if reward.dim() == 3:
            reward = reward.squeeze(-1)
        if td_error is not None and td_error.dim() == 3:
            td_error = td_error.squeeze(-1)
        if value is not None and value.dim() == 3:
            value = value.squeeze(-1)

        align_mask = torch.ones_like(reward, dtype=torch.bool)
        if align_mask.dim() >= 2:
            align_mask[..., 0] = False
            align_mask[..., -1] = False

        # AuxValueNet evaluation and Welford updates are now handled in WorldModel.
        # We just use the provided aux_value directly.
        if aux_value is not None:
            aux_value_all = aux_value.squeeze(-1)
        else:
            aux_value_all = None
            
        distill_loss = latent.new_tensor(0.0)

        # MacroLoss가 비활성화되어 있다면, 대비 학습(Contrastive) 등 나머지 로직은 스킵
        if not self.enabled:
            return distill_loss.new_tensor(0.0), distill_loss, latent.new_tensor(0.0), None, None, None

        if self.trigger_type == "td_error":
            if td_error is None:
                raise ValueError("TriggerType이 'td_error'일 경우 forward에 td_error를 반드시 전달해야 합니다.")
            td_error_float = td_error.detach().to(torch.float32)
            
            with torch.no_grad():
                valid_td = td_error_float[align_mask]
                if valid_td.numel() > 0:
                    current_max_td = torch.max(torch.abs(valid_td))
                    self.running_max_trigger_metric.copy_(torch.max(self.running_max_trigger_metric * self.max_relative_threshold_decay, current_max_td))

            td_mean, td_var = self._update_welford(td_error_float, align_mask)
            td_std = torch.sqrt(td_var + 1e-8)
            trigger_mask_sigma = (torch.abs(td_error_float - td_mean) >= self.sigma_threshold * td_std) & align_mask
            self.debug_metrics['sigma_trigger_count'] = trigger_mask_sigma.sum().item()
            trigger_mask = trigger_mask_sigma
            
            if self.enable_max_relative_threshold:
                trigger_mask_rel = (torch.abs(td_error_float) >= self.max_relative_threshold_ratio * self.running_max_trigger_metric)
                self.debug_metrics['max_rel_trigger_count'] = trigger_mask_rel.sum().item()
                trigger_mask = trigger_mask & trigger_mask_rel
            else:
                self.debug_metrics['max_rel_trigger_count'] = -1

        elif self.trigger_type == "aux_value":
            aux_val_linear = symexp(aux_value_all.detach()).to(torch.float32)
            
            with torch.no_grad():
                valid_aux = aux_val_linear[align_mask]
                if valid_aux.numel() > 0:
                    current_max_aux = torch.max(torch.abs(valid_aux))
                    self.running_max_trigger_metric.copy_(torch.max(self.running_max_trigger_metric * self.max_relative_threshold_decay, current_max_aux))

            # 글로벌 Welford 통계 가져오기 (공통 영역에서 이미 업데이트됨)
            if aux_mean is None or aux_std is None:
                raise ValueError("aux_mean and aux_std must be provided when trigger_type is 'aux_value'")
            
            trigger_mask_sigma = (torch.abs(aux_val_linear - aux_mean) >= self.sigma_threshold * aux_std) & align_mask
            self.debug_metrics['sigma_trigger_count'] = trigger_mask_sigma.sum().item()
            trigger_mask = trigger_mask_sigma
            
            if self.enable_max_relative_threshold:
                trigger_mask_rel = (torch.abs(aux_val_linear) >= self.max_relative_threshold_ratio * self.running_max_trigger_metric)
                self.debug_metrics['max_rel_trigger_count'] = trigger_mask_rel.sum().item()
                trigger_mask = trigger_mask & trigger_mask_rel
            else:
                self.debug_metrics['max_rel_trigger_count'] = -1

        elif self.trigger_type == "aux_value_diff":
            aux_val_linear = symexp(aux_value_all.detach()).to(torch.float32)
            
            # Welford 통계는 공통 영역에서 이미 업데이트됨

            val_diff = torch.zeros_like(aux_val_linear)
            val_diff[..., 1:] = torch.abs(aux_val_linear[..., 1:] - aux_val_linear[..., :-1])
            
            with torch.no_grad():
                valid_aux = aux_val_linear[align_mask]
                if valid_aux.numel() > 0:
                    self.debug_metrics['aux_value_batch_mean'] = valid_aux.mean().item()
                    self.debug_metrics['aux_value_batch_var'] = valid_aux.var(unbiased=False).item() if valid_aux.numel() > 1 else 0.0

                valid_diff = val_diff[align_mask]
                if valid_diff.numel() > 0:
                    current_max_diff = torch.max(torch.abs(valid_diff))
                    self.running_max_trigger_metric.copy_(torch.max(self.running_max_trigger_metric * self.max_relative_threshold_decay, current_max_diff))
                    self.debug_metrics['val_diff_batch_mean'] = valid_diff.mean().item()
                    self.debug_metrics['val_diff_batch_var'] = valid_diff.var(unbiased=False).item() if valid_diff.numel() > 1 else 0.0

            # 글로벌 Welford 통계 (aux_value 기준)
            if aux_mean is None or aux_std is None:
                raise ValueError("aux_mean and aux_std must be provided when trigger_type is 'aux_value_diff'")
            
            self.debug_metrics['aux_value_global_mean'] = aux_mean.item()
            self.debug_metrics['aux_value_global_var'] = (aux_std ** 2).item()
            
            trigger_mask_sigma = (val_diff >= self.sigma_threshold * aux_std) & align_mask
            self.debug_metrics['sigma_trigger_count'] = trigger_mask_sigma.sum().item()
            trigger_mask = trigger_mask_sigma
            
            if self.enable_max_relative_threshold:
                trigger_mask_rel = (torch.abs(val_diff) >= self.max_relative_threshold_ratio * self.running_max_trigger_metric)
                self.debug_metrics['max_rel_trigger_count'] = trigger_mask_rel.sum().item()
                trigger_mask = trigger_mask & trigger_mask_rel
            else:
                self.debug_metrics['max_rel_trigger_count'] = -1
        # -----------------------------------------------------------------

        else:
            with torch.no_grad():
                if reward.numel() > 0:
                    current_max_reward = torch.max(torch.abs(reward))
                    self.running_max_trigger_metric.copy_(torch.max(self.running_max_trigger_metric * self.max_relative_threshold_decay, current_max_reward.float()))
            
            trigger_mask_sigma = (torch.abs(reward - reward_mean) >= self.sigma_threshold * reward_std) & align_mask
            self.debug_metrics['sigma_trigger_count'] = trigger_mask_sigma.sum().item()
            trigger_mask = trigger_mask_sigma
            
            if self.enable_max_relative_threshold:
                trigger_mask_rel = (torch.abs(reward) >= self.max_relative_threshold_ratio * self.running_max_trigger_metric)
                self.debug_metrics['max_rel_trigger_count'] = trigger_mask_rel.sum().item()
                trigger_mask = trigger_mask & trigger_mask_rel
            else:
                self.debug_metrics['max_rel_trigger_count'] = -1

        # --- [추가된 부분] 인과관계(원인) 학습을 위해 직전 프레임(t-1) 마스크 확장 ---
        # 가치가 튄 시점(결과, t) 뿐만 아니라, 그 직전 프레임(원인, t-1)도 핵심 정보로 등록합니다.
        trigger_mask_original = trigger_mask.clone()
        trigger_mask_prev = torch.zeros_like(trigger_mask)
        if self.include_prev_frame:
            trigger_mask_prev[..., :-1] = trigger_mask[..., 1:]
            # 두 시점을 모두 트리거 마스크에 포함시킵니다.
            trigger_mask = trigger_mask | trigger_mask_prev
        
        # 시퀀스 맨 앞(t=0)이나 끝 등 align_mask에서 제외된 부분은 다시 걸러냅니다.
        trigger_mask = trigger_mask & align_mask
        # -----------------------------------------------------------

        trigger_obs = obs[trigger_mask] if torch.any(trigger_mask) else None

        if not torch.any(trigger_mask) or sum(len(b) for b in self.buckets) == 0:
            self.debug_metrics['trigger_count'] = trigger_mask.sum().item() if trigger_mask is not None else 0
            self.debug_metrics['pair_count'] = 0
            self._update_memory(obs, reward, latent, reward_mean, reward_std, td_error, value, aux_value_all, indexes)
            return distill_loss, distill_loss, latent.new_tensor(0.0), trigger_obs, trigger_mask, trigger_mask_original

        latent_trigger = latent[trigger_mask]
        logits_trigger = logits[trigger_mask]
        
        if self.diff_type == "td_error":
            metric_trigger_full = td_error.clone()
        elif self.diff_type == "value":
            metric_trigger_full = value.clone()
        elif self.diff_type == "aux_value":
            metric_trigger_full = aux_value_all.clone()
        else:
            metric_trigger_full = reward.clone()
            
        # [추가] include_prev_frame 설정이 True일 때만 타겟 덮어쓰기 로직 수행
        if self.include_prev_frame:
            metric_trigger_full[..., :-1] = torch.where(
                trigger_mask_prev[..., :-1] & ~trigger_mask_original[..., :-1],
                metric_trigger_full[..., 1:],
                metric_trigger_full[..., :-1]
            )
        metric_trigger = metric_trigger_full[trigger_mask]
            
        keys = self._hash_keys(latent_trigger.detach())
        keys_int = keys.to(torch.int64)
        
        M = len(keys_int)
        K = self.max_past_samples
        
        if M == 0:
            pair_count = 0
        else:
            if self.lazy_rebuild_enable:
                # Vectorized LazyRebuild Logic
                K_fetch = self.max_past_samples * self.lazy_rebuild_multiplier
                keys_list = keys_int.tolist()
                
                pool_global_indices_list = []
                pool_curr_batch_idx_list = []
                
                for i, key in enumerate(keys_list):
                    buckets_to_check = [key]
                    if self.lazy_rebuild_use_similar:
                        buckets_to_check.extend([key ^ (1 << bit) for bit in range(self.hash_bits)])
                        
                    K_per_bucket = max(1, K_fetch // len(buckets_to_check))
                    
                    for b_key in buckets_to_check:
                        # 물리적으로 샘플링하고 버킷에서 완전히 삭제 (consume)
                        sampled = self.buckets[b_key].sample_and_remove(K_per_bucket)
                        for s in sampled:
                            pool_global_indices_list.append(s)
                            pool_curr_batch_idx_list.append(i)
                            
                # Python list를 Tensor로 변환하여 기존 Vectorized 로직과 완벽히 호환되게 함
                if pool_global_indices_list:
                    pool_global_indices = torch.tensor(pool_global_indices_list, dtype=torch.int64, device=keys.device)
                    pool_curr_batch_idx = torch.tensor(pool_curr_batch_idx_list, dtype=torch.int64, device=keys.device)
                else:
                    pool_global_indices = torch.tensor([], dtype=torch.int64, device=keys.device)
                    pool_curr_batch_idx = torch.tensor([], dtype=torch.int64, device=keys.device)
                
                if len(pool_global_indices) > 0:
                    unique_pool_indices, inverse_indices = torch.unique(pool_global_indices, return_inverse=True)
                    self.debug_metrics['lazy_pool_size'] = len(unique_pool_indices)
                    past_index_list = unique_pool_indices.tolist()
                    
                    if replay_buffer.store_on_gpu:
                        past_obs_raw = replay_buffer.obs_buffer[unique_pool_indices]
                    else:
                        idx_tensor = unique_pool_indices.cpu().long()
                        past_obs_raw = torch.from_numpy(replay_buffer.obs_buffer[idx_tensor])
                        
                    past_obs_pool = past_obs_raw.to(device=latent.device, dtype=latent.dtype) / 255.0
                    past_obs_pool = rearrange(past_obs_pool, "N H W C -> N 1 C H W")
                    
                    with torch.no_grad():
                        encoded = encode_fn(past_obs_pool)
                        if len(encoded) == 3:
                            past_latent_pool, past_logits_pool, past_latent_full_pool = encoded
                            past_latent_full_pool = past_latent_full_pool.squeeze(1)
                        else:
                            past_latent_pool, past_logits_pool = encoded
                            past_latent_full_pool = past_latent_pool
                        past_latent_pool, past_logits_pool = past_latent_pool.squeeze(1), past_logits_pool.squeeze(1)
                    
                    past_latent_pool = past_latent_pool.detach()
                    past_logits_pool = past_logits_pool.detach()
                    past_latent_full_pool = past_latent_full_pool.detach()
                    
                    # Lazy Update Memory
                    pool_new_keys = self._hash_keys(past_latent_pool).to(torch.int32)
                    self.item_latents[unique_pool_indices] = past_latent_pool.to(self.item_latents.dtype)
                    self.item_bucket[unique_pool_indices] = pool_new_keys
                    
                    # Re-match with original trigger keys
                    new_keys_for_flat_pool = pool_new_keys[inverse_indices]
                    required_keys = keys_int[pool_curr_batch_idx].to(torch.int32)
                    match_mask = (new_keys_for_flat_pool == required_keys)
                    
                    valid_global_indices_matched = pool_global_indices[match_mask]
                    valid_curr_batch_idx_matched = pool_curr_batch_idx[match_mask]
                    inverse_matched = inverse_indices[match_mask]
                    
                    # 100% GPU Vectorized Group-by & Limit (No CPU Sync!)
                    N_matches = valid_curr_batch_idx_matched.size(0)
                    if N_matches > 0:
                        # [M, N_matches] boolean matrix indicating which match belongs to which trigger
                        match_matrix = (valid_curr_batch_idx_matched.unsqueeze(0) == torch.arange(M, device=keys.device).unsqueeze(1))
                        
                        # Cumulative sum to count occurrences per trigger
                        match_counts = match_matrix.cumsum(dim=1)
                        
                        # Keep only the first max_past_samples for each trigger
                        keep_matrix = match_matrix & (match_counts <= self.max_past_samples)
                        
                        # Collapse back to 1D mask for the matches
                        keep_mask_1d = keep_matrix.any(dim=0)
                        
                        valid_global_indices = valid_global_indices_matched[keep_mask_1d]
                        valid_curr_batch_idx = valid_curr_batch_idx_matched[keep_mask_1d]
                        final_unique_idx = inverse_matched[keep_mask_1d]
                        
                        pair_count = len(valid_global_indices)
                        
                        if pair_count > 0:
                            past_latent = past_latent_pool[final_unique_idx]
                            past_logits_clean = past_logits_pool[final_unique_idx]
                            past_latent_full = past_latent_full_pool[final_unique_idx]
                            past_obs = past_obs_pool[final_unique_idx]
                    else:
                        pair_count = 0
                else:
                    pair_count = 0
            else:
                # Direct Fetch (No LazyRebuild)
                valid_global_indices_list = []
                valid_curr_batch_idx_list = []
                keys_list = keys_int.tolist()
                
                for i, key in enumerate(keys_list):
                    sampled = self.buckets[key].sample_and_remove(K)
                    for s in sampled:
                        valid_global_indices_list.append(s)
                        valid_curr_batch_idx_list.append(i)
                
                if valid_global_indices_list:
                    valid_global_indices = torch.tensor(valid_global_indices_list, dtype=torch.int64, device=keys.device)
                    valid_curr_batch_idx = torch.tensor(valid_curr_batch_idx_list, dtype=torch.int64, device=keys.device)
                else:
                    valid_global_indices = torch.tensor([], dtype=torch.int64, device=keys.device)
                    valid_curr_batch_idx = torch.tensor([], dtype=torch.int64, device=keys.device)
                
                pair_count = len(valid_global_indices)

        if pair_count == 0:
            self.debug_metrics['trigger_count'] = trigger_mask.sum().item()
            self.debug_metrics['pair_count'] = 0
            self._update_memory(obs, reward, latent, reward_mean, reward_std, td_error, value, aux_value_all, indexes)
            return distill_loss, distill_loss, latent.new_tensor(0.0), trigger_obs, trigger_mask, trigger_mask_original

        past_index_list = valid_global_indices.tolist()
        past_metric_list = self.item_metrics[valid_global_indices].tolist()
        past_old_latent_tensor = self.item_latents[valid_global_indices]

        curr_logits = logits_trigger[valid_curr_batch_idx]
        curr_obs_paired = trigger_obs[valid_curr_batch_idx]
        curr_metric_list = list(metric_trigger[valid_curr_batch_idx].unbind(0))

        if not self.lazy_rebuild_enable:
            if replay_buffer.store_on_gpu:
                past_obs_raw = replay_buffer.obs_buffer[valid_global_indices]
            else:
                idx_tensor = valid_global_indices.cpu().long()
                past_obs_raw = torch.from_numpy(replay_buffer.obs_buffer[idx_tensor])
                
            past_obs = past_obs_raw.to(device=latent.device, dtype=latent.dtype) / 255.0
            past_obs = rearrange(past_obs, "N H W C -> N 1 C H W")
            
            with torch.no_grad():
                encoded = encode_fn(past_obs)
                if len(encoded) == 3:
                    past_latent, past_logits_clean, past_latent_full = encoded
                    past_latent_full = past_latent_full.squeeze(1)
                else:
                    past_latent, past_logits_clean = encoded
                    past_latent_full = past_latent
                past_latent, past_logits_clean = past_latent.squeeze(1), past_logits_clean.squeeze(1)

            past_latent = past_latent.detach()
            past_logits_clean = past_logits_clean.detach()
            past_latent_full = past_latent_full.detach()

        # --- Global Rebuild Check ---
        if self.enable_global_rebuild and global_step is not None:
            if global_step - self.last_rebuild_step >= self.rebuild_cooldown:
                latent_diff = 1.0 - F.cosine_similarity(past_latent, past_old_latent_tensor, dim=-1)
                avg_latent_diff = latent_diff.mean().item()
                self.debug_metrics['avg_latent_diff'] = avg_latent_diff
                
                if avg_latent_diff > self.rebuild_threshold:
                    print(f"[MacroLoss] Rebuilding all hash buckets... (step: {global_step}, diff: {avg_latent_diff:.4f})")
                    
                    extra_items = list(zip(past_index_list, past_metric_list, past_old_latent_tensor.cpu().unbind(0)))
                    self._rebuild_all_hash_buckets(replay_buffer, encode_fn, latent_dtype=latent.dtype, extra_items=extra_items)
                    self.last_rebuild_step = global_step
                    
                    self.debug_metrics['trigger_count'] = trigger_mask.sum().item()
                    self.debug_metrics['pair_count'] = pair_count
                    self.debug_metrics['rebuild_triggered'] = 1.0

                    self._update_memory(obs, reward, latent, reward_mean, reward_std, td_error, value, aux_value_all, indexes)
                    return distill_loss, distill_loss, latent.new_tensor(0.0), trigger_obs, trigger_mask, trigger_mask_original

        # --- Multi-Masking Cutout Logic ---
        use_cutout_now = self.use_cutout and self.num_masks > 0
        if use_cutout_now and (self.cutout_disable_after < 0 or global_step <= self.cutout_disable_after):
            M_cut = self.num_masks
            N_cut = pair_count
            
            curr_obs_flat = curr_obs_paired
            past_obs_flat = past_obs.squeeze(1)
            _, C, H, W = curr_obs_flat.shape
            
            curr_obs_M = curr_obs_flat.unsqueeze(1).expand(-1, M_cut, -1, -1, -1).reshape(N_cut * M_cut, C, H, W)
            past_obs_M = past_obs_flat.unsqueeze(1).expand(-1, M_cut, -1, -1, -1).reshape(N_cut * M_cut, C, H, W)
            
            from utils import batch_cutout_mask
            masks = batch_cutout_mask(N_cut * M_cut, H, W, curr_obs_paired.device, p=self.cutout_p, scale=self.cutout_scale, ratio=self.cutout_ratio)
            masks_expanded = masks.expand(-1, C, -1, -1)
            
            cutout_val_tensor = torch.tensor(self.cutout_value, dtype=curr_obs_paired.dtype, device=curr_obs_paired.device)
            curr_obs_masked = torch.where(masks_expanded, cutout_val_tensor, curr_obs_M)
            past_obs_masked = torch.where(masks_expanded, cutout_val_tensor, past_obs_M)
            
            curr_obs_masked = curr_obs_masked.unsqueeze(1)
            past_obs_masked = past_obs_masked.unsqueeze(1)
            
            encoded_curr = encode_fn(curr_obs_masked)
            curr_logits_M = encoded_curr[1].squeeze(1) if len(encoded_curr) == 3 else encoded_curr[1].squeeze(1)
            
            with torch.no_grad():
                encoded_past = encode_fn(past_obs_masked)
                past_logits_M = encoded_past[1].squeeze(1) if len(encoded_past) == 3 else encoded_past[1].squeeze(1)
            past_logits_M = past_logits_M.detach()
            
            target_curr_logits = curr_logits_M
            target_past_logits = past_logits_M
        else:
            target_curr_logits = curr_logits
            target_past_logits = past_logits_clean
            M = 1
            N = pair_count
        # ----------------------------------

        curr_metric = torch.stack(curr_metric_list, dim=0)

        if self.diff_type == "aux_value":
            if aux_value_fn is None:
                raise ValueError("aux_value_fn must be provided when diff_type is 'aux_value'")
            past_aux_pred = aux_value_fn(past_latent_full).squeeze(-1)
            # [수정] SymLog → Linear 변환 후 차이 계산 (SymLog 압축으로 인한 후반부 가중치 소실 방지)
            curr_linear = symexp(curr_metric.detach())
            past_linear = symexp(past_aux_pred.detach())
            metric_diff = torch.abs(curr_linear - past_linear)
            if aux_std is None:
                raise ValueError("aux_std must be provided when diff_type is 'aux_value'")
            sigma = aux_std
        elif self.diff_type == "td_error":
            past_metric = torch.tensor(past_metric_list, device=latent.device, dtype=curr_metric.dtype)
            metric_diff = torch.abs(curr_metric.detach() - past_metric)
            sigma = torch.sqrt(self.td_error_var + 1e-8)
        elif self.diff_type == "reward":
            past_metric = torch.tensor(past_metric_list, device=latent.device, dtype=curr_metric.dtype)
            metric_diff = torch.abs(curr_metric.detach() - past_metric)
            sigma = reward_std
        else:
            past_metric = torch.tensor(past_metric_list, device=latent.device, dtype=curr_metric.dtype)
            metric_diff = torch.abs(curr_metric.detach() - past_metric)
            sigma = torch.tensor(1e-5, device=latent.device)

        # 설정된 시그마 비율보다 적은 차이는 0으로 무시하여 불필요하게 밀어내는 현상 방지
        metric_diff_zero_mask = metric_diff < self.contrastive_sigma_ratio * sigma
        self.debug_metrics['metric_diff_zero_ratio'] = metric_diff_zero_mask.float().mean().item()
        metric_diff = torch.where(metric_diff_zero_mask, torch.zeros_like(metric_diff), metric_diff)

        cosine_sim = F.cosine_similarity(target_curr_logits, target_past_logits, dim=-1, eps=1e-8)
        self.debug_metrics['cosine_sim_mean'] = cosine_sim.mean().item()
        self.debug_metrics['pair_count'] = pair_count
        self.debug_metrics['trigger_count'] = trigger_mask.sum().item()
        
        if self.distance_metric == "dbc":
            # DBC Style: 가치 차이에 비례한 거리 손실 (MSE with Target Cosine)
            normalized_diff = metric_diff / (sigma + 1e-8)
            target_cos = torch.clamp(1.0 - self.dbc_alpha * normalized_diff, min=-1.0, max=1.0)
            
            if use_cutout_now and (self.cutout_disable_after < 0 or global_step <= self.cutout_disable_after):
                target_cos_M = target_cos.unsqueeze(1).expand(-1, M).reshape(N * M)
                contrastive_loss_raw = F.mse_loss(cosine_sim, target_cos_M.detach(), reduction='none')
                contrastive_loss_raw = contrastive_loss_raw.view(N, M)
                contrastive_loss_min, _ = torch.min(contrastive_loss_raw, dim=1)
                contrastive_loss = contrastive_loss_min.mean() * self.loss_scale
            else:
                contrastive_loss = F.mse_loss(cosine_sim, target_cos.detach()) * self.loss_scale
        else:
            # Margin Style: 기존의 무조건적 척력(Repulsion) 기반
            with torch.no_grad():
                # [수정] running_max_metric도 Linear 스케일로 관리 (metric_diff와 스케일 일치)
                current_max = torch.max(torch.abs(symexp(curr_metric.detach())))
                self.running_max_metric.copy_(torch.max(self.running_max_metric, current_max.float()))
                
            metric_diff_scaled = metric_diff / self.running_max_metric
            margin = self.margin
            self.debug_metrics['margin_passed_ratio'] = (cosine_sim > margin).float().mean().item()
            
            if use_cutout_now and (self.cutout_disable_after < 0 or global_step <= self.cutout_disable_after):
                metric_diff_M = metric_diff_scaled.unsqueeze(1).expand(-1, M).reshape(N * M)
                contrastive_loss_raw = metric_diff_M * F.relu(cosine_sim - margin)
                contrastive_loss_raw = contrastive_loss_raw.view(N, M)
                contrastive_loss_min, _ = torch.min(contrastive_loss_raw, dim=1)
                contrastive_loss = contrastive_loss_min.mean() * self.loss_scale
            else:
                contrastive_loss = (metric_diff_scaled * F.relu(cosine_sim - margin)).mean() * self.loss_scale
            
        loss = contrastive_loss + distill_loss

        self._update_memory(obs, reward, latent, reward_mean, reward_std, td_error, value, aux_value_all, indexes, aux_mean, aux_std)
        return loss, distill_loss, contrastive_loss, trigger_obs, trigger_mask, trigger_mask_original
