import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.ops.triton.layer_norm import RMSNorm
from torch.distributions import OneHotCategorical, Normal
# import torchvision.transforms as T
import kornia.augmentation as K
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
from torch.cuda.amp import autocast
from sub_models.laprop import LaProp
from pytorch_warmup import LinearWarmup
# from nfnets import AGC

from sub_models.functions_losses import SymLogTwoHotLoss
from sub_models.attention_blocks import get_subsequent_mask_with_batch_length, get_subsequent_mask
from sub_models.transformer_model import StochasticTransformerKVCache
from mamba_ssm import MambaWrapperModel, MambaConfig, InferenceParams, update_graph_cache
import agents
from line_profiler import profile
from torch.distributions.independent import Independent
import numpy as np
from tools import weight_init
import cv2
try:
    from sub_models.macro_loss import MacroLoss
except ImportError:
    MacroLoss = None
    
class Encoder(nn.Module):
    def __init__(self, depth=128, mults=(1, 2, 4, 2), norm='rms', act='SiLU', kernel=4, padding='same',
                 first_stride=True, input_size=(3, 64, 64), dtype=None, device=None) -> None:
        super().__init__()
        act = getattr(nn, act)
        self.depths = [depth * mult for mult in mults]
        self.kernel = kernel
        self.stride = 2
        self.padding = (kernel - 1) // 2 if padding == 'same' else padding

        backbone = []
        current_channels, current_height, current_width = input_size

        # Define convolutional layers for image inputs
        for i, depth in enumerate(self.depths):
            stride = 1 if i == 0 and first_stride else self.stride
            conv = nn.Conv2d(in_channels=current_channels, out_channels=depth, kernel_size=kernel, stride=stride,
                             padding=self.padding, dtype=dtype, device=device)
            backbone.append(conv)
            backbone.append(nn.BatchNorm2d(depth, dtype=dtype, device=device))
            backbone.append(act())

            current_height, current_width = self._compute_output_dim(current_height, current_width, kernel, stride,
                                                                     self.padding)
            current_channels = depth

        self.backbone = nn.Sequential(*backbone)
        self.backbone.apply(weight_init)
        self.last_channels = self.depths[-1]
        self.output_dim = (self.last_channels, current_height, current_width)
        self.output_flatten_dim = self.last_channels * current_height * current_width

    def _compute_output_dim(self, height, width, kernel_size, stride, padding):
        new_height = (height - kernel_size + 2 * padding) // stride + 1
        new_width = (width - kernel_size + 2 * padding) // stride + 1
        return new_height, new_width

    def forward(self, x):
        batch_size = x.shape[0]
        x = rearrange(x, "B L C H W -> (B L) C H W")
        x = self.backbone(x)
        x = rearrange(x, "(B L) C H W -> B L (C H W)", B=batch_size)
        return x


class Decoder(nn.Module):
    def __init__(self, stoch_dim, depth=128, mults=(1, 2, 4, 2), norm='rms', act='SiLU', kernel=4, padding='same',
                 first_stride=True, last_output_dim=(256, 4, 4), input_size=(3, 64, 64), cnn_sigmoid=False, dtype=None,
                 device=None) -> None:
        super().__init__()
        act = getattr(nn, act)
        self.depths = [depth * mult for mult in mults]
        self.kernel = kernel
        self.stride = 2
        self.padding = (kernel - 1) // 2 if padding == 'same' else padding
        self.output_padding = self.stride // 2 if padding == 'same' else 0
        self._cnn_sigmoid = cnn_sigmoid

        backbone = []
        # stem
        backbone.append(
            nn.Linear(stoch_dim, last_output_dim[0] * last_output_dim[1] * last_output_dim[2], bias=True, dtype=dtype,
                      device=device))
        backbone.append(Rearrange('B L (C H W) -> (B L) C H W', C=last_output_dim[0], H=last_output_dim[1]))
        backbone.append(nn.BatchNorm2d(last_output_dim[0], dtype=dtype, device=device))
        backbone.append(act())
        # residual_layer
        # backbone.append(ResidualStack(last_channels, 1, last_channels//4))
        # layers
        current_channels, current_height, current_width = last_output_dim
        # Define convolutional layers for image inputs
        for i, depth in reversed(list(enumerate(self.depths[:-1]))):
            conv = nn.ConvTranspose2d(in_channels=current_channels, out_channels=depth, kernel_size=kernel,
                                      stride=self.stride, padding=self.padding, output_padding=self.output_padding,
                                      dtype=dtype, device=device)
            backbone.append(conv)
            backbone.append(nn.BatchNorm2d(depth, dtype=dtype, device=device))
            backbone.append(act())
            current_height, current_width = self._compute_transposed_output_dim(current_height, current_width, kernel,
                                                                                self.stride, self.padding,
                                                                                self.output_padding)
            current_channels = depth

        stride = 1 if first_stride else self.stride
        output_padding = 0 if i == 0 else self.output_padding

        backbone.append(
            nn.ConvTranspose2d(
                in_channels=self.depths[0],
                out_channels=input_size[0],
                kernel_size=kernel,
                stride=stride,
                padding=self.padding,
                output_padding=output_padding,
                dtype=dtype, device=device
            )
        )

        current_height, current_width = self._compute_transposed_output_dim(
            current_height, current_width, kernel,
            stride, self.padding, output_padding
        )
        self.final_output_dim = (input_size[0], current_height, current_width)
        self.backbone = nn.Sequential(*backbone)
        self.backbone.apply(weight_init)

    def _compute_transposed_output_dim(self, height, width, kernel_size, stride, padding, output_padding):
        new_height = (height - 1) * stride - 2 * padding + kernel_size + output_padding
        new_width = (width - 1) * stride - 2 * padding + kernel_size + output_padding
        return new_height, new_width

    def forward(self, sample):
        batch_size = sample.shape[0]
        obs_hat = self.backbone(sample)
        obs_hat = rearrange(obs_hat, "(B L) C H W -> B L C H W", B=batch_size)
        if self._cnn_sigmoid:
            obs_hat = F.sigmoid(obs_hat)
        else:
            obs_hat += 0.5
        return obs_hat

class DistHead(nn.Module):
    '''
    Dist: abbreviation of distribution
    '''
    def __init__(self, image_feat_dim, hidden_state_dim, categorical_dim, class_dim, unimix_ratio=0.01, dtype=None, device=None) -> None:
        super().__init__()
        self.stoch_dim = categorical_dim
        self.post_head = nn.Linear(image_feat_dim, categorical_dim*class_dim, dtype=dtype, device=device)
        self.prior_head = nn.Linear(hidden_state_dim, categorical_dim*class_dim, dtype=dtype, device=device)
        self.unimix_ratio = unimix_ratio
        self.dtype=dtype
        self.device=device

    def unimix(self, logits, mixing_ratio=0.01):
        # uniform noise mixing
        if mixing_ratio > 0:
            probs = F.softmax(logits, dim=-1)
            mixed_probs = mixing_ratio * torch.ones_like(probs) / self.stoch_dim + (1-mixing_ratio) * probs
            logits = torch.log(mixed_probs).to(dtype=logits.dtype)
        return logits

    def forward_post(self, x):
        logits = self.post_head(x)
        logits = rearrange(logits, "B L (K C) -> B L K C", K=self.stoch_dim)
        logits = self.unimix(logits, self.unimix_ratio)
        return logits

    def forward_prior(self, x):
        logits = self.prior_head(x)
        logits = rearrange(logits, "B L (K C) -> B L K C", K=self.stoch_dim)
        logits = self.unimix(logits, self.unimix_ratio)
        return logits



    

class RewardHead(nn.Module):
    def __init__(self, num_classes, inp_dim, hidden_units, act, layer_num, dtype=None, device=None) -> None:
        super().__init__()
        act = getattr(nn, act)

        # Create the backbone with dynamic number of layers
        layers = []
        for _ in range(layer_num):
            layers.append(nn.Linear(inp_dim, hidden_units, bias=True, dtype=dtype, device=device))
            layers.append(RMSNorm(hidden_units, dtype=dtype, device=device))
            layers.append(act())

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_units, num_classes, dtype=dtype, device=device)

    def forward(self, feat):
        feat = self.backbone(feat)
        reward = self.head(feat)
        return reward




class TerminationHead(nn.Module):
    def __init__(self, inp_dim, hidden_units, act, layer_num, dtype=None, device=None) -> None:
        super().__init__()
        act = getattr(nn, act)

        # Create the backbone with dynamic number of layers
        layers = []
        for _ in range(layer_num):
            layers.append(nn.Linear(inp_dim, hidden_units, bias=True, dtype=dtype, device=device))
            layers.append(RMSNorm(hidden_units, dtype=dtype, device=device))
            layers.append(act())

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_units, 1, dtype=dtype, device=device)

    def forward(self, feat):
        feat = self.backbone(feat)
        termination = self.head(feat)
        termination = termination.squeeze(-1)  # remove last 1 dim
        return termination

class AuxValueModule(nn.Module):
    def __init__(self, latent_dim, dropout_p=0.2):
        super().__init__()
        layers = [
            nn.Dropout(p=dropout_p),
            nn.Linear(latent_dim, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, 1)
        ]
        self.net = nn.Sequential(*layers)
        
        import copy
        self.slow_net = copy.deepcopy(self.net)
        for param in self.slow_net.parameters():
            param.requires_grad = False
            
        self.register_buffer("aux_value_mean", torch.tensor(0.0, dtype=torch.float64))
        self.register_buffer("aux_value_var", torch.tensor(1.0, dtype=torch.float64))
        self.register_buffer("aux_value_count", torch.tensor(1e-4, dtype=torch.float64))

    @torch.no_grad()
    def update_slow_target(self, decay=0.98):
        for slow_param, param in zip(self.slow_net.parameters(), self.net.parameters()):
            slow_param.data.copy_(slow_param.data * decay + param.data * (1 - decay))

    def update_welford(self, aux_value, valid_mask):
        valid_aux = aux_value[valid_mask].to(torch.float64)
        if valid_aux.numel() == 0:
            return self.aux_value_mean.to(torch.float32), self.aux_value_var.to(torch.float32)

        batch_mean = torch.mean(valid_aux)
        batch_var = torch.var(valid_aux, unbiased=False)
        batch_count = torch.tensor(valid_aux.numel(), dtype=torch.float64, device=aux_value.device)

        tot_count = self.aux_value_count + batch_count
        self.aux_value_count.copy_(tot_count)

        decay = 0.999
        alpha = torch.clamp(batch_count / tot_count, min=1.0 - decay, max=1.0)
        old_mean = self.aux_value_mean.clone()
        new_mean = (1.0 - alpha) * old_mean + alpha * batch_mean
        new_var = (1.0 - alpha) * self.aux_value_var + alpha * batch_var + alpha * (1.0 - alpha) * ((batch_mean - old_mean) ** 2)

        self.aux_value_mean.copy_(new_mean)
        self.aux_value_var.copy_(new_var)
        return self.aux_value_mean.to(torch.float32), self.aux_value_var.to(torch.float32)

    def forward(self, x):
        return self.net(x)


class MSELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, obs_hat, obs, reduction='mean'):
        distance = (obs_hat - obs)**2
        loss = reduce(distance, "B L C H W -> B L", "sum")
        if reduction == 'none':
            return loss  # [B, L] shape for per-frame weighting
        return loss.mean()


class CategoricalKLDivLossWithFreeBits(nn.Module):
    def __init__(self, free_bits) -> None:
        super().__init__()
        self.free_bits = free_bits

    def forward(self, p_logits, q_logits):
        p_dist = OneHotCategorical(logits=p_logits)
        q_dist = OneHotCategorical(logits=q_logits)
        kl_div = torch.distributions.kl.kl_divergence(p_dist, q_dist)
        kl_div = reduce(kl_div, "B L D -> B L", "sum")
        kl_div = kl_div.mean()
        real_kl_div = kl_div
        kl_div = torch.max(torch.ones_like(kl_div)*self.free_bits, kl_div)
        return kl_div, real_kl_div


class WorldModel(nn.Module):
    def __init__(self, action_dim, config, device, is_discrete=True):
        super().__init__()
        self.config = config
        self.hidden_state_dim = config.Models.WorldModel.HiddenStateDim
        self.final_feature_width = config.Models.WorldModel.Transformer.FinalFeatureWidth
        self.categorical_dim = config.Models.WorldModel.CategoricalDim
        self.class_dim = config.Models.WorldModel.ClassDim
        self.stoch_flattened_dim = self.categorical_dim*self.class_dim
        self.use_amp = config.BasicSettings.Use_amp
        self.use_cg = config.BasicSettings.Use_cg
        self.tensor_dtype = torch.bfloat16 if self.use_amp and not self.use_cg else config.Models.WorldModel.dtype
        self.save_every_steps = config.JointTrainAgent.SaveEverySteps
        self.imagine_batch_size = -1
        self.imagine_batch_length = -1
        self.action_dim = action_dim
        self.is_discrete = is_discrete
        self.device = device # Maybe it's not needed
        self.model = config.Models.WorldModel.Backbone
        self.max_grad_norm = config.Models.WorldModel.Max_grad_norm  
        max_seq_length = max(config.JointTrainAgent.BatchLength, 
                             config.JointTrainAgent.ImagineContextLength + config.JointTrainAgent.ImagineBatchLength, 
                             config.JointTrainAgent.RealityContextLength)
        self.encoder = Encoder(
            depth=config.Models.WorldModel.Encoder.Depth,
            mults=config.Models.WorldModel.Encoder.Mults, 
            norm=config.Models.WorldModel.Encoder.Norm, 
            act=config.Models.WorldModel.Act, 
            kernel=config.Models.WorldModel.Encoder.Kernel,
            padding=config.Models.WorldModel.Encoder.Padding,
            input_size=config.Models.WorldModel.Encoder.InputSize,
            dtype=config.Models.WorldModel.dtype, device=device
        )
        if self.model == 'Transformer':
            self.sequence_model = StochasticTransformerKVCache(
                stoch_dim=self.stoch_flattened_dim,
                action_dim=action_dim,
                feat_dim=self.hidden_state_dim,
                num_layers=config.Models.WorldModel.Transformer.NumLayers,
                num_heads=config.Models.WorldModel.Transformer.NumHeads,
                max_length=max_seq_length,
                dropout=config.Models.WorldModel.Dropout
            )
        elif self.model == 'Mamba':
            mamba_config = MambaConfig(
                d_model=self.hidden_state_dim, 
                d_intermediate=config.Models.WorldModel.Mamba.d_intermediate,
                n_layer=config.Models.WorldModel.Mamba.n_layer,
                stoch_dim=self.stoch_flattened_dim,
                action_dim=action_dim,
                dropout_p=config.Models.WorldModel.Dropout,
                ssm_cfg={
                    'd_state': config.Models.WorldModel.Mamba.ssm_cfg.d_state,
                    }
                )                                
            self.sequence_model = MambaWrapperModel(mamba_config)
        elif self.model == 'Mamba2':
            mamba_config = MambaConfig(
                d_model=self.hidden_state_dim, 
                d_intermediate=config.Models.WorldModel.Mamba.d_intermediate,
                n_layer=config.Models.WorldModel.Mamba.n_layer,
                stoch_dim=self.stoch_flattened_dim,
                action_dim=action_dim,
                is_discrete=is_discrete,
                dropout_p=config.Models.WorldModel.Dropout,
                ssm_cfg={
                    'd_state': config.Models.WorldModel.Mamba.ssm_cfg.d_state, 
                    'layer': 'Mamba2'}
                )
            self.sequence_model = MambaWrapperModel(mamba_config)                      
        else:
            raise ValueError(f"Unknown dynamics model: {self.model}")               
        

        self.dist_head = DistHead(
            image_feat_dim=self.encoder.output_flatten_dim,
            hidden_state_dim=self.hidden_state_dim,
            categorical_dim=self.categorical_dim,
            class_dim=self.class_dim,
            unimix_ratio=config.Models.WorldModel.Unimix_ratio,
            dtype=config.Models.WorldModel.dtype, device=device
        )      
        self.image_decoder = Decoder(
            stoch_dim=self.stoch_flattened_dim,
            depth=config.Models.WorldModel.Decoder.Depth, 
            mults=config.Models.WorldModel.Decoder.Mults, 
            norm=config.Models.WorldModel.Decoder.Norm, 
            act=config.Models.WorldModel.Act, 
            kernel=config.Models.WorldModel.Decoder.Kernel, 
            padding=config.Models.WorldModel.Decoder.Padding, 
            first_stride=config.Models.WorldModel.Decoder.FirstStrideOne, 
            last_output_dim=self.encoder.output_dim,
            input_size=config.Models.WorldModel.Decoder.InputSize,
            cnn_sigmoid=config.Models.WorldModel.Decoder.FinalLayerSigmoid,
            dtype=config.Models.WorldModel.dtype, device=device
        )
        
        self.reward_decoder = RewardHead(
            num_classes=255,
            inp_dim=self.hidden_state_dim,
            hidden_units=config.Models.WorldModel.Reward.HiddenUnits,
            act=config.Models.WorldModel.Act,
            layer_num=config.Models.WorldModel.Reward.LayerNum,
            dtype=config.Models.WorldModel.dtype, device=device
        )
        self.reward_decoder.apply(weight_init)
        self.termination_decoder = TerminationHead(
            inp_dim=self.hidden_state_dim,
            hidden_units=config.Models.WorldModel.Termination.HiddenUnits,
            act=config.Models.WorldModel.Act,
            layer_num=config.Models.WorldModel.Termination.LayerNum,
            dtype=config.Models.WorldModel.dtype, device=device
        )
        self.termination_decoder.apply(weight_init)
 
        self.mse_loss_func = MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_with_logits_loss_func = nn.BCEWithLogitsLoss()
        self.symlog_twohot_loss_func = SymLogTwoHotLoss(num_classes=255, lower_bound=-20, upper_bound=20)
        self.categorical_kl_div_loss = CategoricalKLDivLossWithFreeBits(free_bits=1)
        
        self.aux_cfg = getattr(config.Models.WorldModel, 'AuxValueNet', {})
        self.aux_enable = getattr(self.aux_cfg, 'Enable', False)
        if self.aux_enable:
            self.aux_value_module = AuxValueModule(self.stoch_flattened_dim).to(device)
        else:
            self.aux_value_module = None
            
        if config.Models.WorldModel.Optimiser == 'Laprop':
            self.optimizer = LaProp(self.parameters(), lr=config.Models.WorldModel.Laprop.LearningRate, eps=config.Models.WorldModel.Laprop.Epsilon, weight_decay=config.Models.WorldModel.Weight_decay)
        elif config.Models.WorldModel.Optimiser == 'Adam':
            self.optimizer = torch.optim.AdamW(self.parameters(), lr=config.Models.WorldModel.Adam.LearningRate, weight_decay=config.Models.WorldModel.Weight_decay)
        else:
            raise ValueError(f"Unknown optimiser: {config.Models.WorldModel.Optimiser}")
        # self.optimizer = AGC(self.parameters(), self.optimizer)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda step: 1.0)
        self.warmup_scheduler = LinearWarmup(self.optimizer, warmup_period=config.Models.WorldModel.Warmup_steps)
        # GradScaler is incompatible with bfloat16 autocast and causes artificial gradient explosions.
        self.scaler = torch.cuda.amp.GradScaler(enabled=False)
        
        macro_cfg = getattr(config.Models.WorldModel, 'MacroLoss', {})
        macro_enable = getattr(macro_cfg, 'Enable', False)
        
        dyn_cfg = getattr(config.Models.WorldModel, 'DynWeighting', {})
        self.dyn_cfg = dyn_cfg
        dyn_weighting_enable = getattr(dyn_cfg, 'Enable', False)
        
        # MacroLoss 모듈은 contrastive loss가 켜져 있을 때만 생성
        if MacroLoss is not None and macro_enable:
            buffer_max_length = getattr(config.BasicSettings, 'BufferMaxLength', 100000)
            self.macro_loss = MacroLoss(
                latent_dim=self.stoch_flattened_dim,
                config=macro_cfg,
                dyn_config=dyn_cfg,
                full_latent_dim=self.stoch_flattened_dim,
                buffer_max_length=buffer_max_length
            ).to(device)
            # Add macro_loss parameters to the optimizer if needed, or let it have its own? 
            # In grad_research it was part of world_model.parameters(). 
            # Since self.macro_loss is a submodule, it will be included if initialized before self.optimizer!
            # Wait, self.optimizer is initialized on line 374! 
            # I need to re-initialize the optimizer to include MacroLoss parameters.
            self.optimizer.add_param_group({'params': self.macro_loss.parameters()})
        else:
            self.macro_loss = None

    @profile
    def encode_obs(self, obs, sample_mode="random_sample"):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            if torch.isnan(obs).any() or torch.isinf(obs).any():
                print(f"[DEBUG encode_obs] NaN/Inf in obs! nan={torch.isnan(obs).sum().item()}, inf={torch.isinf(obs).sum().item()}, shape={obs.shape}, dtype={obs.dtype}")
            embedding = self.encoder(obs)
            if torch.isnan(embedding).any() or torch.isinf(embedding).any():
                print(f"[DEBUG encode_obs] NaN/Inf in embedding! nan={torch.isnan(embedding).sum().item()}, inf={torch.isinf(embedding).sum().item()}, shape={embedding.shape}, dtype={embedding.dtype}")
            post_logits = self.dist_head.forward_post(embedding)
            if torch.isnan(post_logits).any() or torch.isinf(post_logits).any():
                print(f"[DEBUG encode_obs] NaN/Inf in post_logits! nan={torch.isnan(post_logits).sum().item()}, inf={torch.isinf(post_logits).sum().item()}, shape={post_logits.shape}, dtype={post_logits.dtype}")
                print(f"[DEBUG encode_obs] post_logits stats: min={post_logits[~torch.isnan(post_logits)].min().item():.4f}, max={post_logits[~torch.isnan(post_logits)].max().item():.4f}")
            sample = self.stright_throught_gradient(post_logits, sample_mode=sample_mode)
            flattened_sample = self.flatten_sample(sample)
        return flattened_sample

    def compute_dynamics_weights(self, latent_full, values_sq=None):
        """Value-diff의 z-score에 비례하는 dynamics loss 가중치를 계산합니다."""
        dyn_weighting_enable = bool(self.dyn_cfg.get("Enable", False))
        if not dyn_weighting_enable:
            return None, {}
            
        dyn_use_aux_value_net = bool(self.dyn_cfg.get("UseAuxValueNet", True))
        dyn_weighting_scale = float(self.dyn_cfg.get("Scale", 25.0))
        dyn_clip_mode = str(self.dyn_cfg.get("ClipMode", "fixed")).lower()
        dyn_weighting_max = float(self.dyn_cfg.get("MaxWeight", 100.0))
        dyn_clip_percentile = float(self.dyn_cfg.get("ClipPercentile", 99.0))
        dyn_z_threshold = float(self.dyn_cfg.get("ZScoreThreshold", 0.0))
        dyn_trigger_mode = str(self.dyn_cfg.get("TriggerMode", "both")).lower()
        
        from sub_models.functions_losses import symexp
        with torch.no_grad():
            if dyn_use_aux_value_net and self.aux_value_module is not None:
                aux_val_symlog = self.aux_value_module(latent_full.detach()).squeeze(-1)
                aux_val_linear = symexp(aux_val_symlog)
            else:
                if values_sq is None:
                    raise ValueError("UseAuxValueNet is false but values_sq is not provided to compute_dynamics_weights")
                aux_val_linear = values_sq.detach().to(torch.float32)
                align_mask = torch.ones_like(aux_val_linear, dtype=torch.bool)
                if align_mask.dim() >= 2:
                    align_mask[..., 0] = False
                    align_mask[..., -1] = False
                if self.aux_value_module is not None:
                    self.aux_value_module.update_welford(aux_val_linear, align_mask)
            
            val_diff = torch.zeros_like(aux_val_linear)
            val_diff_raw = aux_val_linear[:, 1:] - aux_val_linear[:, :-1]
            val_diff[:, 1:] = torch.abs(val_diff_raw)
            
            if self.aux_value_module is not None:
                aux_std = torch.sqrt(self.aux_value_module.aux_value_var.to(torch.float32) + 1e-8)
            else:
                # Fallback if AuxValueModule is completely disabled but DynWeighting uses values_sq
                aux_std = torch.std(val_diff) + 1e-8
                
            z_score = val_diff / aux_std
            
            if dyn_trigger_mode == "increase":
                z_score[:, 1:][val_diff_raw <= 0] = 0.0
            elif dyn_trigger_mode == "decrease":
                z_score[:, 1:][val_diff_raw >= 0] = 0.0
                
            z_clamped = z_score.clamp(min=0.0)
            weights = 1.0 + dyn_weighting_scale * z_clamped
            
            if dyn_clip_mode == "percentile":
                clip_value = torch.quantile(weights.float(), dyn_clip_percentile / 100.0).item()
            else:
                clip_value = dyn_weighting_max
                
            weights = weights.clamp(max=clip_value)
            
            # Apply threshold: if z_score is below threshold, weight is 1.0
            weights[z_score < dyn_z_threshold] = 1.0
            
            metrics = {
                "val_diff_mean": val_diff[:, 1:].mean().item(),
                "val_diff_std": val_diff[:, 1:].std().item() if val_diff[:, 1:].numel() > 1 else 0.0,
                "weight_95_percentile": torch.quantile(weights.float(), 0.95).item(),
                "weight_99_percentile": torch.quantile(weights.float(), 0.99).item(),
                "weight_clip_ratio": (weights >= clip_value).float().mean().item(),
                "weight_clip_value": clip_value
            }
        
        return weights, metrics

    def encode_for_macro(self, obs):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            embedding = self.encoder(obs)
            post_logits = self.dist_head.forward_post(embedding)
            sample = self.stright_throught_gradient(post_logits, sample_mode="random_sample")
            flattened_sample = self.flatten_sample(sample)
            flattened_logits = rearrange(post_logits, "B L K C -> B L (K C)")
        return flattened_sample, flattened_logits, flattened_sample

    @profile
    def calc_last_dist_feat(self, latent, action, inference_params=None):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            if self.model == 'Transformer':
                temporal_mask = get_subsequent_mask(latent)
                dist_feat = self.sequence_model(latent, action, temporal_mask)
            else:
                dist_feat = self.sequence_model(latent, action, inference_params)
            last_dist_feat = dist_feat[:, -1:]
            prior_logits = self.dist_head.forward_prior(last_dist_feat)
            prior_sample = self.stright_throught_gradient(prior_logits, sample_mode="random_sample")
            prior_flattened_sample = self.flatten_sample(prior_sample)
        return prior_flattened_sample, last_dist_feat
    @profile
    def calc_last_post_feat(self, latent, action, current_obs, inference_params=None):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            embedding = self.encoder(current_obs)
            post_logits = self.dist_head.forward_post(embedding)
            sample = self.stright_throught_gradient(post_logits, sample_mode="random_sample")
            flattened_sample = self.flatten_sample(sample)            
            if self.model == 'Transformer':
                temporal_mask = get_subsequent_mask(latent)
                dist_feat = self.sequence_model(latent, action, temporal_mask)
            else:
                dist_feat = self.sequence_model(latent, action, inference_params)
            last_dist_feat = dist_feat[:, -1:]
            shifted_feat = last_dist_feat
            x = torch.cat((shifted_feat, flattened_sample), -1)
            post_feat = self._obs_out_layers(x)
            post_stat = self._obs_stat_layer(post_feat)
            post_logits = post_stat.reshape(list(post_stat.shape[:-1]) + [self.categorical_dim, self.categorical_dim])
            post_sample = self.stright_throught_gradient(post_logits, sample_mode="random_sample")
            post_flattened_sample = self.flatten_sample(post_sample)            

        return post_flattened_sample, post_feat    
    @profile
    # only called when using Transformer
    def predict_next(self, last_flattened_sample, action, log_video=True):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            dist_feat = self.sequence_model.forward_with_kv_cache(last_flattened_sample, action)
            prior_logits = self.dist_head.forward_prior(dist_feat)

            # decoding
            prior_sample = self.stright_throught_gradient(prior_logits, sample_mode="random_sample")
            prior_flattened_sample = self.flatten_sample(prior_sample)
            if log_video:
                obs_hat = self.image_decoder(prior_flattened_sample)
            else:
                obs_hat = None
            reward_hat = self.reward_decoder(dist_feat)
            reward_hat = self.symlog_twohot_loss_func.decode(reward_hat)
            termination_hat = self.termination_decoder(dist_feat)
            termination_hat = termination_hat > 0

        return obs_hat, reward_hat, termination_hat, prior_flattened_sample, dist_feat
    @profile
    def stright_throught_gradient(self, logits, sample_mode="random_sample"):
        dist = OneHotCategorical(logits=logits)
        # dist = Independent(
        #     OneHotDist(logits), 1
        # )        
        if sample_mode == "random_sample":
            sample = dist.sample() + dist.probs - dist.probs.detach()
            # sample = dist.sample()
        elif sample_mode == "mode":
            sample = dist.mode
            # sample = dist.mode()
        elif sample_mode == "probs":
            sample = dist.probs
        return sample.to(dtype=logits.dtype)
    
    
    def flatten_sample(self, sample):
        return rearrange(sample, "B L K C -> B L (K C)")

    def init_imagine_buffer(self, imagine_batch_size, imagine_batch_length, dtype, device):
        '''
        This can slightly improve the efficiency of imagine_data
        But may vary across different machines
        '''
        if self.imagine_batch_size != imagine_batch_size or self.imagine_batch_length != imagine_batch_length:
            print(f"init_imagine_buffer: {imagine_batch_size}x{imagine_batch_length}@{dtype}")
            self.imagine_batch_size = imagine_batch_size
            self.imagine_batch_length = imagine_batch_length
            latent_size = (imagine_batch_size, imagine_batch_length+1, self.stoch_flattened_dim)
            hidden_size = (imagine_batch_size, imagine_batch_length+1, self.hidden_state_dim)
            scalar_size = (imagine_batch_size, imagine_batch_length)
            self.sample_buffer = torch.zeros(latent_size, dtype=dtype, device=device)
            self.dist_feat_buffer = torch.zeros(hidden_size, dtype=dtype, device=device)
            self.action_buffer = torch.zeros(scalar_size, dtype=dtype, device=device)
            self.reward_hat_buffer = torch.zeros(scalar_size, dtype=dtype, device=device)
            self.termination_hat_buffer = torch.zeros(scalar_size, dtype=dtype, device=device)
    @profile
    def imagine_data(self, agent: agents.ActorCriticAgent, sample_obs, sample_action,
                     imagine_batch_size, imagine_batch_length, log_video, logger, global_step,
                     demo_mask=None, future_actions=None):

        self.init_imagine_buffer(imagine_batch_size, imagine_batch_length, dtype=self.tensor_dtype, device=self.device)
        self.sequence_model.reset_kv_cache_list(imagine_batch_size, dtype=self.tensor_dtype)
        obs_hat_list = []
            
        # context
        context_latent = self.encode_obs(sample_obs)

        for i in range(sample_obs.shape[1]):  # context_length is sample_obs.shape[1]
            last_obs_hat, last_reward_hat, last_termination_hat, last_latent, last_dist_feat = self.predict_next(
                context_latent[:, i:i+1],
                sample_action[:, i:i+1],
                log_video=log_video
            )
        self.sample_buffer[:, 0:1] = last_latent
        self.dist_feat_buffer[:, 0:1] = last_dist_feat

        # imagine
        for i in range(imagine_batch_length):
            action, _ = agent.sample(torch.cat([self.sample_buffer[:, i:i+1], self.dist_feat_buffer[:, i:i+1]], dim=-1))
            
            if demo_mask is not None and future_actions is not None:
                demo_mask_expand = demo_mask.unsqueeze(1).to(action.device)
                action = torch.where(demo_mask_expand, future_actions[:, i:i+1], action)
                
            self.action_buffer[:, i:i+1] = action

            last_obs_hat, last_reward_hat, last_termination_hat, last_latent, last_dist_feat = self.predict_next(
                self.sample_buffer[:, i:i+1], self.action_buffer[:, i:i+1], log_video=log_video)

            self.sample_buffer[:, i+1:i+2] = last_latent
            self.dist_feat_buffer[:, i+1:i+2] = last_dist_feat
            self.reward_hat_buffer[:, i:i+1] = last_reward_hat
            self.termination_hat_buffer[:, i:i+1] = last_termination_hat
            if log_video:
                obs_hat_list.append(last_obs_hat[::imagine_batch_size//4] * 255)  # uniform sample vec_env

        if log_video:    
            img_frames = torch.clamp(torch.cat(obs_hat_list, dim=1), 0, 255)
            img_frames = img_frames.permute(1, 2, 3, 0, 4)
            img_frames = img_frames.reshape(imagine_batch_length, 3, 64, 64 * 4).cpu().float().detach().numpy().astype(np.uint8)
            logger.log("Imagine/predict_video", img_frames, global_step=global_step)

        return torch.cat([self.sample_buffer, self.dist_feat_buffer], dim=-1), self.action_buffer, None, None, self.reward_hat_buffer, self.termination_hat_buffer

    @profile
    def imagine_data2(self, agent: agents.ActorCriticAgent, sample_obs, sample_action,
                     imagine_batch_size, imagine_batch_length, log_video, logger, global_step,
                     demo_mask=None, future_actions=None):
        self.init_imagine_buffer(imagine_batch_size, imagine_batch_length, dtype=self.tensor_dtype, device=self.device)
        # context
        context_latent = self.encode_obs(sample_obs)
        batch_size, seqlen_og, embedding_dim = context_latent.shape
        max_length = imagine_batch_length + seqlen_og
        
        if self.use_cg:
            if not hasattr(self.sequence_model, "_decoding_cache"):
                self.sequence_model._decoding_cache = None
            self.sequence_model._decoding_cache = update_graph_cache(
                self.sequence_model,
                self.sequence_model._decoding_cache,
                imagine_batch_size,
                seqlen_og,
                max_length,
                embedding_dim,
            )
            inference_params = self.sequence_model._decoding_cache.inference_params
            with torch.inference_mode():
                inference_params.reset(max_length, imagine_batch_size)
        else:
            inference_params = InferenceParams(max_seqlen=max_length, max_batch_size=imagine_batch_size, key_value_dtype=torch.bfloat16 if self.use_amp else None)

        
        def get_hidden_state(samples, action, inference_params):
            decoding = inference_params.seqlen_offset > 0

            if not self.use_cg or not decoding:
                hidden_state = self.sequence_model(
                    samples, action,
                    inference_params=inference_params,
                    # num_last_tokens=1,
                # ).logits.squeeze(dim=1)
                )
            else:
                hidden_state = self.sequence_model._decoding_cache.run(
                    samples, action, inference_params.seqlen_offset
                )
            return hidden_state        

        def should_stop(current_token, inference_params):
            if inference_params.seqlen_offset == 0:
                return False
            # if eos_token_id is not None and (current_token == eos_token_id).all():
            #     return True
            if inference_params.seqlen_offset >= max_length:
                return True
            return False
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp and not self.use_cg):
            with torch.inference_mode():
                context_dist_feat = get_hidden_state(context_latent, sample_action, inference_params)
            inference_params.seqlen_offset += context_dist_feat.shape[1]
            context_prior_logits = self.dist_head.forward_prior(context_dist_feat)
            context_prior_sample = self.stright_throught_gradient(context_prior_logits)
            context_flattened_sample = self.flatten_sample(context_prior_sample)

            dist_feat_list, sample_list  = [context_dist_feat[:, -1:]], [context_flattened_sample[:, -1:]]
            self.sample_buffer[:, 0:1] = context_flattened_sample[:, -1:]
            self.dist_feat_buffer[:, 0:1] = context_dist_feat[:, -1:]
            action_list, old_logits_list = [], []
            i = 0
            while not should_stop(sample_list[-1], inference_params):
                action, logits = agent.sample(torch.cat([self.sample_buffer[:, i:i+1], self.dist_feat_buffer[:, i:i+1]], dim=-1))
                
                if demo_mask is not None and future_actions is not None:
                    demo_mask_expand = demo_mask.unsqueeze(1).to(action.device)
                    action = torch.where(demo_mask_expand, future_actions[:, i:i+1], action)
                    
                action_list.append(action)
                self.action_buffer[:, i:i+1] = action
                old_logits_list.append(logits)
                with torch.inference_mode():
                    dist_feat = get_hidden_state(sample_list[-1], action_list[-1], inference_params)
                dist_feat_list.append(dist_feat)
                self.dist_feat_buffer[:, i+1:i+2] = dist_feat
                inference_params.seqlen_offset += sample_list[-1].shape[1]
                # if repetition_penalty == 1.0:
                #     sampled_tokens = sample_tokens(scores[-1], inference_params)
                # else:
                #     logits = modify_logit_for_repetition_penalty(
                #         scores[-1].clone(), sequences_cat, repetition_penalty
                #     )
                #     sampled_tokens = sample_tokens(logits, inference_params)
                #     sequences_cat = torch.cat([sequences_cat, sampled_tokens], dim=1)
                prior_logits = self.dist_head.forward_prior(dist_feat_list[-1])
                prior_sample = self.stright_throught_gradient(prior_logits)
                prior_flattened_sample = self.flatten_sample(prior_sample)
                sample_list.append(prior_flattened_sample)
                self.sample_buffer[:, i+1:i+2] = prior_flattened_sample
                i += 1
                    
                        
            # sample_tensor = torch.cat(sample_list, dim=1)
            # dist_feat_tensor = torch.cat(dist_feat_list, dim=1)
            # action_tensor = torch.cat(action_list, dim=1)
            old_logits_tensor = torch.cat(old_logits_list, dim=1)

            reward_hat_tensor = self.reward_decoder(self.dist_feat_buffer[:,:-1])
            self.reward_hat_buffer = self.symlog_twohot_loss_func.decode(reward_hat_tensor)
            termination_hat_tensor = self.termination_decoder(self.dist_feat_buffer[:,:-1])
            self.termination_hat_buffer = termination_hat_tensor > 0


            if log_video:
                obs_hat = self.image_decoder(self.sample_buffer[::imagine_batch_size//4]) * 255
                obs_hat = torch.clamp(obs_hat, 0, 255)
                img_frames = obs_hat.permute(1, 2, 3, 0, 4)
                img_frames = img_frames.reshape(imagine_batch_length+1, 3, 64, 64 * 4).cpu().float().detach().numpy().astype(np.uint8)
                logger.log("Imagine/predict_video", img_frames, global_step=global_step)
        return torch.cat([self.sample_buffer, self.dist_feat_buffer], dim=-1), self.action_buffer, old_logits_tensor, torch.cat([context_flattened_sample, context_dist_feat], dim=-1), self.reward_hat_buffer, self.termination_hat_buffer


    @profile
    def update(self, obs, action, reward, termination, global_step, epoch_step, logger=None, indexes=None, replay_buffer=None, agent=None):
        self.train()
        batch_size, batch_length = obs.shape[:2]
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            # encoding
            embedding = self.encoder(obs)
            post_logits = self.dist_head.forward_post(embedding)
            sample = self.stright_throught_gradient(post_logits, sample_mode="random_sample")
            flattened_sample = self.flatten_sample(sample)

            # decoding image
            obs_hat = self.image_decoder(flattened_sample)

            # dynamics models
            if self.model == 'Transformer':
                temporal_mask = get_subsequent_mask_with_batch_length(batch_length, flattened_sample.device)
                dist_feat = self.sequence_model(flattened_sample, action, temporal_mask)
            else:
                dist_feat = self.sequence_model(flattened_sample, action)
            prior_logits = self.dist_head.forward_prior(dist_feat)

            # decoding reward and termination with dist_feat
            reward_hat = self.reward_decoder(dist_feat)
            termination_hat = self.termination_decoder(dist_feat)

            # --- Dynamics Weighting 계산 ---
            dyn_weighting_enable = bool(self.dyn_cfg.get("Enable", False))
            dyn_use_aux_value_net = bool(self.dyn_cfg.get("UseAuxValueNet", True))
            
            values_sq = None
            if dyn_weighting_enable and (not dyn_use_aux_value_net or self.aux_value_module is None):
                if agent is not None:
                    with torch.no_grad():
                        aligned_input = torch.cat([flattened_sample[:, 1:], dist_feat[:, :-1]], dim=-1)
                        aligned_values = agent.value(aligned_input) 
                        reward_sq = reward.squeeze(-1) if reward.dim() == 3 else reward
                        values_sq = torch.zeros_like(reward_sq)
                        values_sq[:, 1:] = aligned_values.squeeze(-1) if aligned_values.dim() == 3 else aligned_values

            dyn_weights, dyn_metrics = self.compute_dynamics_weights(flattened_sample, values_sq=values_sq)
            
            # AuxValueNet Distillation (if enabled)
            m_distill_loss = torch.tensor(0.0, device=obs.device)
            aux_val_linear_full = None
            if self.aux_value_module is not None:
                from sub_models.functions_losses import symexp, symlog
                aux_val_symlog = self.aux_value_module(flattened_sample.detach()).squeeze(-1)
                aux_val_linear_full = symexp(aux_val_symlog.detach()).to(torch.float32)
                
                with torch.no_grad():
                    gamma = self.aux_cfg.get("AuxGamma", 0.985)
                    lam = self.aux_cfg.get("AuxLam", 0.95)
                    
                    slow_aux_val_symlog = self.aux_value_module.slow_net(flattened_sample.detach()).squeeze(-1)
                    slow_aux_val_linear = symexp(slow_aux_val_symlog.detach())
                    
                    if termination.dim() == 3:
                        termination_sq = termination.squeeze(-1)
                    else:
                        termination_sq = termination
                    inv_termination = (termination_sq * -1) + 1
                        
                    gamma_return = torch.zeros((batch_size, batch_length + 1), dtype=reward.dtype, device=reward.device)
                    gamma_return[:, -1] = slow_aux_val_linear[:, -1]
                    
                    for t in reversed(range(batch_length)):
                        next_val = slow_aux_val_linear[:, t+1] if t + 1 < batch_length else slow_aux_val_linear[:, t]
                        gamma_return[:, t] = \
                            reward[:, t] + \
                            gamma * inv_termination[:, t] * (1 - lam) * next_val + \
                            gamma * inv_termination[:, t] * lam * gamma_return[:, t+1]
                                
                    target_v_linear_full = gamma_return[:, :-1]
                    sym_target_value = symlog(target_v_linear_full)
                    
                align_mask = torch.ones_like(reward, dtype=torch.bool)
                if align_mask.dim() >= 2:
                    align_mask[..., 0] = False
                    align_mask[..., -1] = False
                
                m_distill_loss = F.mse_loss(aux_val_symlog[align_mask], sym_target_value[align_mask])

            # env loss
            if dyn_weights is not None:
                # Per-frame loss 계산 후 가중 평균
                recon_per_frame = self.mse_loss_func(obs_hat[:batch_size], obs[:batch_size], reduction='none')  # [B, L]
                reward_per_frame = self.symlog_twohot_loss_func(reward_hat, reward, reduction='none')  # [B, L]
                
                dyn_apply_recon = bool(self.dyn_cfg.get("ApplyToRecon", True))
                dyn_apply_reward = bool(self.dyn_cfg.get("ApplyToReward", True))
                
                if dyn_apply_recon:
                    reconstruction_loss = (recon_per_frame * dyn_weights).mean()
                else:
                    reconstruction_loss = recon_per_frame.mean()
                    
                if dyn_apply_reward:
                    reward_loss = (reward_per_frame * dyn_weights).mean()
                else:
                    reward_loss = reward_per_frame.mean()
            else:
                reconstruction_loss = self.mse_loss_func(obs_hat[:batch_size], obs[:batch_size])
                reward_loss = self.symlog_twohot_loss_func(reward_hat, reward)
            
            termination_loss = self.bce_with_logits_loss_func(termination_hat, termination)
            # dyn-rep loss (KL에는 dynamics weighting 적용하지 않음)
            dynamics_loss, dynamics_real_kl_div = self.categorical_kl_div_loss(post_logits[:, 1:].detach(), prior_logits[:, :-1])
            representation_loss, representation_real_kl_div = self.categorical_kl_div_loss(post_logits[:, 1:], prior_logits[:, :-1].detach())
            
            total_loss = reconstruction_loss + reward_loss + termination_loss + dynamics_loss + 0.1*representation_loss + m_distill_loss

            macro_loss = torch.tensor(0.0, device=obs.device)
            m_contrastive_loss = torch.tensor(0.0, device=obs.device)

            if self.macro_loss is not None:
                flattened_logits = rearrange(post_logits, "B L K C -> B L (K C)")
                
                aux_mean = self.aux_value_module.aux_value_mean.to(torch.float32) if self.aux_value_module is not None else None
                aux_std = torch.sqrt(self.aux_value_module.aux_value_var.to(torch.float32) + 1e-8) if self.aux_value_module is not None else None
                
                macro_loss, m_distill_loss_macro, m_contrastive_loss, trigger_obs, t_mask, t_mask_orig = self.macro_loss(
                    obs=obs,
                    latent=flattened_sample,
                    logits=flattened_logits,
                    reward=reward,
                    encode_fn=self.encode_for_macro,
                    reward_mean=0.0,
                    reward_std=1.0,
                    value=values_sq,
                    aux_value=aux_val_symlog,
                    termination=termination,
                    indexes=indexes,
                    replay_buffer=replay_buffer,
                    global_step=global_step,
                    latent_full=flattened_sample,
                    aux_mean=aux_mean,
                    aux_std=aux_std,
                    aux_value_fn=self.aux_value_module
                )
                total_loss = total_loss + macro_loss
                
            if logger is not None:
                if self.macro_loss is not None:
                    logger.log("WorldModel/macro_loss", macro_loss.item(), global_step=global_step)
                    logger.log("WorldModel/macro_contrastive_loss", m_contrastive_loss.item(), global_step=global_step)
                if self.aux_value_module is not None:
                    logger.log("WorldModel/aux_distill_loss", m_distill_loss.item(), global_step=global_step)

            # --- Dynamics Weighting wandb 로깅 ---
            if dyn_weights is not None and logger is not None:
                logger.log("DynWeighting/mean_weight", dyn_weights.mean().item(), global_step=global_step)
                logger.log("DynWeighting/max_weight", dyn_weights.max().item(), global_step=global_step)
                logger.log("DynWeighting/boosted_frame_ratio", (dyn_weights > 1.0).float().mean().item(), global_step=global_step)
                if 'dyn_metrics' in locals() and dyn_metrics:
                    for k, v in dyn_metrics.items():
                        logger.log(f"DynWeighting/{k}", v, global_step=global_step)

        # gradient descent
        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)  # for clip grad
        
        # Manually check for NaN/Inf gradients to prevent LaProp from corrupting weights
        grad_is_valid = True
        for p in self.parameters():
            if p.grad is not None and not p.grad.isfinite().all():
                grad_is_valid = False
                break
                
        if grad_is_valid:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.lr_scheduler.step()
            self.warmup_scheduler.dampen()
        else:
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"[WARNING] Skipping WorldModel optimizer step due to NaN/Inf gradients. Loss={total_loss.item()}")
                print(f"          Components: recon={reconstruction_loss.item():.4f}, reward={reward_loss.item():.4f}, term={termination_loss.item():.4f}, dyn={dynamics_loss.item():.4f}, rep={representation_loss.item():.4f}")
                if dyn_weights is not None:
                    print(f"          dyn_weights: max={dyn_weights.max().item():.4f}, min={dyn_weights.min().item():.4f}, nan={torch.isnan(dyn_weights).any().item()}")
            else:
                print(f"[WARNING] Skipping WorldModel optimizer step due to NaN/Inf gradients (Loss is finite: {total_loss.item():.4f})")
            
        self.optimizer.zero_grad(set_to_none=True)
        
        if self.aux_value_module is not None:
            self.aux_value_module.update_slow_target(decay=0.98)

        if logger is not None and hasattr(self.macro_loss, 'debug_metrics'):
            for metric_name, metric_value in self.macro_loss.debug_metrics.items():
                logger.log(f"MacroDebug/{metric_name}", metric_value, global_step=global_step)

        if logger is not None and 't_mask_orig' in locals() and t_mask_orig is not None:
            b_indices, t_indices = torch.where(t_mask_orig)
            if b_indices.numel() > 0:
                num_frames = min(4, b_indices.numel())
                
                b_sel = b_indices[:num_frames]
                t_sel = t_indices[:num_frames]
                t_prev = torch.clamp(t_sel - 1, min=0)
                
                prev_frames = torch.clamp(obs[b_sel, t_prev] * 255, 0, 255).permute(0, 2, 3, 1).cpu().detach().numpy().astype(np.uint8)
                curr_frames = torch.clamp(obs[b_sel, t_sel] * 255, 0, 255).permute(0, 2, 3, 1).cpu().detach().numpy().astype(np.uint8)
                
                row_prev = np.concatenate([prev_frames[i] for i in range(num_frames)], axis=1)
                row_curr = np.concatenate([curr_frames[i] for i in range(num_frames)], axis=1)
                combined_trigger = np.concatenate([row_prev, row_curr], axis=0)
                
                h, w, c = combined_trigger.shape
                resized_trigger = cv2.resize(combined_trigger, (w * 4, h * 4), interpolation=cv2.INTER_AREA)
                logger.log("MacroLoss/Triggered_images", [resized_trigger], global_step=global_step)

        if (global_step + epoch_step) % self.save_every_steps == 0: # and global_step != 0:
            sample_obs = torch.clamp(obs[:3, 0, :]*255, 0, 255).permute(0, 2, 3, 1).cpu().detach().float().numpy().astype(np.uint8)
            sample_obs_hat = torch.clamp(obs_hat[:3, 0, :]*255, 0, 255).permute(0, 2, 3, 1).cpu().detach().float().numpy().astype(np.uint8)

            concatenated_images = []
            for idx in range(3):
                concatenated_image = np.concatenate((sample_obs[idx], sample_obs_hat[idx]), axis=0)  # Concatenate vertically
                concatenated_images.append(concatenated_image)

            # Combine selected images into one image
            final_image = np.concatenate(concatenated_images, axis=1)  # Concatenate horizontally
            height, width, _ = final_image.shape
            scale_factor = 6
            final_image_resized = cv2.resize(final_image, (width * scale_factor, height * scale_factor), interpolation=cv2.INTER_NEAREST)
            logger.log("Reconstruct/Reconstructed images", [final_image_resized], global_step=global_step)
                         

            # ==========================================================
            # [추가] 합성된 64x64 이미지를 이용한 특정 UI 상태(Diver, Oxygen, Lives)의 aux_value 모니터링
            # ==========================================================
            import os
            
            _diff_type = getattr(self.macro_loss, 'diff_type', 'reward') if self.macro_loss is not None else 'reward'
            _trigger_type = getattr(self.macro_loss, 'trigger_type', 'reward') if self.macro_loss is not None else 'reward'
            _has_critic_probe = agent is not None and getattr(agent, 'enable_critic_probe', False)
            
            # Probing은 무거운 작업이므로 매 스텝마다 실행되지 않도록 save_every_steps 단위로만 실행합니다.
            if (global_step + epoch_step) % self.save_every_steps != 0:
                pass
            elif self.aux_value_module is not None or _diff_type in ['aux_value', 'aux_value_diff'] or _trigger_type in ['aux_value', 'aux_value_diff'] or _has_critic_probe:
                try:
                    # 1. 디스크 I/O 병목 방지를 위한 텐서 캐싱 (1회만 로드)
                    if not hasattr(self, "_cached_probing_tensors"):
                        self._cached_probing_tensors = {}
                        current_img_size = self.config.Models.WorldModel.Encoder.InputSize[1]
                        env_name_full = getattr(self.config.BasicSettings, "Env_name", "")
                        game_name = env_name_full.split("/")[-1].split("-")[0].lower() if env_name_full else "seaquest"
                        
                        base_probing_path = f"Probing_Images/{game_name}"
                        
                        if game_name == "frostbite":
                            categories = {
                                "IglooBlocks": f"{base_probing_path}/{current_img_size}_selected_igloos",
                                "Temperature": f"{base_probing_path}/{current_img_size}_selected_temperature",
                                "Lives": f"{base_probing_path}/{current_img_size}_selected_lives"
                            }
                        elif game_name == "hero":
                            categories = {
                                "Dynamite": f"{base_probing_path}/{current_img_size}_selected_dynamites",
                                "Power": f"{base_probing_path}/{current_img_size}_selected_power",
                                "Lives": f"{base_probing_path}/{current_img_size}_selected_lives"
                            }
                        elif game_name == "choppercommand":
                            categories = {
                                "Trucks": f"{base_probing_path}/{current_img_size}_selected_trucks",
                                "Lives": f"{base_probing_path}/{current_img_size}_selected_lives"
                            }
                        else: # seaquest or default
                            categories = {
                                "Diver": f"{base_probing_path}/{current_img_size}_selected_divers",
                                "Oxygen": f"{base_probing_path}/{current_img_size}_selected_oxygen",
                                "Lives": f"{base_probing_path}/{current_img_size}_selected_lives"
                            }
                            if current_img_size == 64:
                                categories["Distance"] = f"{base_probing_path}/64_sea_divers"
                        
                        for category_name, folder_path in categories.items():
                            alt_path = folder_path.split("/")[-1]
                            if os.path.exists(alt_path):
                                folder_path = alt_path
                            elif not os.path.exists(folder_path):
                                continue
                            
                            category_tensors = {}
                            grouped_imgs = {}
                            
                            for img_name in sorted(os.listdir(folder_path)):
                                if not img_name.endswith('.png'): continue
                                try:
                                    if category_name == "Distance":
                                        parts = img_name.split('.')[0].split('_')
                                        state_val = f"{parts[-2]}_{parts[-1]}"
                                    else:
                                        if "_var_" in img_name:
                                            parts = img_name.split('.')[0].split('_')
                                            var_idx = parts.index('var')
                                            state_val = parts[var_idx - 1]
                                        else:
                                            state_val = img_name.split('_')[-1].split('.')[0]
                                except:
                                    continue
                                    
                                img_path = os.path.join(folder_path, img_name)
                                img = cv2.imread(img_path)
                                if img is None: continue
                                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                img = img.transpose(2, 0, 1) # HWC -> CHW (3, 64, 64)
                                
                                if state_val not in grouped_imgs:
                                    grouped_imgs[state_val] = []
                                grouped_imgs[state_val].append(img)
                                
                            for state_val, img_list in grouped_imgs.items():
                                if img_list:
                                    imgs_arr = np.stack(img_list, axis=0) # [M, 3, 64, 64]
                                    t = torch.from_numpy(imgs_arr).to(device=self.device, dtype=self.tensor_dtype) / 255.0
                                    t = rearrange(t, "M C H W -> M 1 C H W")
                                    category_tensors[state_val] = t
                            
                            if category_tensors:
                                self._cached_probing_tensors[category_name] = category_tensors

                    if getattr(self, "_cached_probing_tensors", {}):
                        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
                            self.eval()
                            try:
                                log_dict = {}
                                for cat_name, state_dict in self._cached_probing_tensors.items():
                                    latents_dict = {}
                                    for state_val, t in state_dict.items():
                                        latent = self.encode_obs(t, sample_mode="mode")
                                        latents_dict[state_val] = latent
                                        if self.aux_value_module is not None:
                                            def symexp_local(x):
                                                return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)
                                            aux_value_symlog = self.aux_value_module(latent).squeeze(-1) # [M, 1]
                                            slow_aux_value_symlog = self.aux_value_module.slow_net(latent).squeeze(-1)
                                            
                                            aux_value_linear = symexp_local(aux_value_symlog)
                                            mean_val = aux_value_linear.mean().item()
                                            
                                            slow_aux_value_linear = symexp_local(slow_aux_value_symlog)
                                            slow_mean_val = slow_aux_value_linear.mean().item()
                                            
                                            log_dict[f"Probing/{cat_name}_{state_val}_AuxValue_Mean"] = mean_val
                                            log_dict[f"Probing_Slow/{cat_name}_{state_val}_AuxValue_Mean"] = slow_mean_val
                                        
                                        if agent is not None and hasattr(agent, 'critic_probe_net'):
                                            stateless_latent = latent[..., :agent.stateless_feat_dim]
                                            probe_raw_value = agent.critic_probe_net(stateless_latent)
                                            probe_value = agent.symlog_twohot_loss.decode(probe_raw_value).squeeze(-1)
                                            log_dict[f"Probing_Critic/{cat_name}_{state_val}_Mean"] = probe_value.mean().item()
                                            
                                        recon_obs = self.image_decoder(latent)
                                        recon_mse = F.mse_loss(recon_obs, t).item()
                                        log_dict[f"Probing_Recon_MSE/{cat_name}_{state_val}"] = recon_mse
                                        
                                    if cat_name in ["Diver", "IglooBlocks", "Dynamite", "Trucks", "Distance"]:
                                        if cat_name == "Distance":
                                            import re
                                            counts_and_keys = []
                                            for k in latents_dict.keys():
                                                nums = re.findall(r'\d+', k)
                                                if nums:
                                                    counts_and_keys.append((int(nums[-1]), k))
                                            counts_and_keys.sort(key=lambda x: x[0])
                                            diver_counts = [x[0] for x in counts_and_keys]
                                            key_map = {x[0]: x[1] for x in counts_and_keys}
                                            prefix_detail = "Probing_CosSim_Detail_Distance"
                                            prefix_delta = "Probing_CosSim_Delta_Distance"
                                        else:
                                            diver_counts = sorted([int(k) for k in latents_dict.keys() if k.isdigit()])
                                            key_map = {c: str(c) for c in diver_counts}
                                            prefix_detail = "Probing_CosSim_Detail"
                                            prefix_delta = "Probing_CosSim_Delta"

                                        delta_sims = {}
                                        for i in range(len(diver_counts)):
                                            for j in range(i + 1, len(diver_counts)):
                                                c1 = diver_counts[i]
                                                c2 = diver_counts[j]
                                                
                                                if c1 == c2:
                                                    continue
                                                    
                                                l1 = latents_dict[key_map[c1]]
                                                l2 = latents_dict[key_map[c2]]
                                                
                                                if l1.shape == l2.shape:
                                                    sim = F.cosine_similarity(l1, l2, dim=-1).mean().item()
                                                else:
                                                    l1_exp = l1.unsqueeze(1) # [M1, 1, D]
                                                    l2_exp = l2.unsqueeze(0) # [1, M2, D]
                                                    sim = F.cosine_similarity(l1_exp, l2_exp, dim=-1).mean().item()
                                                    
                                                log_dict[f"{prefix_detail}/{c1}_vs_{c2}"] = sim
                                                
                                                delta = c2 - c1
                                                if delta not in delta_sims:
                                                    delta_sims[delta] = []
                                                delta_sims[delta].append(sim)
                                                        
                                        for delta, sims in delta_sims.items():
                                            if sims:
                                                log_dict[f"{prefix_delta}/Delta_{delta}_Mean"] = sum(sims) / len(sims)
                            finally:
                                self.train() 
                            
                            if logger is not None and log_dict:
                                for k, v in log_dict.items():
                                    logger.log(k, v, global_step=global_step)

                except Exception as e:
                    print(f"Failed to log probing images: {e}")
        return (
            reconstruction_loss.item(),
            reward_loss.item(),
            termination_loss.item(),
            dynamics_loss.item(),
            dynamics_real_kl_div.item(),
            representation_loss.item(),
            representation_real_kl_div.item(),
            total_loss.item(),
            macro_loss.item() if hasattr(macro_loss, 'item') else macro_loss,
            m_distill_loss.item() if hasattr(m_distill_loss, 'item') else m_distill_loss,
            m_contrastive_loss.item() if hasattr(m_contrastive_loss, 'item') else m_contrastive_loss,
        )
