# MACRO & EFB 구현 점검 보고서

논문 `iclr2025_conference-13.tex` 대비 핵심 로직의 구현 충실도를 점검합니다.

---

## 전체 요약

| 구성 요소 | 논문 의도 | 구현 상태 | 비고 |
|-----------|-----------|-----------|------|
| SimHash (Random Projection) | ✅ 정확 | ✅ 정확 | `_hash_keys` 정확히 일치 |
| Event-Triggered Retrieval | ✅ 정확 | ✅ 정확 | EMA-Welford 구현 |
| Contrastive Loss (margin) | ✅ 정확 | ✅ 정확 | `F.relu(cosine_sim - margin)` |
| max normalization | ✅ 정확 | ✅ 정확 | `running_max_metric` |
| stop-gradient (past latent) | ✅ 정확 | ✅ 정확 | `.detach()` 적용 |
| Lazy Rebuild | ✅ 정확 | ✅ 정확 | bucket consume + re-hash |
| Aux Value Network | ✅ 정확 | ✅ 정확 | `aux_value_net` |
| EFB (Expert Forcing Buffer) | ✅ 정확 | ✅ 정확 | demo_mask + BC loss |
| Distillation loss | ⚠️ 논문 미언급 | 구현 있음 | Lambda-return 기반 추가 구현 |
| **DisableAfterStep: 0** | ❌ **버그** | ❌ **버그** | MacroLoss 즉시 비활성화 |

---

## 1. MACRO 핵심 로직

### 1-1. SimHash (✅ 정확)

**논문 수식 (Eq. 1):**
```
k = sum_{i=1}^{B} 2^{i-1} * I(z_t · W_proj > 0)
```

**코드 구현 (`macro_loss.py` L247-254):**
```python
def _hash_keys(self, latent):
    scores = latent.float() @ self.hash_proj.float()
    bits = scores > 0
    bit_values = self.hash_bit_values.to(bits.device)
    keys = (bits.to(torch.int64) * bit_values).sum(dim=-1)
    return keys.detach()
```

`W_proj ~ N(0,1)`은 `__init__`에서 `torch.randn`으로 초기화하며, `2^{i-1}` 가중치는 `hash_bit_values`에 올바르게 저장됩니다. **완벽히 일치.**

---

### 1-2. Event-Triggered Retrieval — EMA-Welford (✅ 정확)

**논문 (Section 3.2):** "temporal value difference $\Delta V_t = |m_t - m_{t-1}|$를 모니터링하여 동적 임계값($\sigma \cdot \text{std}(m)$)을 초과할 때만 트리거"

**코드 구현 (`macro_loss.py` L635-671):**
```python
# trigger_type == "aux_value_diff" 분기
val_diff[..., 1:] = torch.abs(aux_val_linear[..., 1:] - aux_val_linear[..., :-1])
aux_std = torch.sqrt(self.aux_value_var.to(torch.float32) + 1e-8)
trigger_mask_sigma = (val_diff >= self.sigma_threshold * aux_std) & align_mask
```

논문의 "early near-zero values anchored the historical variance" 문제를 EMA로 해결하는 `_update_aux_welford` (`decay=0.999`)가 올바르게 적용됩니다. `align_mask`로 시퀀스 첫/끝 프레임을 제외하는 것도 적절합니다. **의도에 정확히 부합.**

---

### 1-3. Contrastive Loss 수식 (✅ 정확)

**논문 수식 (Eq. 2):**
```
L_hash = (1/N) * sum( |m_t - m_past| / max_{1:t}|m| * max(0, cos(x_t, x_past) - tau) )
```

**코드 구현 (`macro_loss.py` L1008-1024):**
```python
# DistanceMetric == "margin" (현재 configure.yaml 설정)
metric_diff_scaled = metric_diff / self.running_max_metric  # |Δm| / max|m|
contrastive_loss = (metric_diff_scaled * F.relu(cosine_sim - margin)).mean() * self.loss_scale
```

- `running_max_metric`이 `max_{1:t}|m|`을 추적합니다.  
- `F.relu(cosine_sim - margin)`이 `max(0, cos - τ)`에 정확히 대응합니다.
- `metric_diff`에 `.detach()`가 적용되어 논문의 "stop-gradient to value divergence multiplier"를 준수합니다.
- `past_logits`는 `torch.no_grad()` 블록 내에서 계산 후 `.detach()` 처리됩니다.

**완벽히 일치.**

---

### 1-4. Lazy Rebuild (✅ 정확)

**논문 (Section 3.2):** "과거 WM 인코더의 latent가 현재와 크게 다르면 전체 해시 버킷을 재계산"

**코드 구현 (`macro_loss.py` L903-922):**
```python
latent_diff = 1.0 - F.cosine_similarity(past_latent, past_old_latent_tensor, dim=-1)
avg_latent_diff = latent_diff.mean().item()
if avg_latent_diff > self.rebuild_threshold:
    self._rebuild_all_hash_buckets(...)
```

`rebuild_threshold=0.45`, `rebuild_cooldown=3000`으로 설정되어 과도한 재계산 방지도 구현됩니다. **의도에 정확히 부합.**

---

### 1-5. Aux Value Network 설계 (✅ 정확)

**논문:** "visual representation $z_t$에 직접 붙는 독립적 경량 보조 네트워크"

**코드 구현 (`macro_loss.py` L73-80):**
```python
layers.extend([
    nn.Linear(aux_dim, 512),
    nn.LayerNorm(512),
    nn.SiLU(),
    nn.Linear(512, 1)
])
self.aux_value_net = nn.Sequential(*layers)
```

`world_models.py` L724-728에서 `flattened_sample`(latent $z_t$)을 `value` 입력으로 사용하며, `dist_feat`(hidden $h_t$)는 포함하지 않습니다. 논문 의도("strictly based on the current visual embedding")에 **정확히 부합.**

---

## 2. EFB (Expert Forcing Buffer)

### 2-1. 데모 데이터 보호 (✅ 정확)

**논문 (Section 3.1):** "single seed trajectory + online experience로 리플레이 버퍼 구성"

**코드:**
- `train.py` L764-768: `preload_play_demonstrations` → `replay_buffer.protect_size = demo_steps`
- `replay_buffer.py` L75-86: `protect_size` 이하 인덱스를 별도 데모 풀로 샘플링

데모는 순환 덮어쓰기 없이 보존되며, `WorldModelDemoRatio=0.25`로 배치의 25%를 데모에서 강제 샘플링합니다. **의도에 부합.**

### 2-2. Expert Forcing (Imagination 시) (✅ 정확)

**논문 (Section 5, 결론):** "Expert Forcing을 통해 초기 OOD 정렬 오류를 수정"

**코드 (`world_models.py` L543-545):**
```python
if demo_mask is not None and future_actions is not None:
    demo_mask_expand = demo_mask.unsqueeze(1).to(action.device)
    action = torch.where(demo_mask_expand, future_actions[:, i:i+1], action)
```

상상(imagination) 중에 데모 배치는 에이전트 행동 대신 실제 데모 행동을 사용합니다. **의도에 정확히 부합.**

### 2-3. BC (Behavior Cloning) Loss (✅ 정확)

**코드 (`agents.py` L294-304):**
```python
if demo_mask is not None and demo_mask.any():
    demo_logits = logits[:, :-1][demo_mask_expand]
    demo_actions = action[demo_mask_expand]
    bc_loss = F.cross_entropy(demo_logits.reshape(-1, self.action_dim), demo_actions.reshape(-1).long(), ...)
    policy_loss = policy_loss + self.bc_weight * bc_weight_decay_factor * bc_loss
```

`bc_weight_decay_factor`로 성능이 향상될수록 BC 가중치를 감소시키는 커리큘럼이 `train.py` L449-464에 구현됩니다. **의도에 부합.**

---

## 3. 발견된 버그 및 불일치

### 🐛 Bug #1 (Critical): `DisableAfterStep: 0` — MacroLoss 즉시 비활성화

**파일:** [`configure.yaml` L110](file:///media/storage_data/ai2lab/choemj/Drama/config_files/configure.yaml#L110)
```yaml
DisableAfterStep: 0   # ← 이 값으로 인해 MacroLoss가 step 0부터 비활성화됨
```

**코드 (`macro_loss.py` L512-514):**
```python
if global_step is not None and self.disable_after_step > 0:
    if global_step >= self.disable_after_step:
        self.enabled = False
```

`disable_after_step = 0`이면 조건 `> 0`에 걸리지 않아 즉시 비활성화는 안 되지만, **값 의미가 불명확**합니다. 실제로 distill loss만 계산되고 contrastive loss 계산 경로로 진입하지 않을 수 있습니다. 확인이 필요합니다.

> [!CAUTION]
> 실제로는 `Enable: true`이므로 `disable_after_step <= 0`이면 비활성화 조건이 발동되지 않아 정상 동작합니다. 그러나 `DisableAfterStep: 0`은 "0번째 스텝 이후 비활성화"가 의도라면 **contrastive loss가 전혀 동작하지 않을** 위험이 있습니다. 의미 명확화 필요.

---

### ⚠️ Issue #2: `DiffType: aux_value` vs `TriggerType: aux_value_diff` 혼용

**configure.yaml:**
```yaml
TriggerType: aux_value_diff   # 트리거: |V_t - V_{t-1}| 기반
DiffType: aux_value           # 가중치: 현재 aux_value vs 과거 aux_value
```

`DiffType: aux_value`일 때 `macro_loss.py` L967-969:
```python
if self.diff_type == "aux_value":
    past_aux_pred = self.aux_value_net(past_latent_full).squeeze(-1)
    metric_diff = torch.abs(curr_metric.detach() - past_aux_pred.detach())
```

`curr_metric`은 **현재 aux_value** (SymLog 스케일)이고, `past_aux_pred`도 **SymLog 스케일**입니다. 그런데 `running_max_metric`은 **선형 스케일** 값으로 갱신됩니다 (`L1010`에서 `curr_metric`의 절댓값 최대치). SymLog 값 vs 선형 최대치 간의 스케일 불일치가 존재합니다.

> [!WARNING]
> `metric_diff`는 SymLog 스케일이지만, `running_max_metric`이 동일하게 SymLog 스케일로 관리된다면 상쇄될 수 있습니다. 단, 초기 `running_max_metric = 1.0`이 적절한지 검토 필요.

---

### ⚠️ Issue #3: `world_models.py`에서 `macro_loss`가 None일 때 리턴값 오류

**코드 (`world_models.py` L979-981):**
```python
macro_loss.item() if hasattr(macro_loss, 'item') else macro_loss,
m_distill_loss.item() if hasattr(m_distill_loss, 'item') else m_distill_loss,
m_contrastive_loss.item() if hasattr(m_contrastive_loss, 'item') else m_contrastive_loss,
```

`self.macro_loss is None`일 때, `macro_loss`, `m_distill_loss`, `m_contrastive_loss` 변수 자체가 정의되지 않아 `NameError` 발생 가능.

> [!WARNING]
> `self.macro_loss is None`인 경우 세 변수를 `0.0`으로 초기화하는 방어 코드가 없습니다. `world_model.update()` 함수 초반에 기본값 할당이 필요합니다.

---

### ℹ️ Note #4: Distillation Loss — 논문에 없는 추가 구현

**코드 (`macro_loss.py` L543-580):** Lambda-return 기반 aux value network 지도 학습

이는 논문에 기술되지 않은 추가 구현입니다. aux_value_net이 의미있는 값 추정을 하도록 하는 bootstrapping으로, **구현 의도는 타당**하지만 논문 재현 시 ablation 대상이 될 수 있습니다.

---

## 4. 수정 권고 사항

### Priority 1 — `DisableAfterStep` 명확화

```yaml
# configure.yaml
DisableAfterStep: -1   # -1 = 비활성화 없음 (무한 유지)
```

### Priority 2 — `world_models.py` NameError 방어 코드

**파일:** [`world_models.py`](file:///media/storage_data/ai2lab/choemj/Drama/sub_models/world_models.py#L719)

```python
# 기본값 초기화 추가
macro_loss = torch.tensor(0.0)
m_distill_loss = torch.tensor(0.0)
m_contrastive_loss = torch.tensor(0.0)

if self.macro_loss is not None:
    ...
```

---

## 5. 전체 흐름 확인

```
train.py → replay_buffer.sample() → world_model.update()
                                        ↓
                                    encode_obs (z_t)
                                        ↓
                            macro_loss.forward(obs, z_t, logits, reward, ...)
                                        ↓
                            [Trigger] aux_value_diff → val_diff >= σ·std
                                        ↓
                            [Hash] _hash_keys(z_t) → bucket
                                        ↓
                            [LazyRebuild] re-encode past obs
                                        ↓
                            [Loss] metric_diff_scaled * relu(cos_sim - margin)
                                        ↓
                            total_loss += macro_loss → backward()
                                        ↓
                            update_slow_target() (EMA)
```

전반적으로 논문의 핵심 아이디어가 충실하게 구현되어 있습니다.



# dd


좋은 질문입니다. 데이터 흐름을 따라가며 정확히 짚어보겠습니다.

## SymLog vs Linear 스케일 가이드

`aux_value_net`의 출력값이 사용되는 곳은 **3곳**입니다. 각각의 역할에 따라 적절한 스케일이 다릅니다.

### ✅ SymLog으로 유지해야 하는 것

**1) `aux_value_net`의 출력 자체** — 네트워크는 SymLog 공간에서 학습시키니까 출력도 SymLog입니다.

```python
# macro_loss.py L540
aux_value_all = self.aux_value_net(latent_for_aux).squeeze(-1)  # SymLog 출력 ← 유지
```

**2) Distillation Loss** — 타겟도 SymLog이고 예측도 SymLog이니까 같은 공간에서 MSE를 잡는 것이 맞습니다.

```python
# macro_loss.py L579-580
sym_target_value = symlog(target_v_linear_full)              # 타겟을 SymLog으로 변환
distill_loss = F.mse_loss(aux_value_all, sym_target_value)   # 둘 다 SymLog ← 유지
```

**3) Trigger 판정** — 이미 선형으로 변환해서 계산하고 있어서 **현재 정상**입니다.

```python
# macro_loss.py L636-642 (trigger_type == "aux_value_diff")
aux_val_linear = symexp(aux_value_all.detach())  # ← SymLog → Linear 변환 ✅
val_diff[..., 1:] = torch.abs(aux_val_linear[..., 1:] - aux_val_linear[..., :-1])  # Linear 차이
```

---

### ❌ Linear로 전환해야 하는 것

**Contrastive Loss의 가중치(`metric_diff`)를 계산하는 시점부터**입니다. 현재 문제가 되는 코드는 [macro_loss.py L967-1013](file:///media/storage_data/ai2lab/choemj/Drama/sub_models/macro_loss.py#L967-L1013)입니다:

```python
# 현재 (❌ SymLog끼리 빼기)
if self.diff_type == "aux_value":
    past_aux_pred = self.aux_value_net(past_latent_full).squeeze(-1)   # SymLog
    metric_diff = torch.abs(curr_metric.detach() - past_aux_pred.detach())  # SymLog - SymLog

# ...
with torch.no_grad():
    current_max = torch.max(torch.abs(curr_metric.detach()))   # SymLog의 max
    self.running_max_metric.copy_(torch.max(self.running_max_metric, current_max.float()))

metric_diff_scaled = metric_diff / self.running_max_metric   # SymLog / SymLog
```

이것을 다음과 같이 바꿔야 합니다:

```python
# 수정 후 (✅ Linear끼리 빼기)
if self.diff_type == "aux_value":
    past_aux_pred = self.aux_value_net(past_latent_full).squeeze(-1)   # SymLog
    curr_linear = symexp(curr_metric.detach())      # ← Linear로 변환
    past_linear = symexp(past_aux_pred.detach())     # ← Linear로 변환
    metric_diff = torch.abs(curr_linear - past_linear)  # Linear - Linear ✅

# ...
with torch.no_grad():
    current_max = torch.max(torch.abs(symexp(curr_metric.detach())))  # Linear의 max ✅
    self.running_max_metric.copy_(torch.max(self.running_max_metric, current_max.float()))

metric_diff_scaled = metric_diff / self.running_max_metric   # Linear / Linear ✅
```

---

### 요약 도식

```
aux_value_net 출력 (SymLog)
     │
     ├──→ Distillation Loss:  MSE(SymLog, SymLog)     ← SymLog 유지 ✅
     │
     ├──→ Trigger 판정:       symexp() → Linear 차이   ← 이미 Linear ✅
     │
     └──→ Contrastive 가중치: symexp() → Linear 차이   ← 여기만 수정 필요 ❌→✅
                                  │
                                  └→ running_max_metric도 Linear로 관리
```

**핵심 원칙:** "**빼기(차이 계산)가 일어나는 지점에서는 반드시 Linear 스케일이어야 한다**"입니다. SymLog는 압축 함수이기 때문에, SymLog 공간에서의 차이는 실제 가치 차이를 왜곡합니다. 반면 MSE Loss처럼 같은 공간에서 비교하는 것은 문제없습니다.

수정을 적용해 드릴까요?



# 수정 후 설명


## TriggerType vs DiffType: 역할과 비교 대상

두 설정은 MACRO Loss 파이프라인의 **서로 다른 단계**를 제어합니다.

---

### 1단계: `TriggerType: aux_value_diff` — "언제 작동시킬까?"

**비교 대상:** 같은 궤적(trajectory) 안에서 **시간적으로 인접한 두 프레임** ($V_t$ vs $V_{t-1}$)

```python
# macro_loss.py L636-664
aux_val_linear = symexp(aux_value_all.detach())  # 전체 배치를 Linear로 변환

# 시간 축을 따라 연속 프레임 간 차이 계산
val_diff[..., 1:] = |V_t - V_{t-1}|    # ← 같은 궤적의 바로 옆 프레임과 비교

# 동적 임계값 초과 여부 판정
trigger_mask = (val_diff >= σ × std(V))
```

**의미:** 배치 안의 수백 프레임 중에서 "**이 프레임에서 무언가 중요한 일이 일어났다**"고 판단하는 필터입니다.

**구체적 상황:** 에이전트가 다이버를 구출하면, 그 순간 `aux_value_net`이 예측하는 미래 가치가 갑자기 점프합니다. 예를 들어:

```
t=100: V = 50   (다이버 2명 보유)
t=101: V = 50   (변화 없음)  → val_diff = 0  → ❌ 트리거 안 됨
t=102: V = 120  (다이버 3명 획득!) → val_diff = 70 → ✅ 트리거!
t=103: V = 121  (변화 없음)  → val_diff = 1  → ❌ 트리거 안 됨
```

**결과:** `trigger_mask`에는 t=102 같은 "가치 점프" 프레임만 `True`로 표시됩니다.

---

### 2단계: `DiffType: aux_value` — "얼마나 세게 밀어낼까?"

**비교 대상:** 트리거된 현재 프레임과 **해시 버킷에서 꺼낸 과거 프레임** ($V_{현재}$ vs $V_{과거}$)

```python
# macro_loss.py L967-971 (수정 후)
past_aux_pred = self.aux_value_net(past_latent_full)  # 과거 상태를 현재 네트워크로 재평가

curr_linear = symexp(curr_metric.detach())    # 현재 프레임의 가치 (Linear)
past_linear = symexp(past_aux_pred.detach())  # 과거 프레임의 가치 (Linear)
metric_diff = |curr_linear - past_linear|     # ← 시간적으로 무관한 다른 상태와 비교
```

**의미:** 1단계에서 골라진 프레임에 대해, SimHash로 **시각적으로 거의 같게 생긴 과거 프레임**을 불러온 뒤, "**두 상태의 가치가 실제로 얼마나 다른가**"를 측정하여 Contrastive Loss의 **가중치**로 씁니다.

**구체적 상황:**

```
현재 프레임 (t=102): 다이버 3명, V=120
                     ↓ SimHash → 같은 버킷에서 검색
과거 프레임 (t=5830): 다이버 0명, V=15  ← 시각적으로는 비슷! (같은 위치, 비슷한 배경)

metric_diff = |120 - 15| = 105  → 큰 가중치 → "강하게 밀어내라!"
```

```
현재 프레임 (t=102): 다이버 3명, V=120
                     ↓ SimHash → 같은 버킷에서 검색
과거 프레임 (t=2001): 다이버 3명, V=118  ← 시각적으로 비슷하고 가치도 비슷!

metric_diff = |120 - 118| = 2  → 작은 가중치 → "밀어내지 마라" (같은 건 같게)
```

---

### 두 단계의 협업 흐름

```
배치 (B=16, L=128 = 2048 프레임)
         │
    ┌────┴────────────────────────────────────┐
    │  TriggerType: aux_value_diff            │
    │  "연속 프레임 간 가치가 크게 변한 곳만 골라라" │
    │  비교: V_t vs V_{t-1} (같은 궤적, 바로 옆)  │
    └────┬────────────────────────────────────┘
         │ trigger_mask (예: 2048개 중 30개 True)
         ▼
    ┌────┴────────────────────────────────────┐
    │  SimHash: _hash_keys(z_t)               │
    │  "트리거된 30개 프레임과 시각적으로 비슷한    │
    │   과거 프레임을 해시 버킷에서 검색하라"       │
    └────┬────────────────────────────────────┘
         │ (현재, 과거) 쌍 (예: 150개 pair)
         ▼
    ┌────┴────────────────────────────────────┐
    │  DiffType: aux_value                    │
    │  "현재와 과거의 가치 차이로 가중치를 결정하라"│
    │  비교: V_현재 vs V_과거 (다른 시점, 같은 버킷)│
    └────┬────────────────────────────────────┘
         │ metric_diff_scaled (0~1 사이 가중치)
         ▼
    ┌────┴────────────────────────────────────┐
    │  Contrastive Loss                       │
    │  가중치 × max(0, cos_sim - margin)       │
    │  → 가치가 다르면 밀어내고, 같으면 냅둔다     │
    └─────────────────────────────────────────┘
```

**핵심 차이 요약:**

| | TriggerType (`aux_value_diff`) | DiffType (`aux_value`) |
|---|---|---|
| **역할** | 필터 (어떤 프레임을 처리할지) | 가중치 (얼마나 강하게 밀어낼지) |
| **비교 대상** | $V_t$ vs $V_{t-1}$ (시간적 이웃) | $V_{현재}$ vs $V_{과거 버킷}$ (시각적 이웃) |
| **스케일** | Linear (이미 `symexp` 적용) | Linear (방금 수정 적용) |
| **출력** | `trigger_mask` (bool) | `metric_diff_scaled` (float 0~1) |