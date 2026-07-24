
4. Latent Dropout
MACRO 내부적인 안정화 기법으로, 인코더가 특정한 픽셀(노이즈)에 과도하게 의존하는 것을 방지하기 위해 Latent State에 드롭아웃을 적용합니다.

설정 위치: configure.yaml의 MacroLoss.LatentDropout.Enable: true
구현 위치:
graduate_research/sub_models/macro_loss.py 및 graduate_research/agents.py: 설정된 Target(critic 또는 aux_value_net)에 맞추어 Latent 공간에 드롭아웃을 가하도록 구현되어 있습니다.





억지로 0 값을 wandb에 고정 출력하게 해둔 부분 그냥 기록 안 하게 수정하기


# Sonnet에게 논문 보여주고 의도대로 잘 작성되었는지 검토해달라고 하기.

나도 코드 쭈욱 읽어보고 의도와 다른 부분 없는지 검토


# JAX 변환 계획

일단 lazy rebuild를 적용한 것과 적용하지 않은 걸 7시간 들여서 성능을 비교해보자. 그래서 비슷하면 LazyRebuild에서 캐시된 latent(item_latents)를 그대로 쓰고 CNN 재인코딩을 건너뛰도록 하여 jax로 변환하고, 성능이 떨어지면 그냥 7시간 들여서 하는 식으로.