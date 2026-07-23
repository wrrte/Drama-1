

3. Demonstration (단일 시드 궤적을 이용한 Replay Buffer 초기화)
보상이 극도로 희소한 환경에서 에이전트에게 "숨겨진 희소 보상의 존재"와 그 인과적 진행 과정을 알려주기 위해, 단 하나의 시드 궤적(Seed Trajectory)을 버퍼에 포함시키는 기법입니다. 논문에서 언급된 "A Single Seed Trajectory" 개념입니다.

작동 방식: 에이전트와 월드 모델이 학습할 때, 이 시드 궤적 데이터를 일정 비율(Demo Ratio)로 섞어 배치(Batch)를 구성합니다. 점수에 따른 동적 감소 커리큘럼(Dynamic Decay)도 포함되어 있습니다.
설정 위치: configure.yaml의 Demonstration.Enable: true
구현 위치:
graduate_research/replay_buffer.py: Replay Buffer에서 샘플을 추출할 때 WorldModelDemoRatio와 AgentDemoRatio를 바탕으로 데모 데이터(시드 궤적)를 혼합해 가져오도록 구현되어 있습니다.
4. Latent Dropout
MACRO 내부적인 안정화 기법으로, 인코더가 특정한 픽셀(노이즈)에 과도하게 의존하는 것을 방지하기 위해 Latent State에 드롭아웃을 적용합니다.

설정 위치: configure.yaml의 MacroLoss.LatentDropout.Enable: true
구현 위치:
graduate_research/sub_models/macro_loss.py 및 graduate_research/agents.py: 설정된 Target(critic 또는 aux_value_net)에 맞추어 Latent 공간에 드롭아웃을 가하도록 구현되어 있습니다.




따라서 Drama 저장소에 적용해야 할 최우선 대상은 1) macro_loss.py 전체 모듈의 이식, 2) World Model 내 인코더-Latent 출력 구간에 MacroLoss 결합, 그리고 3) replay_buffer.py에 Demonstration 샘플링 비율 로직 추가가 되겠습니다.



# Sonnet에게 논문 보여주고 의도대로 잘 작성되었는지 검토해달라고 하기.

나도 코드 쭈욱 읽어보고 의도와 다른 부분 없는지 검토


# JAX 변환 계획

일단 lazy rebuild를 적용한 것과 적용하지 않은 걸 10시간 들여서 성능을 비교해보자. 그래서 비슷하면 LazyRebuild에서 캐시된 latent(item_latents)를 그대로 쓰고 CNN 재인코딩을 건너뛰도록 하여 jax로 변환하고, 성능이 떨어지면 그냥 10시간 들여서 하는 식으로.