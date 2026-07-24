


# JAX 변환 계획

일단 lazy rebuild를 적용한 것과 적용하지 않은 걸 7시간 들여서 성능을 비교해보자. 그래서 비슷하면 LazyRebuild에서 캐시된 latent(item_latents)를 그대로 쓰고 CNN 재인코딩을 건너뛰도록 하여 jax로 변환하고, 성능이 떨어지면 그냥 7시간 들여서 하는 식으로.