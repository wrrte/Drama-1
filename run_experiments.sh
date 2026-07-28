#!/bin/bash
# 5번 GPU가 존재하는지 확인 (에러 없이 실행되면 존재하는 것)
if nvidia-smi -i 5 >/dev/null 2>&1; then
    export CUDA_VISIBLE_DEVICES=5
    echo "✅ GPU 5 is detected. Using GPU 5 (Robone)."
else
    export CUDA_VISIBLE_DEVICES=3
    echo "⚠️ GPU 5 not found. Falling back to GPU 3 (B200)."
fi
export CUDA_LAUNCH_BLOCKING=1
export MUJOCO_GL="egl"


python train.py --BasicSettings.Env_name "ALE/Seaquest-v5"
python train.py --BasicSettings.Env_name "ALE/Seaquest-v5"

python train.py --BasicSettings.Env_name "ALE/Hero-v5"
python train.py --BasicSettings.Env_name "ALE/Frostbite-v5"


python train.py --BasicSettings.Env_name "ALE/ChopperCommand-v5" # dreamerV3가 drama보다 점수가 낮은 게임
python train.py --BasicSettings.Env_name "ALE/BankHeist-v5"  # dreamerV3가 drama보다 점수가 높은 게임
python train.py --BasicSettings.Env_name "ALE/PrivateEye-v5" # dreamerV3가 drama보다 점수가 높은 게임


python train.py --BasicSettings.Env_name "ALE/Asterix-v5"        # dreamerV3가 drama보다 점수가 낮은 게임

python train.py --BasicSettings.Env_name "ALE/Qbert-v5"      # dreamerV3가 drama보다 점수가 높은 게임
python train.py --BasicSettings.Env_name "ALE/Breakout-v5"   # dreamerV3가 drama보다 점수가 높음 게임

python train.py --BasicSettings.Env_name "ALE/Freeway-v5"        # dreamerV3가 drama보다 점수가 낮은 게임

python train.py --BasicSettings.Env_name "ALE/MsPacman-v5"       # dreamerV3가 drama보다 점수가 낮은 게임



python train.py --BasicSettings.Env_name "ALE/Alien-v5"
python train.py --BasicSettings.Env_name "ALE/Amidar-v5"
