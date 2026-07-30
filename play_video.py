import os
import cv2
import numpy as np
import torch
from einops import rearrange
from collections import deque
from tqdm import tqdm
import yaml
import gymnasium
import warnings
warnings.filterwarnings("ignore")

# Import necessary modules from Drama
from utils import seed_np_torch
from envs.my_atari import Atari
from train import parse_args_and_update_config, DotDict, build_world_model, build_agent, PLAY_KEY_ACTION_MEANING
from eval import build_vec_env, _get_ram_62

def process_visualize(img):
    img = img.astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_AREA)
    return img

def main():
    # Load config
    config_path = 'config_files/configure.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    config = parse_args_and_update_config(config)
    config = DotDict(config)
    
    # Force settings for evaluation video
    config.BasicSettings.Env_name = 'ALE/Seaquest-v5'
    config.Evaluate.NumEnvs = 1
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_np_torch(seed=config.BasicSettings.Seed)
    
    # Load environment
    vec_env = build_vec_env(config.BasicSettings.Env_name, config.BasicSettings.ImageSize, num_envs=1)
    
    # Load model
    action_dim = vec_env.single_action_space.n
    world_model = build_world_model(config, action_dim, device=device)
    agent = build_agent(config, action_dim, device=device)
    
    # Checkpoint path
    ckpt_path = '/media/storage_data/ai2lab/choemj/Drama/saved_models/standard/ALE/Seaquest-v5/9ncwrs6m/ckpt'
    
    print(f"Loading checkpoint from: {ckpt_path}")
    world_model_state = torch.load(f"{ckpt_path}/world_model.pth", map_location=device)
    world_model_state = {k.replace('_orig_mod.', ''): v for k, v in world_model_state.items()}
    world_model.load_state_dict(world_model_state, strict=True)
    
    agent_state = torch.load(f"{ckpt_path}/agent.pth", map_location=device)
    agent_state = {k.replace('_orig_mod.', ''): v for k, v in agent_state.items()}
    agent.load_state_dict(agent_state, strict=True)
    
    world_model.eval()
    agent.eval()
    
    video_path = "seaquest_eval_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 640))
    print(f"Saving video to {video_path}")
    
    current_obs, initial_info = vec_env.reset()
    current_is_first = np.ones(1, dtype=np.float32)
    
    context_obs = deque(maxlen=config.JointTrainAgent.RealityContextLength)
    context_action = deque(maxlen=config.JointTrainAgent.RealityContextLength)
    context_reward = deque(maxlen=config.JointTrainAgent.RealityContextLength)
    context_is_first = deque(maxlen=config.JointTrainAgent.RealityContextLength)
    
    sum_reward = 0
    diver_count = 0
    
    # Play 1 episode
    print("Playing 1 episode...")
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=world_model.use_amp):
        while True:
            if len(context_action) == 0:
                action = np.array([1]) # FIRE instead of random sample
                act_val = int(action[0])
            else:
                context_latent = world_model.encode_obs(torch.cat(list(context_obs), dim=1).to(device))
                model_context_action = np.stack(list(context_action), axis=1)
                model_context_action = torch.Tensor(model_context_action).to(device)
                
                model_context_reward = np.stack(list(context_reward), axis=1)
                model_context_reward = torch.as_tensor(model_context_reward, device=device)
                
                model_context_is_first = np.stack(list(context_is_first), axis=1)
                model_context_is_first = torch.as_tensor(model_context_is_first, device=device)
                
                current_obs_tensor = rearrange(torch.Tensor(current_obs).to(device), "B H W C -> B 1 C H W") / 255
                current_latent = world_model.encode_obs(current_obs_tensor)
                
                _, last_dist_feat = world_model.calc_last_dist_feat(
                    context_latent, model_context_action
                )
                
                actor_input = torch.cat([current_latent, last_dist_feat], dim=-1)
                action = agent.sample_as_env_action(actor_input, greedy=True)
                act_val = int(action[0])
            
            context_obs.append(rearrange(torch.Tensor(current_obs).to(device), "B H W C -> B 1 C H W")/255)
            context_action.append(action)
            context_is_first.append(current_is_first.copy())
            
            obs, reward, done, truncated, info = vec_env.step(action)
            done_flag = np.logical_or(done, truncated)
            
            if 'ram' in info:
                diver_count = _get_ram_62(info['ram'], 0)
            
            # Draw frame
            frame = process_visualize(obs[0])
            act_str = PLAY_KEY_ACTION_MEANING.get(act_val, f"UNKNOWN({act_val})")
            
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (450, 140), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, f"Action:     {act_str}", (20, 45), font, 1.0, (50, 255, 50), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Reward sum: {sum_reward:.0f}", (20, 85), font, 1.0, (255, 255, 50), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Diver(RAM62): {diver_count}", (20, 125), font, 1.0, (50, 150, 255), 2, cv2.LINE_AA)
            
            video_writer.write(frame)
            
            sum_reward += reward[0]
            current_obs = obs
            current_is_first = done_flag.astype(np.float32)
            context_reward.append(reward.copy())
            
            if done_flag[0]:
                print(f"Episode finished! Total score: {sum_reward}")
                break
                
    vec_env.close()
    video_writer.release()
    print(f"Video saved successfully to {video_path}")

if __name__ == '__main__':
    main()
