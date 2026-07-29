import gymnasium
import argparse
from tensorboardX import SummaryWriter
import cv2
import numpy as np
from einops import rearrange
import torch
from collections import deque
from tqdm import tqdm
import colorama
import os

from utils import seed_np_torch, WandbLogger
import env_wrapper
import agents
from sub_models.world_models import WorldModel
import yaml
from utils import WandbLogger
import pandas as pd

def process_visualize(img):
    img = img.astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (640, 640))
    return img

class RAMWrapper(gymnasium.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if hasattr(self.unwrapped, 'ale'):
            ram_buffer = np.zeros(128, dtype=np.uint8)
            self.unwrapped.ale.getRAM(ram_buffer)
            info['ram'] = ram_buffer
        return obs, reward, terminated, truncated, info
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if hasattr(self.unwrapped, 'ale'):
            ram_buffer = np.zeros(128, dtype=np.uint8)
            self.unwrapped.ale.getRAM(ram_buffer)
            info['ram'] = ram_buffer
        return obs, info


def build_single_env(env_name, image_size):
    env = gymnasium.make(env_name, full_action_space=False, render_mode="rgb_array", frameskip=1, repeat_action_probability=0)
    env = env_wrapper.MaxLast2FrameSkipWrapper(env, skip=4)
    env = env_wrapper.AreaResizeObservation(env, shape=image_size)
    env = RAMWrapper(env)
    return env


def build_vec_env(env_name, image_size, num_envs):
    # lambda pitfall refs to: https://python.plainenglish.io/python-pitfalls-with-variable-capture-dcfc113f39b7
    def lambda_generator(env_name, image_size):
        return lambda: build_single_env(env_name, image_size)
    env_fns = []
    env_fns = [lambda_generator(env_name, image_size) for i in range(num_envs)]
    vec_env = gymnasium.vector.AsyncVectorEnv(env_fns=env_fns)
    return vec_env

def _get_ram_62(ram_obj, i):
    if isinstance(ram_obj, (list, tuple)):
        return ram_obj[i][62]
    elif hasattr(ram_obj, 'ndim'):
        if ram_obj.ndim == 2:
            return ram_obj[i, 62]
        elif ram_obj.ndim == 1:
            first_elem = ram_obj[0]
            if isinstance(first_elem, (list, tuple, np.ndarray)):
                return ram_obj[i][62]
            else:
                return ram_obj[62]
    return ram_obj[i][62]



def eval_episodes(config,
                  world_model: WorldModel, agent: agents.ActorCriticAgent, logger: WandbLogger, global_step=None):
    world_model.eval()
    agent.eval()
    vec_env = build_vec_env(config.BasicSettings.Env_name, config.BasicSettings.ImageSize, num_envs=config.Evaluate.NumEnvs)
    # print("Evaluating Env: " + colorama.Fore.YELLOW + f"{config.BasicSettings.Env_name}" + colorama.Style.RESET_ALL)
    sum_reward = np.zeros(config.Evaluate.NumEnvs)
    current_obs, _ = vec_env.reset()
    context_obs = deque(maxlen=config.JointTrainAgent.RealityContextLength)
    context_action = deque(maxlen=config.JointTrainAgent.RealityContextLength)

    atari_benchmark_df = pd.read_csv("atari_performance.csv", index_col='Task', usecols=lambda column: column in ['Task', 'Alien', 'Amidar', 'Assault', 'Asterix', 'BankHeist', 'BattleZone', 'Boxing', 'Breakout', 'ChopperCommand', 'CrazyClimber', 'DemonAttack', 'Freeway', 'Frostbite', 'Gopher', 'Hero', 'Jamesbond', 'Kangaroo', 'Krull', 'KungFuMaster', 'MsPacman', 'Pong', 'PrivateEye', 'Qbert', 'RoadRunner', 'Seaquest', 'UpNDown'])
    atari_pure_name = config.BasicSettings.Env_name.split('/')[-1].split('-')[0]
    game_benchmark_df = atari_benchmark_df.get(atari_pure_name)

    episode_idx = 0
    score_table = {"episode": [], "evaluate/score": [], "evaluate/normalised_score": []}
    for algorithm in game_benchmark_df.index[2:]:
        score_table[f"evaluate/normalised_{algorithm}_score"] = []
        
    episode_rams = [[] for _ in range(config.Evaluate.NumEnvs)]
    episode_values = [[] for _ in range(config.Evaluate.NumEnvs)]
    episode_obs = [[] for _ in range(config.Evaluate.NumEnvs)]
    
    # Store initial RAM if available (actually vec_env.reset() returns a tuple of obs, info)
    # VecEnv returns info as a dict of arrays
    initial_info = _ if isinstance(_, dict) else {}
    if 'ram' in initial_info:
        for i in range(config.Evaluate.NumEnvs):
            episode_rams[i].append(_get_ram_62(initial_info['ram'], i))
            episode_obs[i].append(current_obs[i].copy())
            
    collected_instances = []
            
    with tqdm(total=config.Evaluate.EpisodeNum, desc="Evaluating episodes") as episode_pbar:
        while True:
            with torch.no_grad():
                if len(context_action) == 0:
                    action = vec_env.action_space.sample()
                    # action = np.array([action], dtype=int)
                    # inference_params = InferenceParams(max_seqlen=1, max_batch_size=1)
                else:
                    context_latent = world_model.encode_obs(torch.cat(list(context_obs), dim=1).to(world_model.device))
                    model_context_action = np.stack(list(context_action), axis=1)
                    model_context_action = torch.Tensor(model_context_action).to(world_model.device)
                    # 현재 프레임(current_obs)의 실제 이미지 기반 인코딩 (Posterior 역할)
                    current_obs_tensor = rearrange(torch.Tensor(current_obs).to(world_model.device), "B H W C -> B 1 C H W")/255
                    current_latent = world_model.encode_obs(current_obs_tensor)

                    if world_model.model == 'Transformer':
                        _, last_dist_feat = world_model.calc_last_dist_feat(context_latent, model_context_action)
                    elif world_model.model == 'Mamba' or world_model.model == 'Mamba2':
                        _, last_dist_feat = world_model.calc_last_dist_feat(context_latent, model_context_action)

                    # Get value for DynWeighting
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=world_model.use_amp):
                        full_latent = torch.cat([current_latent, last_dist_feat], dim=-1)
                        value_t = agent.critic(full_latent)
                        value_t = world_model.symlog_twohot_loss_func.decode(value_t).squeeze(-1)
                    value_np = value_t.cpu().numpy()
                    for i in range(config.Evaluate.NumEnvs):
                        episode_values[i].append(value_np[i])

                    # Prior가 아닌 현재 프레임의 관측 결과(current_latent)를 바탕으로 행동 결정
                    action = agent.sample_as_env_action(
                        torch.cat([current_latent, last_dist_feat], dim=-1),
                        greedy=True
                    )

            context_obs.append(rearrange(torch.Tensor(current_obs).to(world_model.device), "B H W C -> B 1 C H W")/255)
            context_action.append(action)

            obs, reward, done, truncated, info = vec_env.step(action)
            # cv2.imshow("current_obs", process_visualize(obs[0]))
            # cv2.waitKey(10)
            # update current_obs, current_info and sum_reward
            sum_reward += reward
            current_obs = obs
            
            if 'ram' in info:
                for i in range(config.Evaluate.NumEnvs):
                    # In VectorEnv, if done, info['ram'] might be the new episode's ram.
                    # terminal ram might be in info['final_info'][i]['ram'].
                    # But for simplicity, we just append info['ram']
                    episode_rams[i].append(_get_ram_62(info['ram'], i))
                    episode_obs[i].append(current_obs[i].copy())

            done_flag = np.logical_or(done, truncated)
            if done_flag.any():
                # inference_params = InferenceParams(max_seqlen=1, max_batch_size=1)
                for i in range(config.Evaluate.NumEnvs):
                    if done_flag[i]:
                        episode_score = sum_reward[i]
                        normalised_score = (episode_score - game_benchmark_df['Random']) / (game_benchmark_df['Human'] - game_benchmark_df['Random'])
                        
                        score_table["episode"].append(episode_idx)
                        score_table["evaluate/score"].append(episode_score)
                        score_table["evaluate/normalised_score"].append(normalised_score)

                        for algorithm in game_benchmark_df.index[2:]:
                            denominator = game_benchmark_df[algorithm] - game_benchmark_df['Random']
                            # Check if the denominator is zero
                            if denominator != 0:
                                normalised_score = (sum_reward[i] - game_benchmark_df['Random']) / denominator
                                score_table[f"evaluate/normalised_{algorithm}_score"].append(normalised_score)
                            else:
                                score_table[f"evaluate/normalised_{algorithm}_score"].append(None)

                        if len(episode_values[i]) > 0:
                            val_seq = torch.tensor(episode_values[i], dtype=torch.float32, device=world_model.device).unsqueeze(0)
                            old_enable = world_model.dyn_cfg.get("Enable", False)
                            world_model.dyn_cfg["Enable"] = True
                            dyn_weights = world_model.compute_dynamics_weights(None, values_sq=val_seq)
                            world_model.dyn_cfg["Enable"] = old_enable
                            
                            if dyn_weights is not None:
                                dyn_weights = dyn_weights.squeeze(0).cpu().numpy()
                                rams = np.array(episode_rams[i])
                                acquisition_weights = []
                                for t in range(1, len(rams)):
                                    if rams[t] > rams[t-1]:
                                        if t < len(dyn_weights):
                                            acquisition_weights.append(dyn_weights[t])
                                        if t + 1 < len(dyn_weights):
                                            acquisition_weights.append(dyn_weights[t+1])
                                            
                                        if len(collected_instances) < 2:
                                            instance_imgs = []
                                            for offset in [-1, 0, 1, 2]:
                                                idx = t + offset
                                                if 0 <= idx < len(episode_obs[i]):
                                                    instance_imgs.append(episode_obs[i][idx])
                                                else:
                                                    instance_imgs.append(np.zeros_like(episode_obs[i][0]))
                                            collected_instances.append(instance_imgs)
                                            
                                if 'evaluate/acquisition_weight' not in score_table:
                                    score_table['evaluate/acquisition_weight'] = []
                                    score_table['evaluate/mean_weight'] = []
                                    
                                if len(acquisition_weights) > 0:
                                    score_table['evaluate/acquisition_weight'].append(np.mean(acquisition_weights))
                                score_table['evaluate/mean_weight'].append(np.mean(dyn_weights))
                                
                        episode_values[i] = []
                        episode_rams[i] = [_get_ram_62(info['ram'], i)] if 'ram' in info else []
                        episode_obs[i] = [current_obs[i].copy()]
                        
                        sum_reward[i] = 0
                        episode_idx += 1
                        episode_pbar.update(1)  # Update the episode progress bar
                        if episode_idx == config.Evaluate.EpisodeNum:
                            if len(collected_instances) > 0:
                                while len(collected_instances) < 2:
                                    collected_instances.append([np.zeros_like(collected_instances[0][0]) for _ in range(4)])
                                row1 = np.concatenate(collected_instances[0], axis=1)
                                row2 = np.concatenate(collected_instances[1], axis=1)
                                stitched = np.concatenate([row1, row2], axis=0)
                                logger.log("evaluate/acquisition_images", [stitched], global_step=global_step)
                                
                            # print("Mean reward: " + colorama.Fore.YELLOW + f"{np.mean(score_table['evaluate/score'])}" + colorama.Style.RESET_ALL)
                            for key, value in score_table.items():
                                if key != 'episode' and not np.array(value).any() == None:
                                    logger.log(key, np.mean(value), global_step=global_step)
                            return score_table




if __name__ == "__main__":
    from train import parse_args_and_update_config, DotDict, build_world_model, build_agent
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # Read the YAML configuration file
    with open('config_files/configure.yaml', 'r') as file:
        config = yaml.safe_load(file)
   
    
    # Parse the arguments and update the configuration
    config = parse_args_and_update_config(config)   

    config = DotDict(config)
    
    # parse arguments
    # print(colorama.Fore.RED + str(config) + colorama.Style.RESET_ALL)

    device = torch.device(config.BasicSettings.Device)

    # set seed
    seed_np_torch(seed=config.BasicSettings.Seed)

    # getting action_dim with dummy env
    dummy_env = build_single_env(config.BasicSettings.Env_name, config.BasicSettings.ImageSize)
    action_dim = dummy_env.action_space.n

    # build world model and agent
    world_model = build_world_model(config, action_dim, device=device)
    config.update_or_create('Models.WorldModel.TotalParamNum', sum([p.numel() for p in world_model.parameters()]))
    config.update_or_create('Models.WorldModel.BackboneParamNum', sum([p.numel() for p in world_model.sequence_model.parameters()]))
    agent = build_agent(config, action_dim, device=device)
    config.update_or_create('Models.Agent.ActorParamNum', sum([p.numel() for p in agent.actor.parameters()]))
    config.update_or_create('Models.Agent.CriticParamNum', sum([p.numel() for p in agent.critic.parameters()]))
    if (config.BasicSettings.Compile and os.name != "nt"):  # compilation is not supported on windows
        world_model = torch.compile(world_model)
        agent = torch.compile(agent)
    logger = WandbLogger(config=config, project=config.Wandb.Init.Project, mode=config.Wandb.Init.Mode)
    logdir = logger.run.dir

    if config.BasicSettings.SavePath != 'None':
        print('Loading models')
        world_model.load_state_dict(torch.load(f"{config.BasicSettings.SavePath}/world_model.pth"))
        agent.load_state_dict(torch.load(f"{config.BasicSettings.SavePath}/agent.pth"))
    
    scores_table = eval_episodes(
        config, world_model=world_model, agent=agent, logger=logger)
    