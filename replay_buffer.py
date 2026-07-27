import numpy as np
import random
import unittest
import torch
from einops import rearrange
import copy
import pickle


class ReplayBuffer():
    def __init__(self, config, device="cuda", action_dim=1, is_discrete=True) -> None:
        self.store_on_gpu = config.BasicSettings.ReplayBufferOnGPU
        max_length = config.JointTrainAgent.BufferMaxLength
        obs_shape = (*config.BasicSettings.ImageSize, config.BasicSettings.ImageChannel)
        self.device = device
        self.is_discrete = is_discrete  # NEW
        self.action_dim = action_dim  # NEW

        # Determine action buffer shape
        if is_discrete:
            action_shape = (max_length,)  # Scalar for discrete
        else:
            action_shape = (max_length, action_dim)  # Vector for continuous

        if self.store_on_gpu:
            self.obs_buffer = torch.empty((max_length, *obs_shape), dtype=torch.uint8, device=device, requires_grad=False)
            self.action_buffer = torch.empty(action_shape, dtype=torch.float32, device=device, requires_grad=False)  # CHANGED
            self.reward_buffer = torch.empty((max_length), dtype=torch.float32, device=device, requires_grad=False)
            self.termination_buffer = torch.empty((max_length), dtype=torch.float32, device=device, requires_grad=False)
            self.target_metric_buffer = torch.empty((max_length), dtype=torch.float32, device=device, requires_grad=False)
            self.sampled_counter = torch.zeros((max_length), dtype=torch.int32, device=device, requires_grad=False)
            self.imagined_counter = torch.zeros((max_length), dtype=torch.int32, device=device, requires_grad=False)
        else:
            self.obs_buffer = np.empty((max_length, *obs_shape), dtype=np.uint8)
            self.action_buffer = np.empty(action_shape, dtype=np.float32)
            self.reward_buffer = np.empty((max_length), dtype=np.float32)
            self.termination_buffer = np.empty((max_length), dtype=np.float32)
            self.target_metric_buffer = np.empty((max_length), dtype=np.float32)
            self.sampled_counter = np.zeros((max_length), dtype=np.int32)
            self.imagined_counter = np.zeros((max_length), dtype=np.int32)

        self.length = 0
        self.last_pointer = -1
        self.max_length = max_length
        self.world_model_warmup_length = config.JointTrainAgent.WorldModelWarmUp
        self.behaviour_warmup_length = config.JointTrainAgent.BehaviourWarmUp
        self.tau = config.JointTrainAgent.Tau
        self.imagination_tau = config.JointTrainAgent.ImaginationTau
        self.alpha = config.JointTrainAgent.Alpha
        self.beta = config.JointTrainAgent.Beta
        self.batch_scale_factor = config.JointTrainAgent.ImagineBatchSize / config.JointTrainAgent.BatchSize
        
        self.world_model_demo_ratio = getattr(config.Demonstration, 'WorldModelDemoRatio', 0.25) if hasattr(config, 'Demonstration') else 0.25
        self.agent_demo_ratio = getattr(config.Demonstration, 'AgentDemoRatio', 0.0) if hasattr(config, 'Demonstration') else 0.0

    def ready(self, model_name='world_model'):
        return self.length  > self.world_model_warmup_length if model_name == 'world_model' else self.length  > self.behaviour_warmup_length

    @torch.no_grad()
    def sample(self, batch_size, batch_length, imagine=False):
        if self.store_on_gpu:
            obs_list, action_list, reward_list, termination_list, target_metric_list = [], [], [], [], []
            if batch_size > 0:
                counts = self.sampled_counter[:self.length + 1 - batch_length]
                imagine_counts = self.imagined_counter[:self.length + 1 - batch_length] / self.batch_scale_factor
                
                if imagine:
                    linear_penalty = torch.maximum(torch.zeros_like(counts), counts - imagine_counts)
                    score = counts - self.alpha * imagine_counts - self.beta * linear_penalty
                    score = score / self.imagination_tau
                    probabilities = torch.softmax(score, dim=0)
                else:
                    logits = -counts / self.tau
                    probabilities = torch.exp(logits) / torch.sum(torch.exp(logits))

                # 혼합 배치 샘플링
                valid_length = len(probabilities)
                demo_size = getattr(self, 'protect_size', 0)
                demo_valid_size = demo_size - batch_length + 1 if demo_size >= batch_length else 0

                if demo_valid_size > 0 and valid_length > demo_valid_size:
                    demo_prob_mass = probabilities[:demo_valid_size].sum().item()
                    if imagine:
                        if self.agent_demo_ratio == 0.0:
                            target_demo_ratio = demo_prob_mass
                        else:
                            target_demo_ratio = self.agent_demo_ratio
                    else:
                        target_demo_ratio = max(self.world_model_demo_ratio, demo_prob_mass)
                    
                    num_demo_samples = int(batch_size * target_demo_ratio)
                    num_agent_samples = batch_size - num_demo_samples
                    
                    if num_demo_samples > 0:
                        demo_probs = probabilities[:demo_valid_size]
                        demo_sum = demo_probs.sum()
                        if demo_sum <= 1e-8:
                            prob_demo = torch.ones_like(demo_probs) / demo_valid_size
                        else:
                            prob_demo = demo_probs / demo_sum
                        demo_indexes = torch.multinomial(prob_demo, num_demo_samples, replacement=True)
                    else:
                        demo_indexes = torch.empty(0, dtype=torch.long, device=self.device)
                        
                    if num_agent_samples > 0:
                        agent_probs = probabilities[demo_valid_size:]
                        agent_sum = agent_probs.sum()
                        if agent_sum <= 1e-8:
                            prob_agent = torch.ones_like(agent_probs) / agent_probs.numel()
                        else:
                            prob_agent = agent_probs / agent_sum
                        agent_indexes = torch.multinomial(prob_agent, num_agent_samples, replacement=True) + demo_valid_size
                    else:
                        agent_indexes = torch.empty(0, dtype=torch.long, device=self.device)
                        
                    start_indexes = torch.cat([demo_indexes, agent_indexes])
                    start_indexes = start_indexes[torch.randperm(batch_size, device=self.device)]
                else:
                    start_indexes = torch.multinomial(probabilities, batch_size, replacement=True)

                if not imagine:
                    self.sampled_counter[start_indexes] += 1
                else:
                    self.imagined_counter[start_indexes] += 1

                indexes = start_indexes.unsqueeze(-1).to(self.device) + torch.arange(batch_length, device=self.device)
                
                obs_list.append(self.obs_buffer[indexes])
                action_list.append(self.action_buffer[indexes])
                reward_list.append(self.reward_buffer[indexes])
                termination_list.append(self.termination_buffer[indexes])
                target_metric_list.append(self.target_metric_buffer[indexes])

            obs = torch.cat(obs_list, dim=0).float() / 255 if obs_list else torch.empty(0, device=self.device)
            obs = rearrange(obs, "B T H W C -> B T C H W")
            action = torch.cat(action_list, dim=0) if action_list else torch.empty(0, device=self.device)
            reward = torch.cat(reward_list, dim=0) if reward_list else torch.empty(0, device=self.device)
            termination = torch.cat(termination_list, dim=0) if termination_list else torch.empty(0, device=self.device)
            target_metric = torch.cat(target_metric_list, dim=0) if target_metric_list else torch.empty(0, device=self.device)
            indexes = indexes if batch_size > 0 else None
        else:
            obs_list, action_list, reward_list, termination_list, target_metric_list = [], [], [], [], []
            indexes = None
            if batch_size > 0:

                counts = self.sampled_counter[:self.length + 1 - batch_length]
                imagine_counts = self.imagined_counter[:self.length + 1 - batch_length] / self.batch_scale_factor

                if imagine:
                    linear_penalty = np.maximum(np.zeros_like(counts), counts - imagine_counts)
                    score = counts - self.alpha * imagine_counts - self.beta * linear_penalty
                    score /= self.imagination_tau
                else:
                    score = -counts / self.tau

                exp_score = np.exp(score - np.max(score))
                probabilities = exp_score / np.sum(exp_score)

                valid_length = len(probabilities)
                demo_size = getattr(self, 'protect_size', 0)
                demo_valid_size = demo_size - batch_length + 1 if demo_size >= batch_length else 0

                if demo_valid_size > 0 and valid_length > demo_valid_size:
                    if imagine:
                        num_demo_samples = int(batch_size * self.agent_demo_ratio)
                        num_agent_samples = batch_size - num_demo_samples
                    else:
                        demo_prob_mass = probabilities[:demo_valid_size].sum()
                        target_demo_ratio = max(self.world_model_demo_ratio, demo_prob_mass)
                        num_demo_samples = int(batch_size * target_demo_ratio)
                        num_agent_samples = batch_size - num_demo_samples
                    
                    if num_demo_samples > 0:
                        demo_probs = probabilities[:demo_valid_size]
                        demo_sum = demo_probs.sum()
                        if demo_sum <= 1e-8:
                            prob_demo = np.ones_like(demo_probs) / demo_valid_size
                        else:
                            prob_demo = demo_probs / demo_sum
                        replace_demo = num_demo_samples > demo_valid_size
                        demo_indexes = np.random.choice(demo_valid_size, size=num_demo_samples, replace=replace_demo, p=prob_demo)
                    else:
                        demo_indexes = np.array([], dtype=np.int64)
                        
                    if num_agent_samples > 0:
                        agent_probs = probabilities[demo_valid_size:]
                        agent_sum = agent_probs.sum()
                        if agent_sum <= 1e-8:
                            prob_agent = np.ones_like(agent_probs) / len(agent_probs)
                        else:
                            prob_agent = agent_probs / agent_sum
                        replace_agent = num_agent_samples > len(prob_agent)
                        agent_indexes = np.random.choice(len(prob_agent), size=num_agent_samples, replace=replace_agent, p=prob_agent) + demo_valid_size
                    else:
                        agent_indexes = np.array([], dtype=np.int64)
                        
                    start_indexes = np.concatenate([demo_indexes, agent_indexes])
                    np.random.shuffle(start_indexes)
                else:
                    replace_all = batch_size > len(probabilities)
                    start_indexes = np.random.choice(len(probabilities), size=(batch_size,), replace=replace_all, p=probabilities)

                if not imagine:
                    self.sampled_counter[start_indexes] += 1
                else:
                    self.imagined_counter[start_indexes] += 1 

                indexes = start_indexes[:, np.newaxis] + np.arange(batch_length)

                obs_seq = self.obs_buffer[indexes]
                action_seq = self.action_buffer[indexes]
                reward_seq = self.reward_buffer[indexes]
                termination_seq = self.termination_buffer[indexes]
                target_metric_seq = self.target_metric_buffer[indexes]

                obs_seq = torch.from_numpy(obs_seq).float().to(self.device) / 255
                obs_seq = rearrange(obs_seq, "B T H W C -> B T C H W")
                action_seq = torch.from_numpy(action_seq).to(self.device)
                reward_seq = torch.from_numpy(reward_seq).to(self.device)
                termination_seq = torch.from_numpy(termination_seq).to(self.device)
                target_metric_seq = torch.from_numpy(target_metric_seq).to(self.device)

                obs_list.append(obs_seq)
                action_list.append(action_seq)
                reward_list.append(reward_seq)
                termination_list.append(termination_seq)
                target_metric_list.append(target_metric_seq)

            obs = torch.cat(obs_list, dim=0) if obs_list else torch.empty(0, device=self.device)
            action = torch.cat(action_list, dim=0) if action_list else torch.empty(0, device=self.device)
            reward = torch.cat(reward_list, dim=0) if reward_list else torch.empty(0, device=self.device)
            termination = torch.cat(termination_list, dim=0) if termination_list else torch.empty(0, device=self.device)
            target_metric = torch.cat(target_metric_list, dim=0) if target_metric_list else torch.empty(0, device=self.device)
            if not self.store_on_gpu and indexes is not None:
                indexes = torch.tensor(indexes, device=self.device)

        return obs, action, reward, termination, target_metric, indexes

    def append(self, obs, action, reward, termination, target_metric=0.0):
        self.last_pointer = (self.last_pointer + 1) % (self.max_length)
        if self.store_on_gpu:
            self.obs_buffer[self.last_pointer] = torch.from_numpy(obs)
            if self.is_discrete:
                self.action_buffer[self.last_pointer] = torch.tensor(action, device=self.device)
            else:
                # Ensure action is a vector
                action_tensor = torch.tensor(action, device=self.device)
                if action_tensor.dim() == 0:
                    action_tensor = action_tensor.unsqueeze(0)
                self.action_buffer[self.last_pointer] = action_tensor
            self.reward_buffer[self.last_pointer] = torch.tensor(reward, device=self.device)
            self.termination_buffer[self.last_pointer] = torch.tensor(termination, device=self.device)
            self.target_metric_buffer[self.last_pointer] = torch.tensor(target_metric, device=self.device)
        else:
            self.obs_buffer[self.last_pointer] = obs
            if self.is_discrete:
                self.action_buffer[self.last_pointer] = action
            else:
                # Ensure action is stored as vector
                if isinstance(action, (int, float)):
                    action = np.array([action])
                self.action_buffer[self.last_pointer] = action
            self.reward_buffer[self.last_pointer] = reward
            self.termination_buffer[self.last_pointer] = termination
            self.target_metric_buffer[self.last_pointer] = target_metric

        if len(self) < self.max_length:
            self.length += 1

    def __len__(self):
        return self.length
