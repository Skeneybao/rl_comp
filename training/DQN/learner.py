import abc
import copy
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from training.replay.PRB import PrioritizedReplayBuffer
from training.replay.ReplayBuffer import ReplayBuffer
from training.util.logger import logger
from training.util.report_running_time import report_time
from training.util.torch_device import auto_get_device


@dataclass
class LearnerConfig:
    batch_size: int = 128
    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 3e-4
    optimizer_type: str = 'SGD'
    device: torch.device = auto_get_device()
    model_save_step: int = 1000
    # not start training until replay buffer has at least this number of transitions
    minimal_buffer_size: int = 1000

    update_target_model_step: int = 100

    reward_steps: int = 1
    grad_max_norm: float = 1.
    l2_reg: float = 0.

    qrdqn: bool = False
    kappa: float = 1.0

    cyclic_learning_rate: bool = False

    def __post_init__(self):
        self.lr = float(self.lr)
        self.l2_reg = float(self.l2_reg)


class Learner(abc.ABC):
    @abc.abstractmethod
    def step(self) -> Optional[float]:
        pass


NOT_SAVING = 'not_saving'


class DQNLearner(Learner):
    def __init__(
            self,
            config: LearnerConfig,
            model: nn.Module,
            replay_buffer: ReplayBuffer,
            model_saving_path: str,
    ):
        self.config = config
        self.replay_buffer = replay_buffer
        self.model = model.to(config.device)
        self.target_model = copy.deepcopy(model).to(config.device)
        self.optimizer = self.__get_optimizer()
        self.lr_scheduler = self.__get_lr_scheduler(self.optimizer)
        if model_saving_path != NOT_SAVING:
            self.saving = True
            self.model_saving_path = os.path.join(model_saving_path, 'models')
            if not os.path.exists(os.path.join(self.model_saving_path, 'models')):
                os.makedirs(os.path.join(self.model_saving_path, 'models'))
        else:
            self.saving = False
        self.step_cnt = 0
        self.latest_model_num = None

    def save_model(self, path, model, optimizer=None):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        if os.path.exists(path):
            logger.warning(f'File {path} already exists, will overwrite it')
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, path)

    def load_model(self, path, model, optimizer=None):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return model, optimizer

    def __get_optimizer(self):
        if self.config.optimizer_type == 'AdamW':
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.l2_reg)
        elif self.config.optimizer_type == 'Adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.l2_reg)
        elif self.config.optimizer_type == 'SGD':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.l2_reg)
        elif self.config.optimizer_type == 'RMSprop':
            optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.l2_reg)
        else:
            raise ValueError(f'Unknown optimizer type: {self.config.optimizer_type}')
        return optimizer

    def __get_lr_scheduler(self, optimizer):
        if self.config.cyclic_learning_rate:
            return torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=self.config.lr / 100, max_lr=2 * self.config.lr,
                                                     step_size_up=2000, cycle_momentum=False)
        else:
            return None

    @report_time(5000)
    def step(self) -> Optional[float]:
        if self.config.qrdqn:
            return self.step_qrdqn()
        else:
            return self.step_dqn()

    def step_dqn(self) -> Optional[float]:
        """
        Run a single optimization step of a batch with multiple step reward.
        """
        if len(self.replay_buffer.memory) < self.config.minimal_buffer_size:
            return None

        if isinstance(self.replay_buffer, PrioritizedReplayBuffer):
            sequences, sample_indices, loss_weights = self.replay_buffer.sample_batched_ordered_sync(
                int(self.config.batch_size),
                int(self.config.reward_steps))
        else:
            sequences = self.replay_buffer.sample_batched_ordered(int(self.config.batch_size),
                                                                  int(self.config.reward_steps))
        # Prepare batches
        state_batch, action_batch, reward_batch, next_state_batch, done_batch, discount_batch = (
            [], [], [], [], [], [])
        for sequence in sequences:
            # Initialize variables for n-step calculations
            accum_reward = 0.0
            discount = 1.0
            final_state = sequence[-1][3]  # The next state after n steps
            done = False

            # Accumulate reward over n steps
            for transition in sequence:
                state, model_output, reward, _, done = transition
                accum_reward += discount * reward
                discount *= self.config.gamma
                if done:
                    final_state = transition[3]  # If done, use the terminal state
                    break

            state_batch.append(state)
            action_batch.append(model_output)
            reward_batch.append(accum_reward)
            next_state_batch.append(final_state)
            done_batch.append(done)
            discount_batch.append(discount)

        state_batch = torch.stack(state_batch).to(self.config.device)
        action_batch = torch.stack(action_batch).argmax(1).unsqueeze(1).to(self.config.device)
        reward_batch = torch.tensor(reward_batch).to(self.config.device)
        next_state_batch = torch.stack(next_state_batch).to(self.config.device)
        done_batch = torch.tensor(done_batch).to(self.config.device)
        discount_batch = torch.tensor(discount_batch).to(self.config.device)

        non_final_mask = done_batch == 0
        non_final_next_states = next_state_batch[non_final_mask]

        # Q(s_t, a)
        state_action_values = self.model(state_batch).gather(1, action_batch)

        # V(s_{t+n}) = max_a Q(s_{t+n}, a)
        next_state_values = torch.zeros(self.config.batch_size, device=self.config.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_model(non_final_next_states).max(1)[0].detach()
        # expected Q values with n-step rewards
        expected_state_action_values = (next_state_values * discount_batch) + reward_batch

        # Huber loss
        losses = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1), reduction='none')
        if isinstance(self.replay_buffer, PrioritizedReplayBuffer):
            losses = losses * torch.tensor(loss_weights).unsqueeze(1).to(self.config.device)
            self.replay_buffer.update_weight_batch(sample_indices, losses.detach().flatten().tolist())
        loss = torch.mean(losses)

        # optimize
        self.optimizer.zero_grad()
        loss.backward()
        # grad clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_max_norm)
        self.optimizer.step()

        self.step_cnt += 1

        if self.saving and self.step_cnt % self.config.model_save_step == 0:
            path = os.path.join(self.model_saving_path, f'{self.step_cnt}.pt')
            logger.info(f'Save model (and optimizer) at step {self.step_cnt} into {path}')
            self.save_model(path, self.model, self.optimizer)
            self.latest_model_num = self.step_cnt

        if self.step_cnt % self.config.update_target_model_step == 0:
            self.update_target_model()

        return loss.item()


    def step_qrdqn(self) -> Optional[float]:
        """
        Run a single optimization step of a batch with multiple step reward.
        """
        if len(self.replay_buffer.memory) < self.config.minimal_buffer_size:
            return None

        if isinstance(self.replay_buffer, PrioritizedReplayBuffer):
            sequences, sample_indices, loss_weights = self.replay_buffer.sample_batched_ordered_sync(
                int(self.config.batch_size),
                int(self.config.reward_steps))
        else:
            sequences = self.replay_buffer.sample_batched_ordered(int(self.config.batch_size),
                                                                  int(self.config.reward_steps))
        # Prepare batches
        state_batch, action_batch, reward_batch, next_state_batch, done_batch, discount_batch = (
            [], [], [], [], [], [])
        for sequence in sequences:
            # Initialize variables for n-step calculations
            accum_reward = 0.0
            discount = 1.0
            final_state = sequence[-1][3]  # The next state after n steps
            done = False

            # Accumulate reward over n steps
            for transition in sequence:
                state, model_output, reward, _, done = transition
                accum_reward += discount * reward
                discount *= self.config.gamma
                if done:
                    final_state = transition[3]  # If done, use the terminal state
                    break

            state_batch.append(state)
            action_batch.append(model_output)
            reward_batch.append(accum_reward)
            next_state_batch.append(final_state)
            done_batch.append(done)
            discount_batch.append(discount)

        state_batch = torch.stack(state_batch).to(self.config.device)
        action_batch = torch.stack(action_batch).mean(-1).argmax(1).unsqueeze(1).to(self.config.device)
        reward_batch = torch.tensor(reward_batch).to(self.config.device)
        next_state_batch = torch.stack(next_state_batch).to(self.config.device)
        done_batch = torch.tensor(done_batch).to(self.config.device)
        discount_batch = torch.tensor(discount_batch).to(self.config.device)

        non_final_mask = done_batch == 0
        non_final_next_states = next_state_batch[non_final_mask]

        # Predict the quantile values for the current states and actions
        current_quantiles = self.model(state_batch)
        action_batch = action_batch.long()
        current_action_quantiles = current_quantiles.gather(1, action_batch.repeat(1, self.model.quant_dim)
                                                            .unsqueeze(-1)).squeeze(-1)

        # Predict the next state quantile values
        with torch.no_grad():
            next_state_quantiles = self.target_model(non_final_next_states)
            # Double DQN update: use the model to select actions in next states
            next_state_actions = self.model(non_final_next_states).mean(dim=2).max(1)[1].unsqueeze(1).unsqueeze(1)
            next_state_actions = next_state_actions.repeat(1, 1, self.model.quant_dim)
            # Select the quantile values for the actions chosen by the model
            next_quantiles = next_state_quantiles.gather(1, next_state_actions).squeeze(1)

        # Calculate expected quantile values for non-final next states
        expected_quantiles = torch.zeros(state_batch.size(0), self.model.quant_dim).to(self.config.device)
        expected_quantiles[non_final_mask] = next_quantiles

        # Calculate target quantile values
        target_quantile_values = (reward_batch.unsqueeze(1) +
                                  discount_batch.unsqueeze(1) * expected_quantiles * (~done_batch).unsqueeze(1))

        # Calculate quantile regression loss
        td_error = target_quantile_values - current_action_quantiles
        abs_td_error = td_error.abs()
        huber_loss = torch.where(abs_td_error <= self.config.kappa,
                                 0.5 * td_error.pow(2),
                                 self.config.kappa * (abs_td_error - 0.5 * self.config.kappa))
        quantile_loss = torch.abs(
            (torch.arange(self.model.quant_dim, device=self.config.device, dtype=torch.float) + 0.5).unsqueeze(0)
            - (td_error.detach() < 0).float()) * huber_loss / self.config.kappa
        loss = quantile_loss.mean()

        if isinstance(self.replay_buffer, PrioritizedReplayBuffer):
            # Update priorities
            new_priorities = abs_td_error.detach().sum(dim=1).cpu().numpy()
            self.replay_buffer.update_weight_batch(sample_indices, new_priorities)
            # Weighted loss for prioritized replay
            loss = (loss_weights.to(self.config.device) * quantile_loss).mean()

        # optimize
        self.optimizer.zero_grad()
        loss.backward()
        # grad clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_max_norm)
        self.optimizer.step()
        if self.config.cyclic_learning_rate:
            self.lr_scheduler.step()

        self.step_cnt += 1

        if self.saving and self.step_cnt % self.config.model_save_step == 0:
            path = os.path.join(self.model_saving_path, f'{self.step_cnt}.pt')
            logger.info(f'Save model (and optimizer) at step {self.step_cnt} into {path}')
            self.save_model(path, self.model, self.optimizer)
            self.latest_model_num = self.step_cnt

        if self.step_cnt % self.config.update_target_model_step == 0:
            self.update_target_model()

        return loss.item()

    def update_target_model(self):
        logger.debug(f'Update target model at learner step {self.step_cnt}')
        target_params = self.target_model.state_dict()
        params = self.model.state_dict()

        for k in target_params.keys():
            target_params[k] = (1 - self.config.tau) * target_params[k] + self.config.tau * params[k]

        self.target_model.load_state_dict(target_params)
