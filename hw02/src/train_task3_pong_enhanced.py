# Spring 2026, 535518 Deep Learning
# hw02: Value-based RL
# Contributors: Kai-Siang Ma and Alison Wen
# Instructor: Ping-Chun Hsieh

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import cv2
import ale_py
import os
from collections import deque
import wandb
import argparse
from tqdm import tqdm

# Register ALE environments such as ALE/Pong-v5.
gym.register_envs(ale_py)

# Set from CLI through apply_global_seed(). This controls Python, NumPy, PyTorch,
# CUDA RNGs, and the action spaces used by the training/evaluation environments.
GLOBAL_SEED = None


def apply_global_seed(seed: int) -> None:
    """Set global random seeds for reproducible ablation experiments."""
    global GLOBAL_SEED
    GLOBAL_SEED = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class DQN(nn.Module):
    """
        Design the architecture of your deep Q network
        - Input size is the same as the state dimension; the output size is the same as the number of actions
        - Feel free to change the architecture (e.g. number of hidden layers and the width of each hidden layer) as you like
        - Feel free to add any member variables/functions whenever needed
    """
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        ########## YOUR CODE HERE (5~10 lines) ##########

        # Keep the original Task 2 Pong architecture for a clean Task 3 ablation:
        # Nature-style 3-layer CNN + single scalar Q head. No dueling head,
        # NoisyNet, C51, scheduler, or other non-required modifications are used.
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )
        
        ########## END OF YOUR CODE ##########

    def forward(self, x):
        return self.network(x / 255.0)


class AtariPreprocessor:
    """
        Preprocesing the state input of DQN for Atari
    """    
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame.copy())
        return np.stack(self.frames, axis=0)


class PrioritizedReplayBuffer:
    """
        Prioritizing samples in replay memory by the Bellman error.
        This implements proportional PER with importance-sampling weights.
    """
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def __len__(self):
        return len(self.buffer)

    def add(self, transition, error=None):
        ########## YOUR CODE HERE (for Task 3) ##########
        if error is None:
            if len(self.buffer) == 0:
                priority = 1.0
            else:
                priority = float(self.priorities[:len(self.buffer)].max())
                if priority <= 0:
                    priority = 1.0
        else:
            priority = abs(float(error)) + 1e-6

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition

        self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity
        ########## END OF YOUR CODE (for Task 3) ##########

    def sample(self, batch_size):
        ########## YOUR CODE HERE (for Task 3) ##########
        current_size = len(self.buffer)
        priorities = self.priorities[:current_size]
        scaled_priorities = priorities ** self.alpha
        priority_sum = scaled_priorities.sum()

        if priority_sum <= 0:
            probabilities = np.ones(current_size, dtype=np.float32) / current_size
        else:
            probabilities = scaled_priorities / priority_sum

        indices = np.random.choice(current_size, batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]

        weights = (current_size * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        weights = weights.astype(np.float32)
        ########## END OF YOUR CODE (for Task 3) ##########
        return samples, indices, weights

    def update_priorities(self, indices, errors):
        ########## YOUR CODE HERE (for Task 3) ##########
        for idx, error in zip(indices, errors):
            self.priorities[idx] = abs(float(error)) + 1e-6
        ########## END OF YOUR CODE (for Task 3) ##########


class DQNAgent:
    def __init__(self, env_name="ALE/Pong-v5", args=None):
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.test_env = gym.make(env_name, render_mode="rgb_array")
        self.num_actions = self.env.action_space.n
        self.train_preprocessor = AtariPreprocessor()
        self.eval_preprocessor = AtariPreprocessor()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        if GLOBAL_SEED is not None:
            self.env.action_space.seed(GLOBAL_SEED)
            self.test_env.action_space.seed(GLOBAL_SEED)

        self.q_net = DQN(self.num_actions).to(self.device)
        self.q_net.apply(init_weights)
        self.target_net = DQN(self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)

        self.use_double_dqn = args.use_double_dqn
        self.use_per = args.use_per
        self.use_n_step = args.use_n_step
        self.n_step = args.n_step
        self.n_step_buffer = deque(maxlen=args.n_step)

        if self.use_per:
            self.memory = PrioritizedReplayBuffer(args.memory_size, alpha=args.per_alpha, beta=args.per_beta_start)
        else:
            self.memory = deque(maxlen=args.memory_size)

        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        self.epsilon = args.epsilon_start
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min
        self.per_beta_start = args.per_beta_start
        self.per_beta_frames = args.per_beta_frames

        self.env_count = 0
        self.train_count = 0
        self.best_reward = -21
        self.max_episode_steps = args.max_episode_steps
        self.max_env_steps = args.max_env_steps
        self.eval_frequency_steps = args.eval_frequency_steps
        self.next_eval_step = args.eval_frequency_steps
        self.snapshot_steps = sorted(args.snapshot_steps)
        self.saved_snapshot_steps = set()
        self.replay_start_size = args.replay_start_size
        self.target_update_frequency = args.target_update_frequency
        self.train_per_step = args.train_per_step
        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item()

    def update_per_beta(self):
        if self.use_per:
            fraction = min(1.0, self.env_count / max(1, self.per_beta_frames))
            self.memory.beta = self.per_beta_start + fraction * (1.0 - self.per_beta_start)

    def add_to_memory(self, transition):
        if self.use_per:
            self.memory.add(transition, error=None)
        else:
            self.memory.append(transition)

    def append_transition(self, state, action, reward, next_state, done):
        if not self.use_n_step:
            self.add_to_memory((state, action, reward, next_state, done, self.gamma))
            return

        self.n_step_buffer.append((state, action, reward, next_state, done))

        while len(self.n_step_buffer) >= self.n_step or (done and len(self.n_step_buffer) > 0):
            reward_sum = 0.0
            actual_n = min(self.n_step, len(self.n_step_buffer))
            next_state_n = self.n_step_buffer[actual_n - 1][3]
            done_n = self.n_step_buffer[actual_n - 1][4]

            for idx in range(actual_n):
                _, _, reward_i, candidate_next_state, candidate_done = self.n_step_buffer[idx]
                reward_sum += (self.gamma ** idx) * reward_i
                if candidate_done:
                    actual_n = idx + 1
                    next_state_n = candidate_next_state
                    done_n = True
                    break

            state_0, action_0 = self.n_step_buffer[0][0], self.n_step_buffer[0][1]
            self.add_to_memory((state_0, action_0, reward_sum, next_state_n, done_n, self.gamma ** actual_n))
            self.n_step_buffer.popleft()

            if not done and len(self.n_step_buffer) < self.n_step:
                break

    def save_due_snapshots(self):
        for step in self.snapshot_steps:
            if self.env_count >= step and step not in self.saved_snapshot_steps:
                model_path = os.path.join(self.save_dir, f"task3_{step}.pt")
                torch.save(self.q_net.state_dict(), model_path)
                self.saved_snapshot_steps.add(step)
                print(f"Saved SC checkpoint to {model_path}")
                wandb.log({
                    "Env Step Count": self.env_count,
                    "Update Count": self.train_count,
                    "Saved Snapshot Step": step,
                })

    def run(self):
        ep = 0
        done = True
        state = None
        total_reward = 0
        step_count = 0

        for _ in tqdm(range(self.max_env_steps), desc="Training"):
            if done or step_count >= self.max_episode_steps:
                if step_count > 0:
                    print(f"[Episode] Ep: {ep} Total Reward: {total_reward} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
                    wandb.log({
                        "Episode": ep,
                        "Total Reward": total_reward,
                        "Env Step Count": self.env_count,
                        "Update Count": self.train_count,
                        "Epsilon": self.epsilon,
                    })
                    ep += 1

                train_reset_seed = None if GLOBAL_SEED is None else int(GLOBAL_SEED + ep)
                obs, _ = self.env.reset(seed=train_reset_seed)
                state = self.train_preprocessor.reset(obs)
                done = False
                total_reward = 0
                step_count = 0
                self.n_step_buffer.clear()

            action = self.select_action(state)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            next_state = self.train_preprocessor.step(next_obs)
            self.append_transition(state, action, reward, next_state, done)

            for _ in range(self.train_per_step):
                self.train()

            state = next_state
            total_reward += reward
            self.env_count += 1
            step_count += 1
            self.update_per_beta()

            if self.env_count % 1000 == 0:
                print(f"[Collect] Ep: {ep} Step: {step_count} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
                log_dict = {
                    "Episode": ep,
                    "Step Count": step_count,
                    "Env Step Count": self.env_count,
                    "Update Count": self.train_count,
                    "Epsilon": self.epsilon,
                    "Replay Size": len(self.memory),
                }
                if self.use_per:
                    log_dict["PER Beta"] = self.memory.beta
                wandb.log(log_dict)

            self.save_due_snapshots()
            while self.env_count >= self.next_eval_step:
                eval_reward = self.evaluate(episodes=20, seed_start=0)
                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    model_path = os.path.join(self.save_dir, "best_model.pt")
                    torch.save(self.q_net.state_dict(), model_path)
                    print(f"Saved new best model to {model_path} with reward {eval_reward}")

                print(f"[TrueEval] Ep: {ep} Eval Reward: {eval_reward:.2f} SC: {self.env_count} UC: {self.train_count}")
                wandb.log({
                    "Env Step Count": self.env_count,
                    "Update Count": self.train_count,
                    "Eval Reward": eval_reward,
                    "Best Eval Reward": self.best_reward,
                })
                self.next_eval_step += self.eval_frequency_steps

        if step_count > 0:
            print(f"[Episode] Ep: {ep} Total Reward: {total_reward} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
            wandb.log({
                "Episode": ep,
                "Total Reward": total_reward,
                "Env Step Count": self.env_count,
                "Update Count": self.train_count,
                "Epsilon": self.epsilon,
            })

    def evaluate(self, episodes=20, seed_start=0):
        rewards = []
        self.q_net.eval()

        for ep in range(episodes):
            seed = seed_start + ep
            obs, _ = self.test_env.reset(seed=seed)
            self.test_env.action_space.seed(seed)

            state = self.eval_preprocessor.reset(obs)
            done = False
            total_reward = 0

            while not done:
                state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action = self.q_net(state_tensor).argmax().item()

                next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
                done = terminated or truncated
                total_reward += reward
                state = self.eval_preprocessor.step(next_obs)

            rewards.append(total_reward)

        self.q_net.train()
        return float(np.mean(rewards))

    def train(self):
        if len(self.memory) < self.replay_start_size:
            return

        # Keep the original multiplicative epsilon schedule from the template.
        # Linear epsilon scheduling is intentionally not added for this ablation.
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        self.train_count += 1

        ########## YOUR CODE HERE (<5 lines) ##########
        # Sample a mini-batch of transitions from either PER or uniform replay.
        if self.use_per:
            batch, indices, weights = self.memory.sample(self.batch_size)
        else:
            batch = random.sample(self.memory, self.batch_size)
            indices = None
            weights = np.ones(self.batch_size, dtype=np.float32)
        states, actions, rewards, next_states, dones, discounts = zip(*batch)
        ########## END OF YOUR CODE ##########

        states = torch.from_numpy(np.array(states).astype(np.float32)).to(self.device)
        next_states = torch.from_numpy(np.array(next_states).astype(np.float32)).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        discounts = torch.tensor(discounts, dtype=torch.float32).to(self.device)
        weights = torch.tensor(weights, dtype=torch.float32).to(self.device)

        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        ########## YOUR CODE HERE (~10 lines) ##########
        with torch.no_grad():
            if self.use_double_dqn:
                next_actions = self.q_net(next_states).argmax(dim=1)
                next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                next_q_values = self.target_net(next_states).max(dim=1).values
            targets = rewards + discounts * next_q_values * (1.0 - dones)

        td_errors = targets - q_values
        elementwise_loss = 0.5 * td_errors.pow(2)
        loss = (weights * elementwise_loss).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        if self.use_per and indices is not None:
            self.memory.update_priorities(indices, td_errors.detach().abs().cpu().numpy())
        ########## END OF YOUR CODE ##########

        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        if self.train_count % 1000 == 0:
            print(f"[Train #{self.train_count}] Loss: {loss.item():.4f} Q mean: {q_values.mean().item():.3f} std: {q_values.std().item():.3f}")
            wandb.log({
                "Loss": loss.item(),
                "Q Mean": q_values.mean().item(),
                "Q Std": q_values.std().item(),
                "TD Abs Mean": td_errors.detach().abs().mean().item(),
                "Env Step Count": self.env_count,
                "Update Count": self.train_count,
            })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, default="./outputs/results/task3")
    parser.add_argument("--wandb-run-name", type=str, default="task3")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--memory-size", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.99999)
    parser.add_argument("--epsilon-min", type=float, default=0.01)
    parser.add_argument("--target-update-frequency", type=int, default=1000)
    parser.add_argument("--replay-start-size", type=int, default=10000)
    parser.add_argument("--max-episode-steps", type=int, default=100000)
    parser.add_argument("--max-env-steps", type=int, default=5000000)
    parser.add_argument("--eval-frequency-steps", type=int, default=50000)
    parser.add_argument(
        "--snapshot-steps",
        type=int,
        nargs="+",
        default=[600000, 1000000, 1500000, 2000000, 2500000],)
    parser.add_argument("--train-per-step", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)

    # Required Task 3 enhancement switches only.
    parser.add_argument("--use-double-dqn", action="store_true", default=True)
    parser.add_argument("--no-double-dqn", dest="use_double_dqn", action="store_false")
    parser.add_argument("--use-per", action="store_true", default=True)
    parser.add_argument("--no-per", dest="use_per", action="store_false")
    parser.add_argument("--use-n-step", action="store_true", default=True)
    parser.add_argument("--no-n-step", dest="use_n_step", action="store_false")
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--per-alpha", type=float, default=0.6)
    parser.add_argument("--per-beta-start", type=float, default=0.4)
    parser.add_argument("--per-beta-frames", type=int, default=1000000)
    args = parser.parse_args()

    apply_global_seed(args.seed)

    wandb.init(project="DLP-hw02-Task3-Pong", name=args.wandb_run_name, config=vars(args), save_code=True)
    wandb.define_metric("Env Step Count")
    wandb.define_metric("*", step_metric="Env Step Count")
    agent = DQNAgent(args=args)
    agent.run()
