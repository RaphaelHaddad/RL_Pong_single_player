# agent.py
import torch
import time
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import json
import os
from collections import deque
import pygame
import matplotlib.pyplot as plt
from pongEnv import PongEnv
from torch.nn import functional as F
from R2D2 import R2D2Network


class R2D2Agent:
    def __init__(self, config):
        self.config = config
        self.device = config['DEVICE']
        
        self.policy_net = R2D2Network(config).to(self.device)
        self.target_net = R2D2Network(config).to("cpu")
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config['LEARNING_RATE'])
        self.memory = deque(maxlen=config['MEMORY_SIZE'])
        
        self.epsilon = config['EPSILON_START']
        self.steps = 0

    def soft_update(self):
        """Soft update target network parameters"""
        for target_param, policy_param in zip(self.target_net.parameters(), 
                                            self.policy_net.parameters()):
            policy_param_data = policy_param.data.to("cpu")
            target_param.data.copy_(
                self.config['TAU'] * policy_param_data + 
                (1.0 - self.config['TAU']) * target_param.data
            )
        
    def select_action(self, state_seq, hidden=None, evaluate=False):
        if not evaluate and np.random.random() < self.epsilon:
            return np.random.randint(self.config['ACTION_SIZE']), hidden
            
        with torch.no_grad():
            state_seq = torch.FloatTensor(state_seq).unsqueeze(0).to(self.device)
            q_values, new_hidden = self.policy_net(state_seq, hidden)
            return q_values.argmax().item(), new_hidden

    def sample(self):
        batch_size = self.config['BATCH_SIZE']
        if len(self.memory) < batch_size:
            return None
        if self.config['SAMPLING_METHOD'] == 'random':
            return random.sample(self.memory, batch_size)
        elif self.config['SAMPLING_METHOD'] == 'sequential':
            return list(self.memory)[-batch_size:]
        elif self.config['SAMPLING_METHOD'] == 'random_sequential':
            start = random.randint(0, len(self.memory) - batch_size)
            return list(self.memory)[start:start + batch_size]
            
    def train_step(self):
        transitions = self.sample()
        if transitions is None:
            return None
            
        batch = list(zip(*transitions))
        
        state_seqs = torch.FloatTensor(np.array(batch[0])).to(self.device)
        actions = torch.LongTensor(batch[1]).to(self.device)
        rewards = torch.FloatTensor(batch[2]).to(self.device)
        next_state_seqs = torch.FloatTensor(np.array(batch[3])).to(self.device)
        dones = torch.FloatTensor(batch[4]).to(self.device)
        # Replace the original line with the following code:
        if all(h is not None for h in batch[5]):
            # Assume each h in batch[5] is a tuple (h_tensor, c_tensor) with shape [num_layers, 1, hidden_size]
            hs = torch.cat([h[0] for h in batch[5]], dim=1)  # shape: [num_layers, batch_size, hidden_size]
            cs = torch.cat([h[1] for h in batch[5]], dim=1)  # same shape as hs
            hidden = (hs, cs)
        else:
            hidden = None
        
        current_q_values, _ = self.policy_net(state_seqs, hidden)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_q_values, _ = self.target_net(next_state_seqs.to("cpu"))
            next_q_values = next_q_values.max(1)[0]
            target_q_values = rewards + (1 - dones) * self.config['GAMMA'] * next_q_values.to(self.device)
        
        loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Soft update target network
        self.soft_update()
        
        self.steps += 1
        self.epsilon = max(
            self.config['EPSILON_END'],
            self.epsilon * self.config['EPSILON_DECAY']
        )
        
        return loss.item()
        
    def save(self, episode, reward):
        os.makedirs(self.config['CHECKPOINT_DIR'], exist_ok=True)
        path = os.path.join(
            self.config['CHECKPOINT_DIR'],
            f'model_episode_{episode}_reward_{reward:.0f}.pth'
        )
        self.policy_net.save(path)
        
    def load(self, path):
        self.policy_net.load(path)
        self.target_net.load_state_dict(self.policy_net.state_dict())


    def load_initial_agent(self, path):
        import os  # ensure os is imported
        # Construct the full checkpoint path
        checkpoint_path = str(path)
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.config["DEVICE"])
            # Determine if checkpoint is a dict with a "state_dict" key or just the state_dict directly.
            state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
            self.policy_net.load_state_dict(state_dict)
            self.policy_net.to(self.config["DEVICE"])
            self.policy_net.eval()
            print(f"Checkpoint loaded from: {checkpoint_path}")
        else:
            print("No checkpoint file found at:", checkpoint_path)


    def save_model(self, path):
        """Save the entire Q-network model to a file."""
        torch.save(self.policy_net, path)
        print(f"Model saved to: {path}")



    def save_initial_agent(self, config_dict):
        """
        Save the agent's initial model and write out its configuration to a text file.
        
        Parameters:
            agent: The Agent instance.
            model_path (str): Path to save the initial model parameters.
            config_path (str): Path to save the configuration text file.
            config_dict (dict): Dictionary of configuration/hyperparameter values.
        """
        # Save the initial model
        os.makedirs(os.path.dirname(self.config["MODEL_PATH"]), exist_ok=True)
        self.save_model(self.config["MODEL_PATH"])
        
        # Save the configuration parameters in a nicely formatted JSON text file
        os.makedirs(os.path.dirname(self.config["CONFIG_PATH"]), exist_ok=True)
        config_dict_copy = {k: (str(v) if isinstance(v, torch.device) else v) 
                    for k, v in config_dict.items()}
        with open(self.config["CONFIG_PATH"], "w") as f:
            json.dump(config_dict_copy, f, indent=4)
            
        print(f"Initial model saved to: {self.config['MODEL_PATH']}")
        print(f"Agent configuration saved to: {self.config['CONFIG_PATH']}")



    def train(self):
        env = PongEnv(self.config)
        
        best_reward = float('-inf')
        rewards_history = deque(maxlen=self.config['LEN_REWARD_HISTORY'])
        
        for episode in range(self.config['NUM_EPISODES']):
            torch.mps.empty_cache()
            state = env.reset()
            state_seq = deque([state] * self.config['SEQUENCE_LENGTH'], maxlen=self.config['SEQUENCE_LENGTH'])
            hidden = None
            episode_reward = 0
            episode_loss = 0
            num_steps = 0

            memory_episode = []
            
            while True:
                state_array = np.array(list(state_seq))
                initial_hidden = (hidden[0].detach(), hidden[1].detach()) if hidden is not None else None
                action, hidden = self.select_action(state_array, hidden)
                next_state, reward, done = env.step(action)
                
                state_seq.append(next_state)
                next_state_seq = np.array(list(state_seq))

                transition = (state_array, action, reward, next_state_seq, done, initial_hidden)
                
                self.memory.append(transition)
                memory_episode.append(transition)
                
                loss = self.train_step()
                if loss is not None:
                    episode_loss = loss
                
                episode_reward += reward
                num_steps += 1
                
                if done:
                    break
                    
                state = next_state
                
            rewards_history.append(episode_reward)
            avg_reward = np.mean(rewards_history)
            
            loss_str = f", Loss={episode_loss:.6f}" if episode_loss != 0 else ""
            print(f"Episode {episode}: Reward={episode_reward:.1f}, "
                f"Avg={avg_reward:.1f}, Epsilon={self.epsilon:.3f}, "
                f"Steps={num_steps}{loss_str}")
            
            if episode_reward > best_reward:
                best_reward = episode_reward
                self.save(episode, episode_reward)
                if self.config['PLAY_AFTER_RECORD'] and episode > self.config['MIN_EPISODE_PLAY']:
                    self.play_episode(memory_episode)
                
            if episode % self.config['SAVE_FREQUENCY'] == 0:
                self.save(episode, episode_reward)

    def play(self):
        import pygame
        env = PongEnv(self.config)
        
        state = env.reset()
        state_seq = deque([state] * self.config['SEQUENCE_LENGTH'], maxlen=self.config['SEQUENCE_LENGTH'])
        hidden = None
        episode_reward = 0
        num_steps = 0
        
        while True:
            # Process pygame events to avoid freezing the window
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return
            
            state_array = np.array(list(state_seq))
            initial_hidden = (hidden[0].detach(), hidden[1].detach()) if hidden is not None else None
            action, hidden = self.select_action(state_array, hidden, evaluate=True)
            next_state, reward, done = env.step(action)
            env.render()  # Render the current frame
            
            state_seq.append(next_state)
            episode_reward += reward
            num_steps += 1
            
            if done:
                # Print info and reset the environment instead of quitting.
                print(f"Episode ended: Final reward: {episode_reward:.1f}, Steps: {num_steps}")
                state = env.reset()
                state_seq = deque([state] * self.config['SEQUENCE_LENGTH'], maxlen=self.config['SEQUENCE_LENGTH'])
                hidden = None
                episode_reward = 0
                num_steps = 0
            else:
                state = next_state





    def play_episode(self, memory_episode):
        clock = pygame.time.Clock()
        env = PongEnv(self.config)
        state = env.reset()
        state_seq = deque([state] * self.config['SEQUENCE_LENGTH'], maxlen=self.config['SEQUENCE_LENGTH'])
        hidden = None
        
        # Create a window if not already created by env.render()
        pygame.display.set_caption("Replaying Recorded Episode")
        
        # Iterate through the recorded transitions
        for transition in memory_episode:
            state_array, action, reward, next_state_seq, done, initial_hidden = transition
            last_state = state_array[-1]
            paddle_y, ball_x, ball_y, ball_dx, ball_dy, score = last_state
            env.paddle_y = paddle_y * env.height
            env.ball_x = ball_x * env.width
            env.ball_y = ball_y * env.height 
            env.ball_dx = ball_dx * 10
            env.ball_dy = ball_dy * 10
            env.score = int(score * 20)
            env.render()
            
            # Process events to allow quitting during replay
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            
            # Slow down the replay (pause in seconds)
            time.sleep(1/self.config.get('FRAME_RATE_EPISODE_GAME', 30))  # Adjust delay as needed (0.5 seconds each frame)
        pygame.quit()



    def multiple_test(self, checkpoints_dict):
        # Initialize pygame main window
        num_agents = len(checkpoints_dict)
        env_width = self.config['SCREEN_WIDTH']
        env_height = self.config['SCREEN_HEIGHT']
        main_width = num_agents * env_width
        main_height = env_height
        main_window = pygame.display.set_mode((main_width, main_height))
        pygame.display.set_caption("Multiple Test - Simultaneous Pong Games")

        # Create agents, envs, and state sequences for each checkpoint
        agents = {}
        envs = {}
        state_seqs = {}
        for cp_path, label in checkpoints_dict.items():
            tmp_agent = R2D2Agent(self.config)
            tmp_agent.load_initial_agent(cp_path)
            # Use evaluation mode to avoid exploration randomness
            agents[label] = tmp_agent
            env = PongEnv(self.config)
            env.reset()
            envs[label] = env
            state = env.reset()
            state_seqs[label] = deque([state] * self.config['SEQUENCE_LENGTH'], maxlen=self.config['SEQUENCE_LENGTH'])
        
        clock = pygame.time.Clock()
        font = pygame.font.SysFont(None, 24)
        running = True

        while running:
            # Handle quit event
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # For each agent, step its environment and render
            for idx, (label, agent) in enumerate(agents.items()):
                env = envs[label]
                seq = state_seqs[label]
                state_array = np.array(list(seq))
                # Select action in evaluation mode
                action, _ = agent.select_action(state_array, hidden=None, evaluate=True)
                next_state, reward, done = env.step(action)
                seq.append(next_state)
                # Get image from environment render (a numpy array)
                image = env.render(return_image=True)
                # Convert numpy image (H, W, 3) to pygame Surface.
                surf = pygame.surfarray.make_surface(np.transpose(image, (1, 0, 2)))
                # Blit env image into the main window at the proper offset.
                main_window.blit(surf, (idx * env_width, 0))
                # Render legend in the top left corner of each sub-window.
                text_surface = font.render(label, True, (255, 255, 255))
                main_window.blit(text_surface, (idx * env_width + 10, 10))
            pygame.display.flip()
            clock.tick(self.config.get('FPS', 10))
        pygame.quit()