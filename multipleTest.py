import pygame
import numpy as np
from collections import deque
from pongEnv import PongEnv
from agent import R2D2Agent

class MultipleTest:
    def __init__(self, config, checkpoints):
        pygame.init()
        self.config = config
        self.checkpoints = checkpoints
        self.num_agents = len(checkpoints)
        self.padding = 20
        self.env_width = config['SCREEN_WIDTH']
        self.env_height = config['SCREEN_HEIGHT']
        self.total_width = self.num_agents * (self.env_width + self.padding) + self.padding + 300
        self.total_height = self.env_height + 2 * self.padding + 200

        pygame.display.set_caption("Multiple Pong Agents")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)

        self.agents = {}
        self.envs = {}
        self.state_seqs = {}
        self.surfaces = {}

        self.env = PongEnv(config)
        initial_state = self.env.reset() # paddle_y, ball_x, ball_y, ball_dx, ball_dy, score
        self.env.width = self.total_width
        self.env.height = self.total_height
        self.env.screen = pygame.display.set_mode((self.total_width, self.total_height))

        for cp_path, label in checkpoints.items():
            agent = R2D2Agent(config)
            agent.load_initial_agent(cp_path)
            self.agents[label] = agent

            self.state_seqs[label] = deque([initial_state]*config['SEQUENCE_LENGTH'], maxlen=config['SEQUENCE_LENGTH'])

            self.surfaces[label] = pygame.Surface((self.env_width, self.env_height))

    def run(self):
        running = True
        while running:
            self.env.screen.fill((0, 0, 0))

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            for idx, (label, agent) in enumerate(self.agents.items()):
                self.env.width = self.env_width
                self.env.height = self.env_height
                state_seq = self.state_seqs[label]

                state_array = np.array(list(state_seq))
                action, _ = agent.select_action(state_array, None, evaluate=True)
                self.env.give_state(state_array[-1])
                next_state, reward, done = self.env.step(action)
                self.state_seqs[label].append(next_state)

                if done:
                    state_seq.clear()
                    new_state = self.env.reset()
                    for _ in range(self.config['SEQUENCE_LENGTH']):
                        state_seq.append(new_state)

                # Calculate the offset for each subplot
                offset_x = self.padding + idx * (self.env_width + self.padding)
                offset_y = self.padding

                # Render the game onto the main screen
                self.env.width = self.total_width
                self.env.height = self.total_height
                self.env.render_surface(offset_x, offset_y, epoch = label, rect_dim = (self.env_width, self.env_height))


            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()