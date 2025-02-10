import numpy as np
import pygame


class PongEnv:
    def __init__(self, config):
        pygame.init()
        self.config = config
        self.width = config['SCREEN_WIDTH']
        self.height = config['SCREEN_HEIGHT']
        self.paddle_width = 15
        self.paddle_height = 60
        self.ball_size = 15
        self.frame_skip = config.get('FRAME_SKIP', 4)
        self.reward_distance = config['REWARD_DISTANCE']
        
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.reset()
        self.last_hit = False

    def reset(self):
        self.paddle_y = self.height // 2 - self.paddle_height // 2
        self.ball_x = self.width // 2
        self.ball_y = self.height // 2
        self.ball_dx = -5
        self.ball_dy = np.random.choice([-4, -3, -2, 2, 3, 4])
        self.score = 0
        return self._get_state()

    def step(self, action):
        reward = 0
        done = False
        
        for _ in range(self.frame_skip):
            if action == 1:
                self.paddle_y = max(0, self.paddle_y - 5)
            elif action == 2:
                self.paddle_y = min(self.height - self.paddle_height, self.paddle_y + 5)

            self.ball_x += self.ball_dx
            self.ball_y += self.ball_dy

            if self.ball_y <= 0 or self.ball_y >= self.height - self.ball_size:
                self.ball_dy *= -1

            if (self.ball_x <= self.paddle_width and 
                self.paddle_y <= self.ball_y <= self.paddle_y + self.paddle_height):
                if not self.last_hit: #ADD THIS CONDITION
                    self.ball_dx *= -1
                    self.score += 1
                    reward += 5.0
                    # Add proximity reward
                    hit_pos = (self.ball_y - self.paddle_y) / self.paddle_height
                    if self.reward_distance:
                        reward += 0.5 * (1.0 - abs(hit_pos - 0.5))
                    self.last_hit = True #SET TO TRUE
            else:
                self.last_hit = False #SET TO FALSE

            if self.ball_x >= self.width - self.ball_size:
                self.ball_dx *= -1

            if self.ball_x <= 0:
                reward = -1.0
                done = True
                break

            # Small reward for paddle following ball
            if not done:
                paddle_center = self.paddle_y + self.paddle_height/2
                ball_dist = abs(self.ball_y - paddle_center) / self.height
                if self.reward_distance:
                    reward += 0.01 * (1.0 - ball_dist)

        return self._get_state(), reward, done

    def _get_state(self):
        return np.array([
            self.paddle_y / self.height,
            self.ball_x / self.width,
            self.ball_y / self.height,
            self.ball_dx / 10,
            self.ball_dy / 10,
            self.score / 20.0  # Normalized score
        ], dtype=np.float32)
    

    def give_state(self, state):
        self.paddle_y = int(state[0] * self.height)
        self.ball_x = int(state[1] * self.width)
        self.ball_y = int(state[2] * self.height)
        self.ball_dx = int(state[3] * 10)
        self.ball_dy = int(state[4] * 10)
        self.score = int(state[5] * 20)
        return self._get_state()


    def render(self, return_image=False):
        self.screen.fill((0, 0, 0))
        pygame.draw.rect(self.screen, (255, 255, 255),
                        (5, self.paddle_y, self.paddle_width, self.paddle_height))
        pygame.draw.rect(self.screen, (255, 255, 255),
                        (self.ball_x, self.ball_y, self.ball_size, self.ball_size))
        
        # Render score
        font = pygame.font.Font(None, 36)
        score_text = font.render(f'Score: {self.score}', True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))
        
        pygame.display.flip()
        self.clock.tick(self.config.get('FPS', 60))
        
        if return_image:
            image = pygame.surfarray.array3d(self.screen)
            return np.transpose(image, (1, 0, 2))
        

    def render_surface(self, offset_x=0, offset_y=0, epoch=None, rect_dim = None):
        # Draw a rectangle around the game
        rect_dim = rect_dim if rect_dim is not None else (self.width, self.height)
        
        pygame.draw.rect(self.screen, (255, 255, 255),
                        (offset_x, offset_y, rect_dim[0], rect_dim[1]), 2)  # 2 is the border width
        pygame.draw.line(self.screen, (0, 0, 0), (offset_x, offset_y), (offset_x, offset_y + rect_dim[1]), 2)
        
        # Draw the paddle
        pygame.draw.rect(self.screen, (255, 255, 255),
                        (5 + offset_x, self.paddle_y + offset_y, self.paddle_width, self.paddle_height))

        # Draw the ball
        pygame.draw.rect(self.screen, (255, 255, 255),
                        (self.ball_x + offset_x, self.ball_y + offset_y, self.ball_size, self.ball_size))

        # Render the score
        font = pygame.font.Font(None, 36)
        score_text = font.render(f'Score: {self.score}', True, (255, 255, 255))
        if epoch is not None:
            score_text = font.render(f'Score: {self.score} , Epoch : {str(epoch)}', True, (255, 255, 255))
        self.screen.blit(score_text, (10 + offset_x, 10 + offset_y))

        # pygame.display.flip() # Remove this line
        self.clock.tick(self.config.get('FPS', 60))

    
    def render_text_on_screen(self, text, x, y):
        font = pygame.font.Font(None, 36)
        text_surface = font.render(str(text), True, (255, 255, 255))
        self.screen.blit(text_surface, (x, y))

    def close(self):
        pygame.quit()