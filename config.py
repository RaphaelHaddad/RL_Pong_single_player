import torch
import os
TARGET_UPDATE_FREQ = 3

CHECKPOINT_DIR = 'checkpoints_final'

CONFIG = {
    # Environment
    'SCREEN_WIDTH': 600,
    'SCREEN_HEIGHT': 400,
    'FPS': 30,
    
    # Training
    'NUM_EPISODES': 2000,
    'BATCH_SIZE': 64,
    'SEQUENCE_LENGTH': 8,
    'MEMORY_SIZE': 50000,
    'GAMMA': 0.99,
    'LEARNING_RATE': 1.5e-3,
    
    # Model
    'STATE_SIZE': 6,
    'HIDDEN_SIZE': 256,
    'ACTION_SIZE': 3,
    'NUM_LSTM_LAYERS': 2,
    
    # Exploration
    'EPSILON_START': 1.0,
    'EPSILON_END': 0.01,
    'EPSILON_DECAY': 0.999,
    
    # Saving/Loading
    'CHECKPOINT_DIR': CHECKPOINT_DIR,
    'SAVE_FREQUENCY': 100,
    'FRAME_SKIP': 6,         # Skip frames in PongEnv
    
    # Device
    'DEVICE': torch.device("mps"),
    'TAU': 0.01,  # Soft update parameter

    # Save initial model
    'MODEL_PATH': os.path.join(CHECKPOINT_DIR, 'model.pth'),
    'CONFIG_PATH': os.path.join(CHECKPOINT_DIR, 'config.json'),

    # Others
    'LEN_REWARD_HISTORY': 100,
    'SAMPLING_METHOD': 'random_sequential',
    'REWARD_DISTANCE': True,
    'PLAY_AFTER_RECORD' : True,
    'MIN_EPISODE_PLAY' : 50,
    'FRAME_RATE_EPISODE_GAME' : 10,
}