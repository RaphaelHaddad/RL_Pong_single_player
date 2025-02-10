# r2d2_network.py
import torch
import torch.nn as nn

class R2D2Network(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.feature_net = nn.Sequential(
            nn.Linear(config['STATE_SIZE'], config['HIDDEN_SIZE']),
            nn.ReLU(),
            nn.Linear(config['HIDDEN_SIZE'], config['HIDDEN_SIZE']),
            nn.ReLU()
        )
        
        self.lstm = nn.LSTM(
            config['HIDDEN_SIZE'], 
            config['HIDDEN_SIZE'],
            num_layers=config['NUM_LSTM_LAYERS'],
            batch_first=True
        )
        
        self.advantage = nn.Linear(config['HIDDEN_SIZE'], config['ACTION_SIZE'])
        self.value = nn.Linear(config['HIDDEN_SIZE'], 1)
        
    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        seq_length = x.size(1)
        
        x = x.view(-1, x.size(-1))
        features = self.feature_net(x)
        features = features.view(batch_size, seq_length, -1)
        lstm_out, hidden = self.lstm(features, hidden)
        lstm_out = lstm_out[:, -1, :]
        
        advantage = self.advantage(lstm_out)
        value = self.value(lstm_out)
        
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values, hidden
        
    def save(self, path):
        torch.save(self.state_dict(), path)
        
    def load(self, path):
        self.load_state_dict(torch.load(path))