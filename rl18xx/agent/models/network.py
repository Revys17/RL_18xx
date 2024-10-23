"""Neural network architecture for 18XX games."""
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class BoardEncoder(nn.Module):
    """Encodes the board state using a CNN."""
    
    def __init__(self, in_channels: int = 54):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Position encoding for hex grid
        self.pos_encoding = nn.Parameter(torch.randn(1, 64, 20, 20))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add positional encoding
        x = self.conv1(x)
        x = x + self.pos_encoding
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.conv3(x))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        
        return x.flatten(1)

class PlayerEncoder(nn.Module):
    """Encodes player states."""
    
    def __init__(self, input_dim: int = 45):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class CorporationEncoder(nn.Module):
    """Encodes corporation states."""
    
    def __init__(self, input_dim: int = 34):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class MarketEncoder(nn.Module):
    """Encodes market state."""
    
    def __init__(self, input_dim: int = 57):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class PhaseEncoder(nn.Module):
    """Encodes game phase information."""
    
    def __init__(self, input_dim: int = 14):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class MultiHeadAttention(nn.Module):
    """Multi-head attention for combining different state components."""
    
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        
    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        # Expect input shape: (seq_len, batch, embed_dim)
        attn_output, _ = self.attention(queries, keys, values)
        return attn_output

class Game18XXNetwork(nn.Module):
    """Complete network architecture for 18XX game."""
    
    def __init__(self, 
                 action_dim: int,
                 board_channels: int = 54,
                 player_dim: int = 45,
                 corporation_dim: int = 34,
                 market_dim: int = 57,
                 phase_dim: int = 14):
        super().__init__()
        
        # Component encoders
        self.board_encoder = BoardEncoder(board_channels)
        self.player_encoder = PlayerEncoder(player_dim)
        self.corporation_encoder = CorporationEncoder(corporation_dim)
        self.market_encoder = MarketEncoder(market_dim)
        self.phase_encoder = PhaseEncoder(phase_dim)
        
        # Attention mechanism
        self.attention = MultiHeadAttention(256, num_heads=4)
        
        # Final processing
        self.fusion_layer = nn.Sequential(
            nn.Linear(1152, 512),  # 256 + 256 + 256 + 256 + 128
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU()
        )
        
        # Action head
        self.action_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
    def forward(self, state_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network.
        
        Args:
            state_dict: Dictionary containing state components:
                - 'board': Board state tensor
                - 'players': Player state tensor
                - 'corporations': Corporation state tensor
                - 'market': Market state tensor
                - 'phase': Phase state tensor
                
        Returns:
            Tuple of (action_logits, value)
        """
        # Encode each component
        board_enc = self.board_encoder(state_dict['board'])
        player_enc = self.player_encoder(state_dict['players'])
        corp_enc = self.corporation_encoder(state_dict['corporations'])
        market_enc = self.market_encoder(state_dict['market'])
        phase_enc = self.phase_encoder(state_dict['phase'])
        
        # Prepare for attention
        # Reshape encodings to (seq_len, batch, features)
        queries = board_enc.unsqueeze(0)
        keys = torch.cat([
            player_enc.unsqueeze(0),
            corp_enc.unsqueeze(0),
            market_enc.unsqueeze(0)
        ], dim=0)
        values = keys.clone()
        
        # Apply attention
        attended = self.attention(queries, keys, values)
        attended = attended.squeeze(0)
        
        # Concatenate all features
        combined = torch.cat([
            attended,
            board_enc,
            market_enc,
            phase_enc
        ], dim=-1)
        
        # Final processing
        features = self.fusion_layer(combined)
        
        # Generate outputs
        action_logits = self.action_head(features)
        value = self.value_head(features)
        
        return action_logits, value
    
    def get_action_mask(self, logits: torch.Tensor, valid_actions: torch.Tensor) -> torch.Tensor:
        """Apply action mask to logits.
        
        Args:
            logits: Raw action logits
            valid_actions: Boolean mask of valid actions
            
        Returns:
            Masked logits (invalid actions set to -inf)
        """
        mask = ~valid_actions
        logits[mask] = float('-inf')
        return logits