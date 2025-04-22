import torch
import torch.nn as nn
import numpy as np

class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, hidden_size=512):
        super(Encoder, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.hidden_size = hidden_size
        
        self.rnn1 = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        
        self.rnn2 = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        
    def forward(self, x):
        # Debug input shape
        print(f"Encoder input shape BEFORE processing: {x.shape}")
        
        # Ensure input has correct dimensions
        if len(x.shape) == 2:  # (batch_size, seq_len)
            x = x.unsqueeze(-1)  # Add feature dimension
        elif len(x.shape) == 3:  # (batch_size, seq_len, features)
            if x.shape[1] != self.seq_len or x.shape[2] != self.n_features:
                # Reshape to (batch_size, seq_len, features)
                x = x.reshape(x.shape[0], self.seq_len, self.n_features)
        
        print(f"Encoder input shape AFTER processing: {x.shape}")
        
        x, (hidden_n, cell_n) = self.rnn1(x)
        x, (hidden_n, cell_n) = self.rnn2(x)
        
        return x, (hidden_n, cell_n)

class Decoder(nn.Module):
    def __init__(self, seq_len, n_features, hidden_size=512):
        super(Decoder, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.hidden_size = hidden_size
        
        self.rnn1 = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        
        self.rnn2 = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        
        self.output_layer = nn.Linear(hidden_size, n_features)
        
    def forward(self, x, hidden):
        print(f"Decoder input shape: {x.shape}")
        
        x, (hidden_n, cell_n) = self.rnn1(x, hidden)
        x, (hidden_n, cell_n) = self.rnn2(x)
        
        x = self.output_layer(x)
        print(f"Decoder output shape: {x.shape}")
        return x

class Autoencoder(nn.Module):
    def __init__(self, seq_len, n_features):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(seq_len, n_features)
        self.decoder = Decoder(seq_len, n_features)
        
    def forward(self, x):
        x, hidden = self.encoder(x)
        x = self.decoder(x, hidden)
        return x

def create_dataset(df):
    # Convert DataFrame to numpy array
    sequence = df.values
    
    # Get actual dimensions
    n_rows, n_features = sequence.shape
    print(f"Dataset shape: {sequence.shape}")
    
    # Validate dimensions
    if n_rows % n_features != 0:
        print(f"Warning: Number of rows ({n_rows}) is not divisible by number of features ({n_features})")
        # Adjust the sequence length to be divisible
        n_rows = (n_rows // n_features) * n_features
        sequence = sequence[:n_rows]
        print(f"Adjusted dataset shape: {sequence.shape}")
    
    return sequence, n_rows, n_features 