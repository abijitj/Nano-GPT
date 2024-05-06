import numpy as np
import keras as k

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 1e-2 #3e-3
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_head = 4
dropout = 0.2
n_layer = 4
# ------------

class Head(k.Model): 
    def __init__(self, head_size): 
        super().__init__()
        self.key = k.layers.Dense(n_embd, head_size, use_bias=False)
        self.query = k.layers.Dense(n_embd, head_size, use_bias=False)
        self.value = k.layers.Dense(n_embd, head_size, use_bias=False)
        
        self.dropout = k.layers.Dropout(dropout)

    def call(self, x): 
        B, T, C = x.shape
        k = self.key(x) # (B, T, C)
        q = self.query(x) # (B, T, C)

        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('inf')) # (B, T, T)
        wei = k.activations.softmax(wei) # (B, T, T)
        wei = self.dropout(wei)

        # perform weighted aggregation of the values
        v = self.value(x)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out 


class MultiHeadAttention(k.Model): 
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size): 
        super().__init__()
        self.heads = [Head(head_size) for _ in range(num_heads)]
        self.proj = k.layers.Dense(n_embd, n_embd)
        self.dropout = k.layers.Dropout(dropout)

    def call(self, x): 
        out = k.layers.concatenate([h(x) for h in self.heads], axis=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(k.Model):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd): 
        super().__init__()
        self.l1 = k.layers.Dense(n_embd, 4 * n_embd)
        self.relu = k.layers.Activation('relu')
        self.l2 = k.layers.Dense(4 * n_embd, n_embd)
        self.dropout = k.layers.Dropout(dropout)
    
    def forward(self, x): 
        x = self.relu(self.l1(n_embd))
        x = self.dropout(self.l2(x))
        return x