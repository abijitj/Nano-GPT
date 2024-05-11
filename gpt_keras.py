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
    
    def call(self, x): 
        x = self.relu(self.l1(n_embd))
        x = self.dropout(self.l2(x))
        return x


class Block(k.Model): 
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head): 
        super().__init__()
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = k.layers.LayerNormalization()
        self.ln2 = k.layers.LayerNormalization()
    
    def call(self, x): 
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    

class GPTModel(k.Model): 
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = k.layers.Embedding(vocab_size, n_embd)
        self.position_embedding_table = k.layers.Embedding(block_size, n_embd)
        self.block = k.models.Sequential()
        for _ in range(n_layer): 
            self.block.add(Block(n_embd, n_head=n_head))
        self.ln_f = k.layers.LayerNormalization(n_embd) # final layer norm
        self.lm_head = k.layers.Dense(n_embd, vocab_size)
    
    def call(self, idx, targets=None): 
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(np.arange(T))#, device=device)) # (T,C)
        x = tok_emb + pos_emb

        x = self.block(x)
        logits = self.lm_head(tok_emb) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = k.losses.CategoricalCrossentropy()(targets, logits)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = k.activations.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = np.random.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = k.layers.concatenate((idx, idx_next), dim=1) # (B, T+1)
        return idx