from einops import einsum
import snntorch as snn 
import torch 
import torch.nn as nn 

# .memory source: https://github.com/facebookresearch/XLM/blob/main/xlm/model
from einops.layers.torch import Rearrange
import torch.nn.functional as F


def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.gamma

class ProductKeyMemory(nn.Module):
    def __init__(self, dim, num_keys):
        super().__init__()
        self.dim = dim
        self.num_keys = num_keys
        self.keys = nn.Parameter(torch.randn(num_keys, dim // 2))
        
    def forward(self, query):
        query = query.view(query.shape[0], 2, -1)
        dots = torch.einsum('bkd,nd->bkn', query, self.keys)
        return dots.view(query.shape[0], -1)

class PEER(nn.Module):
    def __init__(
        self,
        dim,
        *,
        heads=8,
        num_experts=1_000_000,
        num_experts_per_head=16,
        dim_key=None,
        product_key_topk=None,
        separate_embed_per_head=False,
        pre_rmsnorm=False,
        dropout=0.
    ):
        super().__init__()

        self.norm = RMSNorm(dim) if pre_rmsnorm else nn.Identity()

        self.heads = heads
        self.separate_embed_per_head = separate_embed_per_head
        self.num_experts = num_experts

        num_expert_sets = heads if separate_embed_per_head else 1

        self.weight_down_embed = nn.Embedding(num_experts * num_expert_sets, dim)
        self.weight_up_embed = nn.Embedding(num_experts * num_expert_sets, dim)

        self.activation = snn.LIF()

        assert (num_experts ** 0.5).is_integer(), '`num_experts` needs to be a square'
        assert (dim % 2) == 0, 'feature dimension should be divisible by 2'

        dim_key = default(dim_key, dim // 2)
        self.num_keys = int(num_experts ** 0.5)

        self.to_queries = nn.Sequential(
            nn.Linear(dim, dim_key * heads * 2, bias=False),
            Rearrange('b n (p h d) -> p b n h d', p=2, h=heads)
        )

        self.product_key_topk = default(product_key_topk, num_experts_per_head)
        self.num_experts_per_head = num_experts_per_head

        self.keys = nn.Parameter(torch.randn(heads, self.num_keys, 2, dim_key))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm(x)

        queries = self.to_queries(x)

        sim = einsum(queries, self.keys, 'p b n h d, h k p d -> p b n h k')

        (scores_x, scores_y), (indices_x, indices_y) = [s.topk(self.product_key_topk, dim=-1) for s in sim]

        all_scores = scores_x.unsqueeze(-1) + scores_y.unsqueeze(-2)
        all_indices = indices_x.unsqueeze(-1) * self.num_keys + indices_y.unsqueeze(-2)

        all_scores = all_scores.view(*all_scores.shape[:-2], -1)
        all_indices = all_indices.view(*all_indices.shape[:-2], -1)

        scores, pk_indices = all_scores.topk(self.num_experts_per_head, dim=-1)
        indices = all_indices.gather(-1, pk_indices)

        if self.separate_embed_per_head:
            head_expert_offsets = torch.arange(self.heads, device=x.device) * self.num_experts
            indices = indices + head_expert_offsets.view(1, 1, -1, 1)

        weights_down = self.weight_down_embed(pk_indices)
        weights_up = self.weight_up_embed(pk_indices)

        x = einsum(x, weights_down, 'b n d, b n h k d -> b n h k')

        x = self.activation(x) # LIF 
        
        x = self.dropout(x)

        x = x * F.softmax(scores, dim=-1)

        x = einsum(x, weights_up, 'b n h k, b n h k d -> b n d')

        return x



'''
def forward(self, x):
    b, s = x.shape
    positions = torch.arange(s, device=x.device).unsqueeze(0).expand(b, s)
    
    x = self.token_embedding(x) + self.position_embedding(positions)
    
    for layer in self.layers:
        x = layer(x)
    
    x = self.layer_norm(x)
    logits = self.lm_head(x)
    return logits
'''
class MOME_net(nn.Module):
    def __init__(self, num_inputs, num_hidden, beta, num_outputs, num_steps, batch_size):
        super().__init__()
        self.num_steps = num_steps 
        self.batch_size = batch_size 
        self.num_inputs = num_inputs 

        # Initialize layers
        self.fc1 = MOME_layer(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = MOME_layer(num_hidden, num_hidden)
        self.lif2 = snn.Leaky(beta=beta)
        self.fc3 = MOME_layer(num_hidden, num_hidden)
        self.lif3 = snn.Leaky(beta=beta)
        self.fc4 = MOME_layer(num_outputs, num_outputs)
        self.lif4 = snn.Leaky(beta=beta)

    def forward(self, x):

        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        
        # Record the final layer
        spk1_rec = []
        spk2_rec = []
        spk3_rec = []
        spk4_rec = []
        mem4_rec = [] 

        for step in range(self.num_steps):
            cur1 = self.fc1(x[0:self.batch_size, step, 0:self.num_inputs]) 
            spk1, mem1 = self.lif1(cur1, mem1)
            spk1_rec.append(spk1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)
            spk3_rec.append(spk3)
            cur4 = self.fc4(spk3)
            spk4, mem4 = self.lif4(cur4, mem4)
            spk4_rec.append(spk4)
            mem4_rec.append(mem4)
        

        spk1_rec = torch.stack(spk1_rec)
        spk2_rec = torch.stack(spk2_rec)
        spk3_rec = torch.stack(spk3_rec)
        spk4_rec = torch.stack(spk4_rec)
        
        return [spk1_rec, spk2_rec, spk3_rec, spk4_rec], mem4
    


# https://github.com/facebookresearch/XLM/blob/main/xlm/model/transformer.py 
# top k selection involves generating a k value for each expert, that allows evaluation of which expert is best for a specific task 
# this is the most complex portion of a mixture of experts model 
# Each MOME layer must make a selection of experts each time 
'''
        self,
        dim, -> num inputs 
        *,
        heads=8,
        num_experts=1_000_000,-> also num inputs? 
        num_experts_per_head=16, -> other num outputs 
        activation=nn.GELU,
        dim_key=None,
        product_key_topk=None, -> num outputs 
        separate_embed_per_head=False,
        pre_rmsnorm=False,
        dropout=0.
'''
class MOME_layer(nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int): 
        super().__init__()
        self.num_experts = num_inputs
        self.experts = [] 
        self.linear = nn.Linear(num_inputs, num_outputs)
        #remember to register args of hashing memory 

        self.peer = PEER(num_inputs, heads=1, num_experts=num_inputs, product_key_topk=16)

    def forward(self, data):
        data = self.peer(data)
        
        return self.linear(data)

        
        # work must be done to ensure that this output tensor is in the proper shape to pass to next layer 
        # study the output dimensions of a regular nn.Linear of input size input size and output size (output size)


    

class Loop_sequential(nn.Sequential):
    def __init__(self, loop_count: int, **kwargs):
        self.loop_count = loop_count 
        super.__init__(kwargs)
    
    def forward(self, data): 
        for _ in range(self.loop_count):
            data = super.forward(data)
        return data 
    
    
        
if __name__ == "__main__":

    # num_inputs, num_hidden, beta, num_outputs, num_steps, batch_size, params)
    mome = MOME_net(16, 16, .95, 16, 100, 4)