import snntorch as snn 
import torch 
import torch.nn as nn 

# .memory source: https://github.com/facebookresearch/XLM/blob/main/xlm/model
from .memory import HashingMemory

class Net_small(nn.Module):
    def __init__(self, num_inputs, num_hidden, beta, num_outputs, num_steps, batch_size, input_width):
        super().__init__()
        self.num_steps = num_steps 
        self.batch_size = batch_size 
        self.input_width = input_width 
        self.num_inputs = num_inputs 

        # Initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.lif2 = snn.Leaky(beta=beta)
        self.fc3 = nn.Linear(num_hidden, num_hidden)
        self.lif3 = snn.Leaky(beta=beta)
        self.fc4 = nn.Linear(num_outputs, num_outputs)
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
class MOME_layer(nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int): 
        self.num_experts = num_inputs
        self.experts = [] 
        for _ in range(self.num_experts): 
            self.experts.append(nn.Linear(1, num_outputs))

        self.memory = HashingMemory.build(num_inputs, num_outputs, self.params) 
    
    def top_k(self): 
        # select top k experts whose corresponding product keys have the highest inner products with the query vector 
        pass 

    def forward(self, data):
        top_k = self.top_k()
        output = [] 
        data = data + self.memory(data)

        for i in range(len(data)): 
            if self.experts[i] in top_k: 
                output.append(self.experts[i](data[i]))
        
        # work must be done to ensure that this output tensor is in the proper shape to pass to next layer 
        # study the output dimensions of a regular nn.Linear of input size input size and output size (output size)
        return torch.Tensor(output)


    

class Loop_sequential(nn.Sequential):
    def __init__(self, loop_count: int, **kwargs):
        self.loop_count = loop_count 
        super.__init__(kwargs)
    
    def forward_loop(self, data): 
        for _ in range(self.loop_count):
            data = self.forward(data)
        return data 