#num_inputs, num_hidden, beta, num_outputs, num_steps, batch_size, input_width 
import torch.nn as nn
import snntorch as snn 
import torch 
class Neuron_soup(nn.Module):
    def __init__(self, num_inputs, num_hidden, beta, num_outputs, num_steps, batch_size, input_width):
        super().__init__()
        self.num_steps = num_steps 
        self.batch_size = batch_size 
        self.input_width = input_width 
        self.num_layers = 4 
        self.num_possible_paths = (self.num_layers - 2) * num_hidden + num_outputs + num_inputs 

        # Initialize layers
        self.fc1 = nn.Linear(num_inputs + (num_hidden * 2) + num_outputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear((num_hidden * 2) + num_outputs, num_hidden)
        self.lif2 = snn.Leaky(beta=beta)
        self.fc3 = nn.Linear((num_hidden * 2) + num_outputs, num_hidden)
        self.lif3 = snn.Leaky(beta=beta)
        self.fc4 = nn.Linear((num_hidden * 3), num_outputs)
        self.lif4 = snn.Leaky(beta=beta)

        self.prev_spk2 = torch.zeros(batch_size, num_hidden)
        
        #print(f'zeroes_tensor: {self.prev_spk2.size()}')
        self.prev_spk3 = torch.zeros(batch_size, num_hidden) 
        self.prev_spk4 = torch.zeros(batch_size, num_outputs)

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
        prev_spk2 = self.prev_spk2 
        prev_spk3 = self.prev_spk3
        prev_spk4 = self.prev_spk4 

        in_size = x.size()[0]

        if(in_size < self.batch_size): 
            x = torch.cat((x, torch.zeros((self.batch_size - in_size, self.num_steps, self.input_width))))

        for step in range(self.num_steps):
            #print(f'forward tensor: {x[0:self.batch_size, step, 0:self.input_width].size()}')
            #print(f'combined input tensor: {torch.cat((x[0:self.batch_size, step, 0:self.input_width], prev_spk2, prev_spk3, prev_spk4), dim=1).size()}')
            cur1 = self.fc1(torch.cat((x[0:self.batch_size, step, 0:self.input_width], prev_spk2.detach(), prev_spk3.detach(), prev_spk4.detach()), dim=1)) 
            spk1, mem1 = self.lif1(cur1, mem1)
            spk1_rec.append(spk1)
            #print(f'layer 2 input: {torch.cat((spk1, prev_spk3, prev_spk4), dim=1).size()}')
            cur2 = self.fc2(torch.cat((spk1, prev_spk3.detach(), prev_spk4.detach()), dim=1))
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            cur3 = self.fc3(torch.cat((spk1, spk2, prev_spk4.detach()), dim=1))
            spk3, mem3 = self.lif3(cur3, mem3)
            spk3_rec.append(spk3)
            cur4 = self.fc4(torch.cat((spk1, spk2, spk3), dim=1))
            spk4, mem4 = self.lif4(cur4, mem4)
            spk4_rec.append(spk4)
            mem4_rec.append(mem4)
            prev_spk2 = spk2_rec[len(spk2_rec) - 1] 
            prev_spk3 = spk3_rec[len(spk3_rec) - 1]
            prev_spk4 = spk4_rec[len(spk4_rec) - 1]
        
        self.prev_spk2 = prev_spk2
        self.prev_spk3 = prev_spk3
        self.prev_spk4 = prev_spk4
        
        spk1_rec = torch.stack(spk1_rec)
        spk2_rec = torch.stack(spk2_rec)
        spk3_rec = torch.stack(spk3_rec)
        spk4_rec = torch.stack(spk4_rec)

        if(in_size < self.batch_size): 
            mem4 = mem4[0:in_size]
        
        return [spk1_rec, spk2_rec, spk3_rec, spk4_rec], mem4
    



