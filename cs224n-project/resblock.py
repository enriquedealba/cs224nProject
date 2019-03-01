import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint

class ResBlock(nn.Module):
    """ Simple highway network
    """
    def __init__(self, f_embed):
        """ Initialize Highway Network
            @param embed_size (int): embedding size/dimension
        """
        super(ResBlock, self).__init__()
        self.res_block1 = nn.Linear(f_embed, f_embed)
        nn.init.xavier_normal_(self.res_block1.weight, gain=1)
        self.res_block2 = nn.Linear(f_embed, f_embed)
        nn.init.xavier_normal_(self.res_block2.weight, gain=1)
        
    def forward(self, x_convout):
        """ Takes minibatch of x_convout and returns x_highway
            @param x_convout (Tensor): tensor of shape (batch_size, word_embed)
            
            @retuns x_highway (Tensor): tensor of shape (batch_size, word_embed)
        """
        #x_proj = F.relu(self.res_block1(x_convout))
        #x_gate = F.relu(self.res_block2(x_proj))
        #x_highway = torch.mul(x_gate, x_proj) + torch.mul((1. - x_gate), x_convout)
        x_1 = F.relu(self.res_block1(x_convout)) + x_convout
        x = F.relu(self.res_block2(x_1)) + x_1
        return x

### END YOUR CODE 

