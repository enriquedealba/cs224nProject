import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint

class ODEfunc(nn.Module):
    def __init__(self, f_embed):
        """ Initialize Highway Network
            @param embed_size (int): embedding size/dimension
        """
        super(ODEfunc, self).__init__()
        self.ode_block1 = nn.Linear(f_embed+1, f_embed)
        nn.init.xavier_normal_(self.ode_block1.weight, gain=1)
        self.ode_block2 = nn.Linear(f_embed+1, f_embed)
        nn.init.xavier_normal_(self.ode_block2.weight, gain=1)
        
    #def forward_1(self, t, x_convout):
    #    x_1 = F.relu(self.ode_block1(x_convout))
    #    x = F.relu(self.ode_block2(x_1)) + x_1
    #    return x + x_convout

    #def forward_2(self, t, x_convout):
    #    t = float(t)
    #    t = np.sign(t) * np.min([2.5, np.abs(t)])
    #    x_1 = F.relu(t * self.ode_block1(x_convout))
    #    x = F.relu(t * self.ode_block2(x_1)) + (t * x_1)
    #    return x + x_convout

    def forward(self, t, x_convout):
        x = F.relu(x_convout)
        tt = torch.tensor([t] * x.shape[0]).unsqueeze(-1)
        ttx = torch.cat([tt, x], 1)
        x = F.relu(self.ode_block1(ttx))
        tt = torch.tensor([t] * x.shape[0]).unsqueeze(-1)
        ttx = torch.cat([tt, x], 1)
        x = self.ode_block2(ttx)
        return x + x_convout

class ODEblock(nn.Module):
    """ Simple highway network
    """
    def __init__(self, f_embed):
        """ Initialize Highway Network
            @param embed_size (int): embedding size/dimension
        """
        super(ODEblock, self).__init__()
        self.odefunc = ODEfunc(f_embed)
        self.integration_time = torch.tensor([0, 1]).float()
        
    def forward(self, x_convout):
        self.integration_time = self.integration_time.type_as(x_convout)
        out = odeint(self.odefunc, x_convout, self.integration_time, rtol=0.001, atol=0.001)
        return out[1]



