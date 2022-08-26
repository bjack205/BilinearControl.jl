import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import json
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

ratio = 1

class Spaces:
    def __init__(self,shape, high=0, low=0):
        self.shape = shape
        self.high = high
        self.low = low
    def sample(self):
        return np.random.rand(*self.shape)*(self.high-self.low) + self.low

class DeterministicNetwork(nn.Module):
    def __init__(self, num_state, num_actions, num_outs, hidden_dim, out_space=None, use_relu=False):
        super(DeterministicNetwork, self).__init__()
        num_inputs = num_state + num_actions
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_outs)
        self.noise = torch.Tensor(num_outs)

        self.apply(weights_init_)
        self.use_tanh = False
        if use_relu:
            self.activatation = torch.relu
        else:
            self.activatation = torch.tanh

        self.loss_history = []
        self.vloss_history = []
        self.tloss_history = []
        # action rescaling
        # ipdb.set_trace()
        if out_space is None:
            self.out_scale = torch.tensor(1.)
            self.out_bias = torch.tensor(0.)
        elif (out_space.high - out_space.low).sum() > 1e3:
            self.use_tanh = False
        else:
            self.out_scale = torch.FloatTensor(
                (out_space.high - out_space.low) / 2.)
            self.out_bias = torch.FloatTensor(
                (out_space.high + out_space.low) / 2.)

    def forward(self, input):
        x = self.activatation(self.linear1(input))
        x = self.activatation(self.linear2(x))
        if self.use_tanh:
            mean = torch.tanh(self.mean(x)) * self.out_scale + self.out_bias
        else:
            mean = self.mean(x) 
        return mean

    def to(self, device):
        if self.use_tanh:
            self.out_scale = self.out_scale.to(device)
            self.out_bias = self.out_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicNetwork, self).to(device)

    def save_wts(self, alpha, outfile):
        W1 = self.linear1.weight.data.cpu().numpy()
        b1 = self.linear1.bias.data.cpu().numpy()
        W2 = self.linear2.weight.data.cpu().numpy()
        b2 = self.linear2.bias.data.cpu().numpy()
        W3 = self.mean.weight.data.cpu().numpy()
        b3 = self.mean.bias.data.cpu().numpy()

        data = {
            "name": "double integrator",
            "W1": W1.tolist(),
            "b1": b1.tolist(),
            "W2": W2.tolist(),
            "b2": b2.tolist(),
            "W3": W3.tolist(),
            "b3": b3.tolist(),
            "loss": self.loss_history,
            "vloss": self.vloss_history,
            "tloss": self.tloss_history,
            "alpha": alpha,
        }
        f = open(outfile, "w")
        json.dump(data, f, indent=2)

        # np.save(save_dir+'l1wt.npy', self.linear1.weight.data.cpu().numpy())
        # np.save(save_dir+'l2wt.npy', self.linear2.weight.data.cpu().numpy())
        # np.save(save_dir+'l3wt.npy', self.linear3.weight.data.cpu().numpy())

        # np.save(save_dir+'l1bias.npy', self.linear1.bias.data.cpu().numpy())
        # np.save(save_dir+'l2bias.npy', self.linear2.bias.data.cpu().numpy())
        # np.save(save_dir+'l3bias.npy', self.linear3.bias.data.cpu().numpy())