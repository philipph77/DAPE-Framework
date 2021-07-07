import torch
import torch.nn as nn
import torch.nn.functional as F

class ConstrainedConv2d(nn.Conv2d):
    def forward(self, input, max_norm_val=1.):
        return F.conv2d(input, self._max_norm(self.weight, max_norm_val), self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
    
    def _max_norm(self, w, _max_norm_val, _eps=1e-3):
        norm = w.norm(2, dim=0, keepdim=True)
        desired = torch.clamp(norm, 0, _max_norm_val)
        return w * (desired / (_eps + norm))

class ConstrainedLinear(nn.Linear):
    def forward(self, input, max_norm_val=.25):
        return F.linear(input, self._max_norm(self.weight, max_norm_val), self.bias)

    def _max_norm(self, w, _max_norm_val, _eps=1e-3):
        norm = w.norm(2, dim=0, keepdim=True)
        desired = torch.clamp(norm, 0, _max_norm_val)
        return w * (desired / (_eps + norm))