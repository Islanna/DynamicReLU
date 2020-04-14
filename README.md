# DynamicReLU
Implementation of [Dynamic ReLU(types A,B)](https://arxiv.org/abs/2003.10027) on Pytorch.

## Example
```
import torch.nn as nn
from dyrelu import DyReluB

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, 5)
        self.relu = DyReLUB(10, conv_type='2d')

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        return x
```
