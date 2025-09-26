import torch
from torch.nn.modules import Module
from torch.nn.modules import Sequential
from ti_torch import TiLinear
from ti_torch import TiReLU
from ti_torch import TiMaxpool2d
from ti_torch import TiFlat
from ti_torch import TiInt8ToFloat
from ti_torch import TiFloatToInt8
from ti_torch import TiConv2d, TiConv2d_acc23
class TiMobileNet(TiNet):
    def __init__(self):
        super(TiMobileNet, self).__init__()
        self.data2int=TiFloatToInt8()
        self.forward_layers = self._make_layers()
    
    def _make_layers(self):
        layers = []
        layers += [

        ]
        return Sequential(*layers)
