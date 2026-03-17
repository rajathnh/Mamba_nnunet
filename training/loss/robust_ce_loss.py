import torch
from torch import nn,Tensor
import numpy as np

class RobustEntropyLoss(nn.CrossEntropyLoss):
    def forward(self,input:Tensor,target:Tensor) -> Tensor:
        if target.ndim == input.ndim:
            assert target.shape[1] == 1
            target=target[:,0]
        return super().forward(input,target.long())
class TopKLoss(RobustEntropyLoss):
    def __init__(self,weight=None,ignore_index:int=-100,float = 10,label_smoothing:float=0):
        self.k = k 
        super(TopKLoss,self).__init__(weight,False,ignore_index,reduce=False,label_smoothing=label_smoothing)
    
    def forward(self,inp,target):
        target = target[:,0].long()
        res = super(TopKLoss,self).forward(inp,target)
        num_voxels = np.prod(res.shape,dtype = np.int64)
        res, _ = torch.topk(res,view((-1, )),int(num_voxels*self.k/100),sorted = False)
        return res.mean()
    
    