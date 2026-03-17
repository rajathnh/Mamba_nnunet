from torch import nn
class DeepSupervisionWrapper(nn.Module):
    def __init__(self,loss,weight_factors=None):
        super(DeepSupervisionWrapper,self).__init__()
        assert any([x!= 0 for x in weight_factors]),"1 weight should be != 0.0"
        self.weight_factors = tuple(weight_factors)
        self.loss = loss
    def forward(self,*args):
        assert all([isinstance(i,(tuple,list)) for i in args]),f"all args must be tuple or list"
        if self.weight_factors is None:
            weights = (1,)*len(args[0])
        else:
            weights = self.weight_factors
        return sum([weights[i]*self.loss(*inputs) for i, inputs in enumerate(zip(*args)) if weights[i] != 0.0])    