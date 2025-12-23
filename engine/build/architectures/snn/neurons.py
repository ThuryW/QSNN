import torch
from torch import nn
from torch import Tensor
from torch.autograd import Function


class TriangleSG(Function):

    @staticmethod
    def forward(ctx, mem: Tensor):
        spk = (mem >= 1.).float()
        ctx.save_for_backward(mem)
        return spk

    @staticmethod
    def backward(ctx, grad_spk: Tensor):
        mem, = ctx.saved_tensors
        grad_mem = grad_spk * (1. - (mem - 1.).abs()).clamp_min(0.)
        return grad_mem, None


class IFNeuron(nn.Module):

    def __init__(
        self,
        timesteps: int
    ):
        super().__init__()
        self.timesteps = timesteps
        self.sg_func = TriangleSG.apply
        self.vth = nn.Parameter(torch.tensor(1.))

    def forward(self, input: Tensor):
        input_t = input.reshape(self.timesteps, -1, *input.shape[1:]) / self.vth

        spks = []
        mem = torch.zeros_like(input_t[0])
        for t in range(self.timesteps):
            mem = mem + input_t[t]
            spk = self.sg_func(mem)
            mem = mem - spk
            spks.append(spk)

        return torch.cat(spks, dim=0) * self.vth


class Encoder(nn.Module):

    def __init__(
        self,
        timesteps: int
    ):
        super().__init__()
        self.timesteps = timesteps

    def forward(self, input: Tensor):
        return input.repeat(self.timesteps, *([1] * (input.dim() - 1)))
    

class Decoder(nn.Module):

    def __init__(
        self,
        timesteps: int
    ):
        super().__init__()
        self.timesteps = timesteps

    def forward(self, input: Tensor):
        return input.reshape(self.timesteps, -1, *input.shape[1:]).mean(dim=0)


if __name__ == "__main__":
    pass