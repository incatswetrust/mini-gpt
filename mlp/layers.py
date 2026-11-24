"""Own layers in the style of torch.nn: Embedding, Flatten, Linear, BatchNorm1d, Tanh.

They hold plain tensors and implement the math directly, no torch.nn inside.
"""
import torch


class Module:
    training = True

    def parameters(self):
        return []

    def train(self, mode=True):
        self.training = mode
        return self

    def eval(self):
        return self.train(False)

    def to(self, device):
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.to(device))
        return self


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, generator=None):
        self.weight = torch.randn((num_embeddings, embedding_dim), generator=generator)

    def __call__(self, IX):
        self.out = self.weight[IX]
        return self.out

    def parameters(self):
        return [self.weight]


class Flatten(Module):
    def __call__(self, x):
        self.out = x.view(x.shape[0], -1)
        return self.out


class Linear(Module):
    def __init__(self, fan_in, fan_out, bias=True, generator=None):
        # Kaiming-style 1/sqrt(fan_in): output has about the same scale as input.
        self.weight = torch.randn((fan_in, fan_out), generator=generator) / fan_in**0.5
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out = self.out + self.bias
        return self.out

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])


class BatchNorm1d(Module):
    """Normalize each feature to mean 0, std 1 over the batch, then scale and shift
    by trainable gamma, beta. In eval mode use running averages instead of batch
    statistics, so a single example can be processed."""

    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        if self.training:
            mean = x.mean(0, keepdim=True)
            var = x.var(0, keepdim=True, unbiased=False)
            with torch.no_grad():
                # Like torch.nn.BatchNorm1d: biased var to normalize, unbiased for the running estimate.
                n = x.shape[0]
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.squeeze(0)
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.squeeze(0) * n / (n - 1)
        else:
            mean, var = self.running_mean, self.running_var
        xhat = (x - mean) / torch.sqrt(var + self.eps)
        self.out = self.gamma * xhat + self.beta
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]


class Tanh(Module):
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out


class Sequential(Module):
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def train(self, mode=True):
        for layer in self.layers:
            layer.train(mode)
        return super().train(mode)

    def to(self, device):
        for layer in self.layers:
            layer.to(device)
        return self
