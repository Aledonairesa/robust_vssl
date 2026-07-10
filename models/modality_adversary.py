"""Modality-adversarial classifier with a gradient reversal layer."""

import torch
from torch import nn


class GradientReverse(torch.autograd.Function):
    """Identity in the forward pass, sign-reversed gradient in backward."""

    @staticmethod
    def forward(ctx, inputs, scale):
        ctx.scale = scale
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.scale * grad_output, None


def gradient_reverse(inputs, scale=1.0):
    return GradientReverse.apply(inputs, scale)


class ModalityAdversary(nn.Module):
    """MLP that predicts image-vs-audio modality after gradient reversal."""

    def __init__(self, embedding_dim=512, hidden_dim=512, num_layers=5,
                 grl_scale=1.0):
        super(ModalityAdversary, self).__init__()
        if num_layers < 1:
            raise ValueError('num_layers must be at least 1')
        if embedding_dim <= 0:
            raise ValueError('embedding_dim must be positive')
        if hidden_dim <= 0:
            raise ValueError('hidden_dim must be positive')

        self.grl_scale = float(grl_scale)

        layers = []
        input_dim = embedding_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))

        self.classifier = nn.Sequential(*layers)

    def forward(self, embeddings):
        reversed_embeddings = gradient_reverse(embeddings, self.grl_scale)
        return self.classifier(reversed_embeddings).squeeze(-1)
