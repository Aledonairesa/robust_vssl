import math

from torch import nn


class LoRALinear(nn.Module):
    """Frozen linear layer with a trainable low-rank residual update."""

    def __init__(self, base_layer, rank, alpha=None, dropout=0.05):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError('LoRALinear requires an nn.Linear base layer')
        if rank <= 0:
            raise ValueError('LoRA rank must be positive')
        if rank > min(base_layer.in_features, base_layer.out_features):
            raise ValueError(
                'LoRA rank {} exceeds the base layer dimensions ({}, {})'.format(
                    rank, base_layer.in_features, base_layer.out_features))

        self.base_layer = base_layer
        self.rank = rank
        self.alpha = float(2 * rank if alpha is None else alpha)
        self.scaling = self.alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Linear(base_layer.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base_layer.out_features, bias=False)

        # B starts at zero so enabling LoRA initially leaves the pretrained
        # model output unchanged.
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        for parameter in self.base_layer.parameters():
            parameter.requires_grad = False

    def forward(self, inputs):
        update = self.lora_B(self.lora_A(self.dropout(inputs)))
        return self.base_layer(inputs) + self.scaling * update


def inject_dino_qv_lora(model, rank, alpha=None, dropout=0.05):
    """Add LoRA to every DINO attention query and value projection."""
    attention_modules = []
    for module_name, module in model.named_modules():
        if (isinstance(getattr(module, 'q_proj', None), nn.Linear)
                and isinstance(getattr(module, 'v_proj', None), nn.Linear)):
            attention_modules.append((module_name, module))

    if not attention_modules:
        raise RuntimeError(
            'Could not find DINO attention modules with separate q_proj and '
            'v_proj linear layers. The Hugging Face model structure may have '
            'changed or --dino_model_name may not select a compatible model.')

    adapted_names = []
    for module_name, attention in attention_modules:
        attention.q_proj = LoRALinear(
            attention.q_proj, rank=rank, alpha=alpha, dropout=dropout)
        attention.v_proj = LoRALinear(
            attention.v_proj, rank=rank, alpha=alpha, dropout=dropout)
        adapted_names.extend([
            '{}.q_proj'.format(module_name),
            '{}.v_proj'.format(module_name),
        ])

    expected_layers = getattr(model.config, 'num_hidden_layers', None)
    if expected_layers is not None and len(attention_modules) != expected_layers:
        raise RuntimeError(
            'Found {} adaptable DINO attention modules, but the model config '
            'declares {} hidden layers'.format(
                len(attention_modules), expected_layers))

    return adapted_names


def count_lora_parameters(model):
    return sum(
        module.lora_A.weight.numel() + module.lora_B.weight.numel()
        for module in model.modules()
        if isinstance(module, LoRALinear)
    )
