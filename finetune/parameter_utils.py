import torch
import torch.nn as nn
import torch.nn.functional as F

def print_trainable_params_percentage(model):
    orig_param_size = sum(p.numel() for p in model.parameters())
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    trainable_param_size = count_parameters(model)
    percentage = trainable_param_size / orig_param_size * 100
    print(f"Trainable parameters: {percentage:.2f}% ({trainable_param_size}/{orig_param_size})")
    return percentage

def freeze_whole_model(model):
    for n, p in model.named_parameters():
        p.requires_grad = False