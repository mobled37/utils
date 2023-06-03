import torch
import torch.nn as nn
import torch.nn.functional as F

from adapter import Adapter, VisionAdapter

def unfreeze_adapter(model, args):
    for name, sub_module in model.named_modules():

        # Unfreeze VisionAdapter
        if args.use_vis_adapter:
            if isinstance(sub_module, VisionAdapter):
                print(f"{name} is trainable...")
                for param_name, param in sub_module.named_parameters():
                    param.requires_grad = True

        # Unfreeze Adapter
        if args.use_adapter:
            if isinstance(sub_module, Adapter):
                print(f"{name} is trainable...")
                for param_name, param in sub_module.named_parameters():
                    param.requires_grad = True

