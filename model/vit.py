import torch.nn as nn
from torchvision.models.vision_transformer import _vision_transformer


def vit_b_4(image_size, num_classes, num_channels, **kwargs):
    v = _vision_transformer(
        weights=None,
        progress=True,
        image_size=image_size,
        num_classes=num_classes,
        patch_size=4,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
    )

    if num_channels != 3:
        v.conv_proj = nn.Conv2d(
            num_channels,
            out_channels=v.hidden_dim,
            kernel_size=v.patch_size,
            stride=v.patch_size,
        )

    return v
