"""
model.py — VGG-19 based neural style transfer model.
Uses perceptual loss (content + style) to blend artistic style onto content images.
"""
import torch
import torch.nn as nn
import torchvision.models as models

CONTENT_LAYERS = ['conv_4']
STYLE_LAYERS   = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


class VGGFeatureExtractor(nn.Module):
    """Extracts intermediate feature maps from a pretrained VGG-19."""
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.eval()
        for p in vgg.parameters():
            p.requires_grad_(False)

        self.slices, self.layer_names = nn.ModuleList(), []
        current, i_conv = [], 0
        for layer in vgg.children():
            if isinstance(layer, nn.Conv2d):
                i_conv += 1
                name = f'conv_{i_conv}'
            elif isinstance(layer, nn.ReLU):
                name = f'relu_{i_conv}'
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = f'pool_{i_conv}'
            else:
                name = f'other_{i_conv}'
            current.append(layer)
            if name in CONTENT_LAYERS or name in STYLE_LAYERS:
                self.slices.append(nn.Sequential(*current))
                self.layer_names.append(name)
                current = []

    def forward(self, x):
        features = {}
        for slice_, name in zip(self.slices, self.layer_names):
            x = slice_(x)
            features[name] = x
        return features


def gram_matrix(f: torch.Tensor) -> torch.Tensor:
    b, c, h, w = f.shape
    f = f.view(b, c, h * w)
    return torch.bmm(f, f.transpose(1, 2)) / (c * h * w)


def compute_loss(gen_f, content_f, style_f, content_weight=1.0, style_weight=1e6):
    content_loss = sum(
        nn.functional.mse_loss(gen_f[l], content_f[l].detach())
        for l in CONTENT_LAYERS if l in gen_f
    )
    style_loss = sum(
        nn.functional.mse_loss(gram_matrix(gen_f[l]), gram_matrix(style_f[l].detach()))
        for l in STYLE_LAYERS if l in gen_f
    )
    total = content_weight * content_loss + style_weight * style_loss
    return total, {'content': content_loss.item(), 'style': style_loss.item()}
