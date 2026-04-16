"""utils.py — Image loading, saving, and preprocessing utilities."""
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
from PIL import Image

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


def load_image(path: str, transform=None) -> torch.Tensor:
    img = Image.open(path).convert('RGB')
    if transform:
        img = transform(img)
    return img.unsqueeze(0)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    t = tensor.squeeze(0).detach().cpu()
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std  = torch.tensor(STD).view(3, 1, 1)
    t = (t * std + mean).clamp(0, 1)
    return TF.to_pil_image(t)


def save_image(tensor: torch.Tensor, path: str) -> None:
    tensor_to_pil(tensor).save(path)


def get_transform(size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
