"""
metrics.py — Quantitative evaluation of style transfer output.
Metrics: SSIM (structural similarity), PSNR (peak signal-to-noise ratio)
"""
import torch, torch.nn.functional as F, math, json, argparse
from torchvision import transforms
from PIL import Image


def to_tensor(path, size=256):
    t = transforms.Compose([transforms.Resize(size), transforms.CenterCrop(size), transforms.ToTensor()])
    return t(Image.open(path).convert('RGB')).unsqueeze(0)


def psnr(a, b):
    mse = F.mse_loss(a, b).item()
    return round(10 * math.log10(1.0 / mse), 2) if mse > 0 else float('inf')


def ssim(a, b, k=11):
    C1, C2 = 0.01**2, 0.03**2
    mu1 = F.avg_pool2d(a, k, 1, k//2)
    mu2 = F.avg_pool2d(b, k, 1, k//2)
    m1s, m2s, m12 = mu1**2, mu2**2, mu1*mu2
    s1  = F.avg_pool2d(a**2, k, 1, k//2) - m1s
    s2  = F.avg_pool2d(b**2, k, 1, k//2) - m2s
    s12 = F.avg_pool2d(a*b,  k, 1, k//2) - m12
    num = (2*m12+C1)*(2*s12+C2)
    den = (m1s+m2s+C1)*(s1+s2+C2)
    return round(num.div(den).mean().item(), 4)


def evaluate(content_path, generated_path):
    c = to_tensor(content_path)
    g = to_tensor(generated_path)
    return {'ssim_vs_content': ssim(g, c), 'psnr_vs_content': psnr(g, c)}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--content',   required=True)
    p.add_argument('--generated', required=True)
    a = p.parse_args()
    print(json.dumps(evaluate(a.content, a.generated), indent=2))
