"""
train.py — Style transfer optimisation loop.
Usage: python src/train.py --content IMG --style IMG [options]
"""
import argparse, time, torch, torch.optim as optim
from model import VGGFeatureExtractor, compute_loss
from utils  import load_image, save_image, get_transform


def run(content_path, style_path, output_path,
        image_size=512, num_steps=500,
        content_weight=1.0, style_weight=1e6, lr=0.01, log_every=50):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n[style-transfer] device={device} | steps={num_steps} | size={image_size}px\n")
    tf = get_transform(image_size)
    content = load_image(content_path, tf).to(device)
    style   = load_image(style_path,   tf).to(device)
    model   = VGGFeatureExtractor().to(device)

    with torch.no_grad():
        content_feats = model(content)
        style_feats   = model(style)

    generated = content.clone().requires_grad_(True)
    optimizer = optim.Adam([generated], lr=lr)
    history, t0 = [], time.time()

    for step in range(1, num_steps + 1):
        optimizer.zero_grad()
        loss, breakdown = compute_loss(model(generated), content_feats, style_feats, content_weight, style_weight)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            generated.clamp_(-2.5, 2.5)
        if step % log_every == 0 or step == 1:
            print(f"  step {step:>4}/{num_steps} | loss={loss.item():.4f} | {time.time()-t0:.1f}s")
            history.append({'step': step, 'loss': loss.item(), **breakdown})

    save_image(generated, output_path)
    print(f"\n  Saved → {output_path}\n")
    return history


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--content',        required=True)
    p.add_argument('--style',          required=True)
    p.add_argument('--output',         default='data/outputs/result.jpg')
    p.add_argument('--size',           type=int,   default=512)
    p.add_argument('--steps',          type=int,   default=500)
    p.add_argument('--content-weight', type=float, default=1.0)
    p.add_argument('--style-weight',   type=float, default=1e6)
    p.add_argument('--lr',             type=float, default=0.01)
    a = p.parse_args()
    run(a.content, a.style, a.output, a.size, a.steps, a.content_weight, a.style_weight, a.lr)
