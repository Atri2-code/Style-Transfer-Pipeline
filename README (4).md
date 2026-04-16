# Style-transfer-pipeline

> Neural style transfer with quantitative evaluation — built for production-readiness.

Applies the artistic style of any reference image to a content image using a VGG-19 perceptual loss network. Includes a full evaluation suite (SSIM, PSNR) and a clean CLI for batch processing.

---

## Quick start

```bash
git clone https://github.com/YOUR_USERNAME/style-transfer-pipeline.git
cd style-transfer-pipeline
pip install -r requirements.txt

python src/train.py \
  --content data/inputs/photo.jpg \
  --style   data/styles/vangogh.jpg \
  --output  data/outputs/result.jpg \
  --steps   500 \
  --size    512
```

Runs on CPU or GPU automatically.

---

## Evaluate output quality

```bash
python evaluation/metrics.py \
  --content   data/inputs/photo.jpg \
  --style     data/styles/vangogh.jpg \
  --generated data/outputs/result.jpg
```

```json
{
  "ssim_vs_content": 0.7812,
  "psnr_vs_content": 24.31
}
```

---

## Architecture

```
Content image ──┐
                ├──► VGG-19 feature extractor ──► Perceptual loss ──► Adam optimizer ──► Generated image
Style image  ───┘         (frozen weights)         content + style
```

**Loss function:**

```
L_total = α × L_content + β × L_style

L_content = MSE(F_generated, F_content)         # conv_4 features
L_style   = Σ MSE(G_generated_l, G_style_l)     # Gram matrices at conv_1–5
```

where G denotes the Gram matrix.

---

## Project structure

```
style-transfer-pipeline/
├── src/
│   ├── model.py        # VGG-19 feature extractor + loss functions
│   ├── train.py        # Optimisation loop + CLI
│   └── utils.py        # Image I/O and preprocessing
├── evaluation/
│   └── metrics.py      # SSIM, PSNR evaluation
├── data/
│   ├── inputs/         # Content images
│   ├── styles/         # Style reference images
│   └── outputs/        # Generated results (gitignored)
├── requirements.txt
└── README.md
```

---

## Key parameters

| Flag | Default | Effect |
|------|---------|--------|
| `--steps` | 500 | More steps = stronger stylisation |
| `--style-weight` | 1e6 | Higher = more artistic, less content |
| `--content-weight` | 1.0 | Higher = more faithful to original |
| `--size` | 512 | Output resolution (px) |
| `--lr` | 0.01 | Adam learning rate |

---

## Skills demonstrated

| ML competency | Implementation |
|---|---|
| Deep learning (CV) | VGG-19 perceptual loss network |
| Model evaluation | SSIM + PSNR quantitative metrics |
| PyTorch proficiency | Custom forward pass, Gram matrices |
| Production mindset | CLI, GPU/CPU auto-detect, logging |
| Generative modelling | Iterative image synthesis via gradient descent |

---

## Roadmap

- [ ] Fast style transfer (feed-forward network, ~1000× faster)
- [ ] Multi-style blending
- [ ] REST API endpoint (FastAPI)
- [ ] Batch processing script

---

## References

- Gatys et al., *A Neural Algorithm of Artistic Style* (2015)
- Johnson et al., *Perceptual Losses for Real-Time Style Transfer* (2016)

---

## License

MIT
