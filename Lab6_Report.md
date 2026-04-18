# Lab 6 — Deep Generative Models: VAE, GAN, CGAN, DCGAN, and GAN Loss Variants

**Course:** Deep Learning (Semester 6)
**Student:** Aayush Jha
**Datasets:** FashionMNIST (10 classes, 28×28), EMNIST-Balanced (47 classes, 28×28)
**Framework:** PyTorch (CUDA)
**Date:** 2026-04-18

---

## 1. Objective

Study, implement, train and compare the principal families of deep generative models on low-resolution image data, quantify their generative quality with standard metrics (FID, SSIM, Inception Score, reconstruction MSE), and analyze the effect of different adversarial loss formulations on training stability.

The lab is organised around four stages:

1. **Variational Autoencoders** — unconditional VAE and Conditional VAE (CVAE).
2. **Generative Adversarial Networks** — vanilla MLP GAN and Conditional GAN (CGAN).
3. **Deep Convolutional GAN (DCGAN)** — fully convolutional generator/discriminator.
4. **GAN loss variants** — Vanilla, Non-Saturating, LSGAN, WGAN (weight-clipping), WGAN-GP.

---

## 2. Datasets and Pre-processing

| Dataset | Split | #Classes | Image Size | Transform |
|---|---|---|---|---|
| FashionMNIST | train (60k) | 10 | 28×28 grayscale | `ToTensor()` → [0,1] |
| EMNIST-Balanced | train (~112k) | 47 | 28×28 grayscale | `ToTensor()` → [0,1] |

Batch size **128**, shuffled. For GAN training the generator output is rescaled to **[-1, 1]** (via `Tanh`) while the real images remain in **[0, 1]**; only cosmetic rescaling `(x+1)/2` is applied at sampling time. VAE inputs remain in **[0, 1]** because the decoder ends in `Sigmoid` to match a BCE reconstruction loss.

---

## 3. Methodology

### 3.1 Common setup

* Device: `cuda` (verified at runtime).
* Optimiser: **Adam** — learning rate `1e-3` for VAE, `2e-4` for GAN / DCGAN / CGAN.
* DCGAN variants additionally use **β₁ = 0.5, β₂ = 0.999**, the standard DCGAN recipe from Radford et al.
* Number of training epochs: **20** for each base model, **10** for the latent-dim ablation, **5** for the EMNIST transfer experiment.
* Label smoothing for the discriminator: real labels set to **0.9** instead of 1.0, reducing gradient magnitude near the boundary and improving GAN stability.
* Weight initialisation for DCGAN: Conv weights ~ 𝒩(0, 0.02), BatchNorm weights ~ 𝒩(1, 0.02), bias = 0 (the canonical DCGAN init).

### 3.2 VAE training objective

The VAE is trained with the standard evidence-lower-bound (ELBO) loss decomposed into reconstruction + KL divergence:

```
L = BCE(x̂, x)  +  β · D_KL( N(μ, σ²) ‖ N(0, I) )
D_KL = -½ · Σ (1 + log σ² − μ² − σ²)
```

**KL annealing** is used: `β = min(1, epoch / 10)`. This lets the decoder first learn to reconstruct (low-KL phase), then gradually pulls the posterior toward the unit Gaussian prior, preventing early posterior collapse.

### 3.3 GAN training objective

The base adversarial loss is the BCE min-max game. Training alternates:

1. **Discriminator step** — maximise `log D(x) + log(1 − D(G(z)))` (implemented as BCE on real/fake).
2. **Generator step** — minimise the **non-saturating** `−log D(G(z))` form (BCE with real labels) so that gradients do not vanish when the discriminator is confident.

### 3.4 Conditional models

For both CVAE and CGAN, the class label is injected into **both** encoder/generator and decoder/discriminator:

* **CVAE:** one-hot label concatenated with input and with latent.
* **CGAN:** a learned `nn.Embedding(num_classes, 50)` embedding concatenated with `z` (for G) and with the flattened image (for D). Using an embedding (rather than one-hot) is strictly more flexible and lets the discriminator learn label-conditional similarity.

### 3.5 GAN loss variants (Section 7 of the notebook)

Five variants are implemented with a shared DCGAN backbone, and run for 20 epochs each:

| Variant | D loss | G loss | Extra |
|---|---|---|---|
| Vanilla | `−E[log D(x)] − E[log(1−D(G(z)))]` | `E[log(1−D(G(z)))]` (saturating) | — |
| Non-Saturating | same as vanilla | `−E[log D(G(z))]` | — |
| LSGAN | `½·E[(D(x)−1)²] + ½·E[D(G(z))²]` | `½·E[(D(G(z))−1)²]` | — |
| WGAN | `−E[D(x)] + E[D(G(z))]` | `−E[D(G(z))]` | Weight clipping `|w| ≤ 0.01` |
| WGAN-GP | WGAN + `λ · E[(‖∇ D(x̂)‖₂ − 1)²]`, `λ=10` | `−E[D(G(z))]` | Gradient penalty on interpolates |

### 3.6 Evaluation

* **FID** (`torchmetrics.image.fid`, 64-dim Inception features) — distance between distributions of real and generated images.
* **SSIM** (`skimage.metrics.structural_similarity`) — per-image structural similarity for reconstructions (VAE only).
* **Inception Score** (custom, using `torchvision.models.inception_v3`, 1000 samples by default) — quality + diversity proxy.
* **Reconstruction MSE** — pixel-wise MSE between VAE reconstructions and input.
* **Latent visualisation** — t-SNE projection of encoder means `μ` on ~2.5k samples, colour-coded by class.
* **Mismatch detection test** (CGAN only) — feed the discriminator real images with **wrong** labels; accuracy measures how well the discriminator internalises the conditioning.
* **Intra-class diversity** — variance across 100 samples generated for a fixed class.

---

## 4. Architectures

### 4.1 VAE (MLP, latent_dim = 32)

```
Encoder:   784 → 512 (ReLU) → 256 (ReLU) → { μ: 32 , logσ²: 32 }
Reparam :   z = μ + ε · σ ,  ε ~ N(0, I)
Decoder:    32 → 256 (ReLU) → 512 (ReLU) → 784 (Sigmoid)
```

### 4.2 CVAE (latent_dim = 32, num_classes = 10)

```
Encoder:   [x ‖ y_onehot] 794 → 512 (ReLU) → { μ:32 , logσ²:32 }
Decoder:   [z ‖ y_onehot] 42  → 512 (ReLU) → 784 (Sigmoid)
```

### 4.3 Vanilla GAN (MLP, z_dim = 100)

```
Generator:     100 → 256 → 512 → 1024 → 784 , LeakyReLU(0.2) + Tanh
Discriminator: 784 → 1024 → 512 → 256 → 1  , LeakyReLU(0.2) + Dropout(0.3) + Sigmoid
```

### 4.4 CGAN (z_dim = 100, embed_dim = 50)

```
Generator:     [z ‖ emb(y)] 150 → 256(BN,ReLU) → 512(BN,ReLU) → 1024(BN,ReLU) → 784 (Tanh)
Discriminator: [x ‖ emb(y)] 834 → 1024 → 512 → 256 → 1  (LeakyReLU, Sigmoid)
```

### 4.5 DCGAN (convolutional)

```
Generator (z=100):
  FC(100 → 128·7·7) → reshape (128,7,7)
  BatchNorm2d(128)
  ConvT(128→64, k=4, s=2, p=1)   7→14   + BN + ReLU
  ConvT(64 →1 , k=4, s=2, p=1)   14→28  + Tanh

Discriminator:
  Conv(1 →64 , k=4, s=2, p=1)    28→14  + LeakyReLU(0.2)
  Conv(64→128, k=4, s=2, p=1)    14→7   + BN + LeakyReLU
  Conv(128→256,k=3, s=2, p=1)    7 →4   + BN + LeakyReLU
  Flatten → Linear(256·4·4 → 1) + Sigmoid
```

Also explored: a BatchNorm-free generator (`G_NoBN`) and a **spectral-norm** discriminator (`D_Spectral`) as stability ablations.

---

## 5. Step-by-Step Execution

1. **Imports & device check** — CUDA confirmed (`print(device) → cuda`).
2. **Data loaders** — FashionMNIST and EMNIST-Balanced with `ToTensor`.
3. **VAE implementation** — `VAE`, `loss_function` (BCE + β·KL), `train_vae` with KL annealing.
4. **Train VAE on FashionMNIST (20 ep.)** → loss converges from 266.18 → 237.52.
5. **Train VAE on EMNIST (20 ep.)** → loss 173.93 → 147.87.
6. **Reconstruction visualisation** — top row originals, bottom row reconstructions.
7. **Latent interpolation** — 10 steps between `μ₁` and `μ₂` decoded, producing a smooth morph.
8. **Unconditional sampling** — 16 `z ~ N(0, I)` decoded into a 4×4 grid.
9. **Latent-dim ablation** — `latent_dim ∈ {8, 32, 128}`, 10 epochs each. Larger dims reach slightly higher KL (tighter fit to the prior) without clearly better reconstructions at 28×28.
10. **FID** — computed between a freshly sampled batch and the real batch. **FID = 0.27** (64-dim features, per-batch; this is the metric-object's running value — an *optimistic* small-scale estimate, not a full-set FID).
11. **SSIM** — reconstruction SSIM over the training loader: **0.7025**.
12. **t-SNE latent plot** — shows partial class clustering, which is expected for an unconditional VAE with a fairly entangled prior.
13. **CVAE** — trained 20 epochs, loss 253.07 → 237.38. Class-conditional sampling works (e.g. `generate_class(cvae, digit=3)` produces class-3 images).
14. **Vanilla GAN (FashionMNIST, 20 ep.)** — D-loss settles near ~1.0, G-loss near ~1.55 by epoch 20 (healthy equilibrium).
15. **CGAN (FashionMNIST, 20 ep.)** — losses are more oscillatory (epoch 20: D = 0.82, G = 6.96), typical of a discriminator getting intermittently too strong.
16. **Per-class generation + label interpolation** — visually confirms class control.
17. **Mismatch detection test** — D accuracy on mis-labelled real pairs = **0.4221**, indicating the discriminator only *partially* internalised the label conditioning (chance for a binary classifier is 0.5, so the model is weakly rejecting wrong labels but not reliably).
18. **Intra-class diversity (class 3)** = **3.98 × 10⁻⁷** — very low pixel-variance, a warning sign of **partial mode collapse** inside that class.
19. **Transfer CGAN to EMNIST (5 ep.)** — uses a Wasserstein-style loss for the transfer; completes without divergence.
20. **Inception Score (CGAN, 1000 samples)** = **1.99** — modest diversity × confidence score (upper bound for 10 classes is 10; IS ≈ 2 says samples are *recognisable as something* but not strongly class-consistent at the Inception level, which is also partly because Inception was trained on ImageNet, not 28×28 grayscale).
21. **DCGAN (FashionMNIST, 20 ep.)** — D 0.71 → 0.37, G 1.91 → 4.29. G-loss drifting upward while D-loss falls is the textbook sign of the discriminator starting to dominate.
22. **DCGAN (EMNIST, 20 ep.)** — similar behaviour, slightly higher G-loss due to the 47-class complexity.
23. **Stability ablations** — `G_NoBN` and `D_Spectral` defined; progressive-training and latent-arithmetic helpers included.
24. **Aggregated comparison table** (Cell 53–54) — see §6.2.
25. **GAN-variant experiment** — five variants trained with a shared DCGAN backbone (20 ep. each). Per-epoch timing recorded.
26. **Loss plots** and **FID-vs-epoch plots** per variant.

---

## 6. Results

### 6.1 VAE training curves (FashionMNIST, 20 ep.)

| Epoch | Loss | β |
|---|---|---|
| 1 | 266.18 | 0.00 |
| 5 | 235.44 | 0.40 |
| 10 | 238.91 | 0.90 |
| 15 | 238.52 | 1.00 |
| 20 | **237.52** | 1.00 |

A characteristic KL-annealing signature is visible: loss drops quickly in epochs 1–5 (reconstruction phase), then briefly **rises** around epochs 6–11 as the KL term is phased in, then stabilises.

### 6.2 Summary metrics (FashionMNIST, 20 epochs unless noted)

| Model | FID (64-dim, per-batch) | IS (1000 samples) | Recon MSE | Convergence epoch | Notes |
|---|---|---|---|---|---|
| VAE | 0.27 (optimistic) | — | **0.0143** | — | SSIM 0.7025 |
| GAN | n/a | n/a | — | 18 | Oscillating D/G ≈ 1.0 / 1.55 |
| CGAN | n/a | **1.98** | — | — | Mismatch acc 0.4221, intra-class var 4e-7 (mode collapse warning) |
| DCGAN | n/a | n/a | — | 2 | D↘ G↗ drift from epoch ~10 |

> The FID and IS numbers are *indicative*, not publication-grade: FID is taken from the running `torchmetrics` accumulator as wired in the notebook, and IS uses only 1000 samples fed through ImageNet-pretrained Inception-V3 at 299×299 after channel-broadcasting a 28×28 grayscale image — both introduce bias. They are still usable as *relative* signals within this lab.

### 6.3 DCGAN training (FashionMNIST)

| Epoch | D loss | G loss |
|---|---|---|
| 1 | 0.7093 | 1.9058 |
| 5 | 0.4573 | 2.9917 |
| 10 | 0.4551 | 3.5075 |
| 15 | 0.4322 | 3.7327 |
| 20 | **0.3721** | **4.2910** |

### 6.4 GAN-variant experiment (DCGAN backbone, 20 ep., FashionMNIST)

| Variant | Avg epoch time (s) | Observed behaviour |
|---|---|---|
| Vanilla | 10.57 | Saturating loss, large oscillations (D collapses to ~0 on epochs 5–8, then recovers) |
| Non-Saturating | 10.07 | Smooth monotone G-loss growth, D-loss low and stable |
| LSGAN | 9.98 | Very smooth, no divergence |
| WGAN | 10.06 | Meaningful, monotonically-decreasing critic; clipping hurts capacity |
| WGAN-GP | **13.84** | Most stable; 37% slower due to second-order gradient-penalty graph |

Vanilla epoch excerpt showing the saturating-loss pathology:

```
VANILLA Epoch 4  | D: 0.0144 | G: -0.0072
VANILLA Epoch 8  | D: 0.0010 | G: -0.0005     ← G gradient ≈ 0
VANILLA Epoch 10 | D: 0.6429 | G: -0.2992     ← sudden recovery
```

Non-saturating, by contrast, gives a healthy monotone G-loss (1.96 → 4.86 by ep. 16) with D staying around 0.1–0.3.

### 6.5 Qualitative observations (notebook `compare_models`)

* **VAE** reconstructions: structurally correct but **blurry** — a direct consequence of the pixel-wise BCE loss and the Gaussian posterior, which averages plausible completions.
* **Vanilla GAN** samples: sharper than VAE, but with occasional texture artefacts and low intra-batch diversity.
* **CGAN**: clear class separability in the generated grids, but the 4e-7 intra-class variance confirms the mode-collapse within-class that the loss itself does not penalise.
* **DCGAN**: the sharpest and most consistent samples — Conv/ConvT inductive bias matters a lot even at 28×28.

---

## 7. Discussion

* **KL annealing matters**: without it, β = 1 from epoch 1 drives the encoder straight into posterior collapse and the decoder learns a near-mean image. The plateau-then-rise-then-stabilise loss curve in §6.1 is the trademark of a healthy β-schedule.
* **GAN stability ∝ loss geometry, not architecture alone**: swapping the loss function under the *same* DCGAN backbone (§6.4) turned a routinely-diverging Vanilla run into a stable WGAN-GP run. Weight-clipping (plain WGAN) is strictly inferior to gradient penalty — it trades instability for capacity loss.
* **Mismatch detection < 0.5** on CGAN means the discriminator has not fully learnt to reject wrong-label/real pairs. Label-conditional discriminators benefit from **projection discriminator**-style conditioning rather than simple concatenation; that is a natural follow-up.
* **IS on 28×28 grayscale** is a known under-estimator because Inception-V3 expects 299×299 RGB natural images. The **relative** IS ranking across variants is still informative, but absolute values (≈2) should not be compared to CIFAR-10 / ImageNet literature.
* **Reported FID ≈ 0.27 is not a true FID** for the full dataset — it is read from the running `torchmetrics` object inside the per-batch loop. For a publishable number, the updates must be accumulated across the full 60k before a single `fid.compute()`.

---

## 8. Conclusion

The lab successfully implements, trains and evaluates the main families of likelihood-based and adversarial generative models on FashionMNIST and EMNIST, and runs a controlled comparison of five adversarial-loss formulations under a shared DCGAN backbone.

**Ranking observed in this lab:**

* **Image quality:**   DCGAN  >  CGAN  >  GAN  >  VAE
* **Training stability:**   VAE  >  WGAN-GP  >  LSGAN  >  DCGAN/CGAN  >  Vanilla GAN
* **Controllability:**   CVAE ≈ CGAN  >  VAE ≈ GAN (class conditioning)

**Key takeaway:** among adversarial methods, **WGAN-GP** is the most reliable setup — at the cost of ~37% per-epoch overhead — and the **non-saturating BCE** loss is the best drop-in replacement for the original saturating loss at no computational cost. Among likelihood models, the VAE remains the most *stable* trainer but is fundamentally limited in sharpness; CVAE adds useful label control at negligible extra cost.

---

## 9. Files

| File | Role |
|---|---|
| `Lab6.ipynb` | Executable notebook (68 cells) |
| `gan_samples_fashion/epoch_*.png` | GAN samples saved every 5 epochs |
| `dcgan_fashion_samples/epoch_*.png` | DCGAN (FashionMNIST) samples |
| `dcgan_emnist_samples/epoch_*.png` | DCGAN (EMNIST) samples |
| `Lab6_Report.md` | **This report** |
