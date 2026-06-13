# Method Details &mdash; Hierarchical Music-to-Dance Generation

This document describes the technical details of the two components I add on top
of a transformer-based dance diffusion model: **(1) hierarchical (global &rarr;
local) conditioning** and **(2) a hierarchical time-segment &times; body-part
loss**. It also documents the classifier-free guidance handling and the
sampling-time global constraint hook.

---

## 1. Background and notation

The backbone is a FiLM transformer denoiser that, at diffusion timestep `t`,
predicts the noise for a motion tensor

```
x       : (B, T, D)      # B batch, T = 150 frames (5 s @ 30 fps), D = 151 motion channels
cond    : (B, T, F)      # per-frame music features, F = 4800 (Jukebox)
times   : (B,)           # diffusion timestep
```

The motion channel layout `D = 151` is: `4` foot-contact + `3` root translation
+ `24 x 6` joint rotations (6D rotation representation), i.e. for joint `j` the
rotation occupies channels `7 + 6j : 13 + 6j`.

The baseline conditions **every frame only on its local music features** `cond`.
There is no explicit representation of the *whole-clip* style.

---

## 2. Hierarchical conditioning (global &rarr; local)

Implemented in `model/model.py` (`DanceDecoder`).

### 2.1 Global style vector

The whole-clip music feature is **mean-pooled over time** and encoded by an MLP
into a global *style* vector:

```
global_pool   = cond.mean(dim=1)               # (B, F)   whole-clip summary
global_hidden = global_encoder(global_pool)    # (B, d)   global style vector
```

```python
self.global_encoder = nn.Sequential(
    nn.Linear(cond_feature_dim, latent_dim),
    nn.SiLU(),
    nn.Linear(latent_dim, latent_dim),
)
```

`global_hidden` is the **first hierarchical feature**: it captures the overall
character (genre / energy) of the clip, independent of any single frame.

### 2.2 Global style modulates the local tokens (FiLM)

The local per-frame condition tokens `cond_tokens = (B, T, d)` are
**feature-wise affine modulated** by the global style through a `DenseFiLM`
layer, so the global style scales and shifts every local token:

```python
global_hidden = self.global_encoder(global_pool)            # (B, d)
cond_tokens   = featurewise_affine(cond_tokens,
                                   self.global_film(global_hidden))
```

This is the core "global guides local" mechanism: the same local beats are
re-expressed under the chosen global style.

### 2.3 Dual-channel injection

Beyond FiLM modulation of the local tokens, the global token is injected in two
more places so the decoder can attend to it directly:

1. **Cross-attention memory.** The global token is appended to the attention
   memory `c = [cond_tokens ; t_tokens ; global_token]`.
2. **FiLM timestep condition.** The global style is added to the timestep
   conditioning vector `t`.

```python
c = torch.cat((cond_tokens, t_tokens), dim=-2)
global_token = global_hidden.unsqueeze(1)                       # (B, 1, d)
global_token = torch.where(keep_mask_embed, global_token, null_global_token)
c = torch.cat((c, global_token), dim=-2)                        # appended to memory
t = t + torch.where(keep_mask_hidden, global_hidden, 0)         # added to FiLM cond
```

### 2.4 Classifier-free guidance (CFG)

A learned `null_global_token` replaces the global token when the condition is
dropped (probability `cond_drop_prob` during training), enabling classifier-free
guidance at sampling time exactly as for the local condition:

```python
self.null_global_token = nn.Parameter(torch.randn(1, 1, latent_dim))
# keep_mask is False -> use null token (dropped condition)
global_token = torch.where(keep_mask_embed, global_token, null_global_token)
```

At inference, `guided_forward` mixes the unconditional and conditional passes:

```
out = unc + (cond - unc) * guidance_weight
```

### 2.5 Sampling-time global constraint (`global_pool_override`)

A hook lets us **override the global style at sampling time**, instead of
computing it from the music. This is the *first-level* (global) constraint that
local inpainting cannot express:

```python
global_pool = cond.mean(dim=1)
override = getattr(self, "global_pool_override", None)
if override is not None:
    global_pool = override.to(global_pool.device, global_pool.dtype).expand_as(global_pool)
global_hidden = self.global_encoder(global_pool)
```

Setting `model.global_pool_override` to the pooled features of *another* clip
swaps the global style while keeping the local beats &mdash; the basis of the
style-swap experiment.

---

## 3. Hierarchical loss (time-segment &times; body-part)

Implemented in `model/diffusion.py` (`_hierarchical_weights`, `p_losses`).

### 3.1 Weighting field

Given a config `{frames: (a, b), joints: [...], weight: w}`, we build two
multiplicative weight fields equal to `1.0` everywhere except inside the chosen
**time segment AND chosen body parts**, where they equal `w`:

```python
w_extra   = weight - 1.0
time_mask = 1 on frames[a:b] else 0                 # (1, T, 1)
chan_mask = 1 on channels of selected joints        # (1, 1, D)
joint_mask= 1 on selected joints                    # (1, 1, 24, 1)

chan_w  = 1.0 + w_extra * time_mask * chan_mask      # for reconstruction loss
joint_w = 1.0 + w_extra * time_mask * joint_mask     # for FK loss
```

For joint `j`, its rotation channels `7 + 6j : 13 + 6j` are selected; if the
root joint `0` is selected, its translation + contact channels `0:7` are also
included.

### 3.2 Applied to reconstruction and FK losses

```python
loss = loss_fn(model_out, target, reduction="none")
loss = loss * chan_w                  # emphasize chosen frames x channels

fk_loss = loss_fn(model_xp, target_xp, reduction="none")
fk_loss = fk_loss * joint_w           # emphasize chosen frames x joints (3D positions)
```

This lets the model allocate more representational fidelity to user-chosen
frames / joints (e.g. the arms during the first half of the clip).

### 3.3 CLI

```bash
--hierarchical_loss_frames 0,75            # time segment [0, 75)
--hierarchical_loss_joints 16,17,18,19,20  # SMPL joint indices (e.g. shoulders/elbows/wrist)
--hierarchical_loss_weight 1.5             # emphasis factor w
```

SMPL joint indices used in this repo (see `dataset/masks.py`): `0` root,
`16/17` shoulders, `18/19` elbows, `20/21` wrists, lower-body `1,2,4,5,7,8,10,11`.

---

## 4. Why this is not redundant with EDGE editing

EDGE's editing (`inpaint_loop` + masks in `dataset/masks.py`) is a **hard,
local** constraint applied **at sampling time**: chosen joints/frames are forced
to given values while the rest is inpainted. It requires the target values up
front and has no notion of global style.

My contributions are complementary and operate at a different level:

| Aspect | EDGE inpainting | Hierarchical conditioning | Hierarchical loss |
|---|---|---|---|
| Level | local (joints/frames) | **global (whole-clip style)** | training-time region emphasis |
| Time | sampling | sampling (`global_pool_override`) | training |
| Type | hard replace | style guidance | soft fidelity weighting |
| Needs target motion | yes | no (just a style vector) | no |

The hierarchical conditioning adds a **global control axis** EDGE does not have;
the hierarchical loss is a **training prior** rather than a runtime hard mask.

---

## 5. Evaluation protocol

- Dataset: AIST++ test set, 20 unique slices, identical sampling noise across
  variants for fair comparison.
- Metrics:
  - **PFC** (Physical Foot Contact) &mdash; foot-skating, lower is better.
  - **Beat Align** &mdash; kinematic-beat vs music-beat alignment, higher is better.
  - **FID_k** &mdash; Frechet distance of kinetic features vs ground truth, lower is better.
  - **Div_k** &mdash; kinetic diversity; best when **closest to the GT value 10.21**
    (too low = mode collapse, too high = distorted/diverging).

See the root `README.md` for the controlled ablation table and figures, and
`eval/` for the scripts that produce every number and plot.
