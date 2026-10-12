# Model weights

DiagAD is **training-free**: it needs no trained weights of its own. The only
model file it requires is the **frozen DINOv2 ViT-B/14 backbone** (with
register tokens), published by Meta AI under the Apache-2.0 license.

The weight file itself (~346 MB) is **not committed** to this repository.
Download it once from the official source below — the evaluation scripts load
it automatically via `torch.hub`.

## DINOv2 ViT-B/14 (registers) — the backbone used by DiagAD

- Official repository: <https://github.com/facebookresearch/dinov2>
- Direct download (Apache-2.0):
  <https://dl.fbarespubliciles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth>
  - File: `dinov2_vitb14_reg4_pretrain.pth` (~346 MB)
  - SHA-256: `73182a088cf94833c94b1666d1c99e02fe87e2007bff57b564fb6206e25dba71`
  - License: Apache-2.0 (© Meta Platforms, Inc.)

## How the weights are used

`code/src/models/backbone.py` loads the backbone with

```python
torch.hub.load("facebookresearch/dinov2:main", "dinov2_vitb14_reg", pretrained=True)
```

On first use, `torch.hub` downloads the checkpoint automatically into the
local torch hub cache (Linux/macOS: `~/.cache/torch/hub/checkpoints/`,
Windows: `%USERPROFILE%\.cache\torch\hub\checkpoints\`). No manual step is
required when the machine has internet access.

For an **offline** setup, download the file above manually and place it in
that cache directory under the exact filename
`dinov2_vitb14_reg4_pretrain.pth` — `torch.hub` then finds it locally and
skips the download. (The torch hub cache directory can be relocated with the
`TORCH_HOME` environment variable.)

`code/run_diagad.bat` (Windows) and `code/run_diagad.sh` (Linux/macOS)
perform this check-and-download step automatically.

## Not used by DiagAD

`weights/mobileclip_s1/` and the MobileCLIP backbone path in the code belong
to an earlier version of the project and are **not** part of the DiagAD
method (the frozen configuration is `backbone: dinov2`). They are documented
here only for completeness and are not required for reproduction.
