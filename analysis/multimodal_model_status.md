# Multimodal MySA Development Status

## Overview
The latest upgrade replaces the feature-engineering pipeline with an end-to-end
multimodal training stack. Raw tabular, imaging, and sensor assets are ingested
via manifest files, aligned by canonicalised identifiers, and streamed into the
MySA survival model through modality-specific PyTorch encoders. A learnable
fusion block combines encoder outputs before the discrete-time survival head, so
all modality parameters are optimised jointly for the TEXGI loss.

## Key Capabilities
- **Canonical ID alignment** – The upload workflow and training code now
  normalise identifiers across tabular, image, and sensor sources to prevent
  drift between manifests and raw CSV data. This eliminates the missing image
  feature issue observed in aggregated previews.
- **Asset-backed datasets** – `AssetBackedMultiModalDataset` resolves manifests
  to file paths and streams batches that include raw images and sequences while
  keeping survival supervision tensors in memory. Missing assets are represented
  by zeroed modality tensors so batches remain well-formed.
- **Encoder integration** – Each modality routes through a learnable encoder
  (ResNet-based for images, 1D CNN/TCN for sensors, MLP for tabular data) that
  feeds a shared fusion module and the discrete-time hazard head. Gradients flow
  from the survival loss into every encoder.
- **Attribution smoothing option** – Training exposes `λ_texgi_smooth`, letting
  practitioners add temporal smoothing to TEXGI attributions in addition to the
  existing hazard smoothness and expert-prior terms.

## Remaining Work
- **Backbone configurability** – Only ResNet-18/50 and the default CNN sensor
  encoder are wired up today. Adding Vision Transformers or Transformer-based
  sequence models would broaden modality coverage.
- **On-device augmentations** – The image pipeline applies only centre crops and
  normalisation. Introducing stochastic augmentations (flips, colour jitter) can
  improve robustness without leaving the end-to-end training loop.
- **Advanced fusion strategies** – The current mid-level fusion uses gated
  averaging. Future iterations could incorporate cross-modal attention or
  contrastive alignment objectives to better exploit complementary signals.

Overall, the implementation now satisfies the paper's definition of a
"genuinely multimodal" TEXGI model: every modality is differentiably linked to
survival outputs, and attribution regularisers operate on the joint encoder
representations.
