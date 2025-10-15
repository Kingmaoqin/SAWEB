# TEXGI Multimodal Pipeline Assessment

## Executive Summary
- The Streamlit workflow now preserves per-modality metadata (IDs, manifests, raw asset roots) and forwards it to the model configuration so multimodal runs can invoke the asset-backed path instead of simple DataFrame concatenation.【F:pages_logic/run_models.py†L488-L600】
- When raw manifests are supplied, `_prepare_multimodal_inputs_raw` constructs `AssetBackedMultiModalDataset` instances that stream tabular, image, and sensor tensors together, populate modality masks, and hand batches directly to `MultiModalMySAModel` for joint optimisation.【F:models/mysa.py†L1200-L1467】【F:multimodal_assets.py†L1-L227】
- `MultiModalMySAModel` equips each modality with a learnable encoder, projection head, and gating mechanism, ensuring gradients from the TEXGI-enhanced survival loss flow end-to-end through every modality during training.【F:models/mysa.py†L203-L360】【F:models/mysa.py†L695-L860】

## Front-End Data Preparation and Configuration
The web interface collects tabular, image, and sensor tables through the data manager, tracks the canonical identifier column, and bundles modality descriptors (feature columns plus optional raw manifest metadata) under `multimodal_sources`. This structure allows downstream code to detect whether raw assets are available and whether each modality contains features worth training on.【F:pages_logic/run_models.py†L488-L600】

When users upload raw archives, helper pipelines (`images_to_dataframe`, `sensors_to_dataframe`) still export previewable feature tables, but the session state simultaneously retains manifest paths and roots. As a result, the training configuration includes both the precomputed features and the raw asset descriptors; the latter trigger the asset-backed pipeline in `models/mysa.py` so the model ingests pixels/sequences rather than frozen embeddings.【F:pages_logic/run_models.py†L520-L577】【F:image_pipeline.py†L1-L160】

A fallback remains for purely tabular or feature-level multimodal data: `_prepare_multimodal_inputs` aligns per-modality DataFrames by canonical IDs, concatenates them, and trains the flat-mode variant of `MultiModalMySAModel`. This path mirrors the earlier "concatenate feature tables" behaviour and is automatically selected whenever raw manifests are absent.【F:models/mysa.py†L1101-L1179】

## Asset-Backed Multimodal Training Stack
Supplying raw manifests shifts execution into `_prepare_multimodal_inputs_raw`. The routine canonicalises IDs, resolves asset paths against the provided roots, builds per-modality availability masks, and constructs `AssetBackedMultiModalDataset` objects that lazily load images and sensor sequences on demand while keeping tabular tensors in memory.【F:models/mysa.py†L1200-L1388】【F:multimodal_assets.py†L1-L227】 The resulting PyTorch dataloaders yield batches of modality dictionaries, supervision tensors, and modality masks, which the trainer consumes without precomputing static embeddings.

`MultiModalMySAModel` inspects the supplied `raw_modalities` and instantiates modality-specific encoders: a tabular MLP, a ResNet-based image head, and a 1D CNN/TCN sensor head. Each encoder feeds into a projection block and gating layer so modality contributions are weighted adaptively before reaching the discrete-time survival head. Because the encoders live inside the model graph, gradient updates from the loss, TEXGI penalties, and expert priors backpropagate into every modality-specific parameter.【F:models/mysa.py†L203-L360】

## TEXGI and Expert Prior Integration
`MySATrainer` wraps the multimodal model with the full TEXGI toolkit: it subsamples batches for time-dependent integrated gradients, fits the extreme-baseline generator when expert priors are active, and enforces TEXGI smoothness or rule-based penalties using the modality-aware embeddings produced by the encoders.【F:models/mysa.py†L695-L860】【F:models/mysa.py†L420-L528】 Because the trainer operates on raw-mode batches, the integrated gradients procedure differentiates through the image and sensor encoders as well as the fusion block, yielding modality-coupled attributions.

## Gaps and Recommendations
1. **Clarify fallback semantics.** When only precomputed feature tables are available, training reverts to the flat concatenation path. Documenting this behaviour in the UI would prevent users from mistaking the fallback for the fully joint multimodal model.【F:models/mysa.py†L1101-L1179】
2. **Revisit generator warm-up data.** `_collect_embeddings` snapshots encoder outputs before training to seed the TEXGI generator, so the generator initially sees untrained embeddings. Refitting or refreshing these statistics after several epochs could stabilise expert-prior regularisation when raw encoders evolve substantially.【F:models/mysa.py†L1438-L1450】【F:models/mysa.py†L695-L860】
3. **Expose modality-specific health checks.** Surfacing diagnostics from `AssetBackedMultiModalDataset` (e.g., missing asset counts, sensor column detection) in the web UI would make it easier to verify that raw manifests are wired correctly before training.【F:multimodal_assets.py†L36-L266】

Overall, the platform already contains the hooks needed for a "genuinely multimodal" TEXGI training loop: raw assets flow through learnable encoders, fusion happens inside the survival model, and TEXGI/expert priors differentiate through the entire stack. The primary upgrades now lie in UX clarity and trainer ergonomics rather than fundamental architectural changes.
