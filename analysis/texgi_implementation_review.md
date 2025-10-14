# TEXGI Implementation vs. Paper Specification

This note compares the TEXGI/MySA implementation in this repository with the
methodology described in the provided Section III excerpt. Each subsection
summarizes the observed behavior in code, contrasts it with the manuscript, and
labels the change as an **upgrade**, **optimization**, or **regression** from the
reference design.

## 1. Discrete-time formulation and supervision
- **What the code does.** Survival times are discretized with `pd.qcut` so that
  each interval contains roughly the same number of samples, and supervision
  tensors are built with masked binary cross-entropy on per-bin hazards.【F:data.py†L10-L74】【F:models/mysa.py†L71-L95】
- **Paper expectation.** The manuscript also groups samples into equal-count
  intervals (Eq. 2) and optimizes a masked logistic loss (Eq. 5).
- **Assessment.** *Optimization.* The implementation matches the intent while
  using Pandas utilities to handle ties and edge cases automatically; this keeps
  the discretization stable without deviating from the paper.

## 2. Network architecture
- **What the code does.** Tabular runs rely on a shallow MLP trunk, while the
  multimodal path uses per-modality encoders with softmax gating to fuse latent
  features before a linear hazard layer.【F:model.py†L7-L34】【F:models/mysa.py†L201-L256】
- **Paper expectation.** TEXGI’s network is described as a Dense Block followed
  by optional Table Attention and a Residual Shared Layer that feeds the hazard
  head.
- **Assessment.** *Regression.* The current architecture omits the dense
  concatenations, attention re-weighting, and shared residual stack, reducing the
  model capacity and the specific inductive biases argued for in the paper. The
  gating fusion benefits multimodal inputs but does not replace the richer
  tabular backbone that TEXGI originally proposed.

## 3. TEXGI baselines and path construction
- **What the code does.** Rather than sampling an extreme baseline directly from
  a tail distribution, the trainer fits an adversarial generator that produces a
  high-risk baseline conditioned on each input and a Pareto “extreme code”; this
  synthesized baseline is used for Integrated Gradients. A high-quantile backup
  is only used if the generator is unavailable.【F:models/mysa.py†L96-L369】
- **Paper expectation.** The definition of TEXGI integrates gradients along the
  straight line between the input and a sampled extreme example drawn from a
  user-controlled high-risk tail (Algorithm 1).
- **Assessment.** *Upgrade.* Training a generator keeps baselines near the data
  manifold while steering them toward risky directions, which operationalizes
  the “extreme yet plausible” requirement more flexibly than manual tail
  sampling. The fallback Pareto code still echoes the paper’s intent.

## 4. Temporal smoothness regularization
- **What the code does.** The only built-in smoothness term penalizes temporal
  differences in predicted hazards via a smooth L1 penalty. No regularizer is
  applied directly to the attribution trajectories.【F:models/mysa.py†L85-L95】【F:models/mysa.py†L642-L669】
- **Paper expectation.** Equation (15) introduces a smoothness prior on the TEXGI
  attributions themselves (Ω<sub>smooth</sub>(Φ)) so that feature importances vary
  gradually across time bins, and Equation (16) combines it with the survival
  loss.
- **Assessment.** *Regression.* Penalizing hazard volatility can stabilize
  predictions, but it does not guarantee smooth, interpretable attribution
  sequences as required by Ω<sub>smooth</sub>(Φ), meaning the code falls short of the
  stated interpretability goal.

## 5. Expert feedback integration
- **What the code does.** Expert rules operate on the global mean absolute and
  directional TEXGI scores, supporting inequality, sign, minimum-magnitude, and
  weighting constraints. These penalties are enforced through ReLU slack terms
  inside the training loop.【F:models/mysa.py†L405-L669】【F:models/mysa.py†L918-L989】
- **Paper expectation.** Equation (18) constrains the ℓ₂ norm of each expert
  feature’s attribution trajectory relative to the average while also shrinking
  non-expert features.
- **Assessment.** *Mixed.* The current design drops the ℓ₂-based constraint in
  favor of mean-based rules, which may weaken guarantees about trajectory-level
  strength (regression), yet it adds directional and per-feature weighting
  options that increase practical flexibility (upgrade). Overall, it is a
  functional but non-equivalent substitute.

## 6. Risk scoring and evaluation
- **What the code does.** Validation uses the sum of predicted hazards as a risk
  score when computing the concordance index, and the training loop checkpoints
  on that metric.【F:models/mysa.py†L685-L706】【F:models/mysa.py†L1005-L1040】
- **Paper expectation.** While the manuscript does not detail risk summarization,
  the discrete-time formulation naturally implies hazard-based survival curves.
- **Assessment.** *Optimization.* Summing hazards is a standard proxy for early
  event risk and aligns with the discrete-time setup without contradicting the
  paper.

## Summary Table

| Component | Implementation Behavior | Paper Specification | Verdict |
|-----------|-------------------------|---------------------|---------|
| Interval construction | Quantile binning via `qcut` | Equal-count batches (Eq. 2) | Optimization |
| Backbone architecture | Shallow MLP / modality gating | Dense Block + Table Attention + Residual Layer | Regression |
| Extreme baseline | Learned adversarial generator (Pareto-coded) | Sampled extreme tail baseline | Upgrade |
| Smoothness prior | Smooth L1 on hazard deltas | Smoothness on TEXGI trajectories | Regression |
| Expert priors | Mean-based, directional, weighted penalties | ℓ₂-norm inequality + sparsity (Eq. 18) | Mixed |
| Risk scoring | Sum of hazards for C-index | Not specified (consistent) | Optimization |

