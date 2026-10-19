# Digits ensemble diffusion: loss-weighted vs. mean score aggregation

## Question
Can weighting ensemble members of a small score-based diffusion model by their
training-set score-matching loss lower the Frechet distance (PCA-20 feature space)
of generated digits relative to the uniform-mean ensemble?

## Method
- Data: scikit-learn `load_digits` (1797 rows), scaled to [0,1], stratified 70/30 split (seed 42).
  Test set is never used for fitting, weighting or selection.
- Shared: 5 MLP noise-prediction members per seed (seeds 42, 43, 44), linear beta schedule
  with T=100, DDIM-style sampler (eta=0, 25 steps), PCA(20) fitted on training data.
- Baseline: uniform average of member noise predictions (scores).
- Improved: weights proportional to 1/MSE of each member's noise prediction on the training set.
  No pilot sampling and no additional training, so sampling cost matches the baseline.
- Metric: Frechet distance between Gaussians fitted to PCA features of 1000 generated samples
  and the test digits (lower is better). Mean and std reported over three seeds.

## Run
```
pip install -r requirements.txt
python analysis.py
```
Writes `results.json` and `figures/fd_comparison.png`.

## Caveats
- Toy setting; PCA-Frechet distance is not comparable to FID.
- Weights come from in-sample training loss and may favour members that overfit less or more;
  the sign of the effect is not guaranteed.
- Only three seeds, so the standard deviation is wide relative to small differences.

## Reference

This project is motivated by the following paper:

> R Razafindralambo, R Sun, F Precioso (2026). *When Are Two Scores Better Than One? Investigating Ensembles of Diffusion Models*. arXiv preprint arXiv. https://arxiv.org/abs/2601.11444

Paper author: Prof. Damien Garreau, Julius-Maximilians-Universität Würzburg (damien.garreau@uni-wuerzburg.de)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
