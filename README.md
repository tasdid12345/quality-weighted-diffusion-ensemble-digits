# Confidence-Weighted Score Ensembles for Small Diffusion Models

Motivation: "When Are Two Scores Better Than One? Investigating Ensembles of Diffusion Models" studies how to aggregate the scores of ensembles of score-based diffusion models, and reports that the arithmetic mean is a strong default. This experiment asks whether a different aggregation rule can do better on a much smaller problem.

Question: Can weighting ensemble members of a small score-based diffusion model by a perceptual-quality proxy, rather than by score-matching loss, lower the Fréchet distance of generated samples relative to the arithmetic-mean ensemble?

Dataset: scikit-learn's bundled digits dataset (load_digits), scaled to [0,1] and split once with a stratified, seeded train/test split. This is not the paper's dataset (the paper uses FFHQ-256).

Baseline: a DDPM-style ensemble of small scikit-learn MLP noise predictors, sampled with a DDIM-style deterministic sampler and combined by the arithmetic mean of member scores.

Improved model: identical members, sampler, seeds and step count. Each member's weight is a softmax over the mean digit-classifier confidence of a pilot batch drawn from that member alone, and the weighted mean of scores is used.

How to run:

    pip install -r requirements.txt
    python analysis.py

Results: results are written to results.json (and a chart to figures/).

## Reference

This project is motivated by the following paper:

> R Razafindralambo, R Sun, F Precioso (2026). *When Are Two Scores Better Than One? Investigating Ensembles of Diffusion Models*. arXiv preprint arXiv. https://arxiv.org/abs/2601.11444

Paper author: Prof. Damien Garreau, Julius-Maximilians-Universität Würzburg (damien.garreau@uni-wuerzburg.de)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
