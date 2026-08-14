<h1 align="center">Machine Learning Algorithms from Scratch</h1>

<p align="center">
  <i>Ten classical and modern ML algorithms implemented in NumPy and PyTorch primitives —<br/>
  no scikit-learn in the algorithm bodies, no autograd where the point is the gradient.</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white">
  <img src="https://img.shields.io/badge/algorithms-10-0b5b39?style=flat">
</p>

---

## Why this repo exists

Calling `sklearn.fit()` teaches you an API. Writing the update rule teaches you why the
algorithm behaves the way it does — where it converges, where it stalls, and what each
hyperparameter is actually trading off.

Every algorithm below was written against a provided test harness (USC CSCI 567, Spring 2025):
the data loaders and unit tests came with the assignment, the algorithm bodies are mine. Where
a result is quoted, it comes from my own recorded run, not from the literature.

## What's implemented

| # | Algorithm | What was written by hand |
|---|---|---|
| 1 | **K-Nearest Neighbours** | Euclidean / Minkowski-L3 / cosine distances, min-max and L2 scalers, F1 scoring, and the k × distance × scaler tuning grid |
| 2 | **Linear & Ridge Regression** | Closed-form normal equations, λ search over 2⁻⁴⁰…2⁰, polynomial feature maps p = 2…5 |
| 3 | **Perceptron, Logistic & Softmax Regression** | Hand-derived gradients; binary GD, and multiclass softmax by both **full-batch GD and SGD** |
| 4 | **Multilayer Perceptron** | Full **forward and backward passes** — linear, ReLU, tanh, inverted **dropout** (train/test aware), softmax cross-entropy, mini-batch **SGD with momentum** and step decay. No autograd |
| 5 | **AdaBoost** | Sample reweighting, β = ½·ln((1−ε)/ε), weighted-vote prediction, over decision-tree weak learners |
| 6 | **K-Means & K-Means++** | Lloyd's algorithm, k-means++ seeding, k-means as a classifier, and image compression by vector quantization |
| 7 | **PCA Word Embeddings** | Embeddings from a 3000×3000 Wikipedia co-occurrence matrix; analogy solver, synonym/antonym cosine tests, gender-bias projection |
| 8 | **Hidden Markov Model** | Forward, backward, sequence likelihood, posterior γ, pairwise ξ, and **Viterbi decoding** — plus a POS tagger (MLE π/A/B, unseen-word smoothing) |
| 9 | **Decoder-only Transformer** | Scaled dot-product attention with a causal mask, multi-head concat, pre-LN residual blocks, token + position embeddings, autoregressive sampling |
| 10 | **Tabular Q-Learning** | ε-greedy policy, vectorized TD update, and an experience replay buffer over a finite MDP |

## Results worth reading

These are the numbers I actually care about — not accuracy for its own sake, but what the
implementation revealed.

**SGD vs full-batch gradient descent** (10-class MNIST, softmax regression):

| Optimizer | Wall clock | Test accuracy |
|---|---|---|
| SGD | **0.024 s** | 73.0% |
| Full-batch GD | 4.04 s (**170× slower**) | **89.6%** |

The tradeoff in one table: SGD gets you a usable model almost immediately, full-batch spends
170× the compute to buy 16 points. Which one is correct depends entirely on your budget.

**Dropout is worth more than depth** (from-scratch MLP, MNIST 5k/1k/1k, 784→1000→10):

| Configuration | Validation accuracy |
|---|---|
| ReLU + dropout 0.5 | **96.0%** (train 98.8%) |
| ReLU, no dropout | 91.5% |

A **+4.5 point** gain from one regularizer, measured across a ReLU/tanh × dropout {0, 0.25, 0.5}
× momentum {0, 0.9} grid.

**Linear classifiers** (`work 3`): Two-Moons — perceptron 84.0% test, logistic 86.7%.
Binarized MNIST — perceptron 82.8%, logistic 83.4%.

**HMM POS tagging** (`work 8`): Brown corpus, 49,469 sentences, 12 universal tags.

## Datasets

UCI Cleveland heart disease (303 × 13) · UCI white wine (4,399 rows) · MNIST ·
Brown corpus · tiny-Shakespeare (1.1 MB, character-level) · Wikipedia co-occurrence matrix.

## Layout

```
work 1/   KNN + distance metrics and scalers
work 2/   linear and ridge regression
work 3/   perceptron, logistic, multiclass softmax (GD and SGD)
work 4/   MLP with hand-written backprop, dropout, momentum
work 5/   AdaBoost
work 6/   K-means, K-means++, vector-quantization image compression
work 7/   PCA word embeddings, analogies, bias projection
work 8/   HMM forward/backward/Viterbi + POS tagger
work 9/   decoder-only Transformer
work 10/  tabular Q-learning with experience replay
```

## A note on scope

These were course assignments, so the scaffolding — data loaders, class signatures, unit
tests — was provided. What is mine is the algorithm inside each one, plus the experiments and
ablations that produced the numbers above. I have kept the repo public because the implementations
are the clearest evidence I have of understanding these methods from the inside rather than the
API surface.

---

<p align="center">
  <a href="https://kelidari.com">kelidari.com</a> ·
  <a href="https://github.com/Nikelroid">github.com/Nikelroid</a>
</p>
