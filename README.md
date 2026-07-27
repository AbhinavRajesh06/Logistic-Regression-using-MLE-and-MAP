# Logistic Regression: MLE vs MAP (L1/L2) Comparison

This script trains three variants of logistic regression on the Breast Cancer Wisconsin dataset and compares their **test accuracy** and **learned feature weights**, illustrating how regularization (L1 vs L2) changes a model compared to unregularized (MLE) fitting.

## What It Does

1. **Loads data**: Uses scikit-learn's built-in Breast Cancer dataset (30 numeric features per sample, binary label: malignant or benign).

2. **Splits and scales**:
   - 80% train / 20% test split, stratified so both sets have the same class balance.
   - Features are standardized (mean 0, std 1) using `StandardScaler`, since logistic regression is sensitive to feature scale.

3. **Trains three logistic regression models**:

   | Model | Penalty | Interpretation |
   |---|---|---|
   | **MLE** | `penalty=None` | Plain maximum likelihood fit, no regularization |
   | **MAP (L1)** | `penalty='l1'` | Laplace prior on weights; encourages sparsity (some weights → exactly 0) |
   | **MAP (L2)** | `penalty='l2'` | Gaussian prior on weights; shrinks all weights smoothly toward 0 |

4. **Evaluates accuracy** of each model on the held-out test set.

5. **Plots a 2×2 grid**:
   - **Top-left**: bar chart comparing test accuracy (%) across the three models.
   - **Remaining 3 panels**: line plots of each model's 30 learned feature weights, with a dashed line at zero for reference.

## Requirements

```
matplotlib
scikit-learn
```

Install with:
```bash
pip install matplotlib scikit-learn
```

## Usage

```bash
python mle_vs_map_logreg.py
```

No internet connection needed — the dataset is bundled with scikit-learn.

## Output

A single figure with 4 subplots:
- Accuracy comparison bar chart (top-left)
- Weight plots for MLE, MAP (L1), and MAP (L2) (remaining 3 panels)

**What to look for:**
- L1 weights should show many values pinned at or near zero (automatic feature selection).
- L2 weights should look more uniformly shrunk but rarely exactly zero.
- MLE weights are typically the largest in magnitude, with no shrinkage applied.
- Accuracy differences across the three models are often small on this dataset — regularization's main effect here is on weight behavior, not necessarily raw accuracy.

## Key Concept

Adding a **prior** over the model's weights during training corresponds to **MAP estimation**:
- **L1 penalty** ↔ **Laplace prior** → sparse weights
- **L2 penalty** ↔ **Gaussian prior** → smooth, small weights
- **No penalty** ↔ **MLE** → weights fit purely to the training data, no prior belief incorporated

This mirrors the Bayesian MLE-vs-MAP framing seen in probability estimation more generally, just applied here to a model's parameters instead of word frequencies.

## File Structure

```
.
├── mle_vs_map_logreg.py   # Main script
└── README.md              # This file
```
