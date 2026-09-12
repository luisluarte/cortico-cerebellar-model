---
name: scikit-learn
description: Machine learning pipelines, regularized regression, dimensionality reduction, and cross-validation for neuro-behavioral modeling with scikit-learn.
---

# Scikit-Learn for Scientific Modeling

## 1. Regularized Readout Models
```python
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet
import numpy as np

# Cross-validated L2 Ridge for reservoir state readouts
ridge = RidgeCV(alphas=np.logspace(-4, 4, 50), cv=5)
ridge.fit(reservoir_states, target_outputs)
best_w = ridge.coef_
```

## 2. Dimensionality Reduction & Manifold Projection
```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

pca = PCA(n_components=10)
latent_Z = pca.fit_transform(high_dim_states)
explained_var = pca.explained_variance_ratio_
```

## 3. Subject-Aware Cross-Validation
```python
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

# Leave-One-Subject-Out (LNSO) cross-validation
logo = LeaveOneGroupOut()
for train_idx, test_idx in logo.split(X, y, groups=subject_ids):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
```

## 4. Pipeline & Preprocessing
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", Ridge(alpha=1.0))
])
pipeline.fit(X_train, y_train)
```
