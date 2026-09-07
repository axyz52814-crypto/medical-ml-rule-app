"""
explain.py
----------
Turns a single person's inputs into per-feature SHAP contributions for whatever
model won in the notebook. Kept separate from app.py so the app stays clean.

Works whether the saved best model is Logistic Regression (LinearExplainer,
exact + instant) or a tree model like HistGradientBoosting / RandomForest
(TreeExplainer, fast). You do not need to change anything if the winning model
changes - this picks the right explainer automatically.
"""

import numpy as np
import shap
from sklearn.linear_model import LogisticRegression


def _preprocess(pipeline, X):
    """Apply every pipeline step except the final estimator (e.g. the scaler)."""
    Xt = X
    for _, step in pipeline.steps[:-1]:
        Xt = step.transform(Xt)
    return Xt


def explain_row(pipeline, background_df, row_df, feature_names):
    """
    pipeline       : the saved sklearn Pipeline (scaler + classifier)
    background_df  : small sample of training rows (raw feature values)
    row_df         : single-row DataFrame with the user's inputs
    feature_names  : list of feature names, in order
    returns        : list of (feature_name, shap_value) sorted by |value| desc
    """
    final = pipeline.steps[-1][1]
    bg = _preprocess(pipeline, background_df)
    row = _preprocess(pipeline, row_df)

    if isinstance(final, LogisticRegression):
        explainer = shap.LinearExplainer(final, bg)
        vals = np.array(explainer.shap_values(row))[0]
    else:
        explainer = shap.TreeExplainer(final)
        sv = np.array(explainer.shap_values(row))
        # Normalise possible shapes to a 1-D vector for the positive class.
        if sv.ndim == 3:               # (classes, n, feats) or (n, feats, classes)
            if sv.shape[0] == 2:
                vals = sv[1][0]        # positive class, first (only) row
            else:
                vals = sv[0, :, -1]
        else:                          # (n, feats)
            vals = sv[0]

    pairs = list(zip(feature_names, [float(v) for v in vals]))
    return sorted(pairs, key=lambda x: -abs(x[1]))