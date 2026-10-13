"""
Automated unit tests for Evaluation Metrics.
Validates:
- Cohen's Kappa (unweighted vs quadratic-weighted)
- Bootstrap 95% Confidence Interval calculation
- Context precision and recall functions
"""

import pytest
import numpy as np
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from evaluation.metrics import compute_cohens_kappa, compute_context_precision, compute_context_recall, bootstrap_ci

def test_cohens_kappa_perfect_agreement():
    labels = ["Low", "Medium", "High", "Critical"]
    k_unw = compute_cohens_kappa(labels, labels, weights=None)
    k_quad = compute_cohens_kappa(labels, labels, weights="quadratic")
    assert k_unw == 1.0
    assert k_quad == 1.0

def test_cohens_kappa_quadratic_penalizes_large_discrepancies():
    """
    On ordinal scales, quadratic kappa penalizes distant misclassifications
    (Critical vs Low) more heavily than adjacent ones (Critical vs High).
    """
    human = ["Low", "Low", "Critical", "Critical"]
    adjacent = ["Medium", "Medium", "High", "High"]
    distant = ["Critical", "Critical", "Low", "Low"]

    k_adj_quad = compute_cohens_kappa(human, adjacent, weights="quadratic")
    k_dist_quad = compute_cohens_kappa(human, distant, weights="quadratic")

    assert k_adj_quad > k_dist_quad

def test_bootstrap_ci_coverage():
    """Validates that bootstrap 95% CI bounds contain the true sample mean."""
    rng = np.random.default_rng(42)
    sample = rng.normal(loc=0.5, scale=0.1, size=100).tolist()
    mean, lower, upper = bootstrap_ci(sample, n_resamples=500, seed=42)

    assert lower <= mean <= upper
    assert lower > 0.0
    assert upper < 1.0

def test_context_precision_ranking():
    """Relevant chunks ranked first produce higher AP than ranked last."""
    expected = ["CHUNK-A", "CHUNK-B"]
    ranked_good = ["CHUNK-A", "CHUNK-B", "CHUNK-C", "CHUNK-D"]
    ranked_poor = ["CHUNK-C", "CHUNK-D", "CHUNK-A", "CHUNK-B"]

    prec_good = compute_context_precision(ranked_good, expected)
    prec_poor = compute_context_precision(ranked_poor, expected)

    assert prec_good > prec_poor
