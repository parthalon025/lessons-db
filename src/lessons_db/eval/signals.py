"""Signal extractors and Bayesian fusion for transfer-test evaluation."""

import math
import re as _re
from collections import defaultdict
from typing import Any

# ---------------------------------------------------------------------------
# Response parsers
# ---------------------------------------------------------------------------


def parse_mechanism_triplet(response: str) -> dict[str, str] | None:
    """Parse TRIGGER/TARGET/FIX triplet from mechanism extraction response."""
    if not response:
        return None
    text = _re.sub(r"<think>.*?</think>", "", response, flags=_re.DOTALL | _re.IGNORECASE).strip()
    if "NONE" in text.upper() and len(text) < 50:
        return None
    trigger = _re.search(r"TRIGGER:\s*(.+)", text, _re.IGNORECASE)
    target = _re.search(r"TARGET:\s*(.+)", text, _re.IGNORECASE)
    fix = _re.search(r"FIX:\s*(.+)", text, _re.IGNORECASE)
    if not trigger or not target or not fix:
        return None
    return {
        "trigger": trigger.group(1).strip()[:100],
        "target": target.group(1).strip()[:100],
        "fix": fix.group(1).strip()[:100],
    }


# ---------------------------------------------------------------------------
# Signal extractors — log-likelihood ratios for Bayesian fusion (Task 9)
# ---------------------------------------------------------------------------


def compute_paired_signal(winner: str) -> float:
    """Convert paired comparison outcome to log-likelihood ratio.

    Calibrated from paired tournament experiment:
    - "same": judge picked same-group target → strong positive evidence
    - "diff": judge picked diff-group target → strong negative evidence
    - "neither": judge couldn't decide → uninformative
    """
    return {"same": 2.5, "diff": -2.5, "neither": 0.0}.get(winner, 0.0)


def compute_embedding_signal(cosine_sim: float) -> float:
    """Convert cosine similarity to log-likelihood ratio.

    Thresholds calibrated from embedding AUC=0.707 baseline:
    - >= 0.7: strong semantic overlap → positive
    - >= 0.5: moderate overlap → mild positive
    - >= 0.3: weak overlap → mild negative
    - < 0.3: no meaningful overlap → negative
    """
    if cosine_sim >= 0.7:
        return 1.5
    elif cosine_sim >= 0.5:
        return 0.5
    elif cosine_sim >= 0.3:
        return -0.5
    else:
        return -1.5


def compute_scope_signal(principle_scopes: set, target_scopes: set) -> float:
    """Convert scope tag overlap (Jaccard) to log-likelihood ratio.

    Empty scope on either side → uninformative (0.0).
    """
    if not principle_scopes or not target_scopes:
        return 0.0
    overlap = len(principle_scopes & target_scopes) / len(principle_scopes | target_scopes)
    if overlap >= 0.5:
        return 1.0
    elif overlap > 0:
        return 0.3
    else:
        return -0.5


def compute_mechanism_signal(mechanism_match: bool | None) -> float:
    """Convert mechanism-naming match to log-likelihood ratio.

    Calibrated: P=1.0 (mechanism match → always transfers), R=0.5.
    None means mechanism data unavailable → uninformative.
    """
    if mechanism_match is True:
        return 2.0
    elif mechanism_match is False:
        return -1.5
    else:
        return 0.0


# ---------------------------------------------------------------------------
# Bayesian fusion — compute_transfer_posterior (Task 10)
# ---------------------------------------------------------------------------

# Prior: P(transfers) = 0.25 — most principles DON'T transfer to arbitrary targets
_PRIOR_LOG_ODDS = math.log(0.25 / 0.75)  # ≈ -1.10


def compute_transfer_posterior(
    paired_signal: float,
    embedding_signal: float,
    scope_signal: float,
    mechanism_signal: float,
) -> float:
    """Compute P(transfers | signals) via naive Bayes log-odds fusion.

    Each signal is a log-likelihood ratio from an independent evidence source.
    Combines via addition in log-odds space, then sigmoid to probability.
    Same math as ollama-queue stall detector and ARIA occupancy fusion.
    """
    log_odds = _PRIOR_LOG_ODDS + paired_signal + embedding_signal + scope_signal + mechanism_signal
    return 1.0 / (1.0 + math.exp(-log_odds))


# ---------------------------------------------------------------------------
# Bayesian metrics — AUC via Mann-Whitney U + separation (Task 11)
# ---------------------------------------------------------------------------


def compute_bayesian_metrics(
    scored_pairs: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """Compute AUC and separation metrics from Bayesian fusion posteriors.

    Input: list of dicts with keys: variant, is_same_group (bool), posterior (float)
    Output: per-variant metrics dict with:
        same_mean_posterior, diff_mean_posterior, separation,
        auc (Mann-Whitney U), calibration_error, pair_count
    """
    by_variant: dict[str, list[dict]] = defaultdict(list)
    for entry in scored_pairs:
        by_variant[entry["variant"]].append(entry)

    metrics: dict[str, dict[str, float]] = {}
    for variant_id, entries in sorted(by_variant.items()):
        same_posteriors = [e["posterior"] for e in entries if e["is_same_group"]]
        diff_posteriors = [e["posterior"] for e in entries if not e["is_same_group"]]

        same_mean = sum(same_posteriors) / len(same_posteriors) if same_posteriors else 0.0
        diff_mean = sum(diff_posteriors) / len(diff_posteriors) if diff_posteriors else 0.0

        # AUC via Mann-Whitney U statistic
        if same_posteriors and diff_posteriors:
            u_count = 0
            ties = 0
            for s in same_posteriors:
                for d in diff_posteriors:
                    if s > d:
                        u_count += 1
                    elif s == d:
                        ties += 1
            auc = (u_count + 0.5 * ties) / (len(same_posteriors) * len(diff_posteriors))
        else:
            auc = 0.5  # degenerate: can't compute

        # Calibration error
        all_posteriors = [e["posterior"] for e in entries]
        mean_posterior = sum(all_posteriors) / len(all_posteriors) if all_posteriors else 0.0
        actual_positive_frac = len(same_posteriors) / len(entries) if entries else 0.0
        calibration_error = abs(mean_posterior - actual_positive_frac)

        metrics[variant_id] = {
            "same_mean_posterior": same_mean,
            "diff_mean_posterior": diff_mean,
            "separation": same_mean - diff_mean,
            "auc": auc,
            "calibration_error": calibration_error,
            "pair_count": len(entries),
        }

    return metrics
