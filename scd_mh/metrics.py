"""Evaluation metrics for the SCD-MH framework.

Implements the metrics used in the experimental evaluation (Section 7):

- ``estimate_kl_divergence``: KL divergence between the NSF distribution
  Q_φ and the true conditional P*_φ (Theorem 4.3 / Eq. 9, Section 7.2).
- ``compute_tv_distance``: Total variation distance between sample
  distributions and a target (used in convergence analysis, Section 7.4).
- ``measure_mixing_time``: Empirical mixing time of the SCD-MH chain
  (Section 7.4, Figure 4).
- ``evaluate_reasoning_chain``: Step accuracy and chain length metrics
  for reasoning preservation (Section 7.3, Table 3).

References
----------
- Theorem 4.3 / Eq. 9 (KL divergence bound)
- Theorem 5.4 / Eq. 11 (mixing time bound)
- Section 7.2 (Distribution Distortion Measurement)
- Section 7.3 (Reasoning Preservation, Table 3)
- Section 7.4 (Convergence Analysis, Figure 4)
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from scd_mh.models import AutoregressiveModel
    from scd_mh.oracles import SemanticOracle

logger = logging.getLogger("scd_mh.metrics")


# ---------------------------------------------------------------------------
# KL Divergence  (Theorem 4.3 / Eq. 9 / Section 7.2)
# ---------------------------------------------------------------------------

def estimate_kl_divergence(
    model: "AutoregressiveModel",
    oracle: "SemanticOracle",
    prompt: list[int],
    n_samples: int = 10000,
    max_length: int = 256,
) -> dict[str, float]:
    """Estimate KL divergence between NSF distribution and true conditional.

    Computes KL(Q_φ ‖ P*_φ) via importance sampling, as described in
    Section 7.2:

        "We empirically measure the KL divergence between the NSF
         distribution and the true conditional (estimated via importance
         sampling with 10,000 samples)."

    The KL divergence decomposes as (Theorem 4.3, Eq. 9):

        KL(Q_φ ‖ P*_φ) = Σ_{t=1}^{n} E_{x_{1:t} ~ Q_φ}
                          [log SFS_φ(x_{1:t-1}) / SFS_φ(x_{1:t})]

    which equals zero iff SFS_φ(x_{1:t}) is a deterministic function of t.

    We use the alternative importance-sampling estimator:

        KL ≈ (1/N) Σ_i [log Q_φ(x_i) - log P*_φ(x_i)]

    where x_i ~ Q_φ and log P*_φ(x) = log P(x) - log P(C_φ) is estimated
    from the samples.

    Parameters
    ----------
    model : AutoregressiveModel
        The language model P.
    oracle : SemanticOracle
        The semantic oracle O_φ.
    prompt : list[int]
        Prompt token ids.
    n_samples : int
        Number of importance samples (default 10,000 per Section 7.2).
    max_length : int
        Maximum sequence length.

    Returns
    -------
    dict[str, float]
        Dictionary with keys:
        - ``"kl_divergence"``: Estimated KL(Q_φ ‖ P*_φ).
        - ``"kl_std_error"``: Standard error of the estimate.
        - ``"constraint_mass"``: Estimated P(C_φ) = fraction of
          unconstrained samples satisfying the constraint.
        - ``"n_valid_samples"``: Number of valid (SAT) samples obtained.

    Notes
    -----
    - Theorem 4.3 / Eq. 9: KL decomposition via SFS ratios.
    - Section 7.2: "estimated via importance sampling with 10,000 samples."
    - Table 2: reports mean KL ± standard error across benchmarks.
    """
    from scd_mh.decoding import naive_semantic_filter
    from scd_mh.oracles import OracleResult

    log_ratios: list[float] = []
    n_sat = 0

    for i in range(n_samples):
        # Sample x ~ Q_φ via NSF
        nsf_result = naive_semantic_filter(
            model, oracle, prompt, max_length=max_length, max_retries=1,
        )

        if not nsf_result.is_sat:
            continue

        n_sat += 1

        # log Q_φ(x) — computed during generation
        log_q = nsf_result.log_q

        # log P(x) — from the model
        log_p = model.log_prob(nsf_result.sequence)

        # log(P(x) / Q_φ(x)) — importance weight (unnormalised)
        # KL(Q‖P*) = E_Q[log Q/P*] = E_Q[log Q - log P + log P(C_φ)]
        # We accumulate log(Q/P) and estimate log P(C_φ) separately
        log_ratio = log_q - log_p
        log_ratios.append(log_ratio)

        if (i + 1) % 1000 == 0:
            logger.debug(
                "KL estimation: %d/%d samples processed, %d valid",
                i + 1, n_samples, n_sat,
            )

    if not log_ratios:
        logger.warning("No valid samples for KL estimation")
        return {
            "kl_divergence": float("nan"),
            "kl_std_error": float("nan"),
            "constraint_mass": 0.0,
            "n_valid_samples": 0,
        }

    # Estimate P(C_φ) from unconstrained generation acceptance rate
    constraint_mass = n_sat / n_samples

    # KL(Q_φ ‖ P*_φ) = E_Q[log Q_φ(x)] - E_Q[log P(x)] + log P(C_φ)
    #                  = mean(log_ratios) + log(P(C_φ))
    log_ratios_arr = np.array(log_ratios)
    kl_raw = float(np.mean(log_ratios_arr))
    log_constraint_mass = math.log(max(constraint_mass, 1e-10))
    kl = kl_raw + log_constraint_mass

    # Standard error
    kl_se = float(np.std(log_ratios_arr) / np.sqrt(len(log_ratios_arr)))

    logger.info(
        "KL(Q_φ ‖ P*_φ) ≈ %.4f ± %.4f  (P(C_φ) ≈ %.4f, n_valid=%d)",
        kl, kl_se, constraint_mass, n_sat,
    )

    return {
        "kl_divergence": max(kl, 0.0),  # KL is non-negative
        "kl_std_error": kl_se,
        "constraint_mass": constraint_mass,
        "n_valid_samples": n_sat,
    }


# ---------------------------------------------------------------------------
# Total Variation Distance  (Theorem 5.3 / Section 7.4)
# ---------------------------------------------------------------------------

def compute_tv_distance(
    samples: list[list[int]] | list[str],
    target_dist: dict[str | tuple, float],
) -> float:
    """Compute total variation distance between empirical and target distributions.

    TV(P, Q) = (1/2) Σ_x |P(x) - Q(x)|

    Used in Section 7.4 (Convergence Analysis) to verify that SCD-MH's
    empirical distribution converges to P*_φ as T increases
    (Theorem 5.3, Eq. 10).

    Parameters
    ----------
    samples : list[list[int]] | list[str]
        Empirical samples (token-id sequences or strings).
    target_dist : dict
        Target distribution mapping sequences (as tuples or strings)
        to probabilities. Must sum to ≤ 1.

    Returns
    -------
    float
        Total variation distance ∈ [0, 1].

    Notes
    -----
    - Theorem 5.3: TV(L(x^{(T)}), P*_φ) → 0 as T → ∞.
    - Section 7.4: "We compare the empirical mixing time of SCD-MH
      against the theoretical bound of Theorem 5.4."
    """
    if not samples:
        return 1.0

    # Convert samples to hashable keys
    def to_key(s):
        if isinstance(s, str):
            return s
        return tuple(s)

    # Build empirical distribution
    n = len(samples)
    empirical: dict = {}
    for s in samples:
        k = to_key(s)
        empirical[k] = empirical.get(k, 0) + 1 / n

    # Compute TV distance
    all_keys = set(empirical.keys()) | set(target_dist.keys())
    tv = 0.0
    for k in all_keys:
        p_emp = empirical.get(k, 0.0)
        p_target = target_dist.get(k, 0.0)
        tv += abs(p_emp - p_target)

    return tv / 2.0


# ---------------------------------------------------------------------------
# Mixing Time  (Theorem 5.4 / Eq. 11 / Section 7.4)
# ---------------------------------------------------------------------------

def measure_mixing_time(
    chain: list[list[int]],
    epsilon: float = 0.05,
    window_size: int = 10,
    reference_dist: Optional[dict] = None,
) -> dict[str, float | int]:
    """Measure empirical mixing time of an SCD-MH chain.

    The mixing time is defined as (Section 5.3, below Eq. 11):

        τ_mix(ε) = min{T : TV(L(x^{(T)}), P*_φ) ≤ ε}

    The theoretical upper bound (Theorem 5.4 / Eq. 11) is:

        τ_mix(ε) ≤ (p_max / p_min) · log(1 / (ε · min P*_φ(x)))

    where p_min, p_max are the min/max importance weight ratios
    Q_φ(x) / P*_φ(x).

    We estimate mixing time empirically by monitoring convergence
    diagnostics over the chain trajectory.

    Parameters
    ----------
    chain : list[list[int]]
        Full Markov chain trajectory from SCD-MH
        (SCDMHResult.chain).
    epsilon : float
        TV distance threshold (default 0.05, as in Section 7.4 / Fig. 4).
    window_size : int
        Rolling window size for computing empirical distributions.
    reference_dist : dict, optional
        Reference distribution for TV computation. If None, uses the
        second half of the chain as the reference.

    Returns
    -------
    dict[str, float | int]
        Dictionary with keys:
        - ``"mixing_time"``: Estimated mixing time (iterations).
        - ``"final_tv"``: TV distance at the last iteration.
        - ``"convergence_curve"``: List of TV values over iterations.

    Notes
    -----
    - Theorem 5.4 / Eq. 11: theoretical mixing time upper bound.
    - Section 7.4: "Empirical mixing time (measured as iterations to
      reach TV < 0.05)."
    - Figure 4: empirical vs. theoretical mixing time.
    """
    if len(chain) < 2 * window_size:
        logger.warning(
            "Chain too short (%d) for mixing time estimation with window=%d",
            len(chain),
            window_size,
        )
        return {
            "mixing_time": len(chain),
            "final_tv": 1.0,
            "convergence_curve": [],
        }

    # Build reference distribution from the second half of the chain
    if reference_dist is None:
        mid = len(chain) // 2
        ref_samples = chain[mid:]
        n_ref = len(ref_samples)
        reference_dist = {}
        for seq in ref_samples:
            k = tuple(seq)
            reference_dist[k] = reference_dist.get(k, 0) + 1 / n_ref

    # Compute TV distance over rolling windows
    convergence_curve: list[float] = []
    mixing_time = len(chain)  # default: never mixed

    for start in range(0, len(chain) - window_size + 1):
        window = chain[start : start + window_size]
        tv = compute_tv_distance(window, reference_dist)
        convergence_curve.append(tv)

        if tv <= epsilon and mixing_time == len(chain):
            mixing_time = start + window_size

    final_tv = convergence_curve[-1] if convergence_curve else 1.0

    logger.info(
        "Mixing time (ε=%.3f): %d iterations, final TV=%.4f",
        epsilon,
        mixing_time,
        final_tv,
    )

    return {
        "mixing_time": mixing_time,
        "final_tv": final_tv,
        "convergence_curve": convergence_curve,
    }


# ---------------------------------------------------------------------------
# Reasoning Chain Evaluation  (Section 7.3 / Table 3)
# ---------------------------------------------------------------------------

def evaluate_reasoning_chain(
    steps: list[str],
    oracle: "SemanticOracle",
    gold_steps: Optional[list[str]] = None,
) -> dict[str, float]:
    """Evaluate the quality of a reasoning chain.

    Computes the metrics reported in Table 3 (Section 7.3):

    - **Step Accuracy**: Fraction of intermediate reasoning steps that
      are individually correct (verified by the oracle).
    - **Chain Length**: Number of reasoning steps.
    - **Step Consistency**: Fraction of step transitions that are
      logically consistent with preceding steps.

    Parameters
    ----------
    steps : list[str]
        Reasoning steps extracted from the model's output.
    oracle : SemanticOracle
        The semantic oracle for verifying individual steps.
    gold_steps : list[str], optional
        Ground-truth reasoning steps for comparison (if available).

    Returns
    -------
    dict[str, float]
        Dictionary with keys:
        - ``"step_accuracy"``: Fraction of correct steps.
        - ``"chain_length"``: Number of steps (float for consistency).
        - ``"step_consistency"``: Fraction of consistent transitions.
        - ``"n_correct_steps"``: Count of correct steps.
        - ``"n_total_steps"``: Total number of steps.

    Notes
    -----
    - Section 7.3: "We evaluate the effect of semantic constraints on
      chain-of-thought quality."
    - Table 3: Step Acc., Chain Len. for FOLIO and ProofWriter.
    - Eq. 13 (Reasoning Capacity): RC(P*_φ) = H_{P*_φ}(r | a).
    """
    from scd_mh.oracles import OracleResult

    if not steps:
        return {
            "step_accuracy": 0.0,
            "chain_length": 0.0,
            "step_consistency": 0.0,
            "n_correct_steps": 0,
            "n_total_steps": 0,
        }

    n_total = len(steps)
    n_correct = 0
    n_consistent = 0

    # Evaluate individual step correctness
    for step in steps:
        try:
            result = oracle.verify(step)
            if result == OracleResult.SAT:
                n_correct += 1
        except Exception as e:
            logger.debug("Step verification failed: %s", e)

    # Evaluate step-to-step consistency
    # A transition from step i to step i+1 is consistent if
    # the oracle accepts the concatenation of steps 1..i+1
    for i in range(1, len(steps)):
        partial_chain = " ".join(steps[: i + 1])
        try:
            result = oracle.verify(partial_chain)
            if result == OracleResult.SAT:
                n_consistent += 1
        except Exception:
            pass

    step_accuracy = n_correct / n_total
    n_transitions = max(n_total - 1, 1)
    step_consistency = n_consistent / n_transitions

    # Compare with gold steps if available
    if gold_steps is not None:
        # Compute overlap metric
        gold_set = set(s.strip().lower() for s in gold_steps)
        pred_set = set(s.strip().lower() for s in steps)
        if gold_set:
            overlap = len(gold_set & pred_set) / len(gold_set)
        else:
            overlap = 0.0
    else:
        overlap = float("nan")

    result_dict = {
        "step_accuracy": step_accuracy,
        "chain_length": float(n_total),
        "step_consistency": step_consistency,
        "n_correct_steps": n_correct,
        "n_total_steps": n_total,
    }

    if gold_steps is not None:
        result_dict["gold_overlap"] = overlap

    logger.info(
        "Reasoning chain: %d steps, %.1f%% accurate, %.1f%% consistent",
        n_total,
        step_accuracy * 100,
        step_consistency * 100,
    )

    return result_dict
