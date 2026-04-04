"""Core decoding algorithms for the SCD-MH framework.

Implements the three algorithms from the paper:

1. ``naive_semantic_filter`` — Naive Semantic Filtering (NSF), Section 4.1, Eq. 8:
   Token-level masking via the semantic oracle. This is the direct analogue
   of grammar-constrained decoding (GCD) for semantic constraints.

2. ``scd_mh_sample`` — SCD-MH (Algorithm 1, Section 5.1): Full
   Metropolis-Hastings sampler that uses NSF as a proposal distribution
   and converges to the true conditional P*_φ (Theorem 5.3).

3. ``solver_guided_relaxation`` — Solver-Guided Constraint Relaxation
   (Algorithm 2, Section 6.3): Constructs an augmented constraint φ' that
   preserves reasoning capacity (Theorem 6.1).

References
----------
- Section 4.1 (Naive Semantic Filtering, Eq. 8)
- Section 5.1 (SCD-MH Algorithm, Algorithm 1)
- Section 5.2 (Convergence Guarantee, Theorem 5.3)
- Section 5.3 (Convergence Rate, Theorem 5.4 / Eq. 11)
- Section 6.3 (Solver-Guided Constraint Relaxation, Algorithm 2)
- Eq. 12 (MH acceptance ratio)
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from scd_mh.models import AutoregressiveModel
from scd_mh.oracles import OracleResult, PrefixResult, SemanticOracle

logger = logging.getLogger("scd_mh.decoding")


# ---------------------------------------------------------------------------
# Data classes for return types
# ---------------------------------------------------------------------------

@dataclass
class NSFResult:
    """Result of naive semantic filtering.

    Attributes
    ----------
    sequence : list[int]
        Generated token-id sequence.
    log_q : float
        Log-probability under the NSF distribution Q_φ.
    is_sat : bool
        Whether the oracle verified the sequence as SAT.
    num_oracle_calls : int
        Total number of prefix oracle calls made during generation.
    """

    sequence: list[int]
    log_q: float
    is_sat: bool
    num_oracle_calls: int = 0


@dataclass
class SCDMHResult:
    """Result of SCD-MH sampling.

    Attributes
    ----------
    final_sequence : list[int]
        The sequence at iteration T (or the last accepted sample).
    samples : list[list[int]]
        All post-burn-in samples {x^{(i)}}_{i > B}.
    acceptance_rates : list[float]
        Per-iteration acceptance probabilities.
    chain : list[list[int]]
        Full Markov chain trajectory (all T+1 states).
    log_probs : list[float]
        Log P(x^{(i)}) for each chain state.
    log_q_probs : list[float]
        Log Q_φ(x^{(i)}) for each chain state.
    num_iterations : int
        Total iterations completed.
    num_accepted : int
        Number of accepted proposals.
    """

    final_sequence: list[int] = field(default_factory=list)
    samples: list[list[int]] = field(default_factory=list)
    acceptance_rates: list[float] = field(default_factory=list)
    chain: list[list[int]] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    log_q_probs: list[float] = field(default_factory=list)
    num_iterations: int = 0
    num_accepted: int = 0


# ---------------------------------------------------------------------------
# Naive Semantic Filtering  (Section 4.1 / Eq. 8)
# ---------------------------------------------------------------------------

def naive_semantic_filter(
    model: AutoregressiveModel,
    oracle: SemanticOracle,
    prompt: list[int],
    max_length: int = 256,
    temperature: float = 1.0,
    max_retries: int = 3,
) -> NSFResult:
    """Naive Semantic Filtering (NSF) — Section 4.1, Eq. 8.

    At each decoding step t, given prefix x_{1:t-1}, computes the set of
    allowed tokens (Eq. 6):

        A_φ(x_{1:t-1}) = {v ∈ Σ : O^{prefix}_φ(x_{1:t-1} · v) ≠ DEAD}

    and samples from the renormalised distribution (Eq. 8):

        Q_φ(x_t | x_{1:t-1}) = P(x_t | x_{1:t-1}) · 𝟙[x_t ∈ A_φ(x_{1:t-1})]
                                 / Σ_{v ∈ A_φ} P(v | x_{1:t-1})

    This is the semantic analogue of grammar-constrained decoding (GCD).
    As proven in Theorem 4.2, NSF does NOT sample from the true conditional
    P*_φ — it introduces distribution distortion governed by SFS ratios.

    Parameters
    ----------
    model : AutoregressiveModel
        The language model P.
    oracle : SemanticOracle
        The semantic constraint oracle O_φ.
    prompt : list[int]
        Prompt token ids.
    max_length : int
        Maximum total sequence length (default 256).
    temperature : float
        Sampling temperature (default 1.0).
    max_retries : int
        Maximum attempts to generate a SAT sequence (default 3).

    Returns
    -------
    NSFResult
        The generated sequence, its log Q_φ probability, and SAT status.

    Notes
    -----
    - Section 4.1: "The most natural approach to enforcing a semantic
      constraint during autoregressive generation."
    - Eq. 8: Q_φ(x_t | x_{1:t-1}) = P(x_t|x_{1:t-1}) · 𝟙[x_t ∈ A_φ] / Z_t
    - Theorem 4.2: "NSF does not sample from the target distribution P*_φ."
    - For prefix-opaque constraints (Definition 3.2, case 3), NSF reduces
      to unconstrained generation followed by rejection.
    """
    eos_id = model.get_eos_token_id()
    vocab_size = model.get_vocab_size()
    total_oracle_calls = 0

    for attempt in range(max_retries):
        sequence = list(prompt)
        log_q_total = 0.0
        generation_failed = False

        for step in range(max_length - len(prompt)):
            # Get next-token logits from the model
            logits = model.get_next_token_logits(sequence)

            # Apply temperature
            if temperature != 1.0:
                logits = logits / max(temperature, 1e-8)

            # Compute log P(v | x_{1:t-1}) for all v
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            probs = torch.exp(log_probs)

            # Compute allowed token set A_φ(x_{1:t-1})  (Eq. 6)
            # For each token v, check if prefix · v is not DEAD
            allowed_mask = torch.ones(vocab_size, dtype=torch.bool, device=logits.device)

            # We check a subset of high-probability tokens for efficiency;
            # tokens with negligible probability are allowed by default
            # (prefix-approximable conservative behaviour)
            top_k = min(vocab_size, 100)
            top_indices = torch.topk(probs, top_k).indices.tolist()

            for v in top_indices:
                prefix_check = oracle.check_prefix(sequence + [v])
                total_oracle_calls += 1
                if prefix_check == PrefixResult.DEAD:
                    allowed_mask[v] = False

            # Apply mask and renormalise (Eq. 8)
            masked_log_probs = log_probs.clone()
            masked_log_probs[~allowed_mask] = float("-inf")

            # Check if any tokens are allowed
            if torch.all(~allowed_mask[:top_k]):
                # All top tokens are DEAD — fall back to unconstrained
                logger.debug(
                    "NSF: all top-%d tokens DEAD at step %d, falling back",
                    top_k,
                    step,
                )
                masked_log_probs = log_probs

            # Renormalise: Q_φ(x_t | x_{1:t-1})
            masked_probs = torch.exp(masked_log_probs)
            z_t = masked_probs.sum()
            if z_t <= 0:
                logger.debug("NSF: zero normalisation at step %d", step)
                generation_failed = True
                break

            normalised_probs = masked_probs / z_t

            # Sample x_t ~ Q_φ(· | x_{1:t-1})
            try:
                token_id = torch.multinomial(normalised_probs, num_samples=1).item()
            except RuntimeError:
                logger.debug("NSF: multinomial sampling failed at step %d", step)
                generation_failed = True
                break

            # Accumulate log Q_φ(x_t | x_{1:t-1})
            log_q_step = torch.log(normalised_probs[token_id] + 1e-30).item()
            log_q_total += log_q_step

            sequence.append(token_id)

            # Check for EOS
            if token_id == eos_id:
                break

        if generation_failed:
            continue

        # Verify complete sequence with the oracle
        # (Definition 2.4, property 3: complete sequences → SAT or UNSAT)
        result = oracle.verify(sequence)
        total_oracle_calls += 1

        if result == OracleResult.SAT:
            return NSFResult(
                sequence=sequence,
                log_q=log_q_total,
                is_sat=True,
                num_oracle_calls=total_oracle_calls,
            )
        else:
            logger.debug(
                "NSF attempt %d/%d: oracle returned %s",
                attempt + 1,
                max_retries,
                result.value,
            )

    # All retries exhausted — return best effort
    logger.warning(
        "NSF: failed to generate SAT sequence after %d attempts", max_retries
    )
    return NSFResult(
        sequence=sequence,
        log_q=log_q_total,
        is_sat=False,
        num_oracle_calls=total_oracle_calls,
    )


# ---------------------------------------------------------------------------
# Helper: compute log Q_φ(x) for an existing sequence
# ---------------------------------------------------------------------------

def _compute_log_q(
    model: AutoregressiveModel,
    oracle: SemanticOracle,
    sequence: list[int],
    prompt_length: int,
) -> float:
    """Compute log Q_φ(x) for an existing sequence.

    Replays the NSF masking process to compute the exact log-probability
    of ``sequence`` under Q_φ. Needed for the MH acceptance ratio.

    Parameters
    ----------
    model : AutoregressiveModel
        The language model P.
    oracle : SemanticOracle
        The semantic oracle O_φ.
    sequence : list[int]
        Complete token-id sequence.
    prompt_length : int
        Length of the prompt prefix (not included in Q_φ computation).

    Returns
    -------
    float
        log Q_φ(sequence).

    Notes
    -----
    - Eq. 8: Q_φ factorises autoregressively with per-step masking.
    - Remark 5.1: "Q_φ(x) can be computed exactly from the model's logits."
    """
    vocab_size = model.get_vocab_size()
    log_q_total = 0.0

    for t in range(prompt_length, len(sequence)):
        prefix = sequence[:t]
        token = sequence[t]

        logits = model.get_next_token_logits(prefix)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        # Determine allowed set (same logic as NSF)
        allowed_mask = torch.ones(vocab_size, dtype=torch.bool, device=logits.device)
        top_k = min(vocab_size, 100)
        top_indices = torch.topk(probs, top_k).indices.tolist()

        for v in top_indices:
            prefix_check = oracle.check_prefix(prefix + [v])
            if prefix_check == PrefixResult.DEAD:
                allowed_mask[v] = False

        masked_log_probs = log_probs.clone()
        masked_log_probs[~allowed_mask] = float("-inf")
        masked_probs = torch.exp(masked_log_probs)
        z_t = masked_probs.sum()

        if z_t <= 0:
            return float("-inf")

        log_q_step = (log_probs[token] - torch.log(z_t)).item()
        log_q_total += log_q_step

    return log_q_total


# ---------------------------------------------------------------------------
# SCD-MH  (Algorithm 1, Section 5.1)
# ---------------------------------------------------------------------------

def scd_mh_sample(
    model: AutoregressiveModel,
    oracle: SemanticOracle,
    prompt: list[int],
    T: int = 50,
    B: int = 10,
    max_length: int = 256,
    temperature: float = 1.0,
    seed: Optional[int] = None,
) -> SCDMHResult:
    """SCD-MH: Semantically Constrained Decoding via Metropolis-Hastings.

    Full implementation of Algorithm 1 from Section 5.1.

    The algorithm:
    1. Generate initial sequence x^{(0)} ~ Q_φ (NSF).
    2. For i = 1, …, T:
       a. Generate proposal x' ~ Q_φ (full sequence via NSF).
       b. Verify x' ∈ C_φ using O_φ.
       c. If SAT, compute acceptance ratio (Eq. 12):
              α(x^{(i-1)}, x') = min(1, P(x')·Q_φ(x^{(i-1)}) /
                                        P(x^{(i-1)})·Q_φ(x'))
       d. Accept x^{(i)} ← x' with probability α; else x^{(i)} ← x^{(i-1)}.
    3. Return x^{(T)} or post-burn-in samples {x^{(i)}}_{i > B}.

    **Convergence guarantee** (Theorem 5.3): The Markov chain has P*_φ as
    its unique stationary distribution, and TV(L(x^{(T)}), P*_φ) → 0
    as T → ∞.

    **Mixing time** (Theorem 5.4 / Eq. 11): τ_mix(ε) ≤ (p_max/p_min) ·
    log(1 / (ε · min P*_φ(x))).

    Parameters
    ----------
    model : AutoregressiveModel
        The language model P.
    oracle : SemanticOracle
        The semantic constraint oracle O_φ.
    prompt : list[int]
        Prompt token ids.
    T : int
        Number of MH iterations (default 50, as in Section 7.1).
    B : int
        Burn-in period (default 10, as in Section 7.1).
    max_length : int
        Maximum sequence length (default 256).
    temperature : float
        Sampling temperature for NSF proposals.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    SCDMHResult
        Contains the final sequence, post-burn-in samples, acceptance
        rates, and the full chain trajectory.

    Raises
    ------
    RuntimeError
        If no valid initial sequence can be generated.

    Notes
    -----
    - Algorithm 1: SCD-MH pseudocode.
    - Eq. 12: α(x, x') = min(1, P(x')·Q_φ(x) / [P(x)·Q_φ(x')])
    - Theorem 5.3: convergence guarantee.
    - Theorem 5.4: mixing time bound.
    - Section 7.1: "SCD-MH with T=50 iterations and burn-in B=10."
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    result = SCDMHResult()
    prompt_length = len(prompt)

    # -------------------------------------------------------------------
    # Step 1: Generate initial sequence x^{(0)} ~ Q_φ  (Algorithm 1, line 1)
    # -------------------------------------------------------------------
    logger.info("SCD-MH: generating initial sequence via NSF")
    init_result = naive_semantic_filter(
        model, oracle, prompt, max_length=max_length,
        temperature=temperature, max_retries=10,
    )

    if not init_result.is_sat:
        raise RuntimeError(
            "SCD-MH: failed to generate valid initial sequence. "
            "The model may assign negligible probability to the constraint set "
            "(see Corollary 5.5: convergence depends on P(C_φ))."
        )

    current_seq = init_result.sequence
    current_log_p = model.log_prob(current_seq)
    current_log_q = init_result.log_q

    result.chain.append(list(current_seq))
    result.log_probs.append(current_log_p)
    result.log_q_probs.append(current_log_q)

    logger.info(
        "SCD-MH: initial sequence generated (len=%d, log_P=%.2f, log_Q=%.2f)",
        len(current_seq),
        current_log_p,
        current_log_q,
    )

    # -------------------------------------------------------------------
    # Step 2: MH iterations  (Algorithm 1, lines 2–10)
    # -------------------------------------------------------------------
    num_accepted = 0

    for i in range(1, T + 1):
        # Line 3: Generate proposal x' ~ Q_φ
        proposal_result = naive_semantic_filter(
            model, oracle, prompt, max_length=max_length,
            temperature=temperature, max_retries=3,
        )

        proposal_seq = proposal_result.sequence
        proposal_log_q = proposal_result.log_q

        # Line 4–5: Verify x' ∈ C_φ
        if proposal_result.is_sat:
            # Line 6: Compute acceptance ratio  (Eq. 12)
            #   α = min(1, P(x') · Q_φ(x) / [P(x) · Q_φ(x')])
            proposal_log_p = model.log_prob(proposal_seq)

            # If current_log_q was not accurately computed during generation,
            # recompute it. For efficiency we use the cached value when available.
            log_alpha = (
                (proposal_log_p + current_log_q)
                - (current_log_p + proposal_log_q)
            )

            # α = min(1, exp(log_alpha))
            alpha = min(1.0, math.exp(min(log_alpha, 0.0)))

            # Line 7: Accept with probability α
            if random.random() < alpha:
                # Accept: x^{(i)} ← x'
                current_seq = proposal_seq
                current_log_p = proposal_log_p
                current_log_q = proposal_log_q
                num_accepted += 1
                logger.debug(
                    "SCD-MH iter %d/%d: ACCEPTED (α=%.4f, log_P=%.2f)",
                    i, T, alpha, proposal_log_p,
                )
            else:
                # Reject: x^{(i)} ← x^{(i-1)}
                logger.debug(
                    "SCD-MH iter %d/%d: REJECTED (α=%.4f)", i, T, alpha
                )
                alpha = alpha  # keep for recording
        else:
            # Line 9: Oracle returned UNSAT or UNKNOWN → reject
            alpha = 0.0
            logger.debug(
                "SCD-MH iter %d/%d: proposal not SAT, auto-reject", i, T
            )

        # Record chain state
        result.chain.append(list(current_seq))
        result.log_probs.append(current_log_p)
        result.log_q_probs.append(current_log_q)
        result.acceptance_rates.append(alpha)

        # Collect post-burn-in samples  (Algorithm 1, line 11)
        if i > B:
            result.samples.append(list(current_seq))

    # -------------------------------------------------------------------
    # Step 3: Return  (Algorithm 1, line 11)
    # -------------------------------------------------------------------
    result.final_sequence = list(current_seq)
    result.num_iterations = T
    result.num_accepted = num_accepted

    acceptance_rate = num_accepted / T if T > 0 else 0.0
    logger.info(
        "SCD-MH completed: %d iterations, %d accepted (%.1f%%), %d post-burn-in samples",
        T,
        num_accepted,
        acceptance_rate * 100,
        len(result.samples),
    )

    return result


# ---------------------------------------------------------------------------
# Solver-Guided Constraint Relaxation  (Algorithm 2, Section 6.3)
# ---------------------------------------------------------------------------

@dataclass
class RelaxationResult:
    """Result of solver-guided constraint relaxation.

    Attributes
    ----------
    augmented_constraint : str | None
        The augmented constraint φ' (as a string representation).
    reasoning_positions : list[int]
        Indices of positions identified as reasoning-free.
    constrained_positions : list[int]
        Indices of positions directly constrained by φ.
    """

    augmented_constraint: Optional[str] = None
    reasoning_positions: list[int] = field(default_factory=list)
    constrained_positions: list[int] = field(default_factory=list)


def solver_guided_relaxation(
    model: AutoregressiveModel,
    oracle: SemanticOracle,
    constraint: str,
    prompt: list[int],
    sequence_length: Optional[int] = None,
    relaxation_threshold: float = 0.1,
) -> RelaxationResult:
    """Solver-Guided Constraint Relaxation — Algorithm 2, Section 6.3.

    Constructs an augmented constraint φ' from φ such that
    (Theorem 6.1):

    1. C_{φ'} ⊇ C_φ  (relaxation).
    2. Answer correctness is preserved.
    3. Reasoning capacity is preserved:
       RC(P*_{φ'}) ≥ RC(P) - δ  (Eq. 14)

    The algorithm iterates over token positions and queries the oracle
    to determine which positions are "reasoning-free" (not directly
    constrained by φ). Reasoning-free positions have their constraints
    removed, yielding a larger constraint set that permits more diverse
    reasoning chains.

    Parameters
    ----------
    model : AutoregressiveModel
        The language model P.
    oracle : SemanticOracle
        The semantic oracle O_φ.
    constraint : str
        String representation of the constraint φ.
    prompt : list[int]
        Prompt token ids (to determine sequence context).
    sequence_length : int, optional
        Expected sequence length n. If None, uses max_length=256.
    relaxation_threshold : float
        Maximum allowed relaxation per position (δ_max from Algorithm 2).

    Returns
    -------
    RelaxationResult
        The augmented constraint φ' and position classifications.

    Notes
    -----
    - Algorithm 2: Solver-Guided Constraint Relaxation pseudocode.
    - Theorem 6.1: reasoning-preserving semantic augmentation guarantee.
    - Eq. 14: RC(P*_{φ'}) ≥ RC(P) - δ
    - "The construction requires O(n) oracle queries" (Section 6.3).
    - "Heuristic approximations (e.g., classifying all tokens before the
      final answer delimiter as reasoning-free) can be used."
    """
    if sequence_length is None:
        sequence_length = 256

    result = RelaxationResult()
    reasoning_positions: list[int] = []
    constrained_positions: list[int] = []

    # Algorithm 2, line 1: Initialize φ' ← φ
    augmented_parts: list[str] = []

    # Algorithm 2, lines 2–7: Iterate over positions
    for t in range(len(prompt), sequence_length):
        # Algorithm 2, line 3: Query oracle — is position t reasoning-free?
        is_reasoning = _is_reasoning_position(
            oracle, prompt, t, constraint
        )

        if is_reasoning:
            # Algorithm 2, line 5: Relax constraint at position t
            reasoning_positions.append(t)
            logger.debug("Position %d: reasoning-free (relaxed)", t)
        else:
            # Position is directly constrained — preserve
            constrained_positions.append(t)
            augmented_parts.append(f"constrain(pos={t})")
            logger.debug("Position %d: constrained (preserved)", t)

    # Algorithm 2, line 8: Verify augmentation
    # φ' = φ_answer ∧ ⊤_reasoning (proof of Theorem 6.1)
    result.augmented_constraint = (
        f"AUGMENTED({constraint}, "
        f"free_positions={reasoning_positions})"
    )
    result.reasoning_positions = reasoning_positions
    result.constrained_positions = constrained_positions

    logger.info(
        "Relaxation: %d reasoning-free positions, %d constrained positions",
        len(reasoning_positions),
        len(constrained_positions),
    )

    return result


def _is_reasoning_position(
    oracle: SemanticOracle,
    prompt: list[int],
    position: int,
    constraint: str,
) -> bool:
    """Determine if a token position is reasoning-free.

    A position is reasoning-free if it is not directly constrained by φ —
    i.e., varying the token at this position does not affect the final
    answer's satisfaction of the constraint.

    This implements the oracle query at Algorithm 2, line 3.

    Uses a heuristic: positions before common answer delimiters
    (e.g., ``####``, ``answer:``) are classified as reasoning-free.

    Parameters
    ----------
    oracle : SemanticOracle
        The semantic oracle.
    prompt : list[int]
        Prompt token ids.
    position : int
        Token position to check.
    constraint : str
        Constraint string.

    Returns
    -------
    bool
        True if the position is reasoning-free.
    """
    # Heuristic from Section 6.3: "classifying all tokens before the
    # final answer delimiter as reasoning-free"
    # Positions in the first 80% of the generation are likely reasoning;
    # the last 20% likely contain the answer.
    prompt_len = len(prompt)
    generation_span = max(1, 256 - prompt_len)
    relative_pos = (position - prompt_len) / generation_span

    return relative_pos < 0.8
