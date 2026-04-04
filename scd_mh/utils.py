"""Utility functions for the SCD-MH library.

Provides logging, seed management, device management, and core computational
primitives referenced throughout the paper:

- ``compute_sfs``: Semantic Future Satisfiability (Definition 4.3, Eq. 7)
- ``compute_acceptance_ratio``: Metropolis-Hastings acceptance ratio (Eq. 12)

References
----------
- Section 4.2, Definition 4.3 (Semantic Future Satisfiability)
- Section 5.1, Remark 5.1 (Acceptance ratio simplification)
- Algorithm 1 (SCD-MH), line 6
"""

from __future__ import annotations

import logging
import os
import random
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

if TYPE_CHECKING:
    from scd_mh.models import AutoregressiveModel
    from scd_mh.oracles import SemanticOracle

logger = logging.getLogger("scd_mh")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(level: int = logging.INFO, log_file: Optional[str] = None) -> None:
    """Configure library-wide logging.

    Parameters
    ----------
    level : int
        Logging level (e.g. ``logging.INFO``, ``logging.DEBUG``).
    log_file : str, optional
        If provided, also write logs to this file path.
    """
    fmt = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger("scd_mh")
    root.setLevel(level)

    # Clear existing handlers to avoid duplicates
    root.handlers.clear()

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    if log_file is not None:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        root.addHandler(fh)


# ---------------------------------------------------------------------------
# Seed management
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across Python, NumPy, and PyTorch.

    Parameters
    ----------
    seed : int
        The random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic algorithms may be slower but ensure reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("Random seed set to %d", seed)


# ---------------------------------------------------------------------------
# Device management
# ---------------------------------------------------------------------------

def get_device(prefer_gpu: bool = True) -> torch.device:
    """Return the best available torch device.

    Designed for Colab with a single A100 GPU.

    Parameters
    ----------
    prefer_gpu : bool
        If True and a CUDA device is available, use it.

    Returns
    -------
    torch.device
    """
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA device: %s", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    return device


# ---------------------------------------------------------------------------
# Semantic Future Satisfiability  (Definition 4.3 / Eq. 7)
# ---------------------------------------------------------------------------

def compute_sfs(
    model: "AutoregressiveModel",
    oracle: "SemanticOracle",
    prefix: list[int],
    n_samples: int = 256,
    max_continuation_length: int = 128,
) -> float:
    """Estimate the Semantic Future Satisfiability (SFS) of a prefix.

    SFS_φ(x_{1:t}) = P(∃ y ∈ Σ* : x_{1:t} · y ∈ C_φ | x_{1:t})

    This is the probability, under the unconstrained model, that the given
    prefix can be completed to a sequence satisfying the constraint φ
    (Definition 4.3, Eq. 7 in the paper).

    The SFS generalises the Expected Future Grammaticality (EFG) concept
    from Ugare et al. (2024) — see Eq. 5 — from syntactic to semantic
    constraints.

    We estimate SFS via Monte-Carlo: generate ``n_samples`` unconstrained
    completions from the model conditioned on the prefix, then compute the
    fraction that satisfy the oracle.

    Parameters
    ----------
    model : AutoregressiveModel
        The language model providing P.
    oracle : SemanticOracle
        The semantic constraint oracle O_φ.
    prefix : list[int]
        Token-id prefix x_{1:t}.
    n_samples : int
        Number of Monte-Carlo completion samples (default 256).
    max_continuation_length : int
        Maximum number of tokens to generate per completion.

    Returns
    -------
    float
        Estimated SFS ∈ [0, 1].

    Notes
    -----
    - Section 4.2, Definition 4.3 (Semantic Future Satisfiability)
    - Eq. 7: SFS_φ(x_{1:t}) = P(∃ y : x_{1:t} · y ∈ C_φ | x_{1:t})
    - Generalises EFG (Eq. 5) from syntactic to semantic constraints.
    """
    from scd_mh.oracles import OracleResult

    sat_count = 0
    for _ in range(n_samples):
        try:
            completion = model.generate(
                prefix,
                max_length=len(prefix) + max_continuation_length,
            )
            result = oracle.verify(completion)
            if result == OracleResult.SAT:
                sat_count += 1
        except Exception:
            # Oracle timeout or generation failure — treat as UNKNOWN / not SAT
            logger.debug("SFS sample failed for prefix of length %d", len(prefix))
            continue

    sfs = sat_count / max(n_samples, 1)
    logger.debug("SFS(prefix len=%d) ≈ %.4f  (%d/%d)", len(prefix), sfs, sat_count, n_samples)
    return sfs


# ---------------------------------------------------------------------------
# MH Acceptance Ratio  (Eq. 12 / Algorithm 1, line 6)
# ---------------------------------------------------------------------------

def compute_acceptance_ratio(
    model: "AutoregressiveModel",
    proposal: list[int],
    current: list[int],
    q_proposal: float,
    q_current: float,
) -> float:
    """Compute the Metropolis-Hastings acceptance ratio α(x, x').

    From Algorithm 1 (SCD-MH), line 6 / Eq. 12:

        α(x^{(i-1)}, x') = min(1,  P(x') · Q_φ(x^{(i-1)})  /
                                     P(x^{(i-1)}) · Q_φ(x')    )

    All quantities are computed in log-space for numerical stability.

    As noted in Remark 5.1, this is equivalent to the product of SFS ratios
    (Eq. 12 second form), but in practice we compute P(x) and Q_φ(x)
    directly from model logits so explicit SFS evaluation is not needed.

    Parameters
    ----------
    model : AutoregressiveModel
        The language model providing log P.
    proposal : list[int]
        Token-id sequence x' (the proposed new sequence).
    current : list[int]
        Token-id sequence x^{(i-1)} (the current chain state).
    q_proposal : float
        Log-probability of the proposal under Q_φ: log Q_φ(x').
    q_current : float
        Log-probability of the current state under Q_φ: log Q_φ(x^{(i-1)}).

    Returns
    -------
    float
        Acceptance probability α ∈ [0, 1].

    Notes
    -----
    - Algorithm 1, line 6:
        α = min(1, P(x')·Q_φ(x) / [P(x)·Q_φ(x')] )
    - Eq. 12: equivalent SFS-ratio form (used in theory but not computation).
    - Remark 5.1: P(x) and Q_φ(x) can be computed exactly from logits.
    """
    # Log-probabilities under the unconstrained model P
    log_p_proposal = model.log_prob(proposal)
    log_p_current = model.log_prob(current)

    # Log acceptance ratio (Eq. 12):
    #   log α = log P(x') + log Q_φ(x) - log P(x) - log Q_φ(x')
    log_alpha = (log_p_proposal + q_current) - (log_p_current + q_proposal)

    # α = min(1, exp(log_alpha))
    alpha = min(1.0, float(np.exp(np.clip(log_alpha, -700, 0))))
    return alpha


# ---------------------------------------------------------------------------
# Miscellaneous helpers
# ---------------------------------------------------------------------------

def log_sum_exp(values: list[float]) -> float:
    """Numerically stable log-sum-exp.

    Parameters
    ----------
    values : list[float]
        Log-space values.

    Returns
    -------
    float
        log(sum(exp(v) for v in values))
    """
    if not values:
        return float("-inf")
    max_v = max(values)
    if max_v == float("-inf"):
        return float("-inf")
    return max_v + float(np.log(sum(np.exp(v - max_v) for v in values)))


def tokens_to_text(tokenizer, token_ids: list[int]) -> str:
    """Decode token ids to a string using the provided tokenizer.

    Parameters
    ----------
    tokenizer : transformers.PreTrainedTokenizer
        HuggingFace tokenizer.
    token_ids : list[int]
        Sequence of token ids.

    Returns
    -------
    str
        Decoded text.
    """
    return tokenizer.decode(token_ids, skip_special_tokens=False)


def text_to_tokens(tokenizer, text: str) -> list[int]:
    """Encode a string to token ids using the provided tokenizer.

    Parameters
    ----------
    tokenizer : transformers.PreTrainedTokenizer
        HuggingFace tokenizer.
    text : str
        Input text.

    Returns
    -------
    list[int]
        Token id sequence.
    """
    return tokenizer.encode(text, add_special_tokens=False)
