"""SCD-MH: Semantically Constrained Decoding via Metropolis-Hastings.

A production-quality Python implementation of the algorithms and theory from:

    "Semantically Constrained Decoding: A Formal Theory of
     Distribution-Aligned Neurosymbolic Generation"
    (JMLR 2026)

This library provides:

- **Semantic Constraint Oracles** (``oracles``): Z3-based SMT oracles for
  arithmetic and FOL constraints, Prolog-based oracles for relational
  constraints, and type-checking oracles for code generation.

- **Core Decoding Algorithms** (``decoding``):
    - Naive Semantic Filtering (NSF, Eq. 8)
    - SCD-MH sampler (Algorithm 1) with provable convergence to the true
      conditional distribution P*_φ (Theorem 5.3)
    - Solver-Guided Constraint Relaxation (Algorithm 2) for reasoning
      preservation (Theorem 6.1)

- **Evaluation Metrics** (``metrics``): KL divergence estimation, total
  variation distance, mixing time measurement, and reasoning chain
  evaluation.

- **Benchmark Loaders** (``benchmarks``): FOLIO, GSM-Symbolic, ProofWriter,
  and HumanEval-typed dataset loaders.

- **Model Wrappers** (``models``): HuggingFace Transformers integration for
  Llama-3-8B, Mistral-7B, and other autoregressive models.

Quick Start
-----------
>>> from scd_mh import scd_mh_sample, Z3Oracle, HuggingFaceModel
>>> model = HuggingFaceModel("meta-llama/Meta-Llama-3-8B-Instruct")
>>> oracle = Z3Oracle(constraint_formula="answer == 42", domain="arithmetic")
>>> result = scd_mh_sample(model, oracle, prompt=model.encode("Solve: ..."), T=50, B=10)
>>> print(model.decode(result.final_sequence))
"""

__version__ = "0.1.0"
__author__ = "[Author Name]"

# ---------------------------------------------------------------------------
# Top-level imports for convenience
# ---------------------------------------------------------------------------

# Oracles
from scd_mh.oracles import (
    OracleResult,
    PrefixResult,
    SemanticOracle,
    Z3Oracle,
    PrologOracle,
    TypeCheckOracle,
)

# Core decoding algorithms
from scd_mh.decoding import (
    naive_semantic_filter,
    scd_mh_sample,
    solver_guided_relaxation,
    NSFResult,
    SCDMHResult,
    RelaxationResult,
)

# Metrics
from scd_mh.metrics import (
    estimate_kl_divergence,
    compute_tv_distance,
    measure_mixing_time,
    evaluate_reasoning_chain,
)

# Model wrappers
from scd_mh.models import (
    AutoregressiveModel,
    HuggingFaceModel,
    AirLLMModel,
    TurboQuantModel,
    load_model,
)

# Benchmarks
from scd_mh.benchmarks import (
    load_folio,
    load_gsm_symbolic,
    load_proofwriter,
    load_humaneval_typed,
)

# Utilities
from scd_mh.utils import (
    setup_logging,
    set_seed,
    get_device,
    compute_sfs,
    compute_acceptance_ratio,
)

__all__ = [
    # Version
    "__version__",
    # Oracles
    "OracleResult",
    "PrefixResult",
    "SemanticOracle",
    "Z3Oracle",
    "PrologOracle",
    "TypeCheckOracle",
    # Decoding
    "naive_semantic_filter",
    "scd_mh_sample",
    "solver_guided_relaxation",
    "NSFResult",
    "SCDMHResult",
    "RelaxationResult",
    # Metrics
    "estimate_kl_divergence",
    "compute_tv_distance",
    "measure_mixing_time",
    "evaluate_reasoning_chain",
    # Models
    "AutoregressiveModel",
    "HuggingFaceModel",
    "AirLLMModel",
    "TurboQuantModel",
    "load_model",
    # Benchmarks
    "load_folio",
    "load_gsm_symbolic",
    "load_proofwriter",
    "load_humaneval_typed",
    # Utilities
    "setup_logging",
    "set_seed",
    "get_device",
    "compute_sfs",
    "compute_acceptance_ratio",
]
