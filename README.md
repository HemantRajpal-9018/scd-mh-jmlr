# Semantically Constrained Decoding (SCD-MH)

**Paper:** *Semantically Constrained Decoding: A Formal Theory of Distribution-Aligned Neurosymbolic Generation*
**Target:** Journal of Machine Learning Research (JMLR)

## Repository Structure

```
├── jmlr_draft/
│   └── main.tex              # Full JMLR paper draft (1,629 lines)
│                               5 theorems, 2 algorithms, 4 benchmarks
│                               29 bibliography entries
│
├── scd_mh/                    # Core Python library (3,570 lines)
│   ├── scd_mh/
│   │   ├── __init__.py        # Package init, v0.1.0
│   │   ├── oracles.py         # Z3Oracle, PrologOracle, TypeCheckOracle
│   │   ├── decoding.py        # NSF, SCD-MH (Alg 1), Solver-Guided Relaxation (Alg 2)
│   │   ├── models.py          # HuggingFace model wrapper (Llama-3-8B, Mistral-7B)
│   │   ├── metrics.py         # KL divergence, TV distance, mixing time
│   │   ├── benchmarks.py      # FOLIO, GSM-Symbolic, ProofWriter, HumanEval-typed
│   │   └── utils.py           # SFS computation, acceptance ratio, logging
│   ├── setup.py
│   └── requirements.txt
│
└── colab/
    └── SCD_MH_Experiments.ipynb  # Complete experiment notebook (47 cells)
                                    Sections 0-8 covering all paper experiments
                                    Designed for Google Colab + single A100 GPU
```

## Next Steps

1. **Validate Theorem 1** counterexample with Z3 on FOLIO
2. **Formalize proof sketches** (Theorem 1 → Theorem 3)
3. **Run experiments** on Google Colab (hrajpal.ai@gmail.com)
4. **Replace X.XX placeholders** with real experimental data
5. **Complete appendix proofs** (Theorems 1 & 3)
6. **Add author/affiliation** for camera-ready version

## Models

- Llama-3-8B-Instruct (4-bit quantized)
- Mistral-7B-Instruct (4-bit quantized)

## Benchmarks

| Benchmark | Domain | Oracle | Test Size |
|-----------|--------|--------|-----------|
| FOLIO | Logical reasoning | Z3 (FOL) | 204 |
| GSM-Symbolic | Arithmetic | Z3 (SMT) | 500 |
| ProofWriter | Multi-step deduction | Prolog | 600 |
| HumanEval-typed | Code generation | Type checker | 164 |
