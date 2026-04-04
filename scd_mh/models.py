"""Model wrappers for autoregressive language models.

Provides the ``AutoregressiveModel`` abstraction used throughout the SCD-MH
framework, with three concrete backends:

1. **HuggingFaceModel** — Standard HuggingFace Transformers + optional
   bitsandbytes 4-bit quantization.  The original backend.
2. **AirLLMModel** — Layer-wise inference via ``airllm``.  Loads one
   transformer layer at a time so that even 70B-parameter models fit in
   <4 GB VRAM (ideal for Colab free-tier T4).
3. **TurboQuantModel** — INT4 + AWQ-style activation-aware weight
   quantization via ``TurboQuant-v3``.  Best quality-preserving INT4 on
   A100 / T4 GPUs (<0.001 MSE vs FP16).

A convenience factory ``load_model`` selects the best available backend
automatically (or lets the caller choose explicitly).

The key methods — ``log_prob``, ``conditional_log_prob``, ``generate``, and
``get_next_token_logits`` — directly support the quantities needed for:

- The autoregressive factorisation (Eq. 1):
      P(x_{1:n}) = ∏_{t=1}^{n} P(x_t | x_{1:t-1})
- The MH acceptance ratio (Algorithm 1, line 6 / Eq. 12):
      α(x, x') = min(1, P(x')·Q_φ(x) / P(x)·Q_φ(x'))
- Proposal generation via naive semantic filtering (Eq. 8).

Models evaluated in the paper (Section 7.1): Llama-3-8B and Mistral-7B
(instruction-tuned variants), designed to run on a single A100 GPU.

References
----------
- Section 2.1 (Language Models and Autoregressive Generation, Eq. 1)
- Section 5.1, Remark 5.1 (acceptance ratio from logits)
- Section 7.1 (Models: Llama-3-8B, Mistral-7B)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
import torch

logger = logging.getLogger("scd_mh.models")

# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class AutoregressiveModel(ABC):
    """Abstract base class for autoregressive language models.

    Encapsulates the probability distribution P over Σ* via the chain-rule
    factorisation (Eq. 1):

        P(x_{1:n}) = ∏_{t=1}^n P(x_t | x_{1:t-1})

    Subclasses must implement ``log_prob``, ``conditional_log_prob``,
    ``generate``, ``get_next_token_logits``, ``get_vocab_size``, and
    ``get_eos_token_id``.
    """

    @abstractmethod
    def log_prob(self, sequence: list[int]) -> float:
        """Compute the full-sequence log probability log P(x_{1:n}).

        Uses the autoregressive factorisation (Eq. 1):
            log P(x_{1:n}) = Σ_{t=1}^n log P(x_t | x_{1:t-1})

        Parameters
        ----------
        sequence : list[int]
            Token-id sequence x_{1:n}.

        Returns
        -------
        float
            Log-probability of the sequence under P.

        Notes
        -----
        - Eq. 1: P(x_{1:n}) = ∏_t P(x_t | x_{1:t-1})
        - Required for the MH acceptance ratio (Algorithm 1, line 6).
        """
        ...

    @abstractmethod
    def conditional_log_prob(self, token: int, prefix: list[int]) -> float:
        """Compute the conditional next-token log probability.

        log P(x_t = token | x_{1:t-1} = prefix)

        Parameters
        ----------
        token : int
            The token id x_t.
        prefix : list[int]
            The prefix x_{1:t-1}.

        Returns
        -------
        float
            log P(token | prefix).

        Notes
        -----
        - Used in naive semantic filtering (Eq. 8) for per-step masking.
        """
        ...

    @abstractmethod
    def generate(
        self,
        prompt: list[int],
        max_length: int = 256,
        temperature: float = 1.0,
    ) -> list[int]:
        """Generate a sequence autoregressively from P.

        Samples from the unconstrained model distribution starting from the
        given prompt prefix.

        Parameters
        ----------
        prompt : list[int]
            Prompt token ids.
        max_length : int
            Maximum total sequence length (prompt + generation).
        temperature : float
            Sampling temperature (default 1.0).

        Returns
        -------
        list[int]
            Generated token-id sequence (including prompt).
        """
        ...

    @abstractmethod
    def get_next_token_logits(self, prefix: list[int]) -> torch.Tensor:
        """Get the raw logits for the next token given a prefix.

        Parameters
        ----------
        prefix : list[int]
            The prefix x_{1:t-1}.

        Returns
        -------
        torch.Tensor
            Logit vector of shape (vocab_size,).
        """
        ...

    @abstractmethod
    def get_vocab_size(self) -> int:
        """Return the vocabulary size |Σ|.

        Returns
        -------
        int
        """
        ...

    @abstractmethod
    def get_eos_token_id(self) -> int:
        """Return the EOS token id.

        Returns
        -------
        int
        """
        ...

    # ------------------------------------------------------------------
    # Convenience helpers shared by all backends
    # ------------------------------------------------------------------

    def get_next_token_log_probs(self, prefix: list[int]) -> torch.Tensor:
        """Get log-probability distribution over the next token.

        Convenience method returning log-softmax of logits.

        Parameters
        ----------
        prefix : list[int]
            Prefix x_{1:t-1}.

        Returns
        -------
        torch.Tensor
            Log-probability vector of shape (vocab_size,).
        """
        logits = self.get_next_token_logits(prefix)
        return torch.nn.functional.log_softmax(logits, dim=-1)

    def encode(self, text: str) -> list[int]:
        """Encode text to token ids.

        Parameters
        ----------
        text : str

        Returns
        -------
        list[int]
        """
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, token_ids: list[int]) -> str:
        """Decode token ids to text.

        Parameters
        ----------
        token_ids : list[int]

        Returns
        -------
        str
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)

    @property
    def tokenizer(self):  # noqa: D401
        """Access the underlying tokenizer."""
        return self._tokenizer


# ---------------------------------------------------------------------------
# Backend 1 – HuggingFace Transformers (original, bitsandbytes 4-bit)
# ---------------------------------------------------------------------------


class HuggingFaceModel(AutoregressiveModel):
    """Wrapper for HuggingFace Transformers autoregressive models.

    Supports models evaluated in the paper (Section 7.1):
    - Llama-3-8B (``meta-llama/Meta-Llama-3-8B-Instruct``)
    - Mistral-7B (``mistralai/Mistral-7B-Instruct-v0.3``)

    Designed for single A100 GPU execution on Google Colab.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.
    device : str | torch.device | None
        Device to load the model on.  If *None*, auto-selects GPU when
        available.
    dtype : torch.dtype
        Model precision (default ``torch.float16`` for A100 efficiency).
    max_batch_size : int
        Maximum batch size for batched operations (default 8).

    Examples
    --------
    >>> model = HuggingFaceModel("meta-llama/Meta-Llama-3-8B-Instruct")
    >>> log_p = model.log_prob([1, 234, 567, 2])  # log P(sequence)
    >>> logits = model.get_next_token_logits([1, 234])  # next-token logits

    Notes
    -----
    - Section 7.1: "Both models are used in their instruction-tuned variants."
    - Handles tokenisation, logit computation, and probability extraction
      needed for the MH acceptance ratio (Algorithm 1, line 6).
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str | torch.device] = None,
        dtype: torch.dtype = torch.float16,
        max_batch_size: int = 8,
    ) -> None:
        self.model_name = model_name
        self.dtype = dtype
        self.max_batch_size = max_batch_size

        # Resolve device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._model = None
        self._tokenizer = None
        self._load_model()

    def _load_model(self) -> None:
        """Load model and tokenizer from HuggingFace."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info("Loading model: %s", self.model_name)

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
                device_map="auto" if self.device.type == "cuda" else None,
                trust_remote_code=True,
            )
            if self.device.type != "cuda":
                self._model = self._model.to(self.device)

            self._model.eval()
            logger.info(
                "Model loaded: %s (device=%s, dtype=%s)",
                self.model_name,
                self.device,
                self.dtype,
            )

        except ImportError:
            raise ImportError(
                "transformers library required. Install with: pip install transformers"
            )
        except Exception as e:
            logger.error("Failed to load model %s: %s", self.model_name, e)
            raise

    # ---- AutoregressiveModel interface ------------------------------------

    def log_prob(self, sequence: list[int]) -> float:
        """Compute log P(x_{1:n}) via the autoregressive factorisation.

        log P(x_{1:n}) = Σ_{t=1}^{n} log P(x_t | x_{1:t-1})

        This is the key quantity for the MH acceptance ratio
        (Algorithm 1, line 6 / Eq. 12).

        Parameters
        ----------
        sequence : list[int]
            Token-id sequence x_{1:n}.

        Returns
        -------
        float
            log P(x_{1:n}).

        Notes
        -----
        - Eq. 1: autoregressive factorisation.
        - Remark 5.1: "P(x) … can be computed exactly from the model's
          logits."
        """
        if len(sequence) <= 1:
            return 0.0

        input_ids = torch.tensor([sequence], dtype=torch.long, device=self.device)

        with torch.no_grad():
            outputs = self._model(input_ids)
            # logits shape: (1, seq_len, vocab_size)
            logits = outputs.logits[0]  # (seq_len, vocab_size)

        # Compute log P(x_t | x_{1:t-1}) for t = 1, ..., n
        # logits[t-1] gives the distribution over x_t given x_{1:t-1}
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        total_log_prob = 0.0
        for t in range(1, len(sequence)):
            # log P(x_t | x_{1:t-1}) = log_probs[t-1, x_t]
            total_log_prob += log_probs[t - 1, sequence[t]].item()

        return total_log_prob

    def conditional_log_prob(self, token: int, prefix: list[int]) -> float:
        """Compute log P(x_t = token | x_{1:t-1} = prefix).

        Parameters
        ----------
        token : int
            Target token id x_t.
        prefix : list[int]
            Prefix x_{1:t-1}.

        Returns
        -------
        float
            log P(token | prefix).

        Notes
        -----
        - Used in naive semantic filtering (Eq. 8) for the per-step
          masked distribution Q_φ(x_t | x_{1:t-1}).
        """
        logits = self.get_next_token_logits(prefix)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        return log_probs[token].item()

    def generate(
        self,
        prompt: list[int],
        max_length: int = 256,
        temperature: float = 1.0,
    ) -> list[int]:
        """Generate a sequence from P via ancestral sampling.

        Parameters
        ----------
        prompt : list[int]
            Prompt token ids.
        max_length : int
            Maximum total length (prompt + generation).
        temperature : float
            Sampling temperature. 1.0 = standard, <1.0 = sharper,
            >1.0 = flatter.

        Returns
        -------
        list[int]
            Full generated sequence (prompt + continuation).
        """
        input_ids = torch.tensor([prompt], dtype=torch.long, device=self.device)
        eos_id = self.get_eos_token_id()

        with torch.no_grad():
            output = self._model.generate(
                input_ids,
                max_length=max_length,
                do_sample=True,
                temperature=max(temperature, 1e-8),
                top_k=0,  # No top-k filtering — sample from full distribution
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=eos_id,
            )

        return output[0].tolist()

    def get_next_token_logits(self, prefix: list[int]) -> torch.Tensor:
        """Get raw logits for the next token.

        Parameters
        ----------
        prefix : list[int]
            Prefix x_{1:t-1}.

        Returns
        -------
        torch.Tensor
            Logit vector of shape (vocab_size,).
        """
        if not prefix:
            # Empty prefix: use BOS token if available
            prefix = [self._tokenizer.bos_token_id or 0]

        input_ids = torch.tensor([prefix], dtype=torch.long, device=self.device)

        with torch.no_grad():
            outputs = self._model(input_ids)
            # Last position logits → distribution over next token
            logits = outputs.logits[0, -1, :]  # (vocab_size,)

        return logits

    def get_vocab_size(self) -> int:
        """Return vocabulary size |Σ|."""
        return self._model.config.vocab_size

    def get_eos_token_id(self) -> int:
        """Return the EOS token id."""
        eos = self._tokenizer.eos_token_id
        if eos is None:
            eos = self._model.config.eos_token_id
        if eos is None:
            raise ValueError("Model has no EOS token defined")
        return eos


# ---------------------------------------------------------------------------
# Backend 2 – AirLLM (layer-wise inference, <4 GB VRAM)
# ---------------------------------------------------------------------------


class AirLLMModel(AutoregressiveModel):
    """Wrapper using ``airllm`` for layer-wise inference.

    AirLLM loads one transformer layer at a time, enabling 70B-parameter
    models to run on GPUs with as little as 4 GB VRAM.  This makes it
    ideal for Google Colab free-tier T4 instances.

    Because AirLLM is optimised for *generation* rather than arbitrary
    forward passes, computing log-probabilities requires a generate call
    with ``return_dict_in_generate=True`` and ``output_scores=True`` so
    that per-step logits are returned alongside the generated tokens.

    For ``log_prob`` on *existing* sequences (needed by the MH acceptance
    ratio), we feed the sequence as a forced prefix and collect the
    teacher-forced logits one token at a time.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier (e.g.
        ``"meta-llama/Meta-Llama-3-8B-Instruct"``).
    compression : str
        Quantization mode passed to ``airllm.AutoModel.from_pretrained``.
        Typical value: ``'4bit'`` for 4-bit block quantisation
        (bitsandbytes under the hood).
    max_length : int
        Default maximum sequence length for tokenisation (default 512).

    Examples
    --------
    >>> model = AirLLMModel("meta-llama/Meta-Llama-3-8B-Instruct")
    >>> logits = model.get_next_token_logits([1, 234])
    """

    def __init__(
        self,
        model_name: str,
        compression: str = "4bit",
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.compression = compression
        self.max_length = max_length

        self._model = None
        self._tokenizer = None
        self._vocab_size: Optional[int] = None
        self._load_model()

    def _load_model(self) -> None:
        """Load model via airllm."""
        try:
            from airllm import AutoModel as AirAutoModel
        except ImportError:
            raise ImportError(
                "airllm library required. Install with: pip install airllm"
            )

        logger.info(
            "Loading model via AirLLM: %s (compression=%s)",
            self.model_name,
            self.compression,
        )
        try:
            self._model = AirAutoModel.from_pretrained(
                self.model_name,
                compression=self.compression,
            )
            # airllm exposes the tokenizer on the model object
            self._tokenizer = self._model.tokenizer
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            # Cache vocab size from tokenizer
            self._vocab_size = len(self._tokenizer)
            logger.info(
                "AirLLM model loaded: %s (vocab_size=%d)",
                self.model_name,
                self._vocab_size,
            )
        except Exception as e:
            logger.error(
                "Failed to load model %s via AirLLM: %s", self.model_name, e
            )
            raise

    # ---- Internal helpers -------------------------------------------------

    def _generate_with_scores(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
    ) -> Any:
        """Run AirLLM generation returning per-step scores.

        Parameters
        ----------
        input_ids : torch.Tensor
            Input token ids of shape ``(1, seq_len)`` (on CPU; AirLLM
            handles device placement internally).
        max_new_tokens : int
            Maximum new tokens to generate.

        Returns
        -------
        GenerateOutput
            A ``transformers``-style output with ``.sequences`` and
            ``.scores`` attributes.
        """
        output = self._model.generate(
            input_ids.cuda(),
            max_new_tokens=max_new_tokens,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
        )
        return output

    def _get_logits_for_sequence(self, sequence: list[int]) -> torch.Tensor:
        """Obtain logits at every position in *sequence* using teacher forcing.

        AirLLM does not expose a standard ``forward`` pass, so we
        approximate teacher-forced logits by running a single-step
        generation from every prefix of the sequence and collecting the
        logits.

        For efficiency we attempt a single generation call with
        ``max_new_tokens=1`` on the full sequence (which yields
        next-token logits for the last position only).  For
        full-sequence scoring we iterate over prefixes.

        Parameters
        ----------
        sequence : list[int]
            The full token-id sequence.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(seq_len - 1, vocab_size)`` where row *i*
            contains logits for predicting ``sequence[i + 1]`` given
            ``sequence[:i + 1]``.
        """
        all_logits: list[torch.Tensor] = []
        for t in range(len(sequence) - 1):
            prefix = sequence[: t + 1]
            input_ids = torch.tensor([prefix], dtype=torch.long)
            output = self._generate_with_scores(input_ids, max_new_tokens=1)
            # output.scores is a tuple of tensors, one per generated step.
            # We need the logits from the first (only) generation step.
            if hasattr(output, "scores") and output.scores:
                step_logits = output.scores[0][0]  # (vocab_size,)
            else:
                # Fallback: derive from output tokens (less precise)
                raise RuntimeError(
                    "AirLLM did not return scores; cannot compute log-probs. "
                    "Ensure you have a recent version of airllm that supports "
                    "output_scores=True."
                )
            all_logits.append(step_logits.cpu())

        return torch.stack(all_logits, dim=0)  # (seq_len-1, vocab_size)

    # ---- AutoregressiveModel interface ------------------------------------

    def log_prob(self, sequence: list[int]) -> float:
        """Compute log P(x_{1:n}) by iterating over prefixes.

        Because AirLLM is inference-only (no standard forward pass),
        we obtain per-position logits via single-step generation from
        each prefix and sum the log-probabilities.

        Parameters
        ----------
        sequence : list[int]
            Token-id sequence x_{1:n}.

        Returns
        -------
        float
            log P(x_{1:n}).

        Notes
        -----
        This is O(n) generation calls for a sequence of length n.
        Acceptable for moderate-length sequences used in SCD-MH
        (typically <256 tokens).
        """
        if len(sequence) <= 1:
            return 0.0

        logits = self._get_logits_for_sequence(sequence)  # (n-1, V)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        total = 0.0
        for t in range(1, len(sequence)):
            total += log_probs[t - 1, sequence[t]].item()
        return total

    def conditional_log_prob(self, token: int, prefix: list[int]) -> float:
        """Compute log P(token | prefix) via a single-step generation.

        Parameters
        ----------
        token : int
            Target token id.
        prefix : list[int]
            Prefix token ids.

        Returns
        -------
        float
            log P(token | prefix).
        """
        logits = self.get_next_token_logits(prefix)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        return log_probs[token].item()

    def generate(
        self,
        prompt: list[int],
        max_length: int = 256,
        temperature: float = 1.0,
    ) -> list[int]:
        """Generate a continuation using AirLLM.

        Parameters
        ----------
        prompt : list[int]
            Prompt token ids.
        max_length : int
            Maximum total sequence length.
        temperature : float
            Sampling temperature (applied externally if AirLLM does not
            support it natively; the default AirLLM generate uses
            greedy / sampling internally).

        Returns
        -------
        list[int]
            Full generated sequence (prompt + continuation).
        """
        input_ids = torch.tensor([prompt], dtype=torch.long)
        max_new = max(max_length - len(prompt), 1)

        output = self._model.generate(
            input_ids.cuda(),
            max_new_tokens=max_new,
            use_cache=True,
            return_dict_in_generate=True,
        )

        return output.sequences[0].tolist()

    def get_next_token_logits(self, prefix: list[int]) -> torch.Tensor:
        """Get next-token logits via a single-step AirLLM generation.

        Parameters
        ----------
        prefix : list[int]
            Prefix x_{1:t-1}.

        Returns
        -------
        torch.Tensor
            Logit vector of shape (vocab_size,) on CPU.
        """
        if not prefix:
            bos = self._tokenizer.bos_token_id
            prefix = [bos if bos is not None else 0]

        input_ids = torch.tensor([prefix], dtype=torch.long)
        output = self._generate_with_scores(input_ids, max_new_tokens=1)

        if hasattr(output, "scores") and output.scores:
            return output.scores[0][0].cpu()  # (vocab_size,)

        raise RuntimeError(
            "AirLLM did not return scores.  Upgrade airllm or use a "
            "different backend for logit-level access."
        )

    def get_vocab_size(self) -> int:
        """Return vocabulary size |Σ|."""
        if self._vocab_size is not None:
            return self._vocab_size
        return len(self._tokenizer)

    def get_eos_token_id(self) -> int:
        """Return the EOS token id."""
        eos = self._tokenizer.eos_token_id
        if eos is None:
            raise ValueError("Tokenizer has no EOS token defined")
        return eos


# ---------------------------------------------------------------------------
# Backend 3 – TurboQuant-v3 (INT4 + AWQ)
# ---------------------------------------------------------------------------


class TurboQuantModel(AutoregressiveModel):
    """Wrapper using TurboQuant-v3 for INT4+AWQ weight quantization.

    TurboQuant-v3 replaces every ``nn.Linear`` with a ``QuantizedLinear``
    layer using INT4 + activation-aware scaling + protected FP16 outlier
    channels.  After quantization the model is a standard HuggingFace
    ``PreTrainedModel`` and supports normal forward passes, so
    ``log_prob`` works identically to ``HuggingFaceModel``.

    The wrapper can operate in two modes:

    1. **Quantize on the fly** — provide a HuggingFace model id and an
       optional ``TurboQuantConfig``.  The model will be downloaded,
       quantized, and used directly.
    2. **Load a pre-quantized checkpoint** — provide the path to a
       directory previously saved with ``save_quantized_model``.

    Parameters
    ----------
    model_name : str
        HuggingFace model id *or* local path to a pre-quantized checkpoint.
    quant_config : dict | None
        Keyword arguments forwarded to ``TurboQuantConfig``.  If *None*,
        sensible defaults are used (4-bit, group_size=128, GEMM kernel,
        activation-aware, outlier ratio 0.02, SVD rank 8).
    device : str | torch.device | None
        Target device.  Defaults to CUDA if available.
    dtype : torch.dtype
        Precision for the base model before quantization (default FP16).
    load_quantized : bool
        If *True*, treat ``model_name`` as a path to a pre-quantized
        checkpoint and skip quantization.

    Examples
    --------
    >>> model = TurboQuantModel("meta-llama/Llama-2-7b-hf")
    >>> log_p = model.log_prob([1, 234, 567, 2])

    >>> # Load a previously quantized checkpoint
    >>> model = TurboQuantModel("./quantized-llama/", load_quantized=True)
    """

    # Default quantization parameters (matches paper evaluation)
    _DEFAULT_QUANT_CONFIG: Dict[str, Any] = {
        "bits": 4,
        "group_size": 128,
        "version": "gemm",
        "zero_point": True,
        "activation_aware": True,
        "outlier_keep_ratio": 0.02,
        "rank": 8,
    }

    def __init__(
        self,
        model_name: str,
        quant_config: Optional[Dict[str, Any]] = None,
        device: Optional[str | torch.device] = None,
        dtype: torch.dtype = torch.float16,
        load_quantized: bool = False,
    ) -> None:
        self.model_name = model_name
        self.quant_config_kwargs = quant_config or self._DEFAULT_QUANT_CONFIG.copy()
        self.dtype = dtype
        self.load_quantized = load_quantized

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._model = None
        self._tokenizer = None
        self._load_model()

    def _load_model(self) -> None:
        """Load and (optionally) quantize the model via TurboQuant-v3."""
        try:
            from turboquant import (
                TurboQuantConfig,
                load_quantized_model,
                quantize_model,
            )
        except ImportError:
            raise ImportError(
                "TurboQuant-v3 is required. Install with:\n"
                "  git clone https://github.com/Kubenew/TurboQuant-v3.git\n"
                "  cd TurboQuant-v3 && pip install -e ."
            )

        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self.load_quantized:
            # ---- Load a pre-quantized checkpoint --------------------------
            logger.info(
                "Loading pre-quantized TurboQuant model from: %s", self.model_name
            )
            self._model = load_quantized_model(
                self.model_name, device_map="auto"
            )
            # Tokenizer must be saved alongside the checkpoint
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )
        else:
            # ---- Download FP16 model and quantize on the fly --------------
            logger.info(
                "Loading base model for TurboQuant quantization: %s",
                self.model_name,
            )
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
                trust_remote_code=True,
            )

            logger.info(
                "Quantizing with TurboQuantConfig(%s)", self.quant_config_kwargs
            )
            config = TurboQuantConfig(**self.quant_config_kwargs)
            self._model = quantize_model(base_model, quantization_config=config)

            # Move to device
            if self.device.type == "cuda":
                self._model = self._model.to(self.device)

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model.eval()
        logger.info(
            "TurboQuant model ready: %s (device=%s)", self.model_name, self.device
        )

    # ---- Persistence helpers ----------------------------------------------

    def save(self, output_dir: str) -> None:
        """Save the quantized model and tokenizer to *output_dir*.

        Parameters
        ----------
        output_dir : str
            Destination directory.
        """
        from turboquant import TurboQuantConfig, save_quantized_model

        config = TurboQuantConfig(**self.quant_config_kwargs)
        save_quantized_model(self._model, output_dir, config)
        self._tokenizer.save_pretrained(output_dir)
        logger.info("Quantized model saved to %s", output_dir)

    # ---- AutoregressiveModel interface ------------------------------------

    def log_prob(self, sequence: list[int]) -> float:
        """Compute log P(x_{1:n}) via a standard forward pass.

        After TurboQuant quantization the model behaves like a normal
        HuggingFace model with ``QuantizedLinear`` layers, so a single
        forward pass yields all logits.

        Parameters
        ----------
        sequence : list[int]
            Token-id sequence x_{1:n}.

        Returns
        -------
        float
            log P(x_{1:n}).
        """
        if len(sequence) <= 1:
            return 0.0

        input_ids = torch.tensor([sequence], dtype=torch.long, device=self.device)

        with torch.no_grad():
            outputs = self._model(input_ids)
            logits = outputs.logits[0]  # (seq_len, vocab_size)

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        total_log_prob = 0.0
        for t in range(1, len(sequence)):
            total_log_prob += log_probs[t - 1, sequence[t]].item()

        return total_log_prob

    def conditional_log_prob(self, token: int, prefix: list[int]) -> float:
        """Compute log P(token | prefix).

        Parameters
        ----------
        token : int
            Target token id.
        prefix : list[int]
            Prefix token ids.

        Returns
        -------
        float
            log P(token | prefix).
        """
        logits = self.get_next_token_logits(prefix)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        return log_probs[token].item()

    def generate(
        self,
        prompt: list[int],
        max_length: int = 256,
        temperature: float = 1.0,
    ) -> list[int]:
        """Generate a sequence from the quantized model.

        Parameters
        ----------
        prompt : list[int]
            Prompt token ids.
        max_length : int
            Maximum total length.
        temperature : float
            Sampling temperature.

        Returns
        -------
        list[int]
            Full generated sequence (prompt + continuation).
        """
        input_ids = torch.tensor([prompt], dtype=torch.long, device=self.device)
        eos_id = self.get_eos_token_id()

        with torch.no_grad():
            output = self._model.generate(
                input_ids,
                max_length=max_length,
                do_sample=True,
                temperature=max(temperature, 1e-8),
                top_k=0,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=eos_id,
            )

        return output[0].tolist()

    def get_next_token_logits(self, prefix: list[int]) -> torch.Tensor:
        """Get next-token logits via a forward pass.

        Parameters
        ----------
        prefix : list[int]
            Prefix x_{1:t-1}.

        Returns
        -------
        torch.Tensor
            Logit vector of shape (vocab_size,).
        """
        if not prefix:
            prefix = [self._tokenizer.bos_token_id or 0]

        input_ids = torch.tensor([prefix], dtype=torch.long, device=self.device)

        with torch.no_grad():
            outputs = self._model(input_ids)
            logits = outputs.logits[0, -1, :]  # (vocab_size,)

        return logits

    def get_vocab_size(self) -> int:
        """Return vocabulary size |Σ|."""
        return self._model.config.vocab_size

    def get_eos_token_id(self) -> int:
        """Return the EOS token id."""
        eos = self._tokenizer.eos_token_id
        if eos is None:
            eos = self._model.config.eos_token_id
        if eos is None:
            raise ValueError("Model has no EOS token defined")
        return eos


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

_BACKEND_ORDER = ("turboquant", "airllm", "hf")
"""Preference order for ``backend='auto'``.  TurboQuant yields the best
quality-preserving quantization; AirLLM allows very large models on
limited VRAM; HuggingFace is the baseline fallback."""


def load_model(
    model_name: str,
    backend: str = "auto",
    **kwargs: Any,
) -> AutoregressiveModel:
    """Factory: load a model using the specified (or best available) backend.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier or local checkpoint path.
    backend : str
        One of ``'auto'``, ``'turboquant'``, ``'airllm'``, or ``'hf'``.

        - ``'auto'`` (default): tries TurboQuant → AirLLM → HuggingFace,
          using the first backend whose dependencies are installed.
        - ``'turboquant'``: INT4+AWQ via TurboQuant-v3 (best quality on
          A100 / T4).
        - ``'airllm'``: layer-wise inference via AirLLM (best for very
          limited VRAM, e.g. Colab free T4).
        - ``'hf'``: standard HuggingFace Transformers (original baseline).
    **kwargs
        Additional keyword arguments forwarded to the backend constructor.

    Returns
    -------
    AutoregressiveModel
        An initialised model instance exposing the SCD-MH interface.

    Raises
    ------
    ValueError
        If *backend* is not recognised.
    RuntimeError
        If ``backend='auto'`` and no backend could be loaded.

    Examples
    --------
    >>> model = load_model("meta-llama/Meta-Llama-3-8B-Instruct")
    >>> model = load_model("meta-llama/Llama-2-7b-hf", backend="turboquant")
    >>> model = load_model("meta-llama/Meta-Llama-3-8B-Instruct", backend="airllm", compression="4bit")
    """
    backend = backend.lower().strip()

    if backend not in ("auto", "turboquant", "airllm", "hf"):
        raise ValueError(
            f"Unknown backend '{backend}'. Choose from: "
            "'auto', 'turboquant', 'airllm', 'hf'."
        )

    # ------------------------------------------------------------------
    # Explicit backend selection
    # ------------------------------------------------------------------
    if backend == "hf":
        logger.info("Loading model with HuggingFace backend.")
        return HuggingFaceModel(model_name, **kwargs)

    if backend == "airllm":
        logger.info("Loading model with AirLLM backend.")
        return AirLLMModel(model_name, **kwargs)

    if backend == "turboquant":
        logger.info("Loading model with TurboQuant backend.")
        return TurboQuantModel(model_name, **kwargs)

    # ------------------------------------------------------------------
    # Auto selection — try backends in preference order
    # ------------------------------------------------------------------
    assert backend == "auto"
    errors: list[str] = []

    for candidate in _BACKEND_ORDER:
        try:
            if candidate == "turboquant":
                logger.info("Auto: attempting TurboQuant backend …")
                return TurboQuantModel(model_name, **kwargs)
            elif candidate == "airllm":
                logger.info("Auto: attempting AirLLM backend …")
                return AirLLMModel(model_name, **kwargs)
            else:
                logger.info("Auto: attempting HuggingFace backend …")
                return HuggingFaceModel(model_name, **kwargs)
        except ImportError as exc:
            msg = f"{candidate}: missing dependency – {exc}"
            logger.warning("Auto: %s", msg)
            errors.append(msg)
        except Exception as exc:  # noqa: BLE001
            msg = f"{candidate}: failed – {exc}"
            logger.warning("Auto: %s", msg)
            errors.append(msg)

    raise RuntimeError(
        "load_model(backend='auto') could not initialise any backend.\n"
        + "\n".join(f"  • {e}" for e in errors)
    )
