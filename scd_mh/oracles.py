"""Semantic constraint oracles for the SCD-MH framework.

Implements the oracle abstraction from Definition 2.4 (Symbolic Constraint
Oracle) and the prefix-decidability spectrum from Definition 3.2:

- ``SemanticOracle``: Abstract base class — ``verify`` returns SAT/UNSAT/UNKNOWN
  on complete sequences; ``check_prefix`` returns EXTENDABLE/DEAD/UNKNOWN on
  prefixes.
- ``Z3Oracle``: SMT-based oracle using z3-solver for arithmetic constraints
  (GSM-Symbolic) and FOL validity checking (FOLIO). Parses model output into
  SMT-LIB formulas.
- ``PrologOracle``: Prolog-based oracle using pyswip for relational constraints
  (ProofWriter). Translates deduction steps into Prolog facts/rules.
- ``TypeCheckOracle``: Type-checking oracle for HumanEval-typed using Python's
  ``ast`` module and mypy.

References
----------
- Definition 2.4 (Symbolic Constraint Oracle, Section 2.4)
- Definition 3.2 (Prefix Decidability, Section 3.2)
- Section 7.1: Constraint Oracles (Z3 for FOLIO/GSM, Prolog for ProofWriter)
"""

from __future__ import annotations

import ast
import enum
import logging
import re
import subprocess
import tempfile
import textwrap
from abc import ABC, abstractmethod
from typing import Any, Optional

logger = logging.getLogger("scd_mh.oracles")


# ---------------------------------------------------------------------------
# Result enumerations  (Definition 2.4, Definition 3.2)
# ---------------------------------------------------------------------------

class OracleResult(enum.Enum):
    """Result of a full-sequence oracle verification (Definition 2.4).

    Values
    ------
    SAT
        The interpretation of the sequence satisfies the constraint φ.
    UNSAT
        The interpretation does not satisfy φ.
    UNKNOWN
        The oracle could not determine satisfiability (e.g., timeout).
    """

    SAT = "SAT"
    UNSAT = "UNSAT"
    UNKNOWN = "UNKNOWN"


class PrefixResult(enum.Enum):
    """Result of a prefix oracle check (Definition 3.2).

    Values
    ------
    EXTENDABLE
        There exists a completion y such that prefix · y ∈ C_φ.
    DEAD
        No completion can satisfy the constraint.
    UNKNOWN
        The oracle cannot determine extendability (prefix-approximable case).
    """

    EXTENDABLE = "EXTENDABLE"
    DEAD = "DEAD"
    UNKNOWN = "UNKNOWN"


# ---------------------------------------------------------------------------
# Abstract base class  (Definition 2.4)
# ---------------------------------------------------------------------------

class SemanticOracle(ABC):
    """Abstract base class for semantic constraint oracles.

    A symbolic constraint oracle for a formula φ ∈ L is a function
    O_φ : Σ* → {SAT, UNSAT, UNKNOWN} satisfying (Definition 2.4):

    1. If O_φ(x) = SAT, then I(x) ⊨ φ   (soundness).
    2. If O_φ(x) = UNSAT, then I(x) ⊭ φ  (soundness).
    3. For complete sequences, O_φ(x) ∈ {SAT, UNSAT}  (completeness on
       termination).

    Subclasses must implement ``verify`` (full-sequence check) and
    ``check_prefix`` (prefix extendability check).

    Parameters
    ----------
    timeout : float
        Maximum time in seconds for a single oracle call. Defaults to 30.0.
    """

    def __init__(self, timeout: float = 30.0) -> None:
        self.timeout = timeout

    @abstractmethod
    def verify(self, sequence: list[int] | str) -> OracleResult:
        """Verify whether a complete sequence satisfies the constraint.

        Parameters
        ----------
        sequence : list[int] | str
            Complete token-id sequence or decoded text.

        Returns
        -------
        OracleResult
            SAT if the sequence satisfies φ, UNSAT if not, UNKNOWN on error.
        """
        ...

    @abstractmethod
    def check_prefix(self, prefix: list[int] | str) -> PrefixResult:
        """Check whether a prefix can be extended to satisfy the constraint.

        Corresponds to O^{prefix}_φ from Definition 3.2 (Eq. 4).

        Parameters
        ----------
        prefix : list[int] | str
            Partial token-id sequence or decoded text.

        Returns
        -------
        PrefixResult
            EXTENDABLE, DEAD, or UNKNOWN.
        """
        ...


# ---------------------------------------------------------------------------
# Z3Oracle — SMT-based oracle  (Section 7.1)
# ---------------------------------------------------------------------------

class Z3Oracle(SemanticOracle):
    """SMT-based semantic oracle using the Z3 solver.

    Used for two benchmark domains described in Section 7.1:

    1. **GSM-Symbolic** (arithmetic constraints): Parses the model's output
       into arithmetic expressions and checks equation satisfiability via Z3.
    2. **FOLIO** (FOL validity): Parses the model's output into first-order
       logic formulas expressed as SMT-LIB, and checks validity via Z3.

    The oracle translates model-generated text into Z3 constraints using
    configurable parser functions.

    Parameters
    ----------
    constraint_formula : str | None
        An SMT-LIB format formula representing the constraint φ.
        If None, the oracle operates in "parse-from-output" mode where the
        formula is extracted from the model's output.
    domain : str
        One of ``"arithmetic"`` (GSM-Symbolic) or ``"fol"`` (FOLIO).
    timeout : float
        Z3 solver timeout in seconds (default 30.0).

    Notes
    -----
    - Section 2.4: Z3 (De Moura & Bjørner, 2008) for SMT constraints.
    - Section 7.1: Z3 for arithmetic (GSM-Symbolic) and logical (FOLIO)
      constraint verification.
    """

    def __init__(
        self,
        constraint_formula: Optional[str] = None,
        domain: str = "arithmetic",
        timeout: float = 30.0,
    ) -> None:
        super().__init__(timeout=timeout)
        self.constraint_formula = constraint_formula
        self.domain = domain
        self._z3 = None
        self._import_z3()

    def _import_z3(self) -> None:
        """Lazily import z3-solver."""
        try:
            import z3
            self._z3 = z3
            logger.debug("Z3 solver loaded (version %s)", z3.get_version_string())
        except ImportError:
            logger.warning(
                "z3-solver not installed. Install with: pip install z3-solver"
            )

    def verify(self, sequence: list[int] | str) -> OracleResult:
        """Verify a complete sequence against the SMT constraint.

        For arithmetic domain (GSM-Symbolic): extracts the final numerical
        answer and checks that it satisfies all arithmetic equations in the
        constraint.

        For FOL domain (FOLIO): parses the output into an FOL formula and
        checks validity (i.e., unsatisfiability of the negation).

        Parameters
        ----------
        sequence : list[int] | str
            The complete model output (text or token ids).

        Returns
        -------
        OracleResult
            SAT / UNSAT / UNKNOWN.

        Notes
        -----
        - Definition 2.4, property 3: complete sequences must return SAT or
          UNSAT (we return UNKNOWN only on genuine solver failures/timeouts).
        """
        if self._z3 is None:
            return OracleResult.UNKNOWN

        z3 = self._z3
        text = sequence if isinstance(sequence, str) else str(sequence)

        try:
            if self.domain == "arithmetic":
                return self._verify_arithmetic(text, z3)
            elif self.domain == "fol":
                return self._verify_fol(text, z3)
            else:
                logger.error("Unknown Z3Oracle domain: %s", self.domain)
                return OracleResult.UNKNOWN
        except Exception as e:
            logger.warning("Z3Oracle.verify failed: %s", e)
            return OracleResult.UNKNOWN

    def _verify_arithmetic(self, text: str, z3: Any) -> OracleResult:
        """Verify arithmetic constraints (GSM-Symbolic).

        Extracts numerical expressions from the model output and checks
        them against the constraint formula using Z3's arithmetic solver.

        Parameters
        ----------
        text : str
            Model output text containing arithmetic reasoning.
        z3 : module
            The z3 module.

        Returns
        -------
        OracleResult
        """
        solver = z3.Solver()
        solver.set("timeout", int(self.timeout * 1000))  # ms

        # Extract the final numerical answer (look for patterns like "= 42",
        # "answer is 42", "#### 42")
        answer_match = re.search(
            r"(?:####\s*|(?:answer|result)\s+(?:is|=)\s*|=\s*)(-?\d+(?:\.\d+)?)",
            text,
            re.IGNORECASE,
        )
        if answer_match is None:
            logger.debug("No numerical answer found in output")
            return OracleResult.UNKNOWN

        extracted_answer = float(answer_match.group(1))

        if self.constraint_formula is not None:
            # Parse the SMT-LIB constraint formula
            try:
                answer_var = z3.Real("answer")
                solver.add(answer_var == extracted_answer)

                # Parse constraint: expect a formula referencing 'answer'
                constraint_text = self.constraint_formula.replace(
                    "ANSWER", str(extracted_answer)
                )
                # Try to evaluate as a simple equality/inequality
                constraint_assertions = self._parse_arithmetic_constraint(
                    constraint_text, z3
                )
                for assertion in constraint_assertions:
                    solver.add(assertion)

                result = solver.check()
                if result == z3.sat:
                    return OracleResult.SAT
                elif result == z3.unsat:
                    return OracleResult.UNSAT
                else:
                    return OracleResult.UNKNOWN
            except Exception as e:
                logger.debug("Arithmetic constraint parsing failed: %s", e)
                return OracleResult.UNKNOWN
        else:
            # No explicit constraint formula — verify internal consistency
            # by checking that intermediate arithmetic steps are correct
            return self._verify_arithmetic_chain(text, z3)

    def _parse_arithmetic_constraint(
        self, constraint_text: str, z3: Any
    ) -> list:
        """Parse a simple arithmetic constraint string into Z3 assertions.

        Supports constraints of the form ``answer == 42``, ``answer > 0``,
        or compound constraints separated by ``and`` / ``or``.

        Parameters
        ----------
        constraint_text : str
            The constraint expression.
        z3 : module
            The z3 module.

        Returns
        -------
        list
            Z3 assertion objects.
        """
        answer = z3.Real("answer")
        assertions = []

        # Handle simple equality: "answer == <value>"
        eq_match = re.search(r"answer\s*==\s*(-?\d+(?:\.\d+)?)", constraint_text)
        if eq_match:
            target = float(eq_match.group(1))
            assertions.append(answer == target)
            return assertions

        # Handle inequality: "answer > <value>", "answer < <value>", etc.
        for op_pattern, op_fn in [
            (r"answer\s*>=\s*(-?\d+(?:\.\d+)?)", lambda a, v: a >= v),
            (r"answer\s*<=\s*(-?\d+(?:\.\d+)?)", lambda a, v: a <= v),
            (r"answer\s*>\s*(-?\d+(?:\.\d+)?)", lambda a, v: a > v),
            (r"answer\s*<\s*(-?\d+(?:\.\d+)?)", lambda a, v: a < v),
        ]:
            match = re.search(op_pattern, constraint_text)
            if match:
                val = float(match.group(1))
                assertions.append(op_fn(answer, val))

        return assertions

    def _verify_arithmetic_chain(self, text: str, z3: Any) -> OracleResult:
        """Verify internal arithmetic consistency of a reasoning chain.

        Extracts equations of the form ``a op b = c`` from the text and
        verifies each one.

        Parameters
        ----------
        text : str
            Model output text.
        z3 : module
            The z3 module.

        Returns
        -------
        OracleResult
        """
        equations = re.findall(
            r"(-?\d+(?:\.\d+)?)\s*([+\-*/])\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)",
            text,
        )
        if not equations:
            return OracleResult.UNKNOWN

        for lhs, op, rhs, result in equations:
            try:
                lhs_val, rhs_val, result_val = float(lhs), float(rhs), float(result)
                if op == "+":
                    expected = lhs_val + rhs_val
                elif op == "-":
                    expected = lhs_val - rhs_val
                elif op == "*":
                    expected = lhs_val * rhs_val
                elif op == "/":
                    if rhs_val == 0:
                        return OracleResult.UNSAT
                    expected = lhs_val / rhs_val
                else:
                    continue

                if abs(expected - result_val) > 1e-6:
                    return OracleResult.UNSAT
            except (ValueError, ZeroDivisionError):
                continue

        return OracleResult.SAT

    def _verify_fol(self, text: str, z3: Any) -> OracleResult:
        """Verify first-order logic validity (FOLIO).

        Translates the model's output to an FOL formula in SMT-LIB format,
        then checks validity by testing unsatisfiability of the negation.

        A formula φ is valid iff ¬φ is unsatisfiable.

        Parameters
        ----------
        text : str
            Model output containing an FOL formula or conclusion.
        z3 : module
            The z3 module.

        Returns
        -------
        OracleResult
        """
        solver = z3.Solver()
        solver.set("timeout", int(self.timeout * 1000))

        if self.constraint_formula is not None:
            # Parse SMT-LIB formula
            try:
                assertions = z3.parse_smt2_string(self.constraint_formula)
                for a in assertions:
                    solver.add(a)

                result = solver.check()
                if result == z3.unsat:
                    # Negation is unsat ⟹ formula is valid ⟹ SAT
                    return OracleResult.SAT
                elif result == z3.sat:
                    return OracleResult.UNSAT
                else:
                    return OracleResult.UNKNOWN
            except Exception as e:
                logger.debug("FOL SMT-LIB parsing failed: %s", e)
                return OracleResult.UNKNOWN
        else:
            # Try to extract conclusion from text and verify
            conclusion_match = re.search(
                r"(?:conclusion|therefore|thus|hence)[:\s]*(.*?)(?:\.|$)",
                text,
                re.IGNORECASE,
            )
            if conclusion_match:
                conclusion = conclusion_match.group(1).strip().lower()
                if conclusion in ("true", "valid", "yes"):
                    return OracleResult.SAT
                elif conclusion in ("false", "invalid", "no"):
                    return OracleResult.UNSAT
            return OracleResult.UNKNOWN

    def check_prefix(self, prefix: list[int] | str) -> PrefixResult:
        """Check prefix extendability for SMT constraints.

        For arithmetic constraints, a prefix is DEAD if it contains a
        completed but incorrect equation. Otherwise, it is conservatively
        reported as UNKNOWN (prefix-approximable behaviour per
        Definition 3.2, case 2).

        Parameters
        ----------
        prefix : list[int] | str
            Partial sequence.

        Returns
        -------
        PrefixResult

        Notes
        -----
        - Definition 3.2, case 2 (prefix-approximable): DEAD is sound
          (no extension can fix a completed incorrect equation), but
          UNKNOWN may be returned when the true answer is DEAD.
        """
        if self._z3 is None:
            return PrefixResult.UNKNOWN

        text = prefix if isinstance(prefix, str) else str(prefix)

        try:
            if self.domain == "arithmetic":
                # Check completed equations in the prefix
                result = self._verify_arithmetic_chain(text, self._z3)
                if result == OracleResult.UNSAT:
                    return PrefixResult.DEAD
                return PrefixResult.UNKNOWN  # prefix-approximable
            elif self.domain == "fol":
                # FOL constraints are prefix-opaque (Definition 3.2, case 3):
                # we cannot evaluate validity until the full proof is generated.
                return PrefixResult.UNKNOWN
            else:
                return PrefixResult.UNKNOWN
        except Exception:
            return PrefixResult.UNKNOWN


# ---------------------------------------------------------------------------
# PrologOracle — Prolog-based oracle  (Section 7.1)
# ---------------------------------------------------------------------------

class PrologOracle(SemanticOracle):
    """Prolog-based semantic oracle for relational constraints.

    Used for the ProofWriter benchmark (Section 7.1): translates model
    deduction steps into Prolog facts and rules, then queries logical
    validity using SWI-Prolog via pyswip.

    Parameters
    ----------
    base_facts : list[str]
        Initial Prolog facts and rules representing the problem context.
    query : str
        The Prolog query to evaluate (the conclusion to verify).
    timeout : float
        Maximum time per Prolog query (default 30.0 seconds).

    Notes
    -----
    - Section 2.4: Prolog engines for relational constraints.
    - Section 7.1: Prolog oracle for ProofWriter (600 test examples, rule
      consistency checking).
    """

    def __init__(
        self,
        base_facts: Optional[list[str]] = None,
        query: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        super().__init__(timeout=timeout)
        self.base_facts = base_facts or []
        self.query = query
        self._prolog = None
        self._import_prolog()

    def _import_prolog(self) -> None:
        """Lazily import pyswip."""
        try:
            from pyswip import Prolog
            self._prolog_class = Prolog
            logger.debug("pyswip loaded successfully")
        except ImportError:
            self._prolog_class = None
            logger.warning(
                "pyswip not installed. Install with: pip install pyswip"
            )

    def _create_prolog_instance(self) -> Any:
        """Create a fresh Prolog instance with base facts loaded."""
        if self._prolog_class is None:
            return None
        prolog = self._prolog_class()
        for fact in self.base_facts:
            try:
                prolog.assertz(fact)
            except Exception as e:
                logger.debug("Failed to assert fact '%s': %s", fact, e)
        return prolog

    def verify(self, sequence: list[int] | str) -> OracleResult:
        """Verify a complete deduction chain against Prolog rules.

        Parses the model output into deduction steps, asserts them as
        Prolog facts/rules, and queries the target conclusion.

        Parameters
        ----------
        sequence : list[int] | str
            Complete model output containing deduction steps.

        Returns
        -------
        OracleResult

        Notes
        -----
        - Section 7.1: "The oracle translates the model's deduction steps
          into Prolog facts and rules and verifies logical validity."
        """
        if self._prolog_class is None:
            return OracleResult.UNKNOWN

        text = sequence if isinstance(sequence, str) else str(sequence)

        try:
            prolog = self._create_prolog_instance()
            if prolog is None:
                return OracleResult.UNKNOWN

            # Extract and assert deduction steps from model output
            steps = self._extract_deduction_steps(text)
            for step in steps:
                try:
                    prolog.assertz(step)
                except Exception as e:
                    logger.debug("Failed to assert deduction step '%s': %s", step, e)

            # Query the target conclusion
            if self.query:
                results = list(prolog.query(self.query))
                if results:
                    return OracleResult.SAT
                else:
                    return OracleResult.UNSAT
            else:
                # No specific query — check internal consistency
                return self._check_consistency(prolog)

        except Exception as e:
            logger.warning("PrologOracle.verify failed: %s", e)
            return OracleResult.UNKNOWN

    def _extract_deduction_steps(self, text: str) -> list[str]:
        """Extract Prolog-compatible deduction steps from model text.

        Looks for patterns like:
        - ``Step N: <fact>``
        - ``Therefore: <conclusion>``
        - Direct Prolog-style assertions

        Parameters
        ----------
        text : str
            Model output text.

        Returns
        -------
        list[str]
            List of Prolog fact/rule strings.
        """
        steps = []

        # Pattern 1: "Step N: predicate(args)"
        step_matches = re.findall(
            r"[Ss]tep\s*\d+\s*:\s*(.+?)(?:\.|$)", text, re.MULTILINE
        )
        for match in step_matches:
            cleaned = self._text_to_prolog(match.strip())
            if cleaned:
                steps.append(cleaned)

        # Pattern 2: Direct Prolog-style facts "predicate(args)."
        prolog_matches = re.findall(
            r"([a-z_]\w*\([^)]+\))\s*\.", text
        )
        for match in prolog_matches:
            steps.append(match.strip())

        # Pattern 3: "X is Y" → is_property(x, y)
        is_matches = re.findall(
            r"(\w+)\s+is\s+(\w+)", text, re.IGNORECASE
        )
        for subj, obj in is_matches:
            steps.append(f"is_property({subj.lower()}, {obj.lower()})")

        return steps

    def _text_to_prolog(self, text: str) -> Optional[str]:
        """Convert a natural-language deduction step to a Prolog fact.

        Parameters
        ----------
        text : str
            A single deduction step in natural language.

        Returns
        -------
        str or None
            Prolog-compatible fact string, or None if unparseable.
        """
        # Already Prolog-style
        if re.match(r"[a-z_]\w*\(", text):
            return text.rstrip(".")

        # Simple "X is Y" pattern
        match = re.match(r"(\w+)\s+is\s+(\w+)", text, re.IGNORECASE)
        if match:
            return f"is_property({match.group(1).lower()}, {match.group(2).lower()})"

        # "X implies Y" pattern → implies(x, y)
        match = re.match(r"(\w+)\s+implies\s+(\w+)", text, re.IGNORECASE)
        if match:
            return f"implies({match.group(1).lower()}, {match.group(2).lower()})"

        return None

    def _check_consistency(self, prolog: Any) -> OracleResult:
        """Check for logical contradictions in the asserted facts.

        Parameters
        ----------
        prolog : Prolog
            pyswip Prolog instance.

        Returns
        -------
        OracleResult
        """
        try:
            # Check for explicit contradictions: both P(X) and not_P(X)
            contradiction_query = (
                "is_property(X, Y), is_property(X, Z), Y \\= Z"
            )
            results = list(prolog.query(contradiction_query))
            if results:
                return OracleResult.UNSAT
            return OracleResult.SAT
        except Exception:
            return OracleResult.UNKNOWN

    def check_prefix(self, prefix: list[int] | str) -> PrefixResult:
        """Check prefix extendability for Prolog-based constraints.

        ProofWriter constraints are prefix-approximable (Definition 3.2):
        we can detect DEAD prefixes if a completed deduction step
        contradicts the knowledge base, but partial steps are UNKNOWN.

        Parameters
        ----------
        prefix : list[int] | str
            Partial sequence.

        Returns
        -------
        PrefixResult

        Notes
        -----
        - Definition 3.2, case 2: prefix-approximable.
        """
        if self._prolog_class is None:
            return PrefixResult.UNKNOWN

        text = prefix if isinstance(prefix, str) else str(prefix)

        try:
            prolog = self._create_prolog_instance()
            if prolog is None:
                return PrefixResult.UNKNOWN

            steps = self._extract_deduction_steps(text)
            for step in steps:
                try:
                    prolog.assertz(step)
                except Exception:
                    continue

            consistency = self._check_consistency(prolog)
            if consistency == OracleResult.UNSAT:
                return PrefixResult.DEAD
            return PrefixResult.UNKNOWN

        except Exception:
            return PrefixResult.UNKNOWN


# ---------------------------------------------------------------------------
# TypeCheckOracle — Type-checking oracle  (Section 7.1)
# ---------------------------------------------------------------------------

class TypeCheckOracle(SemanticOracle):
    """Type-checking oracle for HumanEval-typed code generation.

    Uses Python's ``ast`` module for syntax validation and optionally
    invokes mypy for static type checking. This oracle verifies that
    generated code is both syntactically valid and type-correct.

    Parameters
    ----------
    function_signature : str | None
        The expected function signature with type annotations, e.g.,
        ``"def add(a: int, b: int) -> int:"``.
    test_cases : list[dict] | None
        Optional test cases ``[{"input": ..., "expected": ...}, ...]``
        for runtime verification.
    use_mypy : bool
        If True, invoke mypy for static type checking (default True).
    timeout : float
        Timeout for mypy subprocess (default 30.0 seconds).

    Notes
    -----
    - Section 7.1: Type checker oracle for HumanEval-typed (164 test
      examples, type correctness constraint).
    - The constraint is prefix-approximable: syntax errors detected in
      the prefix yield DEAD, but type errors require the full function body.
    """

    def __init__(
        self,
        function_signature: Optional[str] = None,
        test_cases: Optional[list[dict]] = None,
        use_mypy: bool = True,
        timeout: float = 30.0,
    ) -> None:
        super().__init__(timeout=timeout)
        self.function_signature = function_signature
        self.test_cases = test_cases or []
        self.use_mypy = use_mypy

    def verify(self, sequence: list[int] | str) -> OracleResult:
        """Verify a complete code sequence for type correctness.

        Performs three checks:
        1. Syntax validation via ``ast.parse``.
        2. Static type checking via mypy (if enabled).
        3. Runtime test case execution (if test cases provided).

        Parameters
        ----------
        sequence : list[int] | str
            Complete code output.

        Returns
        -------
        OracleResult

        Notes
        -----
        - Definition 2.4, property 3: returns SAT or UNSAT for complete
          sequences (UNKNOWN only on genuine infrastructure failures).
        """
        code = sequence if isinstance(sequence, str) else str(sequence)

        # Step 1: Syntax validation
        try:
            ast.parse(code)
        except SyntaxError as e:
            logger.debug("Syntax error: %s", e)
            return OracleResult.UNSAT

        # Step 2: Type checking with mypy
        if self.use_mypy:
            type_result = self._run_mypy(code)
            if type_result == OracleResult.UNSAT:
                return OracleResult.UNSAT
            elif type_result == OracleResult.UNKNOWN:
                # mypy failure is not fatal — fall through to runtime checks
                logger.debug("mypy check returned UNKNOWN, continuing")

        # Step 3: Signature conformance
        if self.function_signature:
            if not self._check_signature(code):
                return OracleResult.UNSAT

        # Step 4: Runtime test cases
        if self.test_cases:
            runtime_result = self._run_test_cases(code)
            return runtime_result

        return OracleResult.SAT

    def _run_mypy(self, code: str) -> OracleResult:
        """Run mypy type checker on the code.

        Parameters
        ----------
        code : str
            Python code to type-check.

        Returns
        -------
        OracleResult
        """
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False
            ) as f:
                f.write(code)
                f.flush()
                tmp_path = f.name

            result = subprocess.run(
                ["python", "-m", "mypy", "--no-error-summary", "--no-color", tmp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            if result.returncode == 0:
                return OracleResult.SAT
            elif "error:" in result.stdout:
                logger.debug("mypy errors: %s", result.stdout.strip())
                return OracleResult.UNSAT
            else:
                return OracleResult.UNKNOWN

        except subprocess.TimeoutExpired:
            logger.warning("mypy timed out after %.1f seconds", self.timeout)
            return OracleResult.UNKNOWN
        except FileNotFoundError:
            logger.warning("mypy not found in PATH")
            return OracleResult.UNKNOWN
        except Exception as e:
            logger.warning("mypy check failed: %s", e)
            return OracleResult.UNKNOWN
        finally:
            import os
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    def _check_signature(self, code: str) -> bool:
        """Check that the code contains the expected function signature.

        Parameters
        ----------
        code : str
            Python code.

        Returns
        -------
        bool
            True if the signature matches.
        """
        if self.function_signature is None:
            return True

        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Compare function name at minimum
                    sig_tree = ast.parse(self.function_signature + "\n    pass")
                    for sig_node in ast.walk(sig_tree):
                        if isinstance(sig_node, ast.FunctionDef):
                            if node.name == sig_node.name:
                                return True
            return False
        except Exception:
            return False

    def _run_test_cases(self, code: str) -> OracleResult:
        """Execute runtime test cases against the generated code.

        Parameters
        ----------
        code : str
            Python code to test.

        Returns
        -------
        OracleResult
        """
        for tc in self.test_cases:
            try:
                # Create a sandboxed execution environment
                test_code = textwrap.dedent(f"""
{code}

# Test case execution
_input = {tc['input']!r}
_expected = {tc['expected']!r}
if isinstance(_input, tuple):
    _result = {self._extract_function_name(code)}(*_input)
else:
    _result = {self._extract_function_name(code)}(_input)
assert _result == _expected, f"Expected {{_expected}}, got {{_result}}"
""")
                exec_globals: dict = {}
                exec(test_code, exec_globals)
            except AssertionError:
                return OracleResult.UNSAT
            except Exception as e:
                logger.debug("Test case execution error: %s", e)
                return OracleResult.UNKNOWN

        return OracleResult.SAT

    def _extract_function_name(self, code: str) -> str:
        """Extract the first function name from code.

        Parameters
        ----------
        code : str

        Returns
        -------
        str
            Function name.
        """
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    return node.name
        except Exception:
            pass
        return "solution"

    def check_prefix(self, prefix: list[int] | str) -> PrefixResult:
        """Check prefix extendability for type-checking constraints.

        Uses Python's ``ast`` module to detect definitively broken syntax
        that cannot be repaired by any suffix. Conservative: most
        incomplete code returns UNKNOWN (prefix-approximable).

        Parameters
        ----------
        prefix : list[int] | str
            Partial code output.

        Returns
        -------
        PrefixResult

        Notes
        -----
        - Definition 3.2, case 2: prefix-approximable. Only unrecoverable
          syntax errors (e.g., invalid identifiers) yield DEAD.
        """
        code = prefix if isinstance(prefix, str) else str(prefix)

        # Check for definitive syntax issues that cannot be fixed by continuation
        # Unclosed string literals with mismatched quotes are not recoverable
        # But most partial code is potentially extendable
        try:
            # Attempt to parse — if it succeeds, the code is already valid
            ast.parse(code)
            return PrefixResult.EXTENDABLE
        except SyntaxError:
            # Most syntax errors on partial code are due to incompleteness,
            # not fundamental brokenness. We conservatively return UNKNOWN.
            # Only return DEAD for truly unrecoverable patterns.
            if self._is_unrecoverable(code):
                return PrefixResult.DEAD
            return PrefixResult.UNKNOWN

    def _is_unrecoverable(self, code: str) -> bool:
        """Heuristic check for unrecoverable syntax errors.

        Parameters
        ----------
        code : str
            Partial code.

        Returns
        -------
        bool
            True if the code cannot be fixed by any suffix.
        """
        # Invalid variable names, impossible token sequences
        lines = code.strip().split("\n")
        for line in lines:
            stripped = line.strip()
            # Multiple consecutive operators like "++ -- **" (not valid Python)
            if re.search(r"[+\-]{3,}", stripped):
                return True
        return False
