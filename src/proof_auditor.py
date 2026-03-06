"""
Proof auditor — Lane B verification, repair, and decomposition via Axle API.

Separate from Lane A (step-wise tactic search via Pantograph). Operates on
complete or near-complete proofs. All results are tagged with metric category
(raw_success, axle_assisted_success, axle_repair_only) to prevent inflated claims.

Cost controls:
  - Content-hash cache (SHA-256) avoids redundant API calls.
  - Caller filters to top-N candidates or high critic score before calling.
  - Bounded async concurrency via AxleClient.max_concurrency.
  - Graceful degradation on timeout/429/503 (returns AuditResult with error).

Scope caveat (from Axle docs):
  - Axle is designed for simple imports, theorems, and definitions.
  - verify_proof trusts the Lean environment (not adversarial-safe).
  - For high-assurance verification, use Lane C (lean4checker/Comparator/SafeVerify).
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class SuccessCategory(Enum):
    """Metric separation for proof success claims."""

    RAW = "raw_success"  # Proved by Lane A alone, no Axle involvement
    AXLE_ASSISTED = "axle_assisted_success"  # Lane A partial + Axle decompose/repair
    AXLE_REPAIR_ONLY = "axle_repair_only"  # Axle repair_proofs closed remaining goals


@dataclass
class AuditResult:
    """Result of a Lane B proof audit operation."""

    success: bool
    category: SuccessCategory
    verified: bool = False  # True if verify_proof confirmed
    repaired: bool = False  # True if repair_proofs was used
    subgoals: list[str] = field(default_factory=list)  # From sorry2lemma decomposition
    lean_errors: list[str] = field(default_factory=list)
    tool_errors: list[str] = field(default_factory=list)
    timings: dict[str, int] = field(default_factory=dict)
    api_error: str = ""  # Non-empty if Axle call failed (timeout/429/503)
    cached: bool = False  # True if result came from content-hash cache

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "category": self.category.value,
            "verified": self.verified,
            "repaired": self.repaired,
            "subgoals": self.subgoals,
            "lean_errors": self.lean_errors,
            "tool_errors": self.tool_errors,
            "timings": self.timings,
            "api_error": self.api_error,
            "cached": self.cached,
        }


@dataclass
class AuditorConfig:
    """Configuration for the ProofAuditor."""

    environment: str = "lean-4.28.0"
    timeout_seconds: float = 120
    ignore_imports: bool = False  # Strict by default; only True when intentional
    repair_terminal_tactics: list[str] = field(
        default_factory=lambda: ["grind", "aesop", "simp", "omega", "decide"]
    )
    cache_enabled: bool = True
    max_concurrency: int = 10


class ProofAuditor:
    """Lane B: Axle-based proof verification, repair, and decomposition.

    Operates on complete or near-complete proofs produced by Lane A search.
    NOT a replacement for step-wise tactic search — a complementary service.

    Usage:
        async with ProofAuditor(config) as auditor:
            result = await auditor.verify(statement, proof)
            result = await auditor.repair(content)
            result = await auditor.decompose(content)
    """

    def __init__(self, config: AuditorConfig | None = None) -> None:
        self.config = config or AuditorConfig()
        self._client: Any = None  # AxleClient, lazy import
        self._cache: dict[str, AuditResult] = {}

    async def connect(self) -> None:
        """Initialize the Axle client. Call before any audit operations."""
        try:
            from axle import AxleClient
        except ImportError:
            msg = (
                "axiom-axle not installed. Install with: pip install axiom-axle\n"
                "Lane B (Axle auditor) is optional — Lane A search works without it."
            )
            raise ImportError(msg) from None

        self._client = AxleClient(max_concurrency=self.config.max_concurrency)

    async def close(self) -> None:
        """Close the Axle client and release resources."""
        if self._client is not None:
            await self._client.close()
            self._client = None

    async def __aenter__(self) -> ProofAuditor:
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    def _require_client(self) -> Any:
        """Guard: raise clear error if connect() was not called."""
        if self._client is None:
            raise RuntimeError(
                "ProofAuditor.connect() was not called. "
                "Use 'async with ProofAuditor(config) as auditor:' or call connect() first."
            )
        return self._client

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    async def verify(
        self,
        formal_statement: str,
        proof_content: str,
        permitted_sorries: list[str] | None = None,
        category: SuccessCategory = SuccessCategory.RAW,
    ) -> AuditResult:
        """Verify a completed proof against its formal statement.

        Args:
            formal_statement: Lean code with sorry placeholders.
            proof_content: Candidate proof to verify.
            permitted_sorries: Theorem names allowed to contain sorry.
            category: Caller-provided metric category. Use RAW when verifying
                Lane A results, AXLE_ASSISTED when verifying post-repair proofs.
        """
        sorries_key = json.dumps(sorted(permitted_sorries)) if permitted_sorries else ""
        cache_key = self._cache_key("verify", formal_statement, proof_content, sorries_key)
        if cached := self._cache_get(cache_key):
            return cached

        client = self._require_client()
        try:
            result = await client.verify_proof(
                formal_statement=formal_statement,
                content=proof_content,
                environment=self.config.environment,
                permitted_sorries=permitted_sorries,
                ignore_imports=self.config.ignore_imports,
                timeout_seconds=self.config.timeout_seconds,
            )

            audit = AuditResult(
                success=result.okay,
                category=category,
                verified=result.okay,
                lean_errors=result.lean_messages.errors,
                tool_errors=result.tool_messages.errors,
                timings=result.timings,
            )
            self._cache_put(cache_key, audit)
            return audit

        except Exception as e:
            return self._api_error_result(e)

    async def repair(self, content: str) -> AuditResult:
        """Attempt to repair a near-complete proof.

        Uses Axle's repair_proofs with configured terminal tactics.
        Result is categorized as AXLE_REPAIR_ONLY.
        """
        cache_key = self._cache_key("repair", content)
        if cached := self._cache_get(cache_key):
            return cached

        client = self._require_client()
        try:
            result = await client.repair_proofs(
                content=content,
                environment=self.config.environment,
                terminal_tactics=self.config.repair_terminal_tactics,
                ignore_imports=self.config.ignore_imports,
                timeout_seconds=self.config.timeout_seconds,
            )

            has_errors = bool(result.lean_messages.errors)
            audit = AuditResult(
                success=not has_errors,
                category=SuccessCategory.AXLE_REPAIR_ONLY,
                repaired=True,
                lean_errors=result.lean_messages.errors,
                tool_errors=result.tool_messages.errors,
                timings=result.timings,
            )
            self._cache_put(cache_key, audit)
            return audit

        except Exception as e:
            return self._api_error_result(e)

    async def decompose(self, content: str) -> AuditResult:
        """Decompose sorry placeholders into standalone subgoal lemmas.

        Returns subgoal names that the navigator can attack independently.
        Result is categorized as AXLE_ASSISTED.
        """
        cache_key = self._cache_key("decompose", content)
        if cached := self._cache_get(cache_key):
            return cached

        client = self._require_client()
        try:
            result = await client.sorry2lemma(
                content=content,
                environment=self.config.environment,
                extract_sorries=True,
                extract_errors=True,
                include_whole_context=True,
                ignore_imports=self.config.ignore_imports,
                timeout_seconds=self.config.timeout_seconds,
            )

            audit = AuditResult(
                success=bool(result.lemma_names),
                category=SuccessCategory.AXLE_ASSISTED,
                subgoals=result.lemma_names,
                lean_errors=result.lean_messages.errors,
                tool_errors=result.tool_messages.errors,
                timings=result.timings,
            )
            self._cache_put(cache_key, audit)
            return audit

        except Exception as e:
            return self._api_error_result(e)

    async def check(self, content: str) -> AuditResult:
        """Check if Lean code compiles. Fast validity oracle."""
        cache_key = self._cache_key("check", content)
        if cached := self._cache_get(cache_key):
            return cached

        client = self._require_client()
        try:
            result = await client.check(
                content=content,
                environment=self.config.environment,
                ignore_imports=self.config.ignore_imports,
                timeout_seconds=self.config.timeout_seconds,
            )

            audit = AuditResult(
                success=result.okay,
                category=SuccessCategory.RAW,
                verified=result.okay,
                lean_errors=result.lean_messages.errors,
                tool_errors=result.tool_messages.errors,
                timings=result.timings,
            )
            self._cache_put(cache_key, audit)
            return audit

        except Exception as e:
            return self._api_error_result(e)

    async def extract_theorems(self, content: str) -> dict[str, Any]:
        """Extract theorems with dependency metadata for proof network enrichment.

        Returns raw Axle Document data including proof_length, tactic_counts,
        and all 6 dependency flavors (local/external x type/value/syntactic).

        This is a data pipeline operation — no metric categorization.
        """
        client = self._require_client()
        try:
            result = await client.extract_theorems(
                content=content,
                environment=self.config.environment,
                ignore_imports=self.config.ignore_imports,
                timeout_seconds=self.config.timeout_seconds,
            )

            docs = {}
            for name, doc in result.documents.items():
                docs[name] = {
                    "name": doc.name,
                    "signature": doc.signature,
                    "type": doc.type,
                    "type_hash": doc.type_hash,
                    "proof_length": doc.proof_length,
                    "tactic_counts": doc.tactic_counts,
                    "is_sorry": doc.is_sorry,
                    "local_type_dependencies": doc.local_type_dependencies,
                    "local_value_dependencies": doc.local_value_dependencies,
                    "external_type_dependencies": doc.external_type_dependencies,
                    "external_value_dependencies": doc.external_value_dependencies,
                    "local_syntactic_dependencies": doc.local_syntactic_dependencies,
                    "external_syntactic_dependencies": doc.external_syntactic_dependencies,
                }
            return {
                "documents": docs,
                "errors": result.lean_messages.errors,
                "timings": result.timings,
            }

        except Exception as e:
            logger.warning("extract_theorems failed: %s", e)
            return {"documents": {}, "errors": [str(e)], "timings": {}}

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def _cache_key(self, operation: str, *contents: str) -> str:
        """SHA-256 content hash for deduplication."""
        h = hashlib.sha256()
        h.update(operation.encode())
        h.update(self.config.environment.encode())
        for c in contents:
            h.update(c.encode())
        return h.hexdigest()

    def _cache_get(self, key: str) -> AuditResult | None:
        if not self.config.cache_enabled:
            return None
        if stored := self._cache.get(key):
            result = copy.copy(stored)
            result.cached = True
            return result
        return None

    def _cache_put(self, key: str, result: AuditResult) -> None:
        if self.config.cache_enabled:
            self._cache[key] = result

    def cache_stats(self) -> dict[str, int]:
        return {"entries": len(self._cache)}

    def cache_clear(self) -> None:
        self._cache.clear()

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    @staticmethod
    def _api_error_result(error: Exception) -> AuditResult:
        """Graceful degradation: return a failed AuditResult instead of raising."""
        error_type = type(error).__name__
        logger.warning("Axle API error (%s): %s", error_type, error)
        return AuditResult(
            success=False,
            category=SuccessCategory.RAW,
            api_error=f"{error_type}: {error}",
        )
