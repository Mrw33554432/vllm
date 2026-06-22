"""HoleKV Trace Registry — stores and retrieves KV cache trace records.

Every completed request stores a HoleKVTraceRecord. A future request can
reference it via holekv_ref to enable the hole-preserved reuse path.

Security: cache IDs must be unguessable. Cross-user reuse is not allowed
by default.
"""

from __future__ import annotations

import hashlib
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class HoleKVTraceRecord:
    """Complete trace of a request's input prompt KV cache.

    This record is stored after each request completes and can be used by
    a future request via holekv_ref to enable hole-preserved KV reuse.
    """

    cache_id: str
    request_id: str

    # Prompt content (text form for alignment + UTF-8 bytes for byte offsets)
    rendered_prompt_text: str
    rendered_prompt_utf8: bytes

    # Token-level information
    token_ids: list[int]
    token_byte_offsets: list[tuple[int, int]]  # (start_byte, end_byte) per token

    # Block-level KV cache layout
    block_size: int
    block_ids_by_group: list[list[int]]  # [group][block_idx] -> physical block id
    token_ranges_by_block: list[tuple[int, int]]  # (start_token, end_token) per block
    position_ranges_by_block: list[tuple[int, int]]  # (start_pos, end_pos) per block

    # Environment fingerprint — must match for reuse
    model_fingerprint: str
    tokenizer_fingerprint: str
    lora_id: Optional[str]
    dtype: str

    # Access control
    owner_id: Optional[str] = None
    session_id: Optional[str] = None

    # Lifetime
    created_at: float = field(default_factory=time.time)
    ttl_seconds: float = 3600.0  # 1 hour default
    pinned: bool = False

    @property
    def total_tokens(self) -> int:
        """Total number of tokens in the trace."""
        return len(self.token_ids)

    @property
    def is_expired(self) -> bool:
        """Check if the trace has exceeded its TTL."""
        if self.ttl_seconds <= 0:
            return False  # Unlimited TTL
        return (time.time() - self.created_at) > self.ttl_seconds

    @property
    def num_kv_blocks(self) -> int:
        """Total number of KV blocks used by this trace."""
        return sum(len(group) for group in self.block_ids_by_group)

    def to_dict(self) -> dict:
        """Serialize to a JSON-serializable dictionary (minus byte offsets for size)."""
        return {
            "cache_id": self.cache_id,
            "request_id": self.request_id,
            "rendered_prompt_text": self.rendered_prompt_text,
            "token_ids": self.token_ids,
            "block_size": self.block_size,
            "block_ids_by_group": self.block_ids_by_group,
            "token_ranges_by_block": self.token_ranges_by_block,
            "position_ranges_by_block": self.position_ranges_by_block,
            "model_fingerprint": self.model_fingerprint,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "lora_id": self.lora_id,
            "dtype": self.dtype,
            "owner_id": self.owner_id,
            "session_id": self.session_id,
            "created_at": self.created_at,
            "ttl_seconds": self.ttl_seconds,
            "pinned": self.pinned,
        }


def _generate_cache_id() -> str:
    """Generate an unguessable cache ID.

    Uses UUID4 + random bytes hashed together to make guessing infeasible.
    """
    raw = uuid.uuid4().bytes + os.urandom(16)
    return hashlib.sha256(raw).hexdigest()[:24]


class HoleKVTraceRegistry:
    """In-memory registry of HoleKV trace records.

    Provides lookup by cache_id with access control and expiration checks.
    In production, this would be backed by a distributed store.
    """

    def __init__(self, max_entries: int = 10000, default_ttl_seconds: float = 3600.0):
        self._records: dict[str, HoleKVTraceRecord] = {}
        self._max_entries = max_entries
        self._default_ttl_seconds = default_ttl_seconds
        self._block_unpin_fn: Callable[[list[int]], None] | None = None
        """Callback invoked on eviction to unpin KV blocks."""

    def set_block_unpin_fn(self, fn: Callable[[list[int]], None]) -> None:
        """Register a callback for unpinning KV blocks on trace eviction.

        The callback receives the first group's block_ids (list[int]).
        It should call KVCacheManager.unpin_blocks() on them.
        """
        self._block_unpin_fn = fn

    def generate_id(self) -> str:
        """Generate a fresh, unguessable cache ID."""
        return _generate_cache_id()

    def store(self, record: HoleKVTraceRecord) -> None:
        """Store a trace record.

        If the registry is at capacity, evicts the oldest non-pinned entry.
        """
        if len(self._records) >= self._max_entries:
            self._evict_one()
        self._records[record.cache_id] = record
        logger.debug(
            "HoleKVTraceRegistry: stored trace cache_id=%s (tokens=%d, blocks=%d)",
            record.cache_id,
            record.total_tokens,
            record.num_kv_blocks,
        )

    def lookup(
        self,
        cache_id: str,
        *,
        owner_id: Optional[str] = None,
        session_id: Optional[str] = None,
        model_fingerprint: Optional[str] = None,
        tokenizer_fingerprint: Optional[str] = None,
        lora_id: Optional[str] = None,
        dtype: Optional[str] = None,
    ) -> Optional[HoleKVTraceRecord]:
        """Look up a trace by cache_id with access control and fingerprint checks.

        Returns None if:
          - cache_id not found
          - record is expired
          - owner_id mismatch (cross-user access denied)
          - model/tokenizer/dtype fingerprint mismatch
        """
        record = self._records.get(cache_id)
        if record is None:
            logger.debug("HoleKVTraceRegistry: cache_id=%s not found", cache_id)
            return None

        # Expiration check
        if record.is_expired:
            logger.debug(
                "HoleKVTraceRegistry: cache_id=%s expired (age=%.0fs, ttl=%.0fs)",
                cache_id,
                time.time() - record.created_at,
                record.ttl_seconds,
            )
            del self._records[cache_id]
            return None

        # Access control: same owner
        if record.owner_id is not None and owner_id is not None:
            if record.owner_id != owner_id:
                logger.warning(
                    "HoleKVTraceRegistry: cross-user access denied for cache_id=%s "
                    "(owner=%s, requester=%s)",
                    cache_id,
                    record.owner_id,
                    owner_id,
                )
                return None

        # Same session check
        if record.session_id is not None and session_id is not None:
            if record.session_id != session_id:
                logger.warning(
                    "HoleKVTraceRegistry: cross-session access denied for cache_id=%s",
                    cache_id,
                )
                return None

        # Environment fingerprint check
        if model_fingerprint and record.model_fingerprint != model_fingerprint:
            logger.warning(
                "HoleKVTraceRegistry: model fingerprint mismatch for cache_id=%s",
                cache_id,
            )
            return None

        if tokenizer_fingerprint and record.tokenizer_fingerprint != tokenizer_fingerprint:
            logger.warning(
                "HoleKVTraceRegistry: tokenizer fingerprint mismatch for cache_id=%s",
                cache_id,
            )
            return None

        if lora_id is not None and record.lora_id != lora_id:
            logger.warning(
                "HoleKVTraceRegistry: LoRA mismatch for cache_id=%s "
                "(record=%s, request=%s)",
                cache_id,
                record.lora_id,
                lora_id,
            )
            return None

        if dtype and record.dtype != dtype:
            logger.warning(
                "HoleKVTraceRegistry: dtype mismatch for cache_id=%s "
                "(record=%s, request=%s)",
                cache_id,
                record.dtype,
                dtype,
            )
            return None

        return record

    def remove(self, cache_id: str) -> bool:
        """Remove a trace record. Returns True if it existed."""
        if cache_id in self._records:
            del self._records[cache_id]
            return True
        return False

    def pin(self, cache_id: str) -> bool:
        """Pin a trace so it cannot be evicted."""
        record = self._records.get(cache_id)
        if record is None:
            return False
        record.pinned = True
        return True

    def unpin(self, cache_id: str) -> bool:
        """Unpin a trace, allowing eviction."""
        record = self._records.get(cache_id)
        if record is None:
            return False
        record.pinned = False
        return True

    def evict_expired(self) -> int:
        """Remove all expired records. Returns count removed."""
        expired = [
            cid for cid, r in self._records.items() if r.is_expired
        ]
        for cid in expired:
            del self._records[cid]
        if expired:
            logger.debug("HoleKVTraceRegistry: evicted %d expired traces", len(expired))
        return len(expired)

    def _evict_one(self) -> None:
        """Evict the oldest non-pinned entry."""
        candidates = [
            (cid, r) for cid, r in self._records.items() if not r.pinned
        ]
        if not candidates:
            # All are pinned — force-evict the oldest even if pinned
            candidates = list(self._records.items())
        if candidates:
            oldest_id, oldest_record = min(
                candidates, key=lambda x: x[1].created_at
            )
            # Unpin KV blocks before removing the trace
            if (
                self._block_unpin_fn is not None
                and oldest_record.block_ids_by_group
                and oldest_record.block_ids_by_group[0]
            ):
                self._block_unpin_fn(oldest_record.block_ids_by_group[0])
            logger.debug(
                "HoleKVTraceRegistry: evicting trace cache_id=%s (age=%.0fs)",
                oldest_id,
                time.time() - oldest_record.created_at,
            )
            del self._records[oldest_id]

    @property
    def size(self) -> int:
        """Number of stored records."""
        return len(self._records)

    def clear(self) -> None:
        """Remove all records."""
        self._records.clear()


# Global singleton for convenience
_global_registry: Optional[HoleKVTraceRegistry] = None


def get_global_registry() -> HoleKVTraceRegistry:
    """Get or create the global HoleKV trace registry singleton."""
    global _global_registry
    if _global_registry is None:
        _global_registry = HoleKVTraceRegistry()
    return _global_registry
