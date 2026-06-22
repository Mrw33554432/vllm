"""HoleKV Engine Processor — integrates HoleKV processing into the engine core.

Handles the complete HoleKV logic for every request:
  1. Parse inline markers
  2. If holekv_ref is provided:
     a. Look up trace
     b. Align spans against trace
     c. Build hole-preserved active view
     d. Set HoleKV attention metadata on the request
  3. If no holekv_ref:
     a. Compact the prompt (fallback path)
  4. On request completion:
     a. Store a new HoleKV trace record
     b. Return holekv_cache_id in the output
"""

from __future__ import annotations

from typing import Optional

from vllm.logger import init_logger
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest, FinishReason
from vllm.v1.holekv.trace_registry import (
    HoleKVTraceRecord,
    HoleKVTraceRegistry,
    get_global_registry,
)
from vllm.v1.holekv.inline_parser import HoleKVInlineParser, HoleKVParsedPrompt
from vllm.v1.holekv.trace_aligner import HoleKVTraceAligner, HoleKVAlignmentResult
from vllm.v1.holekv.view_builder import HoleKVViewBuilder, HoleKVActiveView
from vllm.v1.holekv.position_map import HoleKVPositionMap
from vllm.v1.holekv.attention_metadata import HoleKVAttentionMetadata
from vllm.v1.request import Request

logger = init_logger(__name__)


class HoleKVEngineProcessor:
    """Processes HoleKV logic for every request.

    Manages trace storage/retrieval, marker parsing, alignment,
    view building, and cache ID assignment.

    Integrated into the engine core's pre-add-request and output processing.
    """

    def __init__(
        self,
        trace_registry: Optional[HoleKVTraceRegistry] = None,
        enabled: bool = True,
    ):
        self.trace_registry = trace_registry or get_global_registry()
        self.parser = HoleKVInlineParser()
        self.aligner = HoleKVTraceAligner()
        self.view_builder = HoleKVViewBuilder()
        self.enabled = enabled

    def generate_cache_id(self) -> str:
        """Generate a unique cache ID."""
        return self.trace_registry.generate_id()

    def assign_cache_id(self, request: Request) -> None:
        """Assign a unique holekv_cache_id to every request.

        Called before preprocess_request so that the cache_id is always set,
        even if HoleKV processing is disabled or falls back.
        The client receives this id in every response for trace lookups.
        """
        if not self.enabled:
            return
        request.holekv_cache_id = self.generate_cache_id()

    def preprocess_request(self, request: Request) -> Request:
        """Process HoleKV markers and alignment for a request.

        Called during preprocess_add_request, before the request enters
        the scheduler.

        Args:
            request: The vLLM Request object.

        Returns:
            The same request, possibly modified with HoleKV metadata.
        """
        if not self.enabled:
            return request

        holekv_ref = request.holekv_ref

        # prompt_text on Request is lost during IPC serialisation
        # (the EngineCoreRequest has the field but downstream code may
        # not propagate it).  Prefer trace_headers which reliably survive.
        prompt_text = None
        th = getattr(request, "trace_headers", None) or {}
        if "_holekv_original_prompt" in th:
            prompt_text = th["_holekv_original_prompt"]
            logger.debug(
                "HoleKV: using original prompt from trace_headers (%d chars)",
                len(prompt_text),
            )

        if prompt_text is None:
            prompt_text = getattr(request, "prompt_text", None)
        if prompt_text is None:
            # Last resort: raw prompt string (this is the full rendered
            # text with chat template, NOT the original marked prompt).
            prompt_text = getattr(request, "prompt", None)
        if prompt_text is None:
            return request

        # Parse inline markers
        parsed = self.parser.parse(prompt_text)

        if not parsed.has_markers:
            # No markers — request is just a normal prompt
            # But we still store the parsed prompt for trace creation later
            request._holekv_parsed = parsed
            return request

        if holekv_ref is None:
            # No-ref fallback: compact the prompt
            compact = parsed.to_compact_prompt()
            # Store the compacted prompt info
            request._holekv_parsed = parsed
            request._holekv_compact_prompt = compact
            request._holekv_mode = "fallback"
            logger.debug(
                "HoleKV fallback mode for request %s: compacted prompt from %d to %d chars",
                request.request_id,
                len(prompt_text),
                len(compact),
            )
            return request

        # HoleKV ref mode: look up trace and build active view
        trace = self.trace_registry.lookup(
            holekv_ref,
            owner_id=request.holekv_owner_id,
            session_id=request.holekv_session_id,
        )

        if trace is None:
            # Trace not found or access denied → fall back
            logger.warning(
                "HoleKV: trace %s not found for request %s, falling back to compact",
                holekv_ref,
                request.request_id,
            )
            compact = parsed.to_compact_prompt()
            request._holekv_parsed = parsed
            request._holekv_compact_prompt = compact
            request._holekv_mode = "fallback"
            request._holekv_ref_failed = True
            return request

        # Align spans against trace
        # Note: actual token IDs come from the processed prompt
        # For now, we store parsed info and trace ref for scheduler processing
        request._holekv_parsed = parsed
        request._holekv_trace = trace
        request._holekv_mode = "ref"

        logger.info(
            "HoleKV ref mode for request %s: using trace %s (%d tokens, %d blocks)",
            request.request_id,
            holekv_ref,
            trace.total_tokens,
            trace.num_kv_blocks,
        )

        return request

    def process_alignment(
        self,
        request: Request,
        compact_token_ids: list[int],
        compact_byte_offsets: list[tuple[int, int]],
    ) -> Optional[HoleKVAlignmentResult]:
        """Called after tokenization to perform alignment against the trace.

        Returns:
            HoleKVAlignmentResult on success, None on failure.
        """
        if not self.enabled:
            return None

        holekv_ref = request.holekv_ref
        parsed = getattr(request, "_holekv_parsed", None)
        trace = getattr(request, "_holekv_trace", None)

        if holekv_ref is None or trace is None or parsed is None:
            return None

        result = self.aligner.align(
            parsed, trace, compact_token_ids, compact_byte_offsets
        )

        if not result.success:
            logger.warning(
                "HoleKV alignment failed for request %s: %s",
                request.request_id,
                result.error_message,
            )
            request._holekv_alignment_failed = True
            return None

        request._holekv_alignment = result
        return result

    def build_active_view(
        self,
        request: Request,
        parsed: HoleKVParsedPrompt,
        alignment: HoleKVAlignmentResult,
        trace: HoleKVTraceRecord,
        compact_token_ids: list[int],
        d_token_ids: list[int],
        block_size: int,
    ) -> Optional[HoleKVActiveView]:
        """Build the hole-preserved active view.

        Returns None if view building fails.
        """
        view = self.view_builder.build(
            parsed, alignment, trace, compact_token_ids, d_token_ids, block_size
        )

        if not view.is_valid:
            logger.warning(
                "HoleKV view building failed for request %s: %s",
                request.request_id,
                view.error_message,
            )
            return None

        return view

    def create_trace_record(
        self,
        request_id: str = "",
        prompt_text: str = "",
        block_ids: list[int] | None = None,
        num_prompt_tokens: int = 0,
        output_token_ids: list[int] | None = None,
        prompt_token_ids: list[int] | None = None,
        prompt_token_byte_offsets: list[int] | None = None,
        cache_id: str | None = None,
        full_decoded_text: str | None = None,
    ) -> Optional[HoleKVTraceRecord]:
        """Create and store a HoleKV trace record for a completed request.

        Called by the scheduler after request completion.

        Args:
            cache_id: If provided, use this as the trace's cache_id.
                      If None, generate a new ID. Should match
                      request.holekv_cache_id for proper ref lookup.
            full_decoded_text: The full rendered chat template text
                      with byte offsets matching its UTF-8 encoding.
                      Composed in async_llm via progressive tokenizer.decode.

        Returns:
            The created trace record with its cache_id, or None if disabled.
        """
        if not self.enabled:
            return None

        # Use the provided cache_id (should match request.holekv_cache_id
        # that was sent to the client) or generate a new one.
        if cache_id is None:
            cache_id = self.trace_registry.generate_id()
        blk_ids = block_ids or []
        out_ids = output_token_ids or []
        pt_ids = prompt_token_ids or []

        # Build token/block metadata from what the scheduler provides.
        num_blocks = len(blk_ids)
        block_size = 0
        if num_blocks > 0 and num_prompt_tokens:
            block_size = num_prompt_tokens // num_blocks

        # Choose the text to store in the trace.
        # If we have full_decoded_text (the full rendered chat template
        # from progressive tokenizer.decode), use it so byte offsets
        # match the stored UTF-8 exactly.
        # Fall back to the compact (marker-stripped) text otherwise.
        if full_decoded_text:
            rendered_text = full_decoded_text
            rendered_utf8 = full_decoded_text.encode("utf-8", errors="replace")
            logger.info(
                "HoleKV: using full_decoded_text (%d bytes) for trace %s",
                len(full_decoded_text), cache_id,
            )
        elif prompt_text:
            parsed = self.parser.parse(prompt_text)
            rendered_text = parsed.to_compact_prompt() if parsed.has_markers else prompt_text
            rendered_utf8 = rendered_text.encode("utf-8", errors="replace")
            logger.warning(
                "HoleKV: full_decoded_text NOT available for trace %s, "
                "using compact text (%d bytes) — byte offsets may mismatch",
                cache_id, len(rendered_text),
            )
        else:
            rendered_text = ""
            rendered_utf8 = b""
            logger.warning(
                "HoleKV: no text available for trace %s", cache_id,
            )

        # Compute per-block token/position ranges from prompt tokens.
        # The scheduler's block_size = num_prompt_tokens // num_blocks.
        _t_ranges: list[tuple[int, int]] = []
        _p_ranges: list[tuple[int, int]] = []
        if num_blocks > 0 and num_prompt_tokens > 0 and block_size > 0:
            for bi in range(num_blocks):
                b_start = bi * block_size
                b_end = min(b_start + block_size, num_prompt_tokens)
                if b_end > b_start:
                    _t_ranges.append((b_start, b_end))
                    _p_ranges.append((b_start, b_end))

        record = HoleKVTraceRecord(
            cache_id=cache_id,
            request_id=request_id,
            rendered_prompt_text=rendered_text,
            rendered_prompt_utf8=rendered_utf8,
            token_ids=list(pt_ids),
            token_byte_offsets=_flat_to_pairs(prompt_token_byte_offsets),
            block_size=block_size,
            block_ids_by_group=[blk_ids] if blk_ids else [],
            token_ranges_by_block=_t_ranges,
            position_ranges_by_block=_p_ranges,
            model_fingerprint="",
            tokenizer_fingerprint="",
            lora_id=None,
            dtype="",
        )

        self.trace_registry.store(record)

        logger.info(
            "HoleKV: stored trace %s for request %s (%d tokens, %d blocks)",
            cache_id,
            request_id,
            num_prompt_tokens,
            num_blocks,
        )

        return record

    def get_cache_id_for_output(
        self,
        request: Request,
    ) -> Optional[str]:
        """Get the holekv_cache_id to include in the output.

        Returns the cached trace ID if available, or None.
        """
        trace = getattr(request, "_holekv_stored_trace", None)
        if trace is not None:
            return trace.cache_id
        return None

    @staticmethod
    def build_position_map(
        a_tokens: int,
        hole_pairs: list[tuple[int, int, int]],
        d_tokens: int,
    ) -> HoleKVPositionMap:
        """Build a position map. Convenience wrapper."""
        return HoleKVPositionMap.build_hole_preserved(a_tokens, hole_pairs, d_tokens)


def _flat_to_pairs(flat_offsets: list[int] | None) -> list[tuple[int, int]]:
    """Convert flat byte offset list [s0, e0, s1, e1, ...] to [(s0, e0), (s1, e1), ...].

    Returns empty list if input is None or empty.
    """
    if not flat_offsets:
        return []
    pairs = []
    for i in range(0, len(flat_offsets) - 1, 2):
        pairs.append((flat_offsets[i], flat_offsets[i + 1]))
    return pairs
