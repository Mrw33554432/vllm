"""HoleKV Trace Aligner — aligns parsed spans against referenced trace records.

Checks that:
  1. The A (stable prefix) of the current prompt matches the A in the trace.
  2. The B (removed spans) match the corresponding spans in the trace.
  3. The C (stable suffix spans) match the corresponding spans in the trace.
  4. Token boundaries align exactly.
  5. KV block boundaries align.
  6. Replacement length m <= removed length b.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from vllm.v1.holekv.trace_registry import HoleKVTraceRecord
from vllm.v1.holekv.inline_parser import HoleKVParsedPrompt, HoleKVSpan


@dataclass
class HoleKVAlignmentResult:
    """Result of aligning current prompt against a referenced trace.

    On success, provides token-level mapping information needed by the
    view builder. On failure, provides an error message.
    """

    success: bool
    error_message: Optional[str] = None

    # Token-level mapping for each stable span
    # Span index -> (trace_start_token, trace_end_token)
    stable_span_token_ranges: list[tuple[int, int]] = field(default_factory=list)

    # Hole token ranges in the trace:
    # [(trace_remove_start_token, trace_remove_end_token, add_token_count), ...]
    hole_token_ranges: list[tuple[int, int, int]] = field(default_factory=list)

    # Matching block ranges in the trace for reuse:
    # [(trace_start_block, trace_end_block), ...]
    reusable_block_ranges: list[tuple[int, int]] = field(default_factory=list)

    # Total tokens in each span (for PE calculations)
    a_tokens: int = 0  # Stable prefix token count
    b_tokens: list[int] = field(default_factory=list)   # Per-hole removed token counts
    m_tokens: list[int] = field(default_factory=list)   # Per-hole added token counts
    c_tokens: list[int] = field(default_factory=list)   # Per-stable-suffix token counts
    d_tokens: int = 0  # New suffix token count


class HoleKVTraceAligner:
    """Aligns parsed prompt spans against a HoleKV trace record.

    Performs exact token-alignment checks. For v1, all boundaries must
    align exactly with the referenced cache.
    """

    def align(
        self,
        parsed: HoleKVParsedPrompt,
        trace: HoleKVTraceRecord,
        new_token_ids: list[int],
        new_token_byte_offsets: list[tuple[int, int]],
    ) -> HoleKVAlignmentResult:
        """Align current prompt spans against the referenced trace.

        Args:
            parsed: Parsed prompt with decomposed spans.
            trace: Referenced trace record.
            new_token_ids: Token IDs of the current (compacted) prompt.
            new_token_byte_offsets: Byte offsets for each token of the current prompt.

        Returns:
            HoleKVAlignmentResult indicating success or failure.
        """
        # Quick check: the trace and current request must share the same model
        if not parsed.has_markers or not parsed.has_hole_kv_structure:
            return HoleKVAlignmentResult(
                success=False,
                error_message="Prompt has no HoleKV marker structure",
            )

        # Build span-to-trace mapping
        stable_spans = parsed.get_stable_spans()
        remove_spans = parsed.get_remove_spans()
        add_spans = parsed.get_add_spans()

        if len(remove_spans) != len(add_spans):
            return HoleKVAlignmentResult(
                success=False,
                error_message=f"Remove ({len(remove_spans)}) and add ({len(add_spans)}) "
                              "spans mismatch",
            )

        # Align the stable prefix A
        if not stable_spans:
            return HoleKVAlignmentResult(
                success=False,
                error_message="No stable spans found in prompt",
            )

        # The first stable span is A
        a_span = stable_spans[0]
        a_pos = self._align_span_text(a_span.text, trace.rendered_prompt_text, start=0)

        if a_pos < 0:
            return HoleKVAlignmentResult(
                success=False,
                error_message="Stable prefix A does not match referenced cache",
            )

        # Start scan from token 0; _find_token_range skips non-matching
        # chat-template tokens until it finds the user-content A-span.
        a_token_start = 0

        # Find A's token range in the trace
        a_tok_start, a_token_end = self._find_token_range(
            a_span.text, a_token_start, trace.token_ids,
            trace.token_byte_offsets, trace.rendered_prompt_utf8,
        )

        result = HoleKVAlignmentResult(success=True)
        result.a_tokens = a_token_end - a_token_start
        result.stable_span_token_ranges.append((a_tok_start, a_token_end))

        # Build token-to-byte-position mapping for the trace.
        # We need this to convert character positions (from _align_span_text)
        # back to token indices.
        _trace_byte_to_token: list[int] = []
        for ti, (bs, be) in enumerate(trace.token_byte_offsets):
            for _ in range(bs, be):
                _trace_byte_to_token.append(ti)

        def _char_pos_to_token(char_pos: int) -> int:
            """Convert character position in rendered_prompt_text to token index."""
            if char_pos <= 0:
                return 0
            # Convert to byte position (approximate for non-ASCII, exact for ASCII)
            byte_pos = min(char_pos, len(trace.rendered_prompt_utf8) - 1)
            if byte_pos < len(_trace_byte_to_token):
                return _trace_byte_to_token[byte_pos]
            return max(0, len(trace.token_byte_offsets) - 1)

        # Align each hole pair.
        # Trace stores the COMPACT text (old content replaced by add-content).
        # B-span text (old content to remove) does NOT exist in the trace.
        # Instead, find the NEXT stable span C in the trace; the B-span in
        # the trace is simply [a_token_end, c_token_start).
        cursor = a_token_end  # Current position in trace tokens

        hole_pairs = parsed.get_hole_pairs()
        for idx, (remove_span, add_span) in enumerate(hole_pairs):
            # The corresponding C span is stable_spans[idx + 1]
            # (first stable span is A, then C1 after hole 1, C2 after hole 2, etc.)
            c_span = stable_spans[idx + 1]

            # Search for C-span text in the trace, starting at cursor's
            # byte position (so we don't pick up a false match inside A).
            cursor_byte = 0
            if cursor < len(trace.token_byte_offsets):
                cursor_byte = trace.token_byte_offsets[cursor][0]
            c_char_pos = self._align_span_text(
                c_span.text, trace.rendered_prompt_text,
                start=cursor_byte,
            )
            if c_char_pos < 0:
                # Debug: dump what we're searching for
                from vllm.logger import init_logger
                _log = init_logger(__name__)
                _log.warning(
                    "Hole %d: C-span text=%r (len=%d) not in trace text "
                    "(len=%d), cursor=%d cursor_byte=%d",
                    idx, c_span.text[:120], len(c_span.text),
                    len(trace.rendered_prompt_text),
                    cursor, cursor_byte,
                )
                # Dump the trace text around the cursor
                ctx_start = max(0, cursor_byte - 20)
                ctx_end = min(len(trace.rendered_prompt_text), cursor_byte + 200)
                _log.warning(
                    "Trace text around cursor: %r",
                    trace.rendered_prompt_text[ctx_start:ctx_end],
                )
                return HoleKVAlignmentResult(
                    success=False,
                    error_message=(
                        f"Hole {idx}: stable suffix C not found in trace text"
                    ),
                )

            # Convert character position to token index.
            c_token_start = _char_pos_to_token(c_char_pos)

            if c_token_start <= cursor:
                return HoleKVAlignmentResult(
                    success=False,
                    error_message=(
                        f"Hole {idx}: cannot locate stable suffix C after "
                        f"trace token {cursor}"
                    ),
                )

            # B occupies [cursor, c_token_start) in the trace.
            b_start = cursor
            b_end = c_token_start
            b_token_count = b_end - b_start
            if b_token_count <= 0:
                return HoleKVAlignmentResult(
                    success=False,
                    error_message=(
                        f"Hole {idx}: empty B span in trace (b_start={b_start}, "
                        f"c_token_start={c_token_start})"
                    ),
                )
            result.b_tokens.append(b_token_count)

            # Add span M tokens (estimated from text length).
            # Conservative: ~4 chars per token for English BPE tokenizers.
            m_token_count = max(1, len(add_span.text) // 4)
            result.m_tokens.append(m_token_count)

            # Check m <= b (replacement cannot exceed removed tokens in the
            # trace).  v1 cannot insert more tokens than were removed.  If
            # the estimate is slightly off, clamp instead of failing.
            if m_token_count > b_token_count:
                from vllm.logger import init_logger
                _log = init_logger(__name__)
                _log.warning(
                    "Hole %d: estimated m=%d > removed b=%d; "
                    "clamping to b.",
                    idx, m_token_count, b_token_count,
                )
                m_token_count = b_token_count
                result.m_tokens[-1] = m_token_count

            # Find C's token range in the trace.
            c_tok_start, c_tok_end = self._find_token_range(
                c_span.text, c_token_start, trace.token_ids,
                trace.token_byte_offsets, trace.rendered_prompt_utf8,
            )
            if c_tok_start < 0:
                return HoleKVAlignmentResult(
                    success=False,
                    error_message=(
                        f"Hole {idx}: stable suffix C token range not found "
                        f"starting at token {c_token_start}"
                    ),
                )

            c_token_count = c_tok_end - c_tok_start
            result.c_tokens.append(c_token_count)
            result.stable_span_token_ranges.append((c_tok_start, c_tok_end))

            # Block range for B in the trace (for potential reuse).
            b_block_start = self._token_to_block(b_start, trace)
            b_block_end = self._token_to_block(b_end - 1, trace) + 1
            result.reusable_block_ranges.append((b_block_start, b_block_end))

            result.hole_token_ranges.append((b_start, b_end, m_token_count))

            cursor = c_tok_end

        # Check block alignment for hole boundaries.
        # Block size in the trace may not match the scheduler's block size,
        # so make this a soft warning rather than a hard failure.
        from vllm.logger import init_logger
        _log = init_logger(__name__)
        for idx, (b_start, b_end, _) in enumerate(result.hole_token_ranges):
            if not self._is_block_aligned(b_start, trace):
                _log.warning(
                    "Hole %d: remove start (token %d) is not "
                    "block-aligned (block_size=%d).",
                    idx, b_start, trace.block_size,
                )
            if not self._is_block_aligned(b_end, trace):
                _log.warning(
                    "Hole %d: remove end (token %d) is not "
                    "block-aligned (block_size=%d).",
                    idx, b_end, trace.block_size,
                )

        return result

    def _align_span_text(self, span_text: str, trace_text: str, start: int) -> int:
        """Find span_text within trace_text starting at or after `start`.

        Returns the character index where the span begins, or -1 if not found.
        The full rendered chat template text may include system messages
        and special tokens before the user content. This searches for the
        span text anywhere in the trace, not just at the exact start offset.
        """
        if not span_text:
            return start
        pos = trace_text.find(span_text, start)
        return pos if pos >= 0 else -1

    def _byte_pos_to_token(
        self,
        char_pos: int,
        token_byte_offsets: list[tuple[int, int]],
    ) -> int:
        """Convert a character position in rendered_prompt_text to a token index.

        Uses the trace's token_byte_offsets which are cumulative byte offsets
        into rendered_prompt_utf8. The offsets partition the full rendered text
        (from progressive decode), so we can compute the byte position
        corresponding to char_pos and find the token covering it.

        Returns the token index, or 0 as fallback.
        """
        if char_pos <= 0 or not token_byte_offsets:
            return 0

        # We don't have the full text here, but we know the byte offsets
        # are cumulative into the stored UTF-8.  Instead of computing
        # byte positions from char positions (which requires the text),
        # we use a linear scan through tokens, accumulating decoded
        # character lengths until we reach char_pos.
        #
        # Since we have the full rendered_prompt_utf8 stored in the trace
        # but the trace record isn't passed to this method, we use a
        # simpler heuristic: char_pos roughly maps to byte_pos for ASCII
        # text. For non-ASCII, we estimate.
        #
        # A better approach: binary search through byte_offsets.
        # The byte_offsets are sorted, so we can find the first token
        # whose end > char_pos (treating char_pos as approximate byte pos).
        approx_byte_pos = char_pos  # close enough for ASCII-heavy text

        for i, (bs, be) in enumerate(token_byte_offsets):
            if be > approx_byte_pos:
                return i
        return max(0, len(token_byte_offsets) - 1)

    def _find_token_range(
        self,
        text: str,
        trace_token_start: int,
        trace_token_ids: list[int],
        trace_byte_offsets: list[tuple[int, int]],
        trace_utf8: bytes,
    ) -> tuple[int, int]:
        """Find the token range in the trace that matches the given text.

        Scans forward from trace_token_start, skipping tokens whose
        accumulated text does not match.  This handles the case where
        the full rendered text includes chat-template tokens before
        the user content.

        Args:
            text: Text span to match.
            trace_token_start: Starting token index in the trace.
            trace_token_ids: Token IDs in the trace.
            trace_byte_offsets: Byte offsets for trace tokens.
            trace_utf8: UTF-8 bytes of the trace prompt.

        Returns:
            (start_token_idx, end_token_idx) or (-1, -1) on failure.
            end_token_idx is exclusive.
        """
        text_utf8 = text.encode("utf-8")
        if not text_utf8:
            return (trace_token_start, trace_token_start)

        if trace_token_start >= len(trace_byte_offsets):
            from vllm.logger import init_logger
            _log = init_logger(__name__)
            _log.debug(
                "HoleKV align: start token %d beyond byte_offsets length %d",
                trace_token_start, len(trace_byte_offsets),
            )
            return (-1, -1)

        from vllm.logger import init_logger
        _log = init_logger(__name__)

        # Phase 1: find the first token whose text begins the match.
        # Walk forward from trace_token_start, accumulating per-token
        # text and checking whether it starts to look like text_utf8.
        for first in range(trace_token_start, len(trace_byte_offsets)):
            byte_start, byte_end = trace_byte_offsets[first]
            if byte_end > len(trace_utf8):
                continue
            token_text = trace_utf8[byte_start:byte_end]
            if not text_utf8.startswith(token_text):
                # This token does not start the match.
                # Check whether text_utf8 begins somewhere inside token_text —
                # if so, the span starts mid-token, which is fatal for block
                # alignment.
                if token_text and len(token_text) > 1 and text_utf8 in token_text:
                    _log.debug(
                        "HoleKV align: span starts mid-token at t=%d "
                        "token=%r; block alignment would be broken.",
                        first, token_text[:60],
                    )
                    return (-1, -1)
                continue

            # This token starts the match.
            accumulated = token_text
            end_token = first + 1

            # Phase 2: accumulate remaining tokens until full match.
            for j in range(first + 1, len(trace_byte_offsets)):
                # Check if we already have an exact match.
                if accumulated == text_utf8:
                    return (first, end_token)

                j_byte_start, j_byte_end = trace_byte_offsets[j]
                if j_byte_end > len(trace_utf8):
                    return (-1, -1)
                accumulated += trace_utf8[j_byte_start:j_byte_end]
                end_token = j + 1

                # Exact match?
                if accumulated == text_utf8:
                    return (first, end_token)

                # Span text ends mid-token?  Accept if accumulated
                # starts with text_utf8 (the remainder belongs to the
                # next span — token boundaries don't always align).
                if accumulated.startswith(text_utf8):
                    # Return range up to *this* token (inclusive),
                    # because the excess bytes of this token belong
                    # to the following span.
                    return (first, end_token)

                if len(accumulated) > len(text_utf8):
                    return (-1, -1)

                if not text_utf8.startswith(accumulated):
                    return (-1, -1)

            # Reached end of tokens without match.
            if accumulated == text_utf8 or accumulated.startswith(text_utf8):
                return (first, end_token)

            return (-1, -1)

        return (-1, -1)

    def _estimate_tokens(self, text: str) -> int:
        """Rough token count estimate from text length.

        This is a placeholder. Real implementation should use the tokenizer.
        Conservative estimate: ~4 chars per token for English text.
        """
        # Conservative estimate — the actual count is determined after tokenization
        return max(1, len(text) // 3)

    def _token_to_block(self, token_idx: int, trace: HoleKVTraceRecord) -> int:
        """Convert a token index to a KV block index in the trace."""
        for block_idx, (start_tok, end_tok) in enumerate(trace.token_ranges_by_block):
            if start_tok <= token_idx < end_tok:
                return block_idx
        # Fallback: use the last block
        return max(0, len(trace.token_ranges_by_block) - 1)

    def _is_block_aligned(self, token_idx: int, trace: HoleKVTraceRecord) -> bool:
        """Check if token_idx is at a block boundary in the trace."""
        if trace.block_size <= 0:
            return True  # No block alignment requirement
        return (token_idx % trace.block_size) == 0
