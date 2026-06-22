"""HoleKV View Builder — builds a hole-preserved active KV view.

Given a parsed prompt and alignment result against a trace, produces
the active view: A M [hole] C_old D_hole → E_hole.

The active view describes which KV blocks to reuse from the trace (C_old),
which to compute fresh (A, M, D_hole), and where holes are placed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from vllm.v1.holekv.trace_registry import HoleKVTraceRecord
from vllm.v1.holekv.inline_parser import HoleKVParsedPrompt
from vllm.v1.holekv.trace_aligner import HoleKVAlignmentResult


@dataclass
class HoleKVBlockAssignment:
    """Describes what to do with a single KV block in the active view.

    Attributes:
        block_index: Logical block index in the active view.
        source: "compute", "reuse", or "hole".
        physical_block_id: Physical block ID (for reuse) or -1 (for compute/hole).
        trace_block_index: Index in the trace for reused blocks.
        start_position: Starting position encoding for this block.
        num_tokens: Number of tokens occupying this block.
    """

    block_index: int
    source: str  # "compute", "reuse", "hole"
    physical_block_id: int = -1
    trace_block_index: int = -1
    start_position: int = 0
    num_tokens: int = 0


@dataclass
class HoleKVActiveView:
    """Complete active view for HoleKV execution.

    Describes the full layout of the active KV cache:
    - Which blocks to compute fresh (A, M, D_hole)
    - Which blocks to reuse from the trace (C_old spans)
    - Where holes are (preserved positions, no content)
    - Position encoding map for all tokens
    """

    # Block assignments in order
    block_assignments: list[HoleKVBlockAssignment] = field(default_factory=list)

    # Position encoding: token_index -> position
    position_map: list[int] = field(default_factory=list)

    # Total token count in the active view
    total_tokens: int = 0

    # Token ranges for each section
    a_range: tuple[int, int] = (0, 0)      # (start, end) in active tokens
    m_ranges: list[tuple[int, int]] = field(default_factory=list)  # per hole
    hole_ranges: list[tuple[int, int]] = field(default_factory=list)  # per hole
    c_old_ranges: list[tuple[int, int]] = field(default_factory=list)  # per stable suffix
    d_hole_range: tuple[int, int] = (0, 0)

    # Blocks to compute (fresh KV)
    compute_blocks: list[int] = field(default_factory=list)

    # Blocks to reuse from trace
    reuse_block_mapping: dict[int, int] = field(default_factory=dict)
    # active_block_index -> trace_physical_block_id

    # Blocks that are holes (skip in attention)
    hole_blocks: list[int] = field(default_factory=list)

    # Is this view valid?
    is_valid: bool = False
    error_message: Optional[str] = None

    @property
    def A_length(self) -> int:
        return self.a_range[1] - self.a_range[0]

    @property
    def M_length(self) -> int:
        return sum(e - s for s, e in self.m_ranges)

    @property
    def C_old_length(self) -> int:
        return sum(e - s for s, e in self.c_old_ranges)

    @property
    def D_hole_length(self) -> int:
        return self.d_hole_range[1] - self.d_hole_range[0]

    @property
    def hole_length(self) -> int:
        return sum(e - s for s, e in self.hole_ranges)


class HoleKVViewBuilder:
    """Builds a HoleKV active view from alignment results and a trace record.

    The active view layout:
        A M [hole] C_old D_hole

    Position encoding (hole-preserved):
        A[0:a)
        M[a:a+m)
        HOLE[a+m:a+b)
        C_old[a+b:a+b+c)
        D_hole[a+b+c:a+b+c+d)
    """

    def build(
        self,
        parsed: HoleKVParsedPrompt,
        alignment: HoleKVAlignmentResult,
        trace: HoleKVTraceRecord,
        compact_token_ids: list[int],
        d_token_ids: list[int],
        block_size: int,
    ) -> HoleKVActiveView:
        """Build the active view.

        Args:
            parsed: Parsed prompt with spans.
            alignment: Successful alignment result.
            trace: Referenced trace record.
            compact_token_ids: Token IDs of the compacted prompt (A M C).
            d_token_ids: Token IDs of the new suffix D.
            block_size: KV cache block size.

        Returns:
            HoleKVActiveView describing the hole-preserved layout.
        """
        if not alignment.success:
            return HoleKVActiveView(
                is_valid=False,
                error_message=alignment.error_message or "Alignment failed",
            )

        view = HoleKVActiveView(is_valid=True)

        # Build position map
        position = 0
        token_idx = 0
        block_idx = 0

        # ---- A: stable prefix ----
        a_start_tok = 0
        a_end_tok = alignment.a_tokens

        for i in range(a_start_tok, a_end_tok):
            view.position_map.append(position)
            position += 1
            token_idx += 1

        view.a_range = (0, token_idx)

        # Allocate blocks for A
        a_blocks = self._tokens_to_blocks(0, alignment.a_tokens, block_size)
        for b in range(a_blocks):
            view.block_assignments.append(HoleKVBlockAssignment(
                block_index=block_idx,
                source="compute",
                start_position=b * block_size,
                num_tokens=min(block_size, alignment.a_tokens - b * block_size),
            ))
            view.compute_blocks.append(block_idx)
            block_idx += 1

        # ---- M and holes for each pair ----
        for hole_idx in range(len(alignment.b_tokens)):
            # M: replacement span
            m_tokens = alignment.m_tokens[hole_idx]
            m_token_start = a_end_tok + sum(alignment.m_tokens[:hole_idx])
            m_token_end = m_token_start + m_tokens

            for i in range(m_tokens):
                view.position_map.append(position)
                position += 1
                token_idx += 1

            m_blocks = self._tokens_to_blocks(token_idx - m_tokens - a_end_tok, m_tokens, block_size)
            # Simplified: allocate M blocks contiguously
            m_start_block = block_idx
            for b in range(m_blocks):
                view.block_assignments.append(HoleKVBlockAssignment(
                    block_index=block_idx,
                    source="compute",
                    start_position=a_end_tok + b * block_size,
                    num_tokens=min(block_size, m_tokens - b * block_size),
                ))
                view.compute_blocks.append(block_idx)
                block_idx += 1

            view.m_ranges.append((token_idx - m_tokens, token_idx))

            # HOLE: preserved positions with no content
            b_tokens = alignment.b_tokens[hole_idx]
            hole_start_pos = position
            hole_end_pos = position + (b_tokens - m_tokens)

            # Hole positions are skipped in the actual KV but preserved in PE
            hole_blocks_needed = self._tokens_to_blocks(0, b_tokens - m_tokens, block_size)
            for b in range(hole_blocks_needed):
                view.block_assignments.append(HoleKVBlockAssignment(
                    block_index=block_idx,
                    source="hole",
                    start_position=hole_start_pos + b * block_size,
                    num_tokens=min(block_size, (b_tokens - m_tokens) - b * block_size),
                ))
                view.hole_blocks.append(block_idx)
                block_idx += 1

            # Advance position past the hole
            position += (b_tokens - m_tokens)
            token_idx += (b_tokens - m_tokens)

            view.hole_ranges.append((token_idx - (b_tokens - m_tokens), token_idx))

            # ---- C_old: reuse stable suffix from trace ----
            if hole_idx < len(alignment.c_tokens):
                c_tokens = alignment.c_tokens[hole_idx]
                # stable_span_token_ranges[0] is A; [1], [2], ... are C spans.
                c_trace_start, c_trace_end = alignment.stable_span_token_ranges[hole_idx + 1]

                # Reuse blocks from the trace
                c_trace_start_block = self._token_to_block(c_trace_start, trace)
                c_trace_end_block = self._token_to_block(c_trace_end - 1, trace) + 1

                # Map trace blocks to active view blocks
                for tb in range(c_trace_start_block, c_trace_end_block):
                    trace_block_id = self._get_physical_block(trace, tb)
                    view.block_assignments.append(HoleKVBlockAssignment(
                        block_index=block_idx,
                        source="reuse",
                        physical_block_id=trace_block_id,
                        trace_block_index=tb,
                        start_position=position,
                        num_tokens=min(block_size, c_tokens - (tb - c_trace_start_block) * block_size),
                    ))
                    view.reuse_block_mapping[block_idx] = trace_block_id
                    block_idx += 1

                # Position encoding: C_old keeps original PE from the trace
                for i in range(c_tokens):
                    orig_pos = trace.position_ranges_by_block[c_trace_start_block][0] + i
                    view.position_map.append(orig_pos)
                    position = orig_pos + 1
                    token_idx += 1

                view.c_old_ranges.append((token_idx - c_tokens, token_idx))
            else:
                # No stable suffix after this hole
                pass

        # ---- D_hole: new suffix ----
        if d_token_ids:
            d_start = token_idx
            for i, _ in enumerate(d_token_ids):
                # D_hole starts after C_old's original PE
                view.position_map.append(position)
                position += 1
                token_idx += 1

            d_blocks = self._tokens_to_blocks(0, len(d_token_ids), block_size)
            for b in range(d_blocks):
                view.block_assignments.append(HoleKVBlockAssignment(
                    block_index=block_idx,
                    source="compute",
                    start_position=d_start + b * block_size,
                    num_tokens=min(block_size, len(d_token_ids) - b * block_size),
                ))
                view.compute_blocks.append(block_idx)
                block_idx += 1

            view.d_hole_range = (d_start, token_idx)

        view.total_tokens = token_idx

        return view

    @staticmethod
    def _tokens_to_blocks(start_token: int, num_tokens: int, block_size: int) -> int:
        """Number of blocks needed for a token range."""
        if num_tokens == 0:
            return 0
        end_token = start_token + num_tokens
        start_block = start_token // block_size
        end_block = (end_token + block_size - 1) // block_size
        return end_block - start_block

    @staticmethod
    def _token_to_block(token_idx: int, trace: HoleKVTraceRecord) -> int:
        """Convert a token index to a KV block index in the trace."""
        for bi, (s, e) in enumerate(trace.token_ranges_by_block):
            if s <= token_idx < e:
                return bi
        return max(0, len(trace.token_ranges_by_block) - 1)

    @staticmethod
    def _get_physical_block(trace: HoleKVTraceRecord, block_idx: int) -> int:
        """Get the physical block ID for a trace block index (uses first group)."""
        if trace.block_ids_by_group:
            group = trace.block_ids_by_group[0]
            if block_idx < len(group):
                return group[block_idx]
        return -1
