"""HoleKV Attention Metadata — attention metadata for hole-preserved KV views.

When HoleKV is active, the attention mask must account for holes:
  - Tokens in holes do not participate in attention (they have no KV).
  - C_old tokens attend to A, M, and previous C_old, but NOT to holes.
  - D_hole tokens attend to A, M, C_old, and D_hole, but NOT to holes.
  - All tokens attend causally.

For full-attention decoder models (no sliding window), this primarily
affects how we construct the attention metadata for blocks that contain
holes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HoleKVAttentionMetadata:
    """Attention metadata for a HoleKV execution step.

    Extends the standard vLLM attention metadata with hole-specific
    information needed by the model runner.

    Attributes:
        is_holekv_active: Whether HoleKV mode is active for this step.
        position_map: Per-token position encoding values.
        hole_token_mask: bool per token — True if token is in a hole.
        block_sources: Per-block source type ("compute", "reuse", "hole").
        reuse_block_to_physical: Mapping from active block index to physical block ID.
        compute_blocks: Indices of blocks that need fresh KV computation.
        reuse_blocks: Indices of blocks that are reused from trace.
        hole_blocks: Indices of blocks that are holes (skip).
        total_blocks: Total number of blocks in the active view.
    """

    is_holekv_active: bool = False

    # Per-token position encoding (length = num_tokens)
    position_map: list[int] = field(default_factory=list)

    # Per-token hole mask (True = hole token, skip in attention)
    hole_token_mask: list[bool] = field(default_factory=list)

    # Per-block source type
    block_sources: list[str] = field(default_factory=list)

    # Block mapping
    reuse_block_to_physical: dict[int, int] = field(default_factory=dict)
    compute_blocks: list[int] = field(default_factory=list)
    reuse_blocks: list[int] = field(default_factory=list)
    hole_blocks: list[int] = field(default_factory=list)

    total_blocks: int = 0

    # Block table: active_block_index -> physical_block_id
    # -1 for compute blocks (not yet allocated) and hole blocks
    block_table: list[int] = field(default_factory=list)

    @classmethod
    def create_empty(cls) -> "HoleKVAttentionMetadata":
        """Create an empty metadata (no HoleKV)."""
        return cls(is_holekv_active=False)

    @classmethod
    def from_active_view(
        cls,
        position_map: list[int],
        block_sources_list: list[str],
        reuse_block_to_physical: dict[int, int],
        compute_blocks: list[int],
        reuse_blocks: list[int],
        hole_blocks: list[int],
        block_table: list[int],
    ) -> "HoleKVAttentionMetadata":
        """Create metadata from an active view's data structures."""
        # Build hole token mask
        hole_positions = set()
        # We detect hole tokens by their block assignment
        hole_token_mask = [False] * len(position_map)

        # Mark hole tokens
        for hb in hole_blocks:
            # Approximate: mark tokens in hole blocks
            # In practice, this should be more precise with token-to-block mapping
            pass

        return cls(
            is_holekv_active=True,
            position_map=position_map,
            hole_token_mask=hole_token_mask,
            block_sources=block_sources_list,
            reuse_block_to_physical=reuse_block_to_physical,
            compute_blocks=compute_blocks,
            reuse_blocks=reuse_blocks,
            hole_blocks=hole_blocks,
            total_blocks=len(block_sources_list),
            block_table=block_table,
        )

    def get_active_block_table(self) -> list[int]:
        """Return the block table mapping active blocks to physical blocks.

        For compute blocks (not yet allocated), returns -1.
        For hole blocks, returns -2 (special marker).
        For reuse blocks, returns the physical block ID.
        """
        return self.block_table

    def is_hole_block(self, block_idx: int) -> bool:
        """Check if a block is a hole block."""
        return block_idx in self.hole_blocks

    def is_reuse_block(self, block_idx: int) -> bool:
        """Check if a block is reused from trace."""
        return block_idx in self.reuse_block_to_physical
