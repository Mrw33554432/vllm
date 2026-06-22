"""HoleKV Position Map — manages position encoding with holes.

In HoleKV mode, position encoding is hole-preserved:
    A[0:a)
    M[a:a+m)
    HOLE[a+m:a+b)
    C_old[a+b:a+b+c)  ← original PE from trace
    D_hole[a+b+c:a+b+c+d)

Unlike the no-ref fallback which uses compact PE, the HoleKV path
preserves C_old's original positions and leaves gaps (holes) where
removed content used to be.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HoleKVPositionMap:
    """Position encoding map for a HoleKV active view.

    Maps each active token index to its position encoding value.

    Attributes:
        positions: List of position values, indexed by active token index.
        hole_intervals: List of (start_pos, end_pos) where holes are.
        total_tokens: Total number of tokens in the active view.
        max_position: Maximum position value used.
    """

    positions: list[int] = field(default_factory=list)
    hole_intervals: list[tuple[int, int]] = field(default_factory=list)
    total_tokens: int = 0
    max_position: int = 0

    @classmethod
    def build_fallback(
        cls,
        num_tokens: int,
    ) -> "HoleKVPositionMap":
        """Build a simple compact position map for the no-ref fallback path.

        Positions are 0-based contiguous: [0, 1, 2, ..., num_tokens-1].
        """
        positions = list(range(num_tokens))
        return cls(
            positions=positions,
            hole_intervals=[],
            total_tokens=num_tokens,
            max_position=num_tokens - 1 if num_tokens > 0 else 0,
        )

    @classmethod
    def build_hole_preserved(
        cls,
        a_tokens: int,
        hole_pairs: list[tuple[int, int, int]],  # [(b_tokens, m_tokens, c_tokens), ...]
        d_tokens: int,
        c_original_positions: Optional[list[list[int]]] = None,
    ) -> "HoleKVPositionMap":
        """Build a hole-preserved position map.

        Args:
            a_tokens: Number of tokens in the stable prefix A.
            hole_pairs: List of (b_tokens, m_tokens, c_tokens) per hole.
            d_tokens: Number of tokens in the new suffix D.
            c_original_positions: Original position values for each C span,
                if available from the trace. Otherwise uses computed positions.

        Returns:
            HoleKVPositionMap with hole-preserved positions.
        """
        positions = []
        hole_intervals = []
        pos = 0

        # A: [0, a)
        for i in range(a_tokens):
            positions.append(pos)
            pos += 1

        for hole_idx, (b, m, c_tok) in enumerate(hole_pairs):
            # M: [a, a+m)
            for i in range(m):
                positions.append(pos)
                pos += 1

            # HOLE: [a+m, a+b)
            hole_start = pos
            hole_size = b - m
            if hole_size > 0:
                hole_intervals.append((pos, pos + hole_size))
                pos += hole_size

            # C_old: [a+b, a+b+c)
            if c_original_positions and hole_idx < len(c_original_positions):
                # Use original positions from the trace
                orig_positions = c_original_positions[hole_idx]
                for orig_p in orig_positions:
                    positions.append(orig_p)
                pos = max(pos, (orig_positions[-1] + 1) if orig_positions else pos)
            else:
                # Use computed positions (contiguous after hole)
                for i in range(c_tok):
                    positions.append(pos)
                    pos += 1

        # D_hole: starts after last C_old position
        for i in range(d_tokens):
            positions.append(pos)
            pos += 1

        total_tokens = len(positions)
        max_position = max(positions) if positions else 0

        return cls(
            positions=positions,
            hole_intervals=hole_intervals,
            total_tokens=total_tokens,
            max_position=max_position,
        )

    def get_position(self, token_idx: int) -> int:
        """Get the position encoding for a token index.

        Returns -1 if the token index is in a hole.
        """
        if token_idx < 0 or token_idx >= len(self.positions):
            return -1
        return self.positions[token_idx]

    def is_in_hole(self, token_idx: int) -> bool:
        """Check if a token index falls within a hole."""
        if token_idx < 0 or token_idx >= len(self.positions):
            return False
        pos = self.positions[token_idx]
        for hole_start, hole_end in self.hole_intervals:
            if hole_start <= pos < hole_end:
                return True
        return False

    def get_position_tensor(self) -> list[int]:
        """Return positions as a flat list suitable for model input."""
        return self.positions

    @property
    def num_holes(self) -> int:
        """Number of hole intervals."""
        return len(self.hole_intervals)

    def compact_positions(self) -> list[int]:
        """Return a compacted version without holes (for fallback comparison)."""
        # Filter out hole positions and renumber
        hole_positions = set()
        for hs, he in self.hole_intervals:
            hole_positions.update(range(hs, he))

        compacted = []
        for p in self.positions:
            if p not in hole_positions:
                compacted.append(p)
        return compacted
