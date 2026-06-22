"""HoleKV Inline Parser — parses <HOLEKV_REMOVE_START/END> and <HOLEKV_ADD_START/END>.

Parses markers from the prompt text before tokenization and produces
a compacted prompt (for no-ref fallback) or structured spans (for HoleKV ref mode).

Markers are stripped before the prompt reaches the model.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("vllm.v1.holekv.inline_parser")

import re
from dataclasses import dataclass, field


# Marker constants
REMOVE_START = "<HOLEKV_REMOVE_START>"
REMOVE_END = "<HOLEKV_REMOVE_END>"
ADD_START = "<HOLEKV_ADD_START>"
ADD_END = "<HOLEKV_ADD_END>"

# Regex patterns for finding markers
_REMOVE_START_RE = re.compile(re.escape(REMOVE_START))
_REMOVE_END_RE = re.compile(re.escape(REMOVE_END))
_ADD_START_RE = re.compile(re.escape(ADD_START))
_ADD_END_RE = re.compile(re.escape(ADD_END))

# Combined pattern to find any marker
_MARKER_RE = re.compile(
    "|".join(
        re.escape(m)
        for m in [REMOVE_START, REMOVE_END, ADD_START, ADD_END]
    )
)


@dataclass
class HoleKVSpan:
    """A text span from the parsed prompt.

    Attributes:
        text: The span content (no markers).
        start_char: Start index in the original marked prompt.
        end_char: End index (exclusive) in the original marked prompt.
        is_removed: True if this span was inside a remove block.
        is_added: True if this span was inside an add block.
        is_stable: True if this span is outside any marker block.
    """

    text: str
    start_char: int
    end_char: int
    is_removed: bool = False
    is_added: bool = False
    is_stable: bool = True

    @property
    def length(self) -> int:
        return len(self.text)


@dataclass
class HoleKVParsedPrompt:
    """Result of parsing a marked prompt.

    Contains the decomposed spans and the compacted (no-marker) prompt.

    In HoleKV notation:
        S0 = stable prefix
        B1 = first removed span
        M1 = first added span
        S1 = stable middle
        B2 = second removed span
        M2 = second added span
        S2 = stable suffix (before any trailing D)
    """

    original_text: str
    has_markers: bool = False

    # Ordered spans in the original text
    spans: list[HoleKVSpan] = field(default_factory=list)

    # Compacted prompt (no markers, no removed spans)
    compact_prompt: str = ""

    # Marker positions in original text: [(start, end, marker_type), ...]
    marker_positions: list[tuple[int, int, str]] = field(default_factory=list)

    def to_compact_prompt(self) -> str:
        """Return the compacted prompt (A + M + C + D)."""
        return self.compact_prompt

    @property
    def has_hole_kv_structure(self) -> bool:
        """True if the prompt has remove/add marker pairs suitable for HoleKV."""
        # Must have at least one remove+add pair
        has_remove = any(s.is_removed for s in self.spans)
        has_add = any(s.is_added for s in self.spans)
        return has_remove and has_add

    @property
    def num_holes(self) -> int:
        """Number of remove/add hole pairs."""
        count = 0
        i = 0
        while i < len(self.spans):
            if self.spans[i].is_removed:
                # Look for matching add span immediately after
                if i + 1 < len(self.spans) and self.spans[i + 1].is_added:
                    count += 1
                    i += 2
                else:
                    i += 1
            else:
                i += 1
        return count

    def get_stable_spans(self) -> list[HoleKVSpan]:
        """Return only stable (non-remove, non-add) spans."""
        return [s for s in self.spans if s.is_stable]

    def get_remove_spans(self) -> list[HoleKVSpan]:
        """Return only remove spans."""
        return [s for s in self.spans if s.is_removed]

    def get_add_spans(self) -> list[HoleKVSpan]:
        """Return only add spans."""
        return [s for s in self.spans if s.is_added]

    def get_hole_pairs(self) -> list[tuple[HoleKVSpan, HoleKVSpan]]:
        """Return (remove_span, add_span) pairs for each hole."""
        pairs = []
        i = 0
        while i < len(self.spans):
            if self.spans[i].is_removed:
                if i + 1 < len(self.spans) and self.spans[i + 1].is_added:
                    pairs.append((self.spans[i], self.spans[i + 1]))
                    i += 2
                else:
                    i += 1
            else:
                i += 1
        return pairs


class HoleKVInlineParser:
    """Parses HoleKV inline markers from prompt text.

    Usage:
        parser = HoleKVInlineParser()
        parsed = parser.parse(prompt_text)
        compact = parsed.to_compact_prompt()   # A M C D
        pairs = parsed.get_hole_pairs()         # [(B1, M1), (B2, M2), ...]
    """

    def __init__(self):
        # Compile the regex that matches all marker types
        self._marker_re = _MARKER_RE

    def has_markers(self, text: str) -> bool:
        """Quick check if text contains any HoleKV markers."""
        result = REMOVE_START in text or ADD_START in text
        if result:
            logger.debug("HoleKVInlineParser.has_markers: found markers in text (len=%d)", len(text))
        return result

    def parse(self, text: str) -> HoleKVParsedPrompt:
        """Parse inline markers from the prompt text.

        Handles multiple remove/add pairs. Processes markers left-to-right.

        Returns a HoleKVParsedPrompt with decomposed spans and the compacted prompt.
        """
        result = HoleKVParsedPrompt(original_text=text)

        if not self.has_markers(text):
            # No markers — whole text is one stable span
            result.spans = [HoleKVSpan(text=text, start_char=0, end_char=len(text))]
            result.compact_prompt = text
            return result

        result.has_markers = True

        # Find all marker positions
        markers: list[tuple[int, int, str]] = []
        for m in self._marker_re.finditer(text):
            match_text = m.group()
            markers.append((m.start(), m.end(), match_text))
            result.marker_positions.append((m.start(), m.end(), match_text))

        result.marker_positions = markers

        # Parse spans using a state machine
        spans: list[HoleKVSpan] = []
        compact_parts: list[str] = []

        cursor = 0
        inside_remove = False
        inside_add = False

        for m_start, m_end, marker_type in markers:
            # Text before this marker
            if cursor < m_start:
                span_text = text[cursor:m_start]
                span = HoleKVSpan(
                    text=span_text,
                    start_char=cursor,
                    end_char=m_start,
                    is_removed=inside_remove,
                    is_added=inside_add,
                    is_stable=not inside_remove and not inside_add,
                )
                spans.append(span)

                if not inside_remove:
                    # Stable or add span → keep in compact prompt
                    compact_parts.append(span_text)

            # Update state based on marker
            if marker_type == REMOVE_START:
                inside_remove = True
            elif marker_type == REMOVE_END:
                inside_remove = False
            elif marker_type == ADD_START:
                inside_add = True
            elif marker_type == ADD_END:
                inside_add = False

            cursor = m_end

        # Trailing text after last marker
        if cursor < len(text):
            span_text = text[cursor:]
            span = HoleKVSpan(
                text=span_text,
                start_char=cursor,
                end_char=len(text),
                is_removed=inside_remove,
                is_added=inside_add,
                is_stable=not inside_remove and not inside_add,
            )
            spans.append(span)

            if not inside_remove:
                compact_parts.append(span_text)

        result.spans = spans
        result.compact_prompt = "".join(compact_parts)

        logger.debug(
            "HoleKVInlineParser.parse: original=%d chars, compact=%d chars, "
            "spans=%d (stable=%d, remove=%d, add=%d), holes=%d",
            len(text),
            len(result.compact_prompt),
            len(spans),
            len([s for s in spans if s.is_stable]),
            len([s for s in spans if s.is_removed]),
            len([s for s in spans if s.is_added]),
            result.num_holes,
        )

        return result

    def validate_markers(self, text: str) -> tuple[bool, Optional[str]]:
        """Validate marker pairing.

        Returns (is_valid, error_message).
        """
        remove_depth = 0
        add_depth = 0
        last_marker = None

        for m in self._marker_re.finditer(text):
            mt = m.group()
            if mt == REMOVE_START:
                if remove_depth > 0:
                    return False, "Nested REMOVE_START not allowed"
                if add_depth > 0:
                    return False, "REMOVE_START inside ADD block not allowed"
                remove_depth += 1
            elif mt == REMOVE_END:
                if remove_depth == 0:
                    return False, "REMOVE_END without matching REMOVE_START"
                remove_depth -= 1
                last_marker = "remove_end"
            elif mt == ADD_START:
                if add_depth > 0:
                    return False, "Nested ADD_START not allowed"
                if remove_depth > 0:
                    return False, "ADD_START inside REMOVE block not allowed"
                if last_marker != "remove_end":
                    return False, "ADD_START must follow REMOVE_END"
                add_depth += 1
            elif mt == ADD_END:
                if add_depth == 0:
                    return False, "ADD_END without matching ADD_START"
                add_depth -= 1
                last_marker = "add_end"

        if remove_depth > 0:
            return False, "Unclosed REMOVE block"
        if add_depth > 0:
            return False, "Unclosed ADD block"

        return True, None
