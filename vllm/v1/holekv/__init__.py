"""HoleKV — Cross-request prefix-derived KV cache reuse optimization for vLLM.

This module implements a unified request path where every request:
  - is a normal vLLM request
  - may optionally provide holekv_ref
  - may contain inline remove/add markers
  - returns a normal response plus holekv_cache_id
"""

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
# NOTE: HoleKVEngineProcessor requires the full vLLM stack (CUDA, msgspec).
# Import it directly: from vllm.v1.holekv.engine_processor import HoleKVEngineProcessor

__all__ = [
    "HoleKVTraceRecord",
    "HoleKVTraceRegistry",
    "get_global_registry",
    "HoleKVInlineParser",
    "HoleKVParsedPrompt",
    "HoleKVTraceAligner",
    "HoleKVAlignmentResult",
    "HoleKVViewBuilder",
    "HoleKVActiveView",
    "HoleKVPositionMap",
    "HoleKVAttentionMetadata",
    "HoleKVEngineProcessor",  # documented but requires full vLLM stack
]
