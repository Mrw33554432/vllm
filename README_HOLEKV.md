# HoleKV — Trace-Based Partial KV-Cache Reuse for vLLM

**Branch:** `holekv`  
**Base:** [vLLM v0.23.0](https://github.com/vllm-project/vllm/releases/tag/v0.23.0)  
**Status:** Functional — passes all unit tests (77/77) and server E2E tests (6/6) on Qwen3.5-9B with RTX 5090

## What is HoleKV?

HoleKV lets vLLM reuse KV-cache blocks from *previous requests* even when the
intermediate prompt content differs.  It matches stable **prefix** (A) and
**suffix** (C) token spans across requests with different middle sections (B/M).

```
Request 1:  [A: stable prefix] [B: old context] [C: stable suffix] [D: assistant]
Request 2:  [A: stable prefix] [M: new context] [C: stable suffix] [D: assistant]
                                        ↑
                              A and C spans reused from Request 1
```

Without HoleKV, every request recomputes everything.  With HoleKV, the C-span
blocks from Request 1 are pinned in memory and imported into Request 2's KV
cache — saving computation for the suffix portion.

## How It Works

1. **Markers in prompt** — The client inserts markers around variable sections:
   ```
   <HOLEKV_REMOVE_START>old content<HOLEKV_REMOVE_END>
   <HOLEKV_ADD_START>new content<HOLEKV_ADD_END>
   ```

2. **Compact re-tokenization** — Markers are stripped before tokenization, so
   the model never sees them.

3. **Trace storage** — After Request 1 finishes, a `HoleKVTraceRecord` is stored
   with token IDs, byte offsets, and pinned KV block IDs.

4. **Alignment** — Request 2 looks up the trace and aligns its compact prompt
   against the trace's token positions, finding A and C span boundaries.

5. **Block import** — The C-span blocks (still alive via `ref_cnt` pinning) are
   imported into Request 2's block table.

6. **Eviction** — When the trace registry evicts an old trace, its pinned blocks
   are released back to the free pool.

## Changes from v0.23.0

### New Modules (`vllm/v1/holekv/`)

| File | Purpose |
|------|---------|
| `inline_parser.py` | Parse `<HOLEKV_*>` markers, render compact prompts |
| `trace_registry.py` | In-memory trace cache with TTL, LRU eviction, owner access control |
| `engine_processor.py` | Lifecycle coordinator: cache ID, lookup, alignment, trace storage |
| `trace_aligner.py` | Byte-offset alignment of compact spans against trace tokens |
| `view_builder.py` | Build `HoleKVActiveView` with A/M/C/D block references |
| `position_map.py` | Hole-preserved position IDs for attention computation |
| `attention_metadata.py` | Block table structures for reuse/compute/hole regions |

### Modified vLLM Files (17 files)

| File | What Changed |
|------|-------------|
| `vllm/v1/core/sched/scheduler.py` | HoleKV view detection, `import_trace_blocks`, block pinning, trace creation |
| `vllm/v1/core/kv_cache_manager.py` | `import_trace_blocks()`, `pin_blocks()`, `unpin_blocks()` |
| `vllm/v1/core/single_type_kv_cache_manager.py` | Relaxed assertion for non-empty computed blocks |
| `vllm/v1/engine/core.py` | `assign_cache_id`, alignment, view building, eviction callback |
| `vllm/v1/engine/async_llm.py` | Marker stripping, byte offsets, force re-tokenization |
| `vllm/v1/engine/input_processor.py` | `prompt_text` threading |
| `vllm/v1/engine/output_processor.py` | `holekv_cache_id` threading |
| `vllm/v1/engine/__init__.py` | `EngineCoreRequest` fields |
| `vllm/v1/request.py` | New HoleKV state fields |
| `vllm/v1/core/sched/output.py` | `holekv_position_map` |
| `vllm/outputs.py` | `holekv_cache_id` in `RequestOutput` |
| `vllm/platforms/__init__.py` | PackageNotFoundError fix |
| `vllm/entrypoints/openai/chat_completion/protocol.py` | API fields |

## API

### Request Fields (Chat Completions)

```json
{
  "messages": [...],
  "holekv_ref": "abc123...",       // optional: reuse KV from this trace
  "holekv_owner_id": "user-42",    // optional: namespace for access control
  "holekv_session_id": "sess-1"    // optional: session grouping
}
```

### Response Fields

```json
{
  "holekv_cache_id": "def456..."   // trace ID for future reuse
}
```

## Quick Start

```bash
# Environment
export VLLM_HOLEKV_ENABLED=1
export VLLM_USE_FLASHINFER_SAMPLER=0

# Server
python -m vllm.entrypoints.openai.api_server \
  --model <model-path> --port 8000 --enforce-eager \
  --trust-remote-code --max-model-len 4096

# Unit tests
python test_holekv_e2e.py

# Server E2E tests
python test_holekv_e2e_server.py
```

## Test Status

| Suite | Result |
|-------|--------|
| Unit tests | 77/77 ✅ |
| Server E2E | 6/6 ✅ (Qwen3.5-9B, RTX 5090) |
| Block import | ✅ `HoleKV imported 18 tokens from trace=…` |
| Block pinning | ✅ Blocks survive across requests |
| Cross-owner control | ✅ Alignment fails for wrong owner |
| Eviction unpin | ✅ Callback wired to `KVCacheManager.unpin_blocks()` |

## Documentation

- [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) — Detailed change log, bugs fixed, test results
- [PLAN.md](PLAN.md) — Architecture overview and design decisions

## Diff with upstream

```bash
git diff v0.23.0 --stat   # 25 files, +4,369 / −246
```

## License

Same as vLLM — Apache 2.0.
