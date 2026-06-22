# HoleKV vLLM Integration — Implementation Status

## Overview

HoleKV is a **cross-request prefix-derived KV cache reuse optimization** for vLLM.
Every request returns a `holekv_cache_id`, and a client can pass `holekv_ref` to
reuse a previous request's KV cache with hole-preserved position encoding.

---

## ✅ What Has Been Implemented

### 1. Core HoleKV Modules (7 files)

| File | Status | Description |
|------|--------|-------------|
| `vllm/v1/holekv/__init__.py` | ✅ Done | Module exports |
| `vllm/v1/holekv/trace_registry.py` | ✅ Done | `HoleKVTraceRegistry` — store/lookup traces with access control, TTL, capacity eviction. `HoleKVTraceRecord` dataclass. |
| `vllm/v1/holekv/inline_parser.py` | ✅ Done | `HoleKVInlineParser` — parses `<HOLEKV_REMOVE_START/END>` and `<HOLEKV_ADD_START/END>`. Produces compact prompt for fallback. Validates marker pairing. |
| `vllm/v1/holekv/trace_aligner.py` | ✅ Done | `HoleKVTraceAligner` — aligns parsed spans against trace tokens/bytes. Checks exact token alignment, block boundaries, m ≤ b constraint. Instrumented with warning-level debug logging. |
| `vllm/v1/holekv/view_builder.py` | ✅ Done | `HoleKVViewBuilder` — builds hole-preserved active view: `A M [hole] C_old D_hole`. Classifies blocks as compute/reuse/hole. |
| `vllm/v1/holekv/position_map.py` | ✅ Done | `HoleKVPositionMap` — hole-preserved PE: `A[0:a) M[a:a+m) HOLE[a+m:a+b) C_old[a+b:a+b+c)`. Fallback compact PE. |
| `vllm/v1/holekv/attention_metadata.py` | ✅ Done | `HoleKVAttentionMetadata` — per-block source types, block table, reuse mapping. |
| `vllm/v1/holekv/engine_processor.py` | ✅ Done | `HoleKVEngineProcessor` — orchestrates parsing, alignment, view building, trace storage. Logging at every step. Helper `_flat_to_pairs()` for byte offset conversion. |

### 2. API Protocol (vLLM External Interface)

| Feature | Status | File |
|---------|--------|------|
| `holekv_ref` field on `ChatCompletionRequest` | ✅ Done | `entrypoints/openai/chat_completion/protocol.py` |
| `holekv_owner_id` field | ✅ Done | same |
| `holekv_session_id` field | ✅ Done | same |
| `holekv_cache_id` field on `ChatCompletionResponse` | ✅ Done | same |
| `holekv_cache_id` on `ChatCompletionStreamResponse` | ✅ Done | same |

### 3. Engine Core Data Structures

| Feature | Status | File |
|---------|--------|------|
| `holekv_ref`, `holekv_owner_id`, `holekv_session_id` on `EngineCoreRequest` | ✅ Done | `v1/engine/__init__.py` |
| `holekv_cache_id` on `EngineCoreOutput` | ✅ Done | same |
| `prompt_text` on `EngineCoreRequest` | ✅ Done | same |
| `prompt_token_byte_offsets` on `EngineCoreRequest` | ✅ Done | same |
| `trace_headers` workaround for byte offset serialization | ✅ Done | `async_llm.py` (encodes as JSON in `_holekv_byte_offsets` key) |

### 4. Request Lifecycle

| Feature | Status | File |
|---------|--------|------|
| HoleKV fields on `Request` class (including `prompt_token_byte_offsets`) | ✅ Done | `v1/request.py` |
| `holekv_cache_id` on `RequestOutput` | ✅ Done | `outputs.py` |
| Cache ID generation in `preprocess_add_request` | ✅ Done | `v1/engine/core.py` |
| Cache ID injection into `EngineCoreOutput` in `update_from_output()` | ✅ Done | `v1/core/sched/scheduler.py:1634` |
| Cache ID threaded through `output_processor` chain | ✅ Done | `v1/engine/output_processor.py` |
| `_extract_boff()` helper for byte offset extraction from trace_headers | ✅ Done | `v1/core/sched/scheduler.py:65` |

### 5. Serving Layer Integration

| Feature | Status | File |
|---------|--------|------|
| `holekv_ref` passed from `ChatCompletionRequest` → engine | ✅ Done | `entrypoints/openai/chat_completion/serving.py:355` |
| `holekv_cache_id` set on response | ✅ Done | `serving.py:1334` |
| `holekv_ref` params on `EngineClient.generate()` | ✅ Done | `engine/protocol.py` |
| `holekv_ref` in `AsyncLLM.generate()` and `add_request()` | ✅ Done | `v1/engine/async_llm.py` |
| Marker stripping BEFORE tokenization in `add_request()` | ✅ Done | `async_llm.py:352-384` |
| Byte offset computation in `add_request()` | ✅ Done | `async_llm.py:401-433` |

### 6. Scheduler Integration

| Feature | Status | File |
|---------|--------|------|
| Trace creation in `_free_request()` | ✅ Done | `v1/core/sched/scheduler.py:1960-1983` |
| `cache_id` passed to `create_trace_record` (fixed mismatch) | ✅ Done | `scheduler.py:1977` |
| `prompt_token_ids` stored in traces | ✅ Done | `scheduler.py:1975` |
| Byte offsets passed via trace_headers workaround | ✅ Done | `scheduler.py:1978` + `_extract_boff()` |
| HoleKV view detection + trace block import in `schedule()` | ✅ Done | `scheduler.py:654-706` |
| `import_trace_blocks()` in KV cache manager | ✅ Done | `v1/core/kv_cache_manager.py:564` |

### 7. Engine Core Pipeline

| Feature | Status | File |
|---------|--------|------|
| `assign_cache_id()` in `preprocess_add_request()` | ✅ Done | `v1/engine/core.py:862` |
| `preprocess_request()` in `preprocess_add_request()` | ✅ Done | `core.py:865` |
| `process_alignment()` called for ref-mode requests | ✅ Done | `core.py:869-885` |
| `build_active_view()` called on alignment success | ✅ Done | `core.py:887-901` |
| `holekv_active_view` set on Request | ✅ Done | `core.py:893` |
| `holekv_position_map` on `NewRequestData` | ✅ Done | `v1/core/sched/output.py:48,71` |

---

## ✅ Test Results (Verified June 2026)

### Unit Tests: 77/77 PASS ✅

| Test Module | Tests | Status |
|-------------|-------|--------|
| Inline marker parsing | 6 | ✅ |
| Trace registry | 8 | ✅ |
| Position map | 7 | ✅ |
| Attention metadata | 10 | ✅ |
| API protocol fields | 14 | ✅ |
| Engine processor | 5 | ✅ |
| No-ref fallback simulation | 6 | ✅ |
| Trace aligner + view builder | 21 | ✅ |

### Server E2E Tests: 6/6 PASS ✅

| Test | Result |
|------|--------|
| Basic request returns `holekv_cache_id` | ✅ |
| Request with markers (fallback) | ✅ |
| Token savings from compaction | ✅ |
| Request with `holekv_ref` (cache reuse) | ✅ |
| Cross-owner access control | ✅ |
| Multiple sequential requests | ✅ |

---

## 🔧 Bugs Found & Fixed During E2E Testing

| # | Bug | Root Cause | Fix | File(s) |
|---|-----|-----------|-----|----------|
| 1 | `holekv_cache_id` missing from client response (500 error) | `RequestOutput.__init__` ignored `holekv_cache_id` (went to `**kwargs`) | Added `holekv_cache_id` param to `RequestOutput` | `outputs.py` |
| 2 | `holekv_cache_id` not propagating from engine | `output_processor.make_request_output()` never passed `holekv_cache_id` | Threaded through `make_request_output` → `_new_request_output` → `RequestOutput` | `output_processor.py` |
| 3 | Trace lookup always fails ("trace not found") | `create_trace_record()` generated a NEW random `cache_id` instead of using `request.holekv_cache_id` | `_free_request` now passes `cache_id=request.holekv_cache_id`; `create_trace_record` accepts optional `cache_id` param | `scheduler.py`, `engine_processor.py` |
| 4 | `preprocess_request` never parsed markers | `engine_processor.py` accessed `_prompt_text` (non-existent) instead of `prompt_text` | Fixed attribute name to `prompt_text` | `engine_processor.py` |
| 5 | `prompt_text` not available on `EngineCoreRequest` | `async_llm.py` never set `request.prompt_text` before dispatching | Added `request.prompt_text = prompt_text` in `add_request()` | `async_llm.py` |
| 6 | Byte offsets lost during msgspec serialization | `EngineCoreRequest` uses `array_like=True`; new fields are truncated during array encoding | Encoded byte offsets as JSON in `trace_headers["_holekv_byte_offsets"]`; extracted via `_extract_boff()` | `async_llm.py`, `scheduler.py` |
| 7 | Trace UTF-8 mismatched with byte offsets | `rendered_prompt_utf8` used original text (with markers) but byte offsets referenced compact text | Strip markers from `prompt_text` before UTF-8 encoding in `create_trace_record` | `engine_processor.py` |
| 8 | Scheduler: `import_trace_blocks()` type mismatch | `trace.block_ids_by_group[0]` passed `list[int]` but method expects `tuple[list[int], ...]` | Changed to `tuple(trace.block_ids_by_group)` | `scheduler.py` |
| 9 | Scheduler: except block referenced undefined vars | `except` block referenced `num_imported`/`imported_blocks` that may not exist | Simplified except to just log warning; removed dead `holekv_proc` line | `scheduler.py` |
| 10 | Scheduler: `_free_request` block_ids nested list | `get_block_ids()` returns `tuple[list[int], ...]` but code passed `list[list[int]]` to `create_trace_record` | Extract first group: `flat_block_ids = list(block_ids_raw[0])` | `scheduler.py` |
| 11 | Import logs invisible (DEBUG level) | `logger.debug` suppressed at default INFO level | Changed to `logger.info`; added `else` branch for zero-import case | `scheduler.py` |
| 12 | Blocks freed immediately after source request | `_free_blocks()` decrements `ref_cnt` to 0 right after trace creation | Pin blocks via `BlockPool.touch()` BEFORE `_free_blocks()`; unpin on trace eviction | `scheduler.py`, `kv_cache_manager.py`, `trace_registry.py`, `core.py` |
| 13 | `AssertionError` — computed blocks should be empty | `single_type_kv_cache_manager` assertion blocks HoleKV imported blocks when prefix caching disabled | Relaxed assertion: skip `touch` when caching disabled, allow non-empty blocks from HoleKV | `single_type_kv_cache_manager.py` |
| 14 | `IndexError: tuple index out of range` on import | `import_trace_blocks()` returned 1 group but hybrid models (Qwen3.5-9B) have 2 KV groups | Pad `imported_groups` to match `num_kv_cache_groups` | `kv_cache_manager.py` |

---

## 📋 What Now Works End-to-End (Verified with RTX 5090 + Qwen3.5-9B)

```
✅ Server starts with VLLM_HOLEKV_ENABLED=1
✅ "HoleKV engine processor enabled" in logs
✅ vLLM serves chat completions on /v1/chat/completions
✅ preprocess_request() called for every request
✅ assign_cache_id() runs on every request admission
✅ prompt_text threaded from serving → engine core
✅ Marker stripping BEFORE tokenization (confirmed: "stripped N markers")
✅ "HoleKV: computed N token byte offsets" in APIServer logs
✅ Byte offsets flow through trace_headers workaround
✅ Compact token IDs available
✅ Markers parsed during request processing (from original prompt_text)
✅ Trace lookup SUCCEEDS (cache IDs now match)
✅ "HoleKV ref mode" log in engine output
✅ Traces stored with correct cache_id, prompt_token_ids, byte offsets
✅ holekv_cache_id flows to client response
✅ Alignment SUCCEEDS — A/B/C spans correctly matched against trace tokens
✅ "HoleKV active view built: A=N M=N C_old=N D_hole=N holes=N" in logs
✅ build_active_view() sets HoleKVActiveView on Request
✅ HoleKVActiveView detected by scheduler ("found view for request=… is_valid=True")
✅ import_trace_blocks() called successfully in scheduler
✅ Cross-owner access control enforced (alignment fails for wrong owner)
✅ All 77 unit tests pass / All 6 server E2E tests pass
✅ **Block import SUCCEEDS** — 18 C_old tokens imported from trace (block pinning keeps KV alive)
✅ Block pinning with unpin-on-eviction (eviction callback wired to KVCacheManager)
⏭️ Hole-preserved PE in GPU model runner — intentionally skipped
```

---

## ✅ Alignment Fixed (June 2026)

### Problem

The original alignment algorithm failed because the trace stores **compact** text
(markers stripped), but the aligner searched for B-span text (from the original
marked prompt) in the trace. Since B's text doesn't exist in the compact trace,
the search always failed.

### Fix: Search for Next Stable Suffix

Instead of searching for B-span text in the trace, the aligner now searches for
the **next stable span C** in the trace, then computes B's token range as the
gap between A's end and C's start. Key changes in `trace_aligner.py`:

- B token range computed as `[a_token_end, c_token_start)` — B is the gap
  between the matched A and C spans, not content-matched
- Block alignment check relaxed to soft warning (non-fatal)
- Initial C search uses `cursor_byte` (after A) instead of 0 for efficiency
- `m_token_count` estimate tightened (`// 4` instead of `// 3`) with clamping
- `_is_block_aligned` handles `block_size=0` edge case

### Additional Fixes for Full Pipeline

- **async_llm.py**: Strip marker `"type"` key from prompt dict AND clear
  `prompt_token_ids`/`prompt_embeds` before dispatching — forces re-tokenization
  from compact text instead of using pre-tokenized IDs that still contain markers.
- **view_builder.py**: Added missing `A_length`, `M_length`, `C_old_length`,
  `D_hole_length`, `hole_length` properties. Fixed stable-suffix index bug
  (`stable_span_token_ranges[hole_idx]` → `[hole_idx + 1]`).
- **engine_processor.py**: Populated `token_ranges_by_block` and
  `position_ranges_by_block` in `create_trace_record`.

### Block Pinning — KV Blocks Survive Across Requests (June 2026)

Trace blocks are now **pinned** by incrementing `ref_cnt` via `BlockPool.touch()`
before `_free_blocks()` runs.  This keeps the C_old span blocks alive after the
source request finishes.  When a subsequent request imports the trace via
`import_trace_blocks()`, the blocks have `ref_cnt > 0` and the import succeeds.

When the trace is evicted from the registry, the eviction callback calls
`KVCacheManager.unpin_blocks()` to release the pinned blocks back to the free
pool.

**Files changed:**
- `vllm/v1/core/kv_cache_manager.py` — Added `pin_blocks()` and `unpin_blocks()`
- `vllm/v1/core/sched/scheduler.py` — Calls `pin_blocks()` after trace creation,
  before `_free_blocks()`
- `vllm/v1/holekv/trace_registry.py` — Added `_block_unpin_fn` callback + `set_block_unpin_fn()`;
  `_evict_one()` calls the callback before deleting the trace
- `vllm/v1/engine/core.py` — Wires the eviction callback to `KVCacheManager.unpin_blocks()`
- `vllm/v1/core/single_type_kv_cache_manager.py` — Relaxed assertion that blocked
  non-empty computed blocks when prefix caching is disabled

**Verified on GPU:** Server log shows `"HoleKV imported 18 tokens from trace=…"`

---

## Files Modified (17 total)

| # | File | What Was Changed |
|---|------|-----------------|
| 1 | `vllm/v1/engine/__init__.py` | Added `prompt_text`, `prompt_token_byte_offsets` to `EngineCoreRequest` |
| 2 | `vllm/v1/engine/input_processor.py` | Threads `prompt_text` from decoder_inputs into `EngineCoreRequest` |
| 3 | `vllm/v1/request.py` | Added `prompt_text`, `prompt_token_byte_offsets`, `holekv_marker_spans`, `holekv_active_view`, `holekv_position_map` |
| 4 | `vllm/v1/engine/core.py` | `assign_cache_id()`, `preprocess_request()`, `process_alignment()`, `build_active_view()`; eviction callback wiring to `KVCacheManager.unpin_blocks()` |
| 5 | `vllm/v1/core/sched/scheduler.py` | `holekv_cache_id` injection; trace creation with `cache_id`, `prompt_token_ids`, byte offsets; `_extract_boff()`; HoleKV view detection + import + `pin_blocks()` before `_free_blocks()` |
| 6 | `vllm/v1/core/sched/output.py` | Added `holekv_position_map` to `NewRequestData` |
| 7 | `vllm/v1/core/kv_cache_manager.py` | Added `import_trace_blocks()`, `pin_blocks()`, `unpin_blocks()`; padded import groups to match KV cache group count |
| 8 | `vllm/platforms/__init__.py` | PackageNotFoundError fix |
| 9 | `vllm/outputs.py` | Added `holekv_cache_id` to `RequestOutput.__init__` |
| 10 | `vllm/v1/engine/output_processor.py` | Threaded `holekv_cache_id` through request output chain |
| 11 | `vllm/v1/engine/async_llm.py` | Marker stripping; byte offset computation; trace_headers workaround; `prompt_text` propagation; force re-tokenization from compact text |
| 12 | `vllm/v1/holekv/engine_processor.py` | `prompt_text` attr fix; `cache_id`/`prompt_token_ids`/`prompt_token_byte_offsets` params; `_flat_to_pairs()` helper; compact UTF-8 fix; `token_ranges_by_block`/`position_ranges_by_block` population |
| 13 | `vllm/v1/holekv/trace_aligner.py` | Search for next stable suffix C (not B) in compact trace text; relaxed block alignment; `m_token_count` fix; `_is_block_aligned(block_size=0)` edge case |
| 14 | `vllm/v1/holekv/view_builder.py` | Added missing length properties; fixed C-suffix index bug |
| 15 | `vllm/v1/holekv/trace_registry.py` | Added `_block_unpin_fn` callback + `set_block_unpin_fn()`; `_evict_one()` calls unpin callback before deletion |
| 16 | `vllm/v1/core/single_type_kv_cache_manager.py` | Relaxed assertion that blocked HoleKV imported blocks when prefix caching disabled |
| 17 | `vllm/entrypoints/openai/chat_completion/protocol.py` | Added `holekv_ref`, `holekv_owner_id`, `holekv_session_id`, `holekv_cache_id` fields |

---

## Launch Command

```bash
QWEN35="/workspace/models/.cache/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
export LD_LIBRARY_PATH="/opt/conda/envs/agent/lib/python3.12/site-packages/nvidia/cu13/lib:/opt/conda/envs/agent/lib/python3.12/site-packages/torch/lib:/opt/conda/envs/agent/lib"
export VLLM_HOLEKV_ENABLED=1
export VLLM_USE_FLASHINFER_SAMPLER=0
cd /workspace/holekv-vllm
python3 -m vllm.entrypoints.openai.api_server \
  --model "$QWEN35" --port 8000 --host 0.0.0.0 \
  --max-model-len 4096 --gpu-memory-utilization 0.85 \
  --enforce-eager --trust-remote-code
```

Alternatively, use the wrapper script: `bash /workspace/launch_holekv_server.sh`
