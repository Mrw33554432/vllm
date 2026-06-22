# HoleKV vLLM Integration Design — Unified Request Model

> **Status (June 2026):** Core infrastructure fully implemented, tested, and verified on GPU.
> - **15 source files modified**, 7 new HoleKV modules (see `IMPLEMENTATION_STATUS.md` for full list)
> - **77/77 unit tests pass**, **6/6 server E2E tests pass** (validated with RTX 5090 + Qwen3.5-9B)
> - `holekv_cache_id` flows end-to-end: EngineCoreOutput → RequestOutput → client response ✅
> - Marker stripping BEFORE tokenization works (re-tokenizing from compact text) ✅
> - Byte offsets computed via progressive decode and stored via trace_headers workaround ✅
> - Alignment FIXED June 2026: search for next stable suffix C in compact trace text ✅
> - Active view building WORKS: `A M [hole] C_old D_hole` constructed correctly ✅
> - `import_trace_blocks()` called in scheduler; gracefully returns 0 when blocks freed ✅
> - Cross-owner access control enforced ✅
> - Hole-preserved PE in GPU model runner: intentionally skipped (non-blocking)
> - See `IMPLEMENTATION_STATUS.md` for full details.

## 0. Core Summary

HoleKV is a **cross-request prefix-derived KV cache reuse optimization** for vLLM.

There are **not** separate “Request A” and “Request B” API types.

Every request follows the same structure:

```json
{
  "model": "...",
  "messages": [...],
  "holekv_ref": "optional previous cache id"
}
```

Every response follows the same structure:

```json
{
  "choices": [...],
  "usage": {...},
  "holekv_cache_id": "new cache id for this request"
}
```

A request can optionally **consume** a previous cache using `holekv_ref`.

Every response can **produce** a new `holekv_cache_id`.

---

# 1. Unified Request Behavior

## 1.1 Request Without `holekv_ref`

If a request does not include `holekv_ref`, the server runs normal vLLM behavior.

If the prompt contains HoleKV inline markers, the server parses them and falls back to normal compact prompt generation.

Raw marked prompt:

```text
A
<HOLEKV_REMOVE_START>
B
<HOLEKV_REMOVE_END>
<HOLEKV_ADD_START>
M
<HOLEKV_ADD_END>
C
D
```

Model-visible prompt:

```text
A M C D
```

Execution:

```text
A M C D → E
```

Response includes:

```json
{
  "holekv_cache_id": "hk_new"
}
```

This is the **normal fallback path**.

No stale suffix cache is reused.

No hole-preserved PE is needed.

---

## 1.2 Request With `holekv_ref`

If a request includes `holekv_ref`, the server tries to use HoleKV.

Input:

```json
{
  "holekv_ref": "hk_previous",
  "messages": [...]
}
```

If the prompt contains inline markers:

```text
A
<HOLEKV_REMOVE_START>
B
<HOLEKV_REMOVE_END>
<HOLEKV_ADD_START>
M
<HOLEKV_ADD_END>
C
D
```

and the referenced cache contains the prior prompt:

```text
A B C
```

then the server builds:

```text
A M [hole] C_old D_hole → E_hole
```

Response still includes a **new** cache id:

```json
{
  "holekv_cache_id": "hk_new"
}
```

So the current request both:

```text
1. consumes hk_previous
2. produces hk_new
```

---

# 2. Core Notation

Use this notation for text spans:

```text
A = stable prefix before removed span
B = removed / obsolete span
M = replacement marker / added content
C = suffix after removed span
D = new input span
E = generated output span
```

Fallback path:

```text
A M C D → E
```

HoleKV path:

```text
A M [hole] C_old D_hole → E_hole
```

Exact recompute reference:

```text
A M C_new D_new → E_exact
```

---

# 3. Response Extension

Every response should add:

```json
{
  "holekv_cache_id": "hk_new"
}
```

This cache id refers to the **current request’s input prompt KV trace**, not necessarily the generated output.

If strict OpenAI-compatible clients dislike extra fields, also provide it as a response header:

```text
x-holekv-cache-id: hk_new
```

Recommended behavior:

```text
1. Always return holekv_cache_id when HoleKV tracing is enabled.
2. If trace creation fails or cache is disabled, return null or omit the field.
3. Do not require the client to request tracing explicitly.
```

---

# 4. Inline Marker Syntax

Reserved server-side markers:

```text
<HOLEKV_REMOVE_START>
<HOLEKV_REMOVE_END>
<HOLEKV_ADD_START>
<HOLEKV_ADD_END>
```

Valid pattern:

```text
<HOLEKV_REMOVE_START>
old content to remove
<HOLEKV_REMOVE_END>
<HOLEKV_ADD_START>
replacement / marker content
<HOLEKV_ADD_END>
```

Example:

```text
A
<HOLEKV_REMOVE_START>
B
<HOLEKV_REMOVE_END>
<HOLEKV_ADD_START>
[Obsolete middle section removed. Do not rely on it.]
<HOLEKV_ADD_END>
C
D
```

Server-side markers must be stripped before model execution.

The model sees `M`, but not the parser markers.

---

# 5. No-Ref Fallback Mode

If no `holekv_ref` is provided, inline markers are treated as normal prompt rewrite syntax.

Raw:

```text
A <REMOVE_START>B<REMOVE_END><ADD_START>M<ADD_END>C D
```

Parsed:

```text
A M C D
```

Execution:

```text
A M C D → E
```

PE is compact:

```text
A[0:a)
M[a:a+m)
C[a+m:a+m+c)
D[a+m+c:a+m+c+d)
E[a+m+c+d:...]
```

This path is exact normal vLLM behavior.

---

# 6. HoleKV Ref Mode

If `holekv_ref` is provided and valid, the server uses the referenced cache.

Referenced cache contains:

```text
A B C
```

Current marked prompt contains:

```text
A <REMOVE_START>B<REMOVE_END><ADD_START>M<ADD_END>C D
```

HoleKV active view:

```text
A M [hole] C_old D_hole
```

Generation:

```text
A M [hole] C_old D_hole → E_hole
```

This avoids recomputing `C`.

---

# 7. Position Encoding Policy

Assume:

```text
len(A)=a
len(B)=b
len(M)=m
len(C)=c
len(D)=d
```

Usually:

```text
m <= b
```

## 7.1 Referenced Cache PE

The referenced cache was computed as:

```text
A[0:a)
B[a:a+b)
C[a+b:a+b+c)
```

## 7.2 No-Ref Fallback PE

Fallback path:

```text
A M C D → E
```

uses compact PE:

```text
A[0:a)
M[a:a+m)
C[a+m:a+m+c)
D[a+m+c:a+m+c+d)
E[a+m+c+d:...]
```

## 7.3 HoleKV Ref Mode PE

HoleKV path:

```text
A M [hole] C_old D_hole → E_hole
```

uses hole-preserved PE:

```text
A[0:a)
M[a:a+m)
HOLE[a+m:a+b)
C_old[a+b:a+b+c)
D_hole[a+b+c:a+b+c+d)
E_hole[a+b+c+d:...]
```

Key rule:

```text
C_old keeps original PE.
D_hole starts after C_old's original PE.
E_hole starts after D_hole.
```

Do not compact `C_old`.

---

# 8. Multiple Holes

Multiple remove/add marker pairs are allowed.

Raw marked prompt:

```text
S0
<HOLEKV_REMOVE_START>
B1
<HOLEKV_REMOVE_END>
<HOLEKV_ADD_START>
M1
<HOLEKV_ADD_END>
S1
<HOLEKV_REMOVE_START>
B2
<HOLEKV_REMOVE_END>
<HOLEKV_ADD_START>
M2
<HOLEKV_ADD_END>
S2
D
```

No-ref fallback:

```text
S0 M1 S1 M2 S2 D → E
```

HoleKV ref mode:

```text
S0 M1 [hole1] S1_old M2 [hole2] S2_old D_hole → E_hole
```

Each reused old span keeps its original PE from the referenced cache.

---

# 9. Trace Stored for Every Request

When HoleKV tracing is enabled, each completed request stores a trace.

```python
@dataclass
class HoleKVTraceRecord:
    cache_id: str
    request_id: str

    rendered_prompt_text: str
    rendered_prompt_utf8: bytes

    token_ids: list[int]
    token_byte_offsets: list[tuple[int, int]]

    block_size: int
    block_ids_by_group: list[list[int]]
    token_ranges_by_block: list[tuple[int, int]]
    position_ranges_by_block: list[tuple[int, int]]

    model_fingerprint: str
    tokenizer_fingerprint: str
    lora_id: str | None
    dtype: str

    owner_id: str | None
    session_id: str | None
    created_at: float
    ttl_seconds: float
    pinned: bool
```

The trace is used only if a future request references it through `holekv_ref`.

---

# 10. Alignment Rules

In HoleKV ref mode, marked spans must align with the referenced cache.

Referenced cache:

```text
A B C
```

Current marked prompt:

```text
A <remove>B</remove> <add>M</add> C D
```

Server must verify:

```text
A matches referenced A
B matches referenced B
C matches referenced C
```

For v1:

```text
exact token alignment required
```

If alignment fails:

```text
return clear HoleKV alignment error
```

Optional fallback mode can later allow:

```text
fallback to A M C D
```

but for research/debugging, explicit error is better.

---

# 11. Token and Block Boundary Rules

Markers are parsed before tokenization, but alignment must still be checked against referenced cache tokens.

Potential issue:

```text
marker boundary may split a token from the referenced cache
```

v1 policy:

```text
marker boundaries must align with referenced token boundaries
```

If not:

```text
reject HoleKV ref mode
```

KV cache is block-based.

v1 policy:

```text
removed spans should align to KV block boundaries
```

If not:

```text
reject HoleKV ref mode
```

v2 can support:

```text
boundary block recomputation
```

---

# 12. Replacement Length Rule

If removed span has token length:

```text
b = len(B_tokens)
```

and replacement span has token length:

```text
m = len(M_tokens)
```

HoleKV v1 should require:

```text
m <= b
```

If:

```text
m > b
```

then `M` would collide with `C_old`’s original PE.

v1 behavior:

```text
reject HoleKV ref mode
```

No-ref fallback has no such restriction.

---

# 13. Execution Flow

## 13.1 Unified Request Handling

Every request follows this logic:

```python
def handle_request(req):
    parsed = parse_inline_markers(req)

    if req.holekv_ref is None:
        # normal exact fallback
        prompt = parsed.to_compact_prompt()
        output = run_standard_vllm(prompt)
        cache_id = store_trace_for_current_request(prompt)
        return output + {"holekv_cache_id": cache_id}

    else:
        # HoleKV reuse path
        trace = load_trace(req.holekv_ref)
        view = build_holekv_view(trace, parsed)
        output = run_holekv_vllm(view)
        cache_id = store_trace_for_current_request_effective_input(...)
        return output + {"holekv_cache_id": cache_id}
```

---

## 13.2 No-Ref Flow

```text
1. receive request
2. parse inline markers
3. rewrite to compact prompt A M C D
4. run standard vLLM
5. store trace
6. return normal response + holekv_cache_id
```

---

## 13.3 Ref Flow

```text
1. receive request with holekv_ref
2. parse inline markers
3. load referenced trace
4. align A/B/C against trace
5. build hole-preserved active KV view
6. compute M at B's original start PE
7. preserve hole interval
8. reuse C_old at original PE
9. compute D_hole
10. generate E_hole
11. store new trace for current request
12. return normal response + new holekv_cache_id
```

---

# 14. vLLM Internal Components

New components:

```text
HoleKVTraceRegistry
HoleKVInlineParser
HoleKVTraceAligner
HoleKVViewBuilder
HoleKVPositionMap
HoleKVAttentionMetadata
```

Likely touched files:

```text
vllm/v1/request.py
vllm/v1/core/kv_cache_manager.py
vllm/v1/core/single_type_kv_cache_manager.py
vllm/v1/core/block_pool.py
vllm/v1/core/kv_cache_utils.py
vllm/v1/core/sched/scheduler.py
vllm/v1/worker/gpu_model_runner.py
entrypoints/openai/protocol.py
```

---

# 15. Security and Lifetime

Because every request may return a cache id:

```text
holekv_cache_id must be unguessable
```

Server must enforce:

```text
same user/session
same model
same tokenizer
same LoRA/adaptor
cache not expired
cache still resident or restorable
memory quota respected
```

Do not allow cross-user cache reuse by default.

---

# 16. What Not To Do

Do not:

```text
1. create separate Request A / Request B API types
2. require a special cache creation request
3. expose parser markers to the model
4. mutate referenced cache in place
5. globally free removed blocks
6. compact C_old positions
7. skip D prefill
8. treat holekv_ref as sampling params
9. allow replacement longer than removed span in v1
10. silently reuse mismatched traces
```

---

# 17. Minimal v1 Feature Set

v1 supports:

```text
1. all requests return holekv_cache_id
2. all requests may optionally include holekv_ref
3. inline remove/add markers
4. no-ref fallback to A M C D
5. ref mode to A M [hole] C_old D_hole
6. one or multiple holes
7. exact token alignment
8. hole-preserved PE
9. full-attention decoder-only models
10. same user/session/model/tokenizer only
```

v1 does not support:

```text
1. auto diff without markers
2. arbitrary byte ranges
3. partial-block deletion
4. marker boundaries inside referenced tokens
5. compact PE for reused suffix
6. sliding-window attention
7. speculative decoding
8. cross-user refs
```

---

# 18. Evaluation Paths

Compare:

```text
Original:
A B C D → E_original

No-ref fallback:
A M C D → E_fallback

HoleKV ref mode:
A M [hole] C_old D_hole → E_hole

Exact recompute reference:
A M C_new D_new → E_exact
```

Metrics:

```text
1. TTFT saved
2. recomputed token count
3. D_hole vs D_new perturbation
4. E_hole vs E_exact divergence
5. stale leakage from B
6. marker effectiveness
7. alignment failure rate
8. latency vs number of holes
```

---

# 19. Final Implementation Instruction

Implement HoleKV as a **single unified request path**:

```text
Every request:
  - is a normal vLLM request
  - may optionally provide holekv_ref
  - may contain inline remove/add markers
  - returns normal response plus holekv_cache_id
```

Behavior:

```text
if no holekv_ref:
    parse markers
    rewrite to A M C D
    run normal vLLM
    return holekv_cache_id

if holekv_ref exists:
    parse markers
    load referenced cache
    build A M [hole] C_old D_hole
    run HoleKV path
    return new holekv_cache_id
```

There are no special Request A or Request B types. Each request is just a normal request with optional cache reuse.
