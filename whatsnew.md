# FluidAudio Upgrade Notes: tag-20260502 → tag-20260509

## Summary

21 commits between the two tags. Changes span TTS (Kokoro crash fix, Mandarin G2P pipeline, Magpie nanocodec precision), diarization (concurrency model, timeline bug fix), and code health (StyleTTS2 add/remove, Float16 guard).

---

## New Features

### Kokoro ANE — Mandarin (v1.1-zh) (#570)

A new Mandarin variant of the Kokoro-82M ANE model is available. The same 7-stage CoreML chain is used (language-agnostic by construction); only the embedding vocabulary (177 → 171 tokens), HuggingFace subdirectory (`ANE/` → `ANE-zh/`), voice file layout (`voices/<voice>.bin`), and default voice (`af_heart` → `zf_001`) differ.

**Paired Mandarin G2P pipeline** (items 1, 2, 3, 4, 5, 6 from issue #572):
- **Erhua merging** (#574): folds trailing `儿` into the preceding syllable so `小孩儿` emits a single r-coloured token.
- **Number/date/currency verbalization** (#573): Chinese numeric expressions are converted to their spoken form before phonemization.
- **Jieba HMM tail** (#575): re-segments OOV single-character runs via 4-state B/M/E/S Viterbi, recovering proper-noun boundaries (`特朗普`, `比特币`).
- **g2pW polyphone disambiguation** (#576): int8 BERT-base CoreML classifier (152 MB) resolves ambiguous Hanzi readings (`行`/`长`/`重`/`朝`) using full sentence context. Falls back to dictionary when model is absent.
- **POS-aware tone sandhi** (#577): grammatical part-of-speech context drives the tone-3 sandhi rule instead of a simple linear scan.
- **User-supplied custom lexicon** (#578): callers can inject application-domain pronunciations that override the default dictionary.

### Magpie Nanocodec — New Precision Variants (#580, #581)

- **nanocodec v2/v3**: two new fp32 decoder builds; `decoder_step` is now pinned to ANE for ~2× wall speedup on M2.
- **nanocodec v4 (fp32 + int8 palettize)**: 8-bit kmeans-palettized weights, ~4× smaller on disk, ~11% lower peak RSS, acoustically transparent vs v3. Runs `.cpuOnly` (ANE refuses fp32 input; GPU is 50%+ slower).
- **Dual-precision API**: `MagpieNanocodecPrecision` enum exposes `.fp32`, `.fp32Pal` (v4), and existing int8 variants.

---

## Bug Fixes

### Kokoro TTS — Metal Crash on Zero-Length Input (#586)

`KokoroSynthesizer.synthesizeChunk()` now throws `TTSError.processingFailed` when `targetTokens == 0` before any MLMultiArray allocation reaches CoreML. Previously a zero-length `put_ids` tensor caused an uncatchable Metal assertion:

```
-[MTLDebugComputeCommandEncoder dispatchThreadgroups:threadsPerThreadgroup:]
    failed assertion `(threadgroupsPerGrid.width(0) * ...) must not be 0.'
```

`KokoroModelCache` also clamps all cached token lengths with `max(1, inferTokenLength(...))` at all three caching sites as defense-in-depth.

**Risk to our code**: We use `KokoroAneManager`, which drives `KokoroSynthesizer` internally. This crash could be triggered by edge-case inputs (empty strings, whitespace-only chunks, very short utterances that phonemize to zero tokens). The fix is now in place upstream; no change needed on our side.

### Diarizer Timeline — Trailing Tentative Segments (#568)

A bug caused the trailing diarizer segment to disappear when `minFramesOff` was nonzero once speech ended. The `DiarizerTimeline` now correctly finalizes segments rather than discarding tentative trailing entries.

**Risk to our code**: We use `OfflineDiarizerManager`, which goes through `DiarizerManager` and `DiarizerTimeline`. This fix improves offline diarization accuracy at the tail of an audio file; no API change required.

### Float16 Guard on non-ARM64 (#582)

Direct `Float16` memory reads in CosyVoice3 and StyleTTS2 synthesizers are now gated with `#if arch(arm64)`. Prevents compile-time errors on x86_64 Simulator builds.

---

## Refactors and Removals

### DiarizerManager De-async + SpeakerManager Struct (#591)

`DiarizerManager.performCompleteDiarization` is now synchronous (no `async`). This was possible because `SpeakerManager` — previously a class actor — is now a value-type struct with copy-on-write semantics. The compiler statically enforces exclusive ownership, eliminating the need for async dispatch.

**Risk to our code**: We do **not** call `DiarizerManager.performCompleteDiarization` directly; we use `OfflineDiarizerManager.process(audio:)`, which remains `async throws`. No change required.

### Magpie Refactor — Drop Non-Native synthesizeStream (#589)

`MagpieSynthesizer.synthesizeStream` (the non-native path that buffered full audio then streamed it) is removed. The async `StyleTTS2Synthesizer.predict` path is now the canonical streaming route.

**Risk to our code**: We do not use `MagpieTtsManager` or `MagpieSynthesizer`. No impact.

### StyleTTS2 Add/Remove Cycle

StyleTTS2 was added as a new CoreML backend (commit `ce59fb1`) and then fully removed (commit `024bd8e`) within the same week. No residual code or model references remain in the library. We never used StyleTTS2; no impact.

---

## Upgrade Risk Assessment

| Area | Risk | Notes |
|------|------|-------|
| Kokoro ANE TTS | **Low** | Crash fix for `targetTokens == 0` is strictly additive. Existing API unchanged. |
| Kokoro Mandarin | **None (unused)** | New model variant; requires separate model download (`ANE-zh/`). |
| Magpie nanocodec | **None (unused)** | We do not use Magpie or CosyVoice. |
| DiarizerManager sync API | **None** | We use `OfflineDiarizerManager` which stays async. |
| Diarizer timeline fix | **Low (beneficial)** | Trailing segment accuracy improved at no API cost. |
| SpeakerManager struct | **None** | Internal to FluidAudio diarization stack. |
| StyleTTS2 removal | **None** | Never used by us. |
| Float16 arm64 guard | **None** | Compile fix; no behavior change on arm64. |

**Overall upgrade risk: Low.** The only behavioral change relevant to our integration is the Kokoro crash fix, which is purely beneficial.

---

## New Speaker Opportunity

**Mandarin Kokoro ANE TTS** is now supported via the `ANE-zh/` model assets and the new Mandarin G2P pipeline. To support it in Privacy AI, a new `FluidAudioKokoroAneZhSpeaker` class would be needed, pointing to the `ANE-zh/` subfolder and setting `defaultVoice = "zf_001"`. The model requires a separate HuggingFace download. This is not yet implemented — inform the user before proceeding.
