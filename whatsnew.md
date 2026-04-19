# FluidAudio What's New: tag-20260412 → tag-20260419

## Commits Included

| Hash | Author | PR | Summary |
|------|--------|----|---------|
| 4ef33f0 | Alex | #521 | Fix Japanese TDT models and consolidate to unified AsrModels API |
| 3dc57c8 | Phoenix | #531 | Fix: Lower ASR minimum audio guard from 1s to 300ms |
| 38a15e4 | wangqi | — | Merge branch 'FluidInference:main' into main |

---

## PR #521 — Fix Japanese TDT models and consolidate to unified AsrModels API

### What Changed

**Three bugs fixed in the Japanese ASR path:**

1. **Model download broken**: `TdtJaModels.getRequiredModelNames()` only returned CTC models from `parakeet-ctc-0.6b-ja-coreml`, omitting the TDT-specific `Decoderv2.mlmodelc` and `Jointerv2.mlmodelc`. Japanese TDT models silently failed to download.

2. **AsrModels file name mismatch**: `AsrModels` used hardcoded names (`Decoder.mlmodelc`, `JointDecision.mlmodelc`) that do not match the Japanese TDT file names (`Decoderv2.mlmodelc`, `Jointerv2.mlmodelc`), blocking use of `AsrManager` with Japanese TDT.

3. **Code duplication**: Four specialized managers (`TdtJaManager`, `CtcJaManager`, `TdtJaModels`, `CtcJaModels`) duplicated functionality already in `AsrModels`/`AsrManager`.

### Breaking API Removals

The following public types were **deleted**:

| Removed Type | Replacement |
|---|---|
| `CtcJaManager` | Use `AsrManager` with `AsrModels.load(version: .tdtJa, ...)` |
| `CtcJaModels` | Removed; load via `AsrModels.load(version: .tdtJa, ...)` |
| `TdtJaManager` | Use `AsrManager` with `AsrModels.load(version: .tdtJa, ...)` |
| `TdtJaModels` | Removed; load via `AsrModels.load(version: .tdtJa, ...)` |
| `AsrModelVersion.ctcJa` | Removed; use `.tdtJa` (TDT is superior, same repo) |

### New Unified API for Japanese ASR

```swift
// Load Japanese TDT models (now same repo as old CTC: parakeet-ctc-0.6b-ja-coreml)
let models = try await AsrModels.load(from: modelDir, version: .tdtJa)
let manager = AsrManager()
try await manager.loadModels(models)

// Transcribe with timing info — previously unavailable via TdtJaManager
var state = TdtDecoderState.make()
let result = try await manager.transcribe(audioSamples, decoderState: &state)
print(result.text)
print(result.timings)  // now available
```

### File System Changes

The old split CI cache path (`parakeet-tdt-ja` + `parakeet-ctc-ja`) is merged to a single path:
- Old: `~/Library/Application Support/FluidAudio/Models/parakeet-tdt-ja`
- Old: `~/Library/Application Support/FluidAudio/Models/parakeet-ctc-ja`
- New: `~/Library/Application Support/FluidAudio/Models/parakeet-ja`

### Impact on iOS App

**High risk — breaking build.** `FluidAudioASR.swift` held `CtcJaManager` and `CtcJaModels` references that no longer compile. Required migration: replace `CtcJaManager`/`CtcJaModels` with `AsrManager` + `AsrModels.load(version: .tdtJa, from:)`.

Benefit: Japanese ASR now provides token timings, previously only available via English TDT models.

---

## PR #531 — Fix: Lower ASR minimum audio guard from 1s to 300ms

### What Changed

Short single-word utterances ("yes", "no", "stop") are typically 500-700 ms and were silently rejected by `AsrManager` with `ASRError.invalidAudioData` before any transcription ran. The guard was `audioSamples.count >= sampleRate` (16,000 samples = 1 s).

**New minimum**: 300 ms = 4,800 samples at 16 kHz.

### New Public API

```swift
// New constant in ASRConstants:
ASRConstants.minimumAudioDurationSeconds   // = 0.3

// New helper — same pattern as calculateEncoderFrames(from:):
ASRConstants.minimumRequiredSamples(forSampleRate: 16_000)   // = 4_800
```

### Changed Behavior

| Location | Old behavior | New behavior |
|---|---|---|
| `AsrManager.transcribe(_:decoderState:)` | Throws if < 16,000 samples | Throws if < 4,800 samples |
| `AsrManager.transcribeDiskBacked(...)` | Throws if < 16,000 samples | Throws if < 4,800 samples |
| `ASRError.invalidAudioData` message | "at least 1 second" | "at least 300ms" |

### Impact on iOS App

**Low risk — no API changes.** The guard is now less strict. If the app pre-validated audio length before calling `transcribe`, those guards may now allow audio through that was previously blocked. Recommended: update any app-side minimum-duration documentation or UI hints that referenced "1 second minimum".

---

## Risk Summary

| Change | Risk Level | Action Required |
|---|---|---|
| PR #521: Removed `CtcJaManager`, `CtcJaModels`, `TdtJaManager`, `TdtJaModels`, `.ctcJa` | **HIGH** | Migrate `FluidAudioASR.swift` to `AsrManager` + `version: .tdtJa` |
| PR #531: ASR minimum guard 1s to 300ms | **LOW** | No code change required; short utterances now transcribed |

---

## No Changes Required For

- `FluidAudioKokoroSpeaker` — TTS pipeline unaffected
- `FluidAudioPocketTTSSpeaker` — TTS pipeline unaffected
- `FluidAudioVAD` — VAD pipeline unaffected
- `FluidAudioDiarizer` — Diarizer pipeline unaffected
- `FluidAudioVoiceCloneProvider` — Voice clone pipeline unaffected
- `FluidAudioParakeetSpeaker` — Delegates to AVFoundationSpeaker; no FluidAudio ASR calls
