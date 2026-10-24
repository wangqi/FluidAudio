// swift-tools-version: 6.2
import PackageDescription
import Foundation

// Tools 6.2+ manifest: identical to Package.swift plus the `NemoTextProcessing`,
// `TTS` and `Diarizer` traits. Keep the two in sync; Package.swift serves
// toolchains < 6.2, which always link the engine and compile every subsystem. (SwiftPM 6.1 accepts the trait syntax but still
// links a trait-conditioned binary target — verified on Xcode 16.4 — so the
// opt-out is gated at 6.2.)

let package = Package(
    name: "FluidAudio",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
        .visionOS(.v2),
        .tvOS(.v17),
    ],
    products: [
        .library(
            name: "FluidAudio",
            targets: ["FluidAudio"]
        ),
        .executable(
            name: "fluidaudiocli",
            targets: ["FluidAudioCLI"]
        ),
    ],
    traits: [
        // Opt out of the NeMo text-normalization engine (~8 MB per slice, a prebuilt
        // Rust staticlib) for ASR/VAD/diarization-only apps, or when the app
        // links its own Rust runtime (#880, #888):
        //   .package(url: ..., traits: [])
        // TTS frontends and `TextNormalizer` then pass text through unchanged
        // and report `isNativeAvailable == false`.
        .trait(
            name: "NemoTextProcessing",
            description: "Link the bundled NeMo text-normalization engine (TTS frontends, ITN)."
        ),
        // Opt out of whole subsystems for apps that never reach them (#990):
        //   .package(url: ..., traits: [])            // ASR/VAD only
        //   .package(url: ..., traits: ["Diarizer"])  // ASR/VAD + diarization
        .trait(
            name: "TTS",
            description: "Text-to-speech backends (Kokoro, PocketTTS, LuxTTS, ...) and their bundled G2P resources."
        ),
        .trait(
            name: "Diarizer",
            description: "Speaker diarization (pyannote, Sortformer, LS-EEND, Nemotron 3)."
        ),
        // NemoTextProcessing off by default in this fork. `NemoTextProcessing.xcframework` is a static-library
        // xcframework, so Xcode unpacks its headers into `$BUILT_PRODUCTS_DIR/include/` — the
        // same place the app's `libgit2.xcframework` (SwiftGit2) puts its own, and both ship a
        // `module.modulemap`, which fails the build with "Multiple commands produce
        // .../include/module.modulemap". The app already ran on `TextNormalizer`'s
        // `isNativeAvailable == false` path (it never linked a NeMo staticlib), so opting out
        // preserves existing behaviour. To turn it back on, first repackage libgit2 as a
        // framework-style xcframework so the two stop sharing `include/`.
        // TTS and Diarizer (#990) stay on: the app uses both.
        // wangqi modified 2026-10-24
        .default(enabledTraits: ["TTS", "Diarizer"]),
    ],
    dependencies: [],
    targets: [
        .target(
            name: "FluidAudio",
            dependencies: [
                "FastClusterWrapper",
                "MachTaskSelfWrapper",
                // The prebuilt xcframework has no visionOS or tvOS slice.
                .target(
                    name: "NemoTextProcessing",
                    condition: .when(platforms: [.macOS, .iOS, .macCatalyst], traits: ["NemoTextProcessing"])
                ),
                .target(name: "LuxTtsG2pResources", condition: .when(traits: ["TTS"])),
            ],
            path: "Sources/FluidAudio",
            exclude: ["ASR/Parakeet/Unified/benchmark.md"],
            // Build under Swift 5 language mode until strict concurrency issues are resolved.
            // This manifest shadows Package.swift on Swift 6.2+ toolchains, so the pin has to be
            // repeated here or the fork's mutable overrides fail to compile. Upstream moved the
            // G2P resources into LuxTtsG2pResources, so the pin now sits on the FluidAudio target.
            // wangqi modified 2026-10-24
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        // Separate target so the ~1 MB G2P bundle drops out with the `TTS` trait.
        .target(
            name: "LuxTtsG2pResources",
            path: "Sources/LuxTtsG2pResources",
            resources: [
                // Keep .process: .copy of a Resources-named directory breaks Apple code signing on iOS.
                .process("Resources")
            ]
        ),
        // Byte-exact NeMo text normalization (FST engine, all 7 languages).
        // Prebuilt xcframework from FluidInference/text-processing-rs v0.3.1
        // (macOS, iOS, iOS Simulator and Mac Catalyst slices).
        .binaryTarget(
            name: "NemoTextProcessing",
            url:
                "https://github.com/FluidInference/text-processing-rs/releases/download/v0.3.1/NemoTextProcessing.xcframework.zip",
            checksum: "5fa8c10d4ec26c1bb2413125f351a7222a4c68a23b74476680fbada7e26fc6aa"
        ),
        .target(
            name: "FastClusterWrapper",
            path: "Sources/FastClusterWrapper",
            publicHeadersPath: "include"
        ),
        .target(
            name: "MachTaskSelfWrapper",
            path: "Sources/MachTaskSelfWrapper",
            publicHeadersPath: "include"
        ),
        .executableTarget(
            name: "FluidAudioCLI",
            dependencies: ["FluidAudio"],
            path: "Sources/FluidAudioCLI",
            exclude: ["README.md"],
            resources: [
                .process("Utils/english.json")
            ]
        ),
        .testTarget(
            name: "FluidAudioTests",
            dependencies: [
                "FluidAudio",
                "FluidAudioCLI",
            ],
            resources: [
                .process("TTS/LuxTts/Resources"),
                .process("TTS/PocketTTS/Fixtures"),
                // Real recordings (cleared for public release by the speaker) for the
                // streaming final-window regression, issue #855.
                .copy("ASR/Parakeet/SlidingWindow/Fixtures"),
            ]
        ),
    ],
    cxxLanguageStandard: .cxx17
)
