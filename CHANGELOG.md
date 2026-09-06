# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.4.1] - 2026-09-06

### Changed

- No-op Hugging Face `Split` pre-tokenizers preserve the 1.3.3 token IDs,
  tokens, offsets, masks, sequence IDs, and decoded text, while now assigning
  Hugging Face-compatible word ID 0 to their single pre-tokenized segment.

### Fixed

- Restored compatibility with Gemma-family Hugging Face `tokenizer.json` files that declare a no-op `Split` pre-tokenizer (#35).
  - The form is accepted only when the normalizer converts literal spaces (`U+0020`) to SentencePiece whitespace (`U+2581`) before pre-tokenization.
  - Other `Split` patterns, behaviors, and inverted matches remain rejected rather than being silently mis-tokenized.

### Added

- Added pinned Hugging Face network fixture coverage for XLM-R, E5, MiniLM, T5, Gemma, SigLIP2, and Llama tokenizer pipelines.
- Added regression coverage for the accepted normalizer-guarded `Split` form and unsupported `Split` variants.

## [1.4.0] - 2026-09-04

### Added

- Expanded Hugging Face `tokenizer.json` compatibility for SentencePiece-based BPE and Unigram tokenizers (#31).
  - Direct and nested `Precompiled` normalizers with ordered `Replace` operations.
  - `WhitespaceSplit` and `Metaspace` pre-tokenizer pipelines, including current and legacy configuration fields.
  - `TemplateProcessing` special tokens with declared BOS/EOS IDs and type IDs.
  - Tokenizer-level padding and truncation settings, including post-processor special-token preservation.
  - Unigram consecutive unknown-token fusion (`fuse_unk`) and declared added-token IDs.
- Added acceptance coverage for official Hugging Face tokenizer.json downloads and MiniLM golden IDs.

### Changed

- Exported `SpAddedToken` for tokenizer JSON metadata interoperability.
- Added configurable padding type IDs and special-token-aware truncation to `Encoding`.

### Fixed

- Malformed precompiled base64 and charsmap data now fail explicitly instead of being silently ignored.
- Corrected SentencePiece protobuf `int32` decoding without ZigZag conversion.
- Preserved BOS/EOS tokens when truncating long single, batch, and parallel-batch encodings.

## [1.3.3] - 2026-09-02

### Fixed

- Fixed loading of Hugging Face BPE `merges` entries represented as `[left, right]` pairs (#26).

### Changed

- Expanded CI matrix coverage for Dart 3.10.7, latest Dart 3.10–3.13 patch releases, and stable.
- Added contributor acknowledgements and a link to the GitHub contributors graph in the README.

## [1.3.2] - 2026-04-07

### Added

- **GitHub Actions CI Pipeline** (#9, #19)
  - `analyze` job - `dart format --set-exit-if-changed` and `dart analyze --fatal-infos`
  - `test` job - Matrix testing across Dart stable and 3.10.7 (minimum supported version)
  - Minimal permissions (`contents: read`) and concurrency group to cancel stale runs

### Changed

- Applied `dart format` to 23 files for consistent code style
- Resolved all `dart analyze --fatal-infos` issues
  - Removed deprecated `avoid_returning_null_for_future` lint rule
  - Added curly braces to if statements, `const` constructors, `final` local variables
- Added documentation comments clarifying `google/sentencepiece` proto spec compliance for default token IDs (`unkId=0, bosId=1, eosId=2, padId=-1`) (#20)

## [1.3.1] - 2026-04-03

### Added

- **HuggingFace `tokenizer.json` Format Support**
  - `HuggingFaceTokenizerLoader` class for loading HuggingFace tokenizer.json files directly
    - `fromJsonString()` / `fromMap()` - Parse from JSON string or pre-parsed map
    - `fromJsonFile()` / `fromJsonFileSync()` - Load from file (async/sync)
  - Supports both **Unigram** (Llama) and **BPE** (Gemma) model types
  - Automatic detection of special tokens (unk, bos, eos, pad) from `added_tokens` section
  - Normalizer settings inference (addDummyPrefix, escapeWhitespaces) from HuggingFace normalizer config
  - Post-processor configuration parsing (addBosToken, addEosToken) from TemplateProcessing
  - Byte fallback detection from decoder configuration
  - Added tokens handling beyond base vocabulary
  - `TokenizerJsonLoader.isHuggingFaceFormat()` - Helper to detect HuggingFace format
  - Auto-detection in `TokenizerJsonLoader` - Automatically delegates to `HuggingFaceTokenizerLoader` when HuggingFace format is detected

## [1.3.0] - 2026-02-02

### Added

- **Streaming API (HuggingFace TextStreamer Compatible)**
  - `BaseStreamer` - Abstract interface for streaming token decoders with `put()` and `end()` methods
  - `TextStreamer` - HuggingFace TextStreamer-compatible class for real-time LLM token decoding
    - `put(int tokenId)` - Add tokens as they are generated
    - `end()` - Signal end of generation and flush remaining content
    - `onFinalizedText` callback for custom text handling
    - `skipSpecialTokens` option to filter BOS/EOS/PAD tokens
    - `skipPrompt` option to skip initial prompt tokens
    - `promptLength` option to skip multiple prompt tokens
    - Word boundary heuristics for clean text emission (newlines, CJK, spaces)
  - `SentencePieceTokenizer.createTextStreamer()` - Factory for TextStreamer
  - `SentencePieceTokenizer.decodeStream()` - Stream-based token decoding
  - `SentencePieceTokenizer.decodeWithCallback()` - Callback-based token decoding

### Usage Examples

**TextStreamer (HuggingFace-compatible):**

```dart
final streamer = tokenizer.createTextStreamer();
for (final id in llmOutput) {
  streamer.put(id);
}
streamer.end();

// With custom callback
final streamer = tokenizer.createTextStreamer(
  onFinalizedText: (text, {required streamEnd}) {
    myTextController.append(text);
    if (streamEnd) myTextController.complete();
  },
);
```

**Stream-based decoding:**

```dart
final textStream = tokenizer.decodeStream(llmTokenStream);
await for (final chunk in textStream) {
  stdout.write(chunk);
}
```

**Callback-based decoding:**

```dart
tokenizer.decodeWithCallback(
  tokenIds,
  (chunk) => stdout.write(chunk),
);
```

## [1.2.2] - 2025-01-28

### Changed

- Extracted duplicate surrogate pair decoding logic in `Trie` into shared `_decodeCodePoint` helper
- Cached computed `sequenceIds` in `Encoding` to avoid O(n) recomputation on repeated access
- Added merge cache size limit (10,000 entries) to `BpeAlgorithm` and `BpeAlgorithmOptimized` to prevent unbounded memory growth
- Replaced manual loops with `fillRange` for padding initialization in `Encoding.withPadding()`

## [1.2.1] - 2025-01-28

### Changed

- Optimized batch `addTokens()` to use single typed array allocation instead of per-token expansion (O(N) instead of O(N²))
- Added input validation and defensive error handling in JSON deserialization (`TokenizerJsonLoader`)
- Consolidated duplicate `_kMaxInputLength` constant declarations

## [1.2.0] - 2025-01-17

### Added

- **JSON Serialization** - HuggingFace-compatible tokenizer.json format
  - `toJson()` - Serialize tokenizer to JSON string
  - `saveToJson()` / `saveToJsonSync()` - Save to file
  - `TokenizerJsonLoader.fromJsonString()` - Load from JSON string
  - `TokenizerJsonLoader.fromJsonFile()` / `fromJsonFileSync()` - Load from file

- **Dynamic Token Addition API**
  - `addTokens(List<String>)` - Add new tokens to vocabulary
  - `addSpecialTokens(Map<String, String>)` - Add special tokens (pad, mask, etc.)
  - `getAddedVocab()` - Get map of dynamically added tokens
  - `isAddedToken(String)` - Check if token was added dynamically
  - `getVocab({withAddedTokens})` - Get full vocabulary as Map<String, int>

- **HuggingFace-compatible Methods**
  - `tokenize(String)` - Returns List<String> of tokens
  - `tokenizeBatch(List<String>)` - Batch tokenization

- **Optimized BPE Algorithm** (`BpeAlgorithmOptimized`)
  - O(n log n) complexity using priority queue (heap)
  - ~35% faster than original algorithm on medium-length text

### Changed

- `SpVocabulary` now uses growable list for dynamic token addition support

## [1.1.0] - 2025-01-04

### Added

- Input length validation (max 500,000 characters) to prevent OOM
- Example usage file (`example/example.dart`)

### Changed

- Improved BPE algorithm efficiency
- Enhanced error messages for input validation

## [1.0.0] - 2025-01-03

### Added

- Initial release of dart_sentencepiece_tokenizer
- Pure Dart implementation with zero external dependencies
- Support for BPE (Byte Pair Encoding) algorithm used by Gemma models
- Support for Unigram algorithm used by Llama models
- Viterbi algorithm implementation for optimal Unigram segmentation
- Byte fallback support for handling unknΩown characters
- Unicode-aware Trie for efficient vocabulary lookup
- Memory-efficient typed arrays (Int32List, Uint8List) for encodings

### Features

- `SentencePieceTokenizer` - Main tokenizer class
  - `fromBytes()` - Load from protobuf bytes
  - `fromModelFile()` / `fromModelFileSync()` - Load from .model file
  - `encode()` - Encode single text
  - `encodeBatch()` - Encode multiple texts
  - `encodeBatchParallel()` - Parallel batch encoding using Isolates
  - `encodePair()` - Encode text pairs for sequence classification
  - `encodePairBatch()` - Batch encode text pairs
  - `decode()` / `decodeBatch()` - Decode token IDs back to text

- `Encoding` class with:
  - `ids` - Token IDs (Int32List)
  - `tokens` - Token strings
  - `typeIds` - Segment type IDs (Uint8List)
  - `attentionMask` - Attention mask (Uint8List)
  - `specialTokensMask` - Special token indicators (Uint8List)
  - `offsets` - Character offsets for each token
  - `withPadding()` / `withTruncation()` - Post-processing methods
  - `truncatePair()` - Static method for pair truncation

- Predefined configurations:
  - `SentencePieceConfig.llama` - Llama-style (BOS only)
  - `SentencePieceConfig.gemma` - Gemma-style (BOS + EOS)

- Truncation strategies:
  - `longestFirst` - Truncate longer sequence first
  - `onlyFirst` - Only truncate first sequence
  - `onlySecond` - Only truncate second sequence
  - `doNotTruncate` - No truncation

- Padding options:
  - Left/right padding direction
  - Fixed length or pad to longest
  - Pad to multiple of N

### Performance

- Efficient Trie-based vocabulary lookup
- Memory-optimized typed arrays reduce memory usage by ~78%
- Parallel batch processing with configurable chunk size
- Lazy evaluation where possible

### Compatibility

- Dart SDK 3.10.7+
- Compatible with Llama, Gemma, and other SentencePiece models
- HuggingFace-compatible API design
