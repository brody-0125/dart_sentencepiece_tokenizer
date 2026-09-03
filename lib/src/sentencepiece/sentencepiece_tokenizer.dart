import 'dart:async';
import 'dart:isolate';
import 'dart:typed_data';

import '../encoding.dart';
import 'algorithm/bpe_algorithm.dart';
import 'algorithm/tokenization_algorithm.dart';
import 'algorithm/unigram_algorithm.dart';
import 'model/model_proto.dart';
import 'model/sentencepiece_model.dart';
import 'normalizer/sp_normalizer.dart';
import 'streaming/text_streamer.dart';
import 'vocabulary/sp_vocabulary.dart';

export '../encoding.dart' show Encoding, TruncationStrategy;
export 'model/model_proto.dart' show ModelType;

const _kMinBatchSizeForParallel = 8;

/// Maximum input text length in characters (1MB of UTF-16 characters).
/// Prevents OOM from extremely large inputs.
const _kMaxInputLength = 500000;

/// Padding direction for batch encoding.
enum SpPaddingDirection { right, left }

/// Padding configuration for SentencePiece tokenizer.
class SpPaddingConfig {
  final SpPaddingDirection direction;
  final int? length;
  final int? padToMultipleOf;
  final int? padTokenId;
  final String? padToken;
  final int padTypeId;

  const SpPaddingConfig({
    this.direction = SpPaddingDirection.right,
    this.length,
    this.padToMultipleOf,
    this.padTokenId,
    this.padToken,
    this.padTypeId = 0,
  });
}

/// Truncation direction for encoding.
enum SpTruncationDirection { right, left }

/// Truncation configuration for SentencePiece tokenizer.
class SpTruncationConfig {
  final int maxLength;
  final SpTruncationDirection direction;
  final TruncationStrategy strategy;

  const SpTruncationConfig({
    required this.maxLength,
    this.direction = SpTruncationDirection.right,
    this.strategy = TruncationStrategy.longestFirst,
  });
}

/// Configuration for SentencePiece tokenizer.
class SentencePieceConfig {
  final bool addBosToken;
  final bool addEosToken;
  final int pairEosTokensBetweenSequences;
  final int pairTypeId;
  final int pairSpecialTypeId;

  const SentencePieceConfig({
    this.addBosToken = false,
    this.addEosToken = false,
    this.pairEosTokensBetweenSequences = 1,
    this.pairTypeId = 1,
    this.pairSpecialTypeId = 1,
  });

  /// Gemma default configuration.
  static const gemma = SentencePieceConfig(
    addBosToken: true,
    addEosToken: true,
  );

  /// Llama default configuration.
  static const llama = SentencePieceConfig(
    addBosToken: true,
    addEosToken: false,
  );
}

/// An added token declared by a Hugging Face tokenizer.json.
class SpAddedToken {
  final int id;
  final String content;
  final bool special;
  final bool singleWord;
  final bool lstrip;
  final bool rstrip;
  final bool normalized;

  const SpAddedToken({
    required this.id,
    required this.content,
    this.special = false,
    this.singleWord = false,
    this.lstrip = false,
    this.rstrip = false,
    this.normalized = true,
  });
}

/// Pure Dart SentencePiece tokenizer.
///
/// Supports both BPE (Gemma) and Unigram (Llama) algorithms. Instances can be
/// created from native SentencePiece models or by
/// `HuggingFaceTokenizerLoader` from a SentencePiece-oriented
/// `tokenizer.json` file.
class SentencePieceTokenizer {
  final SpVocabulary vocab;
  final SentencePieceConfig config;
  final SpNormalizer _normalizer;
  final TokenizationAlgorithm _algorithm;
  final ModelType modelType;
  final PreTokenizerSpec? _preTokenizer;
  final List<SpAddedToken> _addedTokens;

  /// Access normalizer for serialization.
  SpNormalizer get normalizer => _normalizer;

  /// Access the configured pre-tokenizer, when the source JSON defines one.
  PreTokenizerSpec? get preTokenizer => _preTokenizer;

  /// Whether this Unigram model fuses consecutive unknown tokens.
  bool get fuseUnknownTokens {
    final algorithm = _algorithm;
    return algorithm is UnigramAlgorithm && algorithm.fuseUnk;
  }

  SpPaddingConfig? _paddingConfig;
  SpTruncationConfig? _truncationConfig;

  SentencePieceTokenizer._({
    required this.vocab,
    required this.config,
    required SpNormalizer normalizer,
    required TokenizationAlgorithm algorithm,
    required this.modelType,
    required PreTokenizerSpec? preTokenizer,
    required List<SpAddedToken> addedTokens,
  }) : _normalizer = normalizer,
       _algorithm = algorithm,
       _preTokenizer = preTokenizer,
       _addedTokens = List.unmodifiable(addedTokens);

  /// Load tokenizer from a .model file asynchronously.
  static Future<SentencePieceTokenizer> fromModelFile(
    String path, {
    SentencePieceConfig config = const SentencePieceConfig(),
  }) async {
    final model = await SentencePieceModelLoader.fromFile(path);
    return _createFromModel(model, config);
  }

  /// Load tokenizer from a .model file synchronously.
  static SentencePieceTokenizer fromModelFileSync(
    String path, {
    SentencePieceConfig config = const SentencePieceConfig(),
  }) {
    final model = SentencePieceModelLoader.fromFileSync(path);
    return _createFromModel(model, config);
  }

  /// Create tokenizer from model bytes.
  static SentencePieceTokenizer fromBytes(
    List<int> bytes, {
    SentencePieceConfig config = const SentencePieceConfig(),
  }) {
    final data = bytes is Uint8List ? bytes : Uint8List.fromList(bytes);
    final model = SentencePieceModelLoader.fromBytes(data);
    return _createFromModel(model, config);
  }

  static SentencePieceTokenizer _createFromModel(
    SentencePieceModel model,
    SentencePieceConfig config, {
    List<SpAddedToken> addedTokens = const [],
  }) {
    final vocab = SpVocabulary.fromModel(model);
    final normalizer = SpNormalizer.fromSpec(model.normalizerSpec);
    final algorithm = _createAlgorithm(model, vocab);

    return SentencePieceTokenizer._(
      vocab: vocab,
      config: config,
      normalizer: normalizer,
      algorithm: algorithm,
      modelType: model.trainerSpec.modelType,
      preTokenizer: model.preTokenizerSpec,
      addedTokens: addedTokens,
    );
  }

  /// Create tokenizer from a SentencePieceModel.
  ///
  /// This is useful for creating tokenizers from JSON deserialization
  /// or other non-file sources.
  static SentencePieceTokenizer fromModel(
    SentencePieceModel model, {
    SentencePieceConfig config = const SentencePieceConfig(),
    List<SpAddedToken> addedTokens = const [],
  }) {
    return _createFromModel(model, config, addedTokens: addedTokens);
  }

  static TokenizationAlgorithm _createAlgorithm(
    SentencePieceModel model,
    SpVocabulary vocab,
  ) {
    final byteFallback = model.trainerSpec.byteFallback;

    switch (model.trainerSpec.modelType) {
      case ModelType.bpe:
        return BpeAlgorithm(vocab: vocab, byteFallback: byteFallback);
      case ModelType.unigram:
        return UnigramAlgorithm(
          vocab: vocab,
          byteFallback: byteFallback,
          fuseUnk: model.trainerSpec.fuseUnk,
        );
      default:
        throw UnsupportedError(
          'Unsupported model type: ${model.trainerSpec.modelType}',
        );
    }
  }

  /// Get current padding configuration.
  SpPaddingConfig? get padding => _paddingConfig;

  /// Get current truncation configuration.
  SpTruncationConfig? get truncation => _truncationConfig;

  /// Enable padding with the specified configuration.
  SentencePieceTokenizer enablePadding({
    SpPaddingDirection direction = SpPaddingDirection.right,
    int? length,
    int? padToMultipleOf,
    int? padTokenId,
    String? padToken,
    int padTypeId = 0,
  }) {
    _paddingConfig = SpPaddingConfig(
      direction: direction,
      length: length,
      padToMultipleOf: padToMultipleOf,
      padTokenId: padTokenId,
      padToken: padToken,
      padTypeId: padTypeId,
    );
    return this;
  }

  /// Disable padding.
  SentencePieceTokenizer noPadding() {
    _paddingConfig = null;
    return this;
  }

  /// Enable truncation with the specified configuration.
  SentencePieceTokenizer enableTruncation({
    required int maxLength,
    SpTruncationDirection direction = SpTruncationDirection.right,
    TruncationStrategy strategy = TruncationStrategy.longestFirst,
  }) {
    _truncationConfig = SpTruncationConfig(
      maxLength: maxLength,
      direction: direction,
      strategy: strategy,
    );
    return this;
  }

  /// Disable truncation.
  SentencePieceTokenizer noTruncation() {
    _truncationConfig = null;
    return this;
  }

  /// Number of special tokens added during encoding.
  ///
  /// When [isPair] is true, calculates tokens for pair encoding:
  /// - BOS (if enabled) + configured EOS separators + EOS at end (if enabled)
  int numSpecialTokensToAdd({
    bool? addBosToken,
    bool? addEosToken,
    bool isPair = false,
  }) {
    var count = 0;
    final shouldAddBos = addBosToken ?? config.addBosToken;
    final shouldAddEos = addEosToken ?? config.addEosToken;

    if (shouldAddBos && vocab.bosId >= 0) count++;
    if (shouldAddEos && vocab.eosId >= 0) {
      count++; // EOS at end
      if (isPair) count += config.pairEosTokensBetweenSequences;
    }
    return count;
  }

  /// Encode text into token IDs.
  ///
  /// Throws [ArgumentError] if text exceeds maximum input length.
  Encoding encode(String text, {bool? addSpecialTokens}) {
    if (text.length > _kMaxInputLength) {
      throw ArgumentError(
        'Input text too long: ${text.length} characters exceeds maximum of $_kMaxInputLength',
      );
    }

    final shouldAddBos = addSpecialTokens ?? config.addBosToken;
    final shouldAddEos = addSpecialTokens ?? config.addEosToken;

    final tokenized = _tokenizeWithMetadata(text);

    // Build encoding
    final builder = EncodingBuilder();

    // Add BOS token
    if (shouldAddBos && vocab.bosId >= 0) {
      builder.addSpecialToken(
        token: vocab.bosPiece,
        id: vocab.bosId,
        typeId: 0,
      );
    }

    // Add content tokens
    for (final item in tokenized) {
      builder.addToken(
        token: item.token,
        id: item.id,
        typeId: 0,
        offset: item.offset,
        wordId: item.wordId,
      );
    }

    // Add EOS token
    if (shouldAddEos && vocab.eosId >= 0) {
      builder.addSpecialToken(
        token: vocab.eosPiece,
        id: vocab.eosId,
        typeId: 0,
      );
    }

    return _applyPostProcessing(builder.build());
  }

  Encoding _applyPostProcessing(
    Encoding encoding, {
    bool applyTruncation = true,
  }) {
    var result = encoding;

    if (applyTruncation && _truncationConfig != null) {
      result = result.withTruncation(
        maxLength: _truncationConfig!.maxLength,
        truncateFromEnd:
            _truncationConfig!.direction == SpTruncationDirection.right,
        preserveSpecialTokens: true,
      );
    }

    if (_paddingConfig != null) {
      final padOnRight = _paddingConfig!.direction == SpPaddingDirection.right;
      final padTokenId =
          _paddingConfig!.padTokenId ?? (vocab.padId >= 0 ? vocab.padId : 0);
      final padToken = _paddingConfig!.padToken ?? vocab.padPiece;

      if (_paddingConfig!.length != null) {
        result = result.withPadding(
          targetLength: _paddingConfig!.length!,
          padTokenId: padTokenId,
          padToken: padToken,
          padOnRight: padOnRight,
          padTypeId: _paddingConfig!.padTypeId,
        );
      }

      if (_paddingConfig!.padToMultipleOf != null) {
        result = result.withPaddingToMultipleOf(
          multiple: _paddingConfig!.padToMultipleOf!,
          padTokenId: padTokenId,
          padToken: padToken,
          padOnRight: padOnRight,
          padTypeId: _paddingConfig!.padTypeId,
        );
      }
    }

    return result;
  }

  List<Encoding> _applyBatchPostProcessing(
    List<Encoding> encodings, {
    bool applyTruncation = true,
  }) {
    if (encodings.isEmpty) return encodings;

    var results = encodings;

    if (applyTruncation && _truncationConfig != null) {
      results = results
          .map(
            (e) => e.withTruncation(
              maxLength: _truncationConfig!.maxLength,
              truncateFromEnd:
                  _truncationConfig!.direction == SpTruncationDirection.right,
              preserveSpecialTokens: true,
            ),
          )
          .toList();
    }

    if (_paddingConfig != null) {
      final padOnRight = _paddingConfig!.direction == SpPaddingDirection.right;
      final padTokenId =
          _paddingConfig!.padTokenId ?? (vocab.padId >= 0 ? vocab.padId : 0);
      final padToken = _paddingConfig!.padToken ?? vocab.padPiece;

      int targetLength;
      if (_paddingConfig!.length != null) {
        targetLength = _paddingConfig!.length!;
      } else {
        targetLength = results
            .map((e) => e.length)
            .reduce((a, b) => a > b ? a : b);
      }

      if (_paddingConfig!.padToMultipleOf != null) {
        final multiple = _paddingConfig!.padToMultipleOf!;
        final remainder = targetLength % multiple;
        if (remainder != 0) {
          targetLength += multiple - remainder;
        }
      }

      results = results
          .map(
            (e) => e.withPadding(
              targetLength: targetLength,
              padTokenId: padTokenId,
              padToken: padToken,
              padOnRight: padOnRight,
              padTypeId: _paddingConfig!.padTypeId,
            ),
          )
          .toList();
    }

    return results;
  }

  /// Encode a pair of sequences for tasks like question answering or NLI.
  ///
  /// The first sequence gets typeId=0 and the second sequence gets the
  /// configured pair type ID. Special tokens are handled according to config:
  /// - BOS token (if enabled) is added at the start
  /// - Configured EOS separators (if enabled) are added between sequences,
  ///   followed by the final EOS token
  ///
  /// The [strategy] parameter controls how truncation is applied when the
  /// combined sequences exceed [maxLength]:
  /// - [TruncationStrategy.longestFirst]: Truncate the longer sequence first
  /// - [TruncationStrategy.onlyFirst]: Only truncate the first sequence
  /// - [TruncationStrategy.onlySecond]: Only truncate the second sequence
  /// - [TruncationStrategy.doNotTruncate]: Don't truncate (may exceed maxLength)
  Encoding encodePair(
    String text,
    String textPair, {
    bool? addSpecialTokens,
    int? maxLength,
    TruncationStrategy? strategy,
  }) {
    if (text.length > _kMaxInputLength) {
      throw ArgumentError(
        'First input text too long: ${text.length} characters exceeds maximum of $_kMaxInputLength',
      );
    }
    if (textPair.length > _kMaxInputLength) {
      throw ArgumentError(
        'Second input text too long: ${textPair.length} characters exceeds maximum of $_kMaxInputLength',
      );
    }

    final shouldAddBos = addSpecialTokens ?? config.addBosToken;
    final shouldAddEos = addSpecialTokens ?? config.addEosToken;
    final configuredTruncation = _truncationConfig;
    final effectiveMaxLength = maxLength ?? configuredTruncation?.maxLength;
    final effectiveStrategy =
        strategy ??
        configuredTruncation?.strategy ??
        TruncationStrategy.longestFirst;

    // Calculate number of special tokens for truncation
    var numSpecialTokens = 0;
    if (shouldAddBos && vocab.bosId >= 0) numSpecialTokens++;
    if (shouldAddEos && vocab.eosId >= 0) {
      numSpecialTokens += 1 + config.pairEosTokensBetweenSequences;
    }

    // Encode both sequences without special tokens
    final savedPadding = _paddingConfig;
    final savedTruncation = _truncationConfig;
    _paddingConfig = null;
    _truncationConfig = null;

    final encoding1 = _encodeSequence(text, typeId: 0, sequenceId: 0);
    final encoding2 = _encodeSequence(
      textPair,
      typeId: config.pairTypeId,
      sequenceId: 1,
    );

    _paddingConfig = savedPadding;
    _truncationConfig = savedTruncation;

    // Apply pair truncation if maxLength is specified
    Encoding truncated1;
    Encoding truncated2;

    if (effectiveMaxLength != null) {
      (truncated1, truncated2) = Encoding.truncatePair(
        encodingA: encoding1,
        encodingB: encoding2,
        maxLength: effectiveMaxLength,
        strategy: effectiveStrategy,
        numSpecialTokens: numSpecialTokens,
      );
    } else {
      truncated1 = encoding1;
      truncated2 = encoding2;
    }

    // Build final encoding with special tokens
    final builder = EncodingBuilder();

    // Add BOS token
    if (shouldAddBos && vocab.bosId >= 0) {
      builder.addSpecialToken(
        token: vocab.bosPiece,
        id: vocab.bosId,
        typeId: 0,
      );
    }

    // Add first sequence tokens
    for (var i = 0; i < truncated1.length; i++) {
      builder.addToken(
        token: truncated1.tokens[i],
        id: truncated1.ids[i],
        typeId: 0,
        offset: truncated1.offsets[i],
        wordId: truncated1.wordIds[i],
        sequenceId: 0,
      );
    }

    // Add separator EOS tokens (between sequences)
    for (var i = 0; i < config.pairEosTokensBetweenSequences; i++) {
      if (shouldAddEos && vocab.eosId >= 0) {
        builder.addSpecialToken(
          token: vocab.eosPiece,
          id: vocab.eosId,
          typeId: config.pairSpecialTypeId,
        );
      }
    }

    // Add second sequence tokens
    for (var i = 0; i < truncated2.length; i++) {
      builder.addToken(
        token: truncated2.tokens[i],
        id: truncated2.ids[i],
        typeId: config.pairTypeId,
        offset: truncated2.offsets[i],
        wordId: truncated2.wordIds[i],
        sequenceId: 1,
      );
    }

    // Add final EOS token
    if (shouldAddEos && vocab.eosId >= 0) {
      builder.addSpecialToken(
        token: vocab.eosPiece,
        id: vocab.eosId,
        typeId: config.pairSpecialTypeId,
      );
    }

    return _applyPostProcessing(
      builder.build(),
      applyTruncation: effectiveMaxLength == null,
    );
  }

  /// Encode a sequence without special tokens, for internal use.
  Encoding _encodeSequence(
    String text, {
    required int typeId,
    required int sequenceId,
  }) {
    final tokenized = _tokenizeWithMetadata(text);

    final builder = EncodingBuilder();
    for (final item in tokenized) {
      builder.addToken(
        token: item.token,
        id: item.id,
        typeId: typeId,
        offset: item.offset,
        wordId: item.wordId,
        sequenceId: sequenceId,
      );
    }

    return builder.build();
  }

  /// Encode multiple text pairs.
  List<Encoding> encodePairBatch(
    List<(String, String)> textPairs, {
    bool? addSpecialTokens,
    int? maxLength,
    TruncationStrategy? strategy,
  }) {
    final savedPadding = _paddingConfig;
    final savedTruncation = _truncationConfig;
    final effectiveMaxLength = maxLength ?? savedTruncation?.maxLength;
    final effectiveStrategy =
        strategy ??
        savedTruncation?.strategy ??
        TruncationStrategy.longestFirst;
    _paddingConfig = null;
    _truncationConfig = null;

    final encodings = textPairs
        .map(
          (pair) => encodePair(
            pair.$1,
            pair.$2,
            addSpecialTokens: addSpecialTokens,
            maxLength: effectiveMaxLength,
            strategy: effectiveStrategy,
          ),
        )
        .toList();

    _paddingConfig = savedPadding;
    _truncationConfig = savedTruncation;

    return _applyBatchPostProcessing(
      encodings,
      applyTruncation: effectiveMaxLength == null,
    );
  }

  /// Encode multiple texts.
  List<Encoding> encodeBatch(List<String> texts, {bool? addSpecialTokens}) {
    final savedPadding = _paddingConfig;
    final savedTruncation = _truncationConfig;
    _paddingConfig = null;
    _truncationConfig = null;

    final encodings = texts
        .map((text) => encode(text, addSpecialTokens: addSpecialTokens))
        .toList();

    _paddingConfig = savedPadding;
    _truncationConfig = savedTruncation;

    return _applyBatchPostProcessing(encodings);
  }

  /// Encode multiple texts in parallel using Isolates.
  Future<List<Encoding>> encodeBatchParallel(
    List<String> texts, {
    bool? addSpecialTokens,
    int? numWorkers,
  }) async {
    if (_addedTokens.isNotEmpty) {
      return encodeBatch(texts, addSpecialTokens: addSpecialTokens);
    }
    if (texts.length < _kMinBatchSizeForParallel) {
      return encodeBatch(texts, addSpecialTokens: addSpecialTokens);
    }

    final workerCount = numWorkers ?? _getOptimalWorkerCount(texts.length);
    final chunkSize = (texts.length / workerCount).ceil();

    final modelData = _SerializableModelData.fromTokenizer(this);
    final futures = <Future<List<_EncodingData>>>[];

    for (var i = 0; i < workerCount; i++) {
      final start = i * chunkSize;
      if (start >= texts.length) break;

      final end = (start + chunkSize).clamp(0, texts.length);
      final chunk = texts.sublist(start, end);

      futures.add(
        Isolate.run(
          () =>
              _encodeChunkInIsolate(chunk, modelData, config, addSpecialTokens),
        ),
      );
    }

    final results = await Future.wait(futures);

    final encodings = <Encoding>[];
    for (final chunkResults in results) {
      for (final data in chunkResults) {
        encodings.add(data.toEncoding());
      }
    }

    return _applyBatchPostProcessing(encodings);
  }

  int _getOptimalWorkerCount(int batchSize) {
    const maxWorkers = 4;
    const minItemsPerWorker = 4;

    final workersByItems = (batchSize / minItemsPerWorker).floor();
    return workersByItems.clamp(1, maxWorkers);
  }

  /// Decode token IDs back to text.
  String decode(List<int> ids, {bool skipSpecialTokens = true}) {
    final buffer = StringBuffer();

    for (final id in ids) {
      if (skipSpecialTokens && vocab.isSpecialToken(id)) {
        continue;
      }

      final piece = vocab.idToPiece(id);

      // Handle byte tokens
      if (vocab.isByteToken(id)) {
        final byteValue = vocab.byteTokenToValue(id);
        if (byteValue != null) {
          buffer.writeCharCode(byteValue);
        }
        continue;
      }

      buffer.write(piece);
    }

    return _normalizer.denormalize(buffer.toString());
  }

  /// Decode multiple token ID sequences.
  List<String> decodeBatch(
    List<List<int>> idsBatch, {
    bool skipSpecialTokens = true,
  }) {
    return idsBatch
        .map((ids) => decode(ids, skipSpecialTokens: skipSpecialTokens))
        .toList();
  }

  /// Decode a stream of token IDs to a stream of text chunks.
  ///
  /// This is designed for LLM streaming output where tokens arrive one at a time.
  /// The decoder handles incomplete UTF-8 sequences from byte tokens by buffering
  /// until complete characters can be formed.
  ///
  /// Example:
  /// ```dart
  /// final textStream = tokenizer.decodeStream(llmTokenStream);
  /// await for (final chunk in textStream) {
  ///   stdout.write(chunk); // Display incrementally
  /// }
  /// ```
  Stream<String> decodeStream(
    Stream<int> tokenIds, {
    bool skipSpecialTokens = true,
  }) async* {
    final chunks = <String>[];
    final streamer = createTextStreamer(
      skipSpecialTokens: skipSpecialTokens,
      onFinalizedText: (text, {required streamEnd}) {
        if (text.isNotEmpty) chunks.add(text);
      },
    );

    await for (final id in tokenIds) {
      streamer.put(id);
      // Yield any chunks that were emitted
      while (chunks.isNotEmpty) {
        yield chunks.removeAt(0);
      }
    }

    // Flush remaining content
    streamer.end();
    while (chunks.isNotEmpty) {
      yield chunks.removeAt(0);
    }
  }

  /// Decode token IDs with a callback for each text chunk.
  ///
  /// This is useful when you want to process text incrementally without
  /// using streams (e.g., for direct UI updates).
  ///
  /// Example:
  /// ```dart
  /// tokenizer.decodeWithCallback(
  ///   tokenIds,
  ///   (chunk) => stdout.write(chunk),
  /// );
  /// ```
  void decodeWithCallback(
    List<int> ids,
    void Function(String chunk) onChunk, {
    bool skipSpecialTokens = true,
  }) {
    final streamer = createTextStreamer(
      skipSpecialTokens: skipSpecialTokens,
      onFinalizedText: (text, {required streamEnd}) {
        if (text.isNotEmpty) onChunk(text);
      },
    );

    for (final id in ids) {
      streamer.put(id);
    }
    streamer.end();
  }

  /// Create a TextStreamer for HuggingFace-compatible streaming.
  ///
  /// This is the recommended API for LLM token streaming, matching
  /// HuggingFace's `TextStreamer` interface with `put()` and `end()` methods.
  ///
  /// Example:
  /// ```dart
  /// final streamer = tokenizer.createTextStreamer(
  ///   onFinalizedText: (text, {required streamEnd}) => stdout.write(text),
  /// );
  /// for (final tokenId in llmOutput) {
  ///   streamer.put(tokenId);
  /// }
  /// streamer.end();
  /// ```
  ///
  /// [skipSpecialTokens] - Whether to skip special tokens like BOS/EOS.
  /// [skipPrompt] - Whether to skip the initial prompt tokens.
  /// [promptLength] - Number of prompt tokens to skip (when [skipPrompt] is true).
  /// [onFinalizedText] - Callback for finalized text chunks.
  TextStreamer createTextStreamer({
    bool skipSpecialTokens = true,
    bool skipPrompt = false,
    int promptLength = 1,
    OnFinalizedText? onFinalizedText,
  }) {
    return TextStreamer(
      this,
      skipSpecialTokens: skipSpecialTokens,
      skipPrompt: skipPrompt,
      promptLength: promptLength,
      onFinalizedText: onFinalizedText,
    );
  }

  /// Convert tokens to IDs.
  List<int> convertTokensToIds(List<String> tokens) {
    return tokens.map((t) => vocab.pieceToId(t)).toList();
  }

  /// Convert IDs to tokens.
  List<String> convertIdsToTokens(List<int> ids) {
    return ids.map((id) => vocab.idToPiece(id)).toList();
  }

  /// Get vocabulary size.
  int get vocabSize => vocab.size;

  /// Add new tokens to the vocabulary.
  ///
  /// Returns the number of tokens actually added (excluding duplicates).
  /// New tokens can be used in tokenization and will be recognized.
  ///
  /// ```dart
  /// final added = tokenizer.addTokens(['<custom>', '<domain>']);
  /// print('Added $added tokens');
  /// ```
  int addTokens(List<String> tokens) {
    return vocab.addTokens(tokens);
  }

  /// Add special tokens to the vocabulary.
  ///
  /// Special tokens are tokens that should be treated specially during
  /// encoding/decoding (e.g., skip in decode with skipSpecialTokens).
  ///
  /// Supported keys: 'pad_token', 'mask_token', 'sep_token', 'cls_token',
  /// or any custom key.
  ///
  /// ```dart
  /// tokenizer.addSpecialTokens({
  ///   'pad_token': '<pad>',
  ///   'mask_token': '<mask>',
  /// });
  /// ```
  int addSpecialTokens(Map<String, String> specialTokens) {
    var added = 0;
    for (final entry in specialTokens.entries) {
      final key = entry.key;
      final token = entry.value;

      final existingId = vocab.contains(token) ? vocab.pieceToId(token) : null;
      final id = vocab.addSpecialToken(token);

      // Update special token references
      switch (key) {
        case 'pad_token':
          vocab.padId = id;
          vocab.padPiece = token;
          break;
        case 'bos_token':
          vocab.bosId = id;
          vocab.bosPiece = token;
          break;
        case 'eos_token':
          vocab.eosId = id;
          vocab.eosPiece = token;
          break;
      }

      if (existingId == null) added++;
    }
    return added;
  }

  /// Get all dynamically added tokens.
  Map<String, int> getAddedVocab() => vocab.getAddedVocab();

  /// Check if a token was added dynamically.
  bool isAddedToken(String token) => vocab.isAddedToken(token);

  /// Get the full vocabulary as a map from token to ID.
  ///
  /// If [withAddedTokens] is false, only returns the original vocabulary.
  Map<String, int> getVocab({bool withAddedTokens = true}) {
    if (withAddedTokens) {
      return vocab.vocabularyMap;
    }
    // Filter out added tokens
    final result = <String, int>{};
    for (final entry in vocab.vocabularyMap.entries) {
      if (!vocab.isAddedToken(entry.key)) {
        result[entry.key] = entry.value;
      }
    }
    return result;
  }

  /// Tokenize text into token strings without creating full Encoding.
  ///
  /// This is a lightweight alternative to [encode] when you only need
  /// the token strings.
  ///
  /// ```dart
  /// final tokens = tokenizer.tokenize('Hello world');
  /// // → ['▁Hello', '▁world']
  /// ```
  List<String> tokenize(String text) {
    if (text.isEmpty) return [];
    if (text.length > _kMaxInputLength) {
      throw ArgumentError(
        'Input text too long: ${text.length} characters exceeds maximum of $_kMaxInputLength',
      );
    }

    return [for (final token in _tokenizeWithMetadata(text)) token.token];
  }

  List<_TokenizedPiece> _tokenizeWithMetadata(String original) {
    final matches = _findAddedTokenMatches(original);
    if (matches.isEmpty) return _tokenizeWithoutAddedTokens(original);

    final result = <_TokenizedPiece>[];
    var cursor = 0;
    for (final match in matches) {
      if (cursor < match.start) {
        result.addAll(
          _shiftTokenOffsets(
            _tokenizeWithoutAddedTokens(
              _sliceRunes(original, cursor, match.start),
            ),
            cursor,
          ),
        );
      }
      result.add(
        _TokenizedPiece(
          id: match.token.id,
          token: match.token.content,
          offset: (match.start, match.end),
          wordId: null,
        ),
      );
      cursor = match.end;
    }
    if (cursor < original.runes.length) {
      result.addAll(
        _shiftTokenOffsets(
          _tokenizeWithoutAddedTokens(_sliceRunes(original, cursor)),
          cursor,
        ),
      );
    }
    return result;
  }

  List<_TokenizedPiece> _tokenizeWithoutAddedTokens(String original) {
    final normalized = _normalizer.normalize(original);
    final alignment = _alignNormalizedText(original, normalized);
    final chunks = _buildPipelineChunks(normalized, alignment: alignment);
    final result = <_TokenizedPiece>[];

    for (final chunk in chunks) {
      final ids = _algorithm.tokenize(chunk.text);
      final chars = chunk.text.runes.toList();
      var cursor = 0;
      for (final id in ids) {
        final token = vocab.idToPiece(id);
        final tokenChars = token.runes.toList();
        final start = cursor;
        if (_startsWith(chars, cursor, tokenChars)) {
          cursor += tokenChars.length;
        } else if (cursor < chars.length) {
          cursor++;
        }
        final end = cursor;
        result.add(
          _TokenizedPiece(
            id: id,
            token: token,
            offset: _sourceOffset(
              chunk.alignment,
              start,
              end,
              leadingMarkerLength: chunk.leadingMarkerLength,
              includeLeadingMarker: chunk.includeLeadingMarker,
            ),
            wordId: chunk.wordId,
          ),
        );
      }
    }
    return result;
  }

  List<_TokenizedPiece> _shiftTokenOffsets(
    List<_TokenizedPiece> pieces,
    int offset,
  ) {
    return [
      for (final piece in pieces)
        _TokenizedPiece(
          id: piece.id,
          token: piece.token,
          offset: (piece.offset.$1 + offset, piece.offset.$2 + offset),
          wordId: piece.wordId,
        ),
    ];
  }

  List<_AddedTokenMatch> _findAddedTokenMatches(String text) {
    if (_addedTokens.isEmpty || text.isEmpty) return const [];

    final chars = text.runes.toList();
    final matches = <_AddedTokenMatch>[];
    var cursor = 0;
    while (cursor < chars.length) {
      _AddedTokenMatch? best;
      for (final token in _addedTokens) {
        final normalizedContent = token.normalized
            ? _normalizer.normalize(token.content)
            : token.content;
        final tokenChars = normalizedContent.runes.toList();
        if (tokenChars.isEmpty) continue;
        final matchStart = cursor;
        var contentStart = cursor;
        if (token.lstrip) {
          while (contentStart < chars.length &&
              _isWhitespaceRune(chars[contentStart])) {
            contentStart++;
          }
        }
        final contentEnd = _matchAddedTokenContent(
          text,
          chars,
          contentStart,
          normalizedContent,
          normalized: token.normalized,
        );
        if (contentEnd == null) continue;
        var end = contentEnd;
        if (token.rstrip) {
          while (end < chars.length && _isWhitespaceRune(chars[end])) {
            end++;
          }
        }
        if (token.singleWord &&
            ((contentStart > 0 && _isWordRune(chars[contentStart - 1])) ||
                (end < chars.length && _isWordRune(chars[end])))) {
          continue;
        }
        final candidate = _AddedTokenMatch(
          start: matchStart,
          end: end,
          token: token,
        );
        if (best == null ||
            candidate.end - candidate.start > best.end - best.start) {
          best = candidate;
        }
      }
      if (best == null) {
        cursor++;
      } else {
        matches.add(best);
        cursor = best.end;
      }
    }
    return matches;
  }

  int? _matchAddedTokenContent(
    String text,
    List<int> chars,
    int start,
    String content, {
    required bool normalized,
  }) {
    final contentChars = content.runes.toList();
    if (!normalized) {
      return _startsWith(chars, start, contentChars)
          ? start + contentChars.length
          : null;
    }

    // Most tokens match their raw spelling. Try that fast path before the
    // bounded normalization-aware search for compatibility with normalized
    // AddedToken entries such as full-width forms.
    if (_startsWith(chars, start, contentChars)) {
      final rawEnd = start + contentChars.length;
      if (_normalizer.normalize(_sliceRunes(text, start, rawEnd)) == content) {
        return rawEnd;
      }
    }

    final maxLength = contentChars.length + 32;
    final endLimit = start + maxLength < chars.length
        ? start + maxLength
        : chars.length;
    for (var end = start + 1; end <= endLimit; end++) {
      if (_normalizer.normalize(_sliceRunes(text, start, end)) == content) {
        return end;
      }
    }
    return null;
  }

  String _sliceRunes(String text, int start, [int? end]) {
    final chars = text.runes.toList();
    return String.fromCharCodes(chars.sublist(start, end));
  }

  bool _isWordRune(int rune) {
    return rune == 0x5F ||
        rune >= 0x30 && rune <= 0x39 ||
        rune >= 0x41 && rune <= 0x5A ||
        rune >= 0x61 && rune <= 0x7A ||
        rune > 0x7F && !_isWhitespaceRune(rune);
  }

  List<_PipelineChunk> _buildPipelineChunks(
    String normalized, {
    List<(int, int)?>? alignment,
  }) {
    final spec = _preTokenizer;
    final chars = normalized.runes.toList();
    final source = alignment ?? List<(int, int)?>.filled(chars.length, null);
    if (spec == null) {
      return [
        _PipelineChunk(text: normalized, alignment: source, wordId: null),
      ];
    }

    final chunks = <_PipelineChunk>[];
    if (spec.whitespaceSplit) {
      final startsWithWhitespace =
          chars.isNotEmpty && _isWhitespaceRune(chars.first);
      var i = 0;
      var wordIndex = 0;
      while (i < chars.length) {
        while (i < chars.length && _isWhitespaceRune(chars[i])) {
          i++;
        }
        if (i >= chars.length) break;
        final wordChars = <int>[];
        final wordAlignment = <(int, int)?>[];
        while (i < chars.length && !_isWhitespaceRune(chars[i])) {
          wordChars.add(chars[i]);
          wordAlignment.add(source[i]);
          i++;
        }
        if (spec.useMetaspace &&
            _startsWith(wordChars, 0, spec.replacement.runes.toList())) {
          final markerLength = spec.replacement.runes.length;
          wordChars.removeRange(0, markerLength);
          wordAlignment.removeRange(0, markerLength);
        }
        final addMarker =
            spec.addPrefixSpace ||
            wordIndex > 0 ||
            (wordIndex == 0 && startsWithWhitespace);
        final marker = spec.useMetaspace && addMarker
            ? spec.replacement.runes.toList()
            : const <int>[];
        chunks.add(
          _PipelineChunk(
            text: String.fromCharCodes([...marker, ...wordChars]),
            alignment: [
              ...List<(int, int)?>.filled(marker.length, null),
              ...wordAlignment,
            ],
            wordId: wordIndex,
            leadingMarkerLength: marker.length,
            includeLeadingMarker: false,
          ),
        );
        wordIndex++;
      }
    } else if (spec.useMetaspace) {
      final replacement = spec.replacement.runes.toList();
      final transformedChars = <int>[];
      final transformedAlignment = <(int, int)?>[];
      for (var i = 0; i < chars.length; i++) {
        if (_isWhitespaceRune(chars[i])) {
          transformedChars.addAll(replacement);
          transformedAlignment.addAll(
            List<(int, int)?>.filled(replacement.length, source[i]),
          );
        } else {
          transformedChars.add(chars[i]);
          transformedAlignment.add(source[i]);
        }
      }
      if (spec.addPrefixSpace &&
          transformedChars.isNotEmpty &&
          !_startsWith(transformedChars, 0, replacement)) {
        transformedChars.insertAll(0, replacement);
        transformedAlignment.insertAll(
          0,
          List<(int, int)?>.filled(replacement.length, null),
        );
      }

      if (!spec.split) {
        chunks.add(
          _PipelineChunk(
            text: String.fromCharCodes(transformedChars),
            alignment: transformedAlignment,
            wordId: 0,
            leadingMarkerLength: _startsWith(transformedChars, 0, replacement)
                ? replacement.length
                : 0,
            includeLeadingMarker: true,
          ),
        );
      } else {
        var start = 0;
        var wordIndex = 0;
        while (start < transformedChars.length) {
          var markerStart = start;
          if (_startsWith(transformedChars, markerStart, replacement)) {
            markerStart += replacement.length;
          }
          var end = markerStart;
          while (end < transformedChars.length &&
              !_startsWith(transformedChars, end, replacement)) {
            end++;
          }
          if (end > markerStart) {
            final chunkStart = markerStart == start
                ? start
                : markerStart - replacement.length;
            final chunkChars = transformedChars.sublist(chunkStart, end);
            chunks.add(
              _PipelineChunk(
                text: String.fromCharCodes(chunkChars),
                alignment: transformedAlignment.sublist(chunkStart, end),
                wordId: wordIndex,
                leadingMarkerLength: markerStart == start
                    ? 0
                    : replacement.length,
                includeLeadingMarker: true,
              ),
            );
            wordIndex++;
          }
          start = end;
        }
      }
    } else {
      chunks.add(
        _PipelineChunk(text: normalized, alignment: source, wordId: null),
      );
    }
    return chunks;
  }

  List<(int, int)?> _alignNormalizedText(String original, String normalized) {
    final source = original.runes.toList();
    final result = <(int, int)?>[];
    var cursor = 0;
    (int, int)? previous;

    for (final rune in normalized.runes) {
      if (_isWhitespaceRune(rune)) {
        if (cursor < source.length && _isWhitespaceRune(source[cursor])) {
          var end = cursor + 1;
          while (end < source.length && _isWhitespaceRune(source[end])) {
            end++;
          }
          previous = (cursor, end);
          result.add(previous);
          cursor = end;
        } else {
          result.add(null);
        }
        continue;
      }

      var match = cursor;
      while (match < source.length && !_equivalentRune(source[match], rune)) {
        if (!_isWhitespaceRune(source[match])) break;
        match++;
      }
      if (match < source.length && _equivalentRune(source[match], rune)) {
        previous = (match, match + 1);
        result.add(previous);
        cursor = match + 1;
      } else if (cursor < source.length) {
        previous = (cursor, cursor + 1);
        result.add(previous);
        cursor++;
      } else {
        result.add(previous);
      }
    }
    return result;
  }

  (int, int) _sourceOffset(
    List<(int, int)?> alignment,
    int start,
    int end, {
    required int leadingMarkerLength,
    required bool includeLeadingMarker,
  }) {
    final spans = <(int, int)>[];
    for (var i = start; i < end && i < alignment.length; i++) {
      if (!includeLeadingMarker && i < start + leadingMarkerLength) continue;
      final span = alignment[i];
      if (span != null) {
        if (includeLeadingMarker && i < start + leadingMarkerLength) {
          spans.add((span.$2 - 1, span.$2));
        } else {
          spans.add(span);
        }
      }
    }
    if (spans.isNotEmpty) {
      return (
        spans.map((span) => span.$1).reduce((a, b) => a < b ? a : b),
        spans.map((span) => span.$2).reduce((a, b) => a > b ? a : b),
      );
    }
    if (start < end && leadingMarkerLength > 0 && start == 0) {
      return (0, 1);
    }
    return (0, 0);
  }

  bool _startsWith(List<int> chars, int start, List<int> prefix) {
    if (start < 0 || start + prefix.length > chars.length) return false;
    for (var i = 0; i < prefix.length; i++) {
      if (chars[start + i] != prefix[i]) return false;
    }
    return true;
  }

  bool _isWhitespaceRune(int rune) =>
      RegExp(r'^\s$').hasMatch(String.fromCharCode(rune));

  bool _equivalentRune(int source, int normalized) {
    if (source == normalized) return true;
    if (source == 0x3000 && normalized == 0x20) return true;
    if (source >= 0xFF01 && source <= 0xFF5E) {
      return source - 0xFEE0 == normalized;
    }
    return false;
  }

  /// Tokenize multiple texts.
  List<List<String>> tokenizeBatch(List<String> texts) {
    return [for (final text in texts) tokenize(text)];
  }

  @override
  String toString() =>
      'SentencePieceTokenizer(modelType: $modelType, vocabSize: $vocabSize)';
}

class _TokenizedPiece {
  final int id;
  final String token;
  final (int, int) offset;
  final int? wordId;

  const _TokenizedPiece({
    required this.id,
    required this.token,
    required this.offset,
    required this.wordId,
  });
}

class _AddedTokenMatch {
  final int start;
  final int end;
  final SpAddedToken token;

  const _AddedTokenMatch({
    required this.start,
    required this.end,
    required this.token,
  });
}

class _PipelineChunk {
  final String text;
  final List<(int, int)?> alignment;
  final int? wordId;
  final int leadingMarkerLength;
  final bool includeLeadingMarker;

  const _PipelineChunk({
    required this.text,
    required this.alignment,
    required this.wordId,
    this.leadingMarkerLength = 0,
    this.includeLeadingMarker = false,
  });
}

/// Serializable model data for Isolate transfer.
class _SerializableModelData {
  final List<String> pieces;
  final List<double> scores;
  final List<int> types;
  final int unkId;
  final int bosId;
  final int eosId;
  final int padId;
  final String unkPiece;
  final String bosPiece;
  final String eosPiece;
  final String padPiece;
  final bool byteFallback;
  final bool fuseUnk;
  final ModelType modelType;
  final bool addDummyPrefix;
  final bool removeExtraWhitespaces;
  final bool escapeWhitespaces;
  final String normalizerName;
  final Uint8List? precompiledCharsmapBytes;
  final List<String> normalizerOperationTypes;
  final List<String?> normalizerOperationPatterns;
  final List<String?> normalizerOperationReplacements;
  final List<List<int>?> normalizerOperationBytes;
  final bool hasPreTokenizer;
  final bool preTokenizerWhitespaceSplit;
  final bool preTokenizerUseMetaspace;
  final bool preTokenizerAddPrefixSpace;
  final String preTokenizerReplacement;
  final bool preTokenizerSplit;

  _SerializableModelData({
    required this.pieces,
    required this.scores,
    required this.types,
    required this.unkId,
    required this.bosId,
    required this.eosId,
    required this.padId,
    required this.unkPiece,
    required this.bosPiece,
    required this.eosPiece,
    required this.padPiece,
    required this.byteFallback,
    required this.fuseUnk,
    required this.modelType,
    required this.addDummyPrefix,
    required this.removeExtraWhitespaces,
    required this.escapeWhitespaces,
    required this.normalizerName,
    this.precompiledCharsmapBytes,
    this.normalizerOperationTypes = const [],
    this.normalizerOperationPatterns = const [],
    this.normalizerOperationReplacements = const [],
    this.normalizerOperationBytes = const [],
    this.hasPreTokenizer = false,
    this.preTokenizerWhitespaceSplit = false,
    this.preTokenizerUseMetaspace = true,
    this.preTokenizerAddPrefixSpace = true,
    this.preTokenizerReplacement = kSpaceSymbol,
    this.preTokenizerSplit = true,
  });

  factory _SerializableModelData.fromTokenizer(
    SentencePieceTokenizer tokenizer,
  ) {
    return _SerializableModelData(
      pieces: tokenizer.vocab.pieces,
      scores: tokenizer.vocab.scores.toList(),
      types: List.generate(
        tokenizer.vocab.size,
        (i) => tokenizer.vocab.getType(i).value,
      ),
      unkId: tokenizer.vocab.unkId,
      bosId: tokenizer.vocab.bosId,
      eosId: tokenizer.vocab.eosId,
      padId: tokenizer.vocab.padId,
      unkPiece: tokenizer.vocab.unkPiece,
      bosPiece: tokenizer.vocab.bosPiece,
      eosPiece: tokenizer.vocab.eosPiece,
      padPiece: tokenizer.vocab.padPiece,
      byteFallback: tokenizer.vocab.hasByteFallback,
      fuseUnk: tokenizer.fuseUnknownTokens,
      modelType: tokenizer.modelType,
      addDummyPrefix: tokenizer._normalizer.addDummyPrefix,
      removeExtraWhitespaces: tokenizer._normalizer.removeExtraWhitespaces,
      escapeWhitespaces: tokenizer._normalizer.escapeWhitespaces,
      normalizerName: tokenizer._normalizer.normalizerName,
      precompiledCharsmapBytes: tokenizer._normalizer.operations.isEmpty
          ? tokenizer._normalizer.precompiledCharsmapBytes
          : null,
      normalizerOperationTypes: [
        for (final operation in tokenizer._normalizer.operations)
          operation.type,
      ],
      normalizerOperationPatterns: [
        for (final operation in tokenizer._normalizer.operations)
          operation.pattern,
      ],
      normalizerOperationReplacements: [
        for (final operation in tokenizer._normalizer.operations)
          operation.replacement,
      ],
      normalizerOperationBytes: [
        for (final operation in tokenizer._normalizer.operations)
          operation.precompiledCharsmap?.toList(),
      ],
      hasPreTokenizer: tokenizer.preTokenizer != null,
      preTokenizerWhitespaceSplit:
          tokenizer.preTokenizer?.whitespaceSplit ?? false,
      preTokenizerUseMetaspace: tokenizer.preTokenizer?.useMetaspace ?? true,
      preTokenizerAddPrefixSpace:
          tokenizer.preTokenizer?.addPrefixSpace ?? true,
      preTokenizerReplacement:
          tokenizer.preTokenizer?.replacement ?? kSpaceSymbol,
      preTokenizerSplit: tokenizer.preTokenizer?.split ?? true,
    );
  }

  SentencePieceTokenizer recreateTokenizer(SentencePieceConfig config) {
    // Reconstruct the model
    final modelPieces = <SentencePiece>[];
    for (var i = 0; i < pieces.length; i++) {
      modelPieces.add(
        SentencePiece(
          piece: pieces[i],
          score: scores[i],
          type: PieceType.fromValue(types[i]),
        ),
      );
    }

    final operations = <NormalizerOperation>[];
    for (var i = 0; i < normalizerOperationTypes.length; i++) {
      operations.add(
        NormalizerOperation(
          type: normalizerOperationTypes[i],
          pattern: i < normalizerOperationPatterns.length
              ? normalizerOperationPatterns[i]
              : null,
          replacement: i < normalizerOperationReplacements.length
              ? normalizerOperationReplacements[i]
              : null,
          precompiledCharsmap:
              i < normalizerOperationBytes.length &&
                  normalizerOperationBytes[i] != null
              ? Uint8List.fromList(normalizerOperationBytes[i]!)
              : null,
        ),
      );
    }

    final model = SentencePieceModel(
      pieces: modelPieces,
      trainerSpec: TrainerSpec(
        modelType: modelType,
        vocabSize: pieces.length,
        unkId: unkId,
        bosId: bosId,
        eosId: eosId,
        padId: padId,
        unkPiece: unkPiece,
        bosPiece: bosPiece,
        eosPiece: eosPiece,
        padPiece: padPiece,
        byteFallback: byteFallback,
        fuseUnk: fuseUnk,
      ),
      normalizerSpec: NormalizerSpec(
        name: normalizerName,
        precompiledCharsmap: precompiledCharsmapBytes,
        addDummyPrefix: addDummyPrefix,
        removeExtraWhitespaces: removeExtraWhitespaces,
        escapeWhitespaces: escapeWhitespaces,
        operations: operations,
      ),
      preTokenizerSpec: hasPreTokenizer
          ? PreTokenizerSpec(
              whitespaceSplit: preTokenizerWhitespaceSplit,
              useMetaspace: preTokenizerUseMetaspace,
              addPrefixSpace: preTokenizerAddPrefixSpace,
              replacement: preTokenizerReplacement,
              split: preTokenizerSplit,
            )
          : null,
    );

    return SentencePieceTokenizer._createFromModel(model, config);
  }
}

class _EncodingData {
  final List<String> tokens;
  final List<int> ids;
  final List<int> typeIds;
  final List<int> attentionMask;
  final List<int> specialTokensMask;
  final List<List<int>> offsets;
  final List<int?> wordIds;
  final List<int?> sequenceIds;

  const _EncodingData({
    required this.tokens,
    required this.ids,
    required this.typeIds,
    required this.attentionMask,
    required this.specialTokensMask,
    required this.offsets,
    required this.wordIds,
    required this.sequenceIds,
  });

  factory _EncodingData.fromEncoding(Encoding encoding) {
    return _EncodingData(
      tokens: encoding.tokens.toList(),
      ids: encoding.ids.toList(),
      typeIds: encoding.typeIds.toList(),
      attentionMask: encoding.attentionMask.toList(),
      specialTokensMask: encoding.specialTokensMask.toList(),
      offsets: encoding.offsets.map((o) => [o.$1, o.$2]).toList(),
      wordIds: encoding.wordIds.toList(),
      sequenceIds: encoding.sequenceIds.toList(),
    );
  }

  Encoding toEncoding() {
    return Encoding(
      tokens: tokens,
      ids: ids,
      typeIds: typeIds,
      attentionMask: attentionMask,
      specialTokensMask: specialTokensMask,
      offsets: offsets.map((o) => (o[0], o[1])).toList(),
      wordIds: wordIds,
      sequenceIds: sequenceIds,
    );
  }
}

List<_EncodingData> _encodeChunkInIsolate(
  List<String> texts,
  _SerializableModelData modelData,
  SentencePieceConfig config,
  bool? addSpecialTokens,
) {
  final tokenizer = modelData.recreateTokenizer(config);
  final results = <_EncodingData>[];

  for (final text in texts) {
    final encoding = tokenizer.encode(text, addSpecialTokens: addSpecialTokens);
    results.add(_EncodingData.fromEncoding(encoding));
  }

  return results;
}
