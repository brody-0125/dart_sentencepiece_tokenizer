import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import '../model/model_proto.dart';
import '../normalizer/precompiled_charsmap.dart';
import '../sentencepiece_tokenizer.dart';

/// Known variant strings for each special token type.
const _unkTokenVariants = {'<unk>', '[UNK]'};
const _bosTokenVariants = {'<s>', '<bos>', '[BOS]'};
const _eosTokenVariants = {'</s>', '<eos>', '[EOS]'};
const _padTokenVariants = {'<pad>', '[PAD]'};

/// Metadata extracted from HuggingFace tokenizer.json sections
/// (added_tokens, normalizer, decoder, post_processor).
class _HfMetadata {
  final bool addDummyPrefix;
  final bool escapeWhitespaces;
  final bool removeExtraWhitespaces;
  final bool byteFallback;
  final int unkId;
  final int bosId;
  final int eosId;
  final int padId;
  final String unkPiece;
  final String bosPiece;
  final String eosPiece;
  final String padPiece;
  final bool addBosToken;
  final bool addEosToken;
  final int pairEosTokensBetweenSequences;
  final int pairTypeId;
  final int pairSpecialTypeId;
  final List<_HfAddedToken> addedTokens;
  final Set<String> specialContents;
  final List<NormalizerOperation> normalizerOperations;
  final PreTokenizerSpec? preTokenizer;

  const _HfMetadata({
    this.addDummyPrefix = true,
    this.escapeWhitespaces = true,
    this.removeExtraWhitespaces = false,
    this.byteFallback = false,
    this.unkId = 0,
    this.bosId = 1,
    this.eosId = 2,
    this.padId = -1,
    this.unkPiece = '<unk>',
    this.bosPiece = '<s>',
    this.eosPiece = '</s>',
    this.padPiece = '<pad>',
    this.addBosToken = false,
    this.addEosToken = false,
    this.pairEosTokensBetweenSequences = 1,
    this.pairTypeId = 1,
    this.pairSpecialTypeId = 1,
    this.addedTokens = const [],
    this.specialContents = const {},
    this.normalizerOperations = const [],
    this.preTokenizer,
  });
}

class _HfAddedToken {
  final int id;
  final String content;
  final bool special;
  final bool singleWord;
  final bool lstrip;
  final bool rstrip;
  final bool normalized;

  const _HfAddedToken({
    required this.id,
    required this.content,
    required this.special,
    this.singleWord = false,
    this.lstrip = false,
    this.rstrip = false,
    this.normalized = true,
  });
}

/// Loader for HuggingFace `tokenizer.json` format.
///
/// Supports SentencePiece-based BPE and Unigram models as used by
/// Gemma, Llama, and similar HuggingFace models.
class HuggingFaceTokenizerLoader {
  HuggingFaceTokenizerLoader._();

  /// Create tokenizer from HuggingFace tokenizer.json string.
  static SentencePieceTokenizer fromJsonString(
    String json, {
    SentencePieceConfig? config,
  }) {
    final data = jsonDecode(json) as Map<String, dynamic>;
    return fromMap(data, config: config);
  }

  /// Create tokenizer from a pre-parsed HuggingFace tokenizer.json map.
  static SentencePieceTokenizer fromMap(
    Map<String, dynamic> data, {
    SentencePieceConfig? config,
  }) {
    final modelData = data['model'] as Map<String, dynamic>?;
    if (modelData == null) {
      throw const FormatException(
        'Missing model section in HuggingFace tokenizer JSON',
      );
    }

    // Some official SentencePiece exports omit `model.type`; the vocab shape
    // still unambiguously identifies Unigram (array) versus BPE (map).
    final modelType =
        modelData['type'] as String? ??
        switch (modelData['vocab']) {
          final List<dynamic> vocab when vocab.isNotEmpty => 'Unigram',
          final Map<String, dynamic> vocab when vocab.isNotEmpty => 'BPE',
          _ => null,
        };
    if (modelType == null) {
      throw const FormatException(
        'Missing model type in HuggingFace tokenizer JSON',
      );
    }

    final meta = _parseMetadata(data);

    final SentencePieceConfig finalConfig;
    if (config != null) {
      finalConfig = config;
    } else {
      finalConfig = SentencePieceConfig(
        addBosToken: meta.addBosToken,
        addEosToken: meta.addEosToken,
        pairEosTokensBetweenSequences: meta.pairEosTokensBetweenSequences,
        pairTypeId: meta.pairTypeId,
        pairSpecialTypeId: meta.pairSpecialTypeId,
      );
    }

    final SentencePieceModel model;
    switch (modelType) {
      case 'Unigram':
        model = _parseUnigramModel(modelData, meta);
      case 'BPE':
        model = _parseBpeModel(modelData, meta);
      default:
        throw UnsupportedError(
          'Unsupported HuggingFace model type: $modelType. '
          'Only BPE and Unigram are supported.',
        );
    }

    final tokenizer = SentencePieceTokenizer.fromModel(
      model,
      config: finalConfig,
      addedTokens: [
        for (final token in meta.addedTokens)
          SpAddedToken(
            id: token.id,
            content: token.content,
            special: token.special,
            singleWord: token.singleWord,
            lstrip: token.lstrip,
            rstrip: token.rstrip,
            normalized: token.normalized,
          ),
      ],
    );

    _applyAddedTokens(tokenizer, meta, model.vocabSize);
    _applyTokenizerSettings(tokenizer, data);

    return tokenizer;
  }

  /// Load tokenizer from HuggingFace tokenizer.json file asynchronously.
  static Future<SentencePieceTokenizer> fromJsonFile(
    String path, {
    SentencePieceConfig? config,
  }) async {
    final json = await File(path).readAsString();
    return fromJsonString(json, config: config);
  }

  /// Load tokenizer from HuggingFace tokenizer.json file synchronously.
  static SentencePieceTokenizer fromJsonFileSync(
    String path, {
    SentencePieceConfig? config,
  }) {
    final json = File(path).readAsStringSync();
    return fromJsonString(json, config: config);
  }

  static SentencePieceModel _parseUnigramModel(
    Map<String, dynamic> modelData,
    _HfMetadata meta,
  ) {
    final rawVocab = modelData['vocab'] as List?;
    if (rawVocab == null) {
      throw const FormatException('Missing vocab in HuggingFace Unigram model');
    }

    final unkId = modelData['unk_id'] as int? ?? meta.unkId;

    final pieces = <SentencePiece>[];
    for (var i = 0; i < rawVocab.length; i++) {
      final entry = rawVocab[i] as List;
      final piece = entry[0] as String;
      final score = (entry[1] as num).toDouble();

      PieceType type;
      if (i == unkId) {
        type = PieceType.unknown;
      } else if (meta.specialContents.contains(piece)) {
        type = PieceType.control;
      } else if (_isByteToken(piece)) {
        type = PieceType.byte;
      } else {
        type = PieceType.normal;
      }

      pieces.add(SentencePiece(piece: piece, score: score, type: type));
    }

    return SentencePieceModel(
      pieces: pieces,
      trainerSpec: _buildTrainerSpec(
        ModelType.unigram,
        meta,
        pieces.length,
        unkId,
        meta.byteFallback,
        modelData['fuse_unk'] as bool? ?? true,
      ),
      normalizerSpec: _buildNormalizerSpec(meta),
      preTokenizerSpec: meta.preTokenizer,
    );
  }

  static SentencePieceModel _parseBpeModel(
    Map<String, dynamic> modelData,
    _HfMetadata meta,
  ) {
    final rawVocab = modelData['vocab'] as Map<String, dynamic>?;
    if (rawVocab == null) {
      throw const FormatException('Missing vocab in HuggingFace BPE model');
    }

    final rawMerges = modelData['merges'] as List? ?? [];
    final byteFallback =
        modelData['byte_fallback'] as bool? ?? meta.byteFallback;

    final mergeScores = <String, double>{};
    for (var i = 0; i < rawMerges.length; i++) {
      final raw = rawMerges[i];
      final String left;
      final String right;
      if (raw is List) {
        // New HuggingFace `tokenizers` format (>= 0.20, used by recent exports
        // such as SigLIP2 and Gemma-2/3): each merge is a two-element
        // `[left, right]` array instead of a single space-joined string.
        // Skip a malformed entry rather than throw, mirroring the legacy
        // branch's `spaceIdx < 0` skip below (both drop a bad merge and move on).
        if (raw.length < 2 || raw[0] is! String || raw[1] is! String) continue;
        left = raw[0] as String;
        right = raw[1] as String;
      } else {
        // Legacy format: a single `"left right"` string.
        final mergeStr = raw as String;
        final spaceIdx = mergeStr.indexOf(' ');
        if (spaceIdx < 0) continue;
        left = mergeStr.substring(0, spaceIdx);
        right = mergeStr.substring(spaceIdx + 1);
      }
      mergeScores[left + right] = -i.toDouble();
    }

    final baseScore = -(rawMerges.length + 1).toDouble();
    final unkToken = modelData['unk_token'] as String? ?? meta.unkPiece;

    final vocabSize = rawVocab.length;
    final pieces = List<SentencePiece>.filled(
      vocabSize,
      const SentencePiece(piece: ''),
    );

    for (final entry in rawVocab.entries) {
      final piece = entry.key;
      final id = entry.value as int;

      if (id < 0 || id >= vocabSize) continue;

      PieceType type;
      if (piece == unkToken) {
        type = PieceType.unknown;
      } else if (meta.specialContents.contains(piece)) {
        type = PieceType.control;
      } else if (_isByteToken(piece)) {
        type = PieceType.byte;
      } else {
        type = PieceType.normal;
      }

      double score;
      if (type == PieceType.unknown || type == PieceType.control) {
        score = 0.0;
      } else {
        score = mergeScores[piece] ?? baseScore;
      }

      pieces[id] = SentencePiece(piece: piece, score: score, type: type);
    }

    final unkId = rawVocab[unkToken] as int? ?? meta.unkId;

    return SentencePieceModel(
      pieces: pieces,
      trainerSpec: _buildTrainerSpec(
        ModelType.bpe,
        meta,
        vocabSize,
        unkId,
        byteFallback,
        false,
      ),
      normalizerSpec: _buildNormalizerSpec(meta),
      preTokenizerSpec: meta.preTokenizer,
    );
  }

  static TrainerSpec _buildTrainerSpec(
    ModelType modelType,
    _HfMetadata meta,
    int vocabSize,
    int unkId,
    bool byteFallback,
    bool fuseUnk,
  ) {
    return TrainerSpec(
      modelType: modelType,
      vocabSize: vocabSize,
      unkId: unkId,
      bosId: meta.bosId,
      eosId: meta.eosId,
      padId: meta.padId,
      unkPiece: meta.unkPiece,
      bosPiece: meta.bosPiece,
      eosPiece: meta.eosPiece,
      padPiece: meta.padPiece,
      byteFallback: byteFallback,
      fuseUnk: fuseUnk,
    );
  }

  static NormalizerSpec _buildNormalizerSpec(_HfMetadata meta) {
    return NormalizerSpec(
      name: 'identity',
      addDummyPrefix: meta.addDummyPrefix,
      removeExtraWhitespaces: meta.removeExtraWhitespaces,
      escapeWhitespaces: meta.escapeWhitespaces,
      operations: meta.normalizerOperations,
    );
  }

  static _HfMetadata _parseMetadata(Map<String, dynamic> data) {
    final addedTokens = <_HfAddedToken>[];
    final rawAddedTokens = data['added_tokens'] as List?;
    if (rawAddedTokens != null) {
      for (final raw in rawAddedTokens) {
        final entry = raw as Map<String, dynamic>;
        addedTokens.add(
          _HfAddedToken(
            id: entry['id'] as int,
            content: entry['content'] as String,
            special: entry['special'] as bool? ?? false,
            singleWord: entry['single_word'] as bool? ?? false,
            lstrip: entry['lstrip'] as bool? ?? false,
            rstrip: entry['rstrip'] as bool? ?? false,
            normalized: entry['normalized'] as bool? ?? true,
          ),
        );
      }
    }

    // Identify special tokens from added_tokens.
    int unkId = 0;
    int bosId = 1;
    int eosId = 2;
    int padId = -1;
    String unkPiece = '<unk>';
    String bosPiece = '<s>';
    String eosPiece = '</s>';
    String padPiece = '<pad>';

    final specialContents = <String>{};
    for (final token in addedTokens) {
      if (!token.special) continue;
      specialContents.add(token.content);
      final c = token.content;
      if (_unkTokenVariants.contains(c)) {
        unkId = token.id;
        unkPiece = c;
      } else if (_bosTokenVariants.contains(c)) {
        bosId = token.id;
        bosPiece = c;
      } else if (_eosTokenVariants.contains(c)) {
        eosId = token.id;
        eosPiece = c;
      } else if (_padTokenVariants.contains(c)) {
        padId = token.id;
        padPiece = c;
      }
    }

    final normalizerData = data['normalizer'] as Map<String, dynamic>?;
    final normalizer = _parseNormalizer(normalizerData);
    final preTokenizer = _parsePreTokenizer(
      data['pre_tokenizer'] as Map<String, dynamic>?,
    );
    var addDummyPrefix = normalizer.addDummyPrefix;
    var escapeWhitespaces = normalizer.escapeWhitespaces;
    if (preTokenizer?.useMetaspace == true) {
      // Metaspace owns whitespace markers. Keep the decoder's unescape
      // behavior, and remove its leading marker when prefixing is enabled.
      escapeWhitespaces = true;
      if (preTokenizer!.addPrefixSpace) addDummyPrefix = true;
    }

    bool byteFallback = false;
    final decoderData = data['decoder'] as Map<String, dynamic>?;
    if (decoderData != null) {
      byteFallback = _hasDecoderByteFallback(decoderData);
    }

    bool addBosToken = false;
    bool addEosToken = false;
    var pairEosTokensBetweenSequences = 1;
    var pairTypeId = 1;
    var pairSpecialTypeId = 1;
    final postProcessor = data['post_processor'] as Map<String, dynamic>?;
    if (postProcessor != null) {
      final parsed = _parsePostProcessor(postProcessor, bosPiece, eosPiece);
      addBosToken = parsed.$1;
      addEosToken = parsed.$2;
      pairEosTokensBetweenSequences = parsed.$3;
      pairTypeId = parsed.$4;
      pairSpecialTypeId = parsed.$5;
    }

    return _HfMetadata(
      addDummyPrefix: addDummyPrefix,
      escapeWhitespaces: escapeWhitespaces,
      removeExtraWhitespaces: normalizer.removeExtraWhitespaces,
      byteFallback: byteFallback,
      unkId: unkId,
      bosId: bosId,
      eosId: eosId,
      padId: padId,
      unkPiece: unkPiece,
      bosPiece: bosPiece,
      eosPiece: eosPiece,
      padPiece: padPiece,
      addBosToken: addBosToken,
      addEosToken: addEosToken,
      pairEosTokensBetweenSequences: pairEosTokensBetweenSequences,
      pairTypeId: pairTypeId,
      pairSpecialTypeId: pairSpecialTypeId,
      addedTokens: addedTokens,
      specialContents: specialContents,
      normalizerOperations: normalizer.operations,
      preTokenizer: preTokenizer,
    );
  }

  static ({
    bool addDummyPrefix,
    bool escapeWhitespaces,
    bool removeExtraWhitespaces,
    List<NormalizerOperation> operations,
  })
  _parseNormalizer(Map<String, dynamic>? data) {
    if (data == null) {
      return (
        addDummyPrefix: true,
        escapeWhitespaces: true,
        removeExtraWhitespaces: false,
        operations: const [],
      );
    }

    var addDummyPrefix = false;
    var escapeWhitespaces = false;
    const removeExtraWhitespaces = false;
    final operations = <NormalizerOperation>[];

    void visit(Map<String, dynamic> normalizer) {
      final type = normalizer['type'] as String?;
      switch (type) {
        case 'Sequence':
          final rawNormalizers = normalizer['normalizers'];
          if (rawNormalizers is! List) {
            throw const FormatException(
              'Normalizer Sequence is missing normalizers',
            );
          }
          for (final raw in rawNormalizers) {
            if (raw is! Map<String, dynamic>) {
              throw const FormatException(
                'Normalizer operation must be an object',
              );
            }
            visit(raw);
          }
        case 'Precompiled':
          final encoded = normalizer['precompiled_charsmap'];
          if (encoded is! String || encoded.isEmpty) {
            throw const FormatException(
              'Precompiled normalizer is missing precompiled_charsmap',
            );
          }
          final bytes = _decodeCharsmap(encoded);
          // Validate at load time so malformed data cannot degrade to identity.
          PrecompiledCharsmap.fromBytes(bytes);
          operations.add(
            NormalizerOperation(
              type: 'Precompiled',
              precompiledCharsmap: bytes,
            ),
          );
        case 'Prepend':
          final prepend = normalizer['prepend'];
          if (prepend is! String) {
            throw const FormatException(
              'Prepend normalizer is missing prepend',
            );
          }
          if (prepend == '\u2581' || prepend == ' ') addDummyPrefix = true;
          operations.add(
            NormalizerOperation(type: 'Prepend', replacement: prepend),
          );
        case 'Replace':
          final pattern = _parseReplacePattern(normalizer['pattern']);
          final replacement = normalizer['content'];
          if (replacement is! String) {
            throw const FormatException(
              'Replace normalizer is missing string content',
            );
          }
          try {
            RegExp(pattern);
          } on FormatException catch (error) {
            throw FormatException('Invalid Replace normalizer pattern: $error');
          }
          if (replacement == '\u2581') escapeWhitespaces = true;
          operations.add(
            NormalizerOperation(
              type: 'Replace',
              pattern: pattern,
              replacement: replacement,
            ),
          );
        default:
          throw UnsupportedError('Unsupported Hugging Face normalizer: $type');
      }
    }

    visit(data);

    // Keep the legacy flag implementation for old JSON pipelines. Once a
    // Precompiled step exists, every supported step must run in JSON order.
    final hasPrecompiled = operations.any(
      (operation) => operation.type == 'Precompiled',
    );
    return (
      addDummyPrefix: addDummyPrefix,
      escapeWhitespaces: escapeWhitespaces,
      removeExtraWhitespaces: removeExtraWhitespaces,
      operations: hasPrecompiled ? List.unmodifiable(operations) : const [],
    );
  }

  static Uint8List _decodeCharsmap(String encoded) {
    try {
      return base64Decode(encoded);
    } on FormatException catch (error) {
      throw FormatException('Invalid Precompiled charsmap base64: $error');
    }
  }

  static String _parseReplacePattern(Object? rawPattern) {
    if (rawPattern is! Map<String, dynamic>) {
      throw const FormatException('Replace normalizer is missing pattern');
    }
    final regex = rawPattern['Regex'];
    if (regex is String) return regex;
    final string = rawPattern['String'];
    if (string is String) return RegExp.escape(string);
    throw UnsupportedError(
      'Unsupported Hugging Face Replace pattern; expected Regex or String',
    );
  }

  static PreTokenizerSpec? _parsePreTokenizer(Map<String, dynamic>? data) {
    if (data == null) return null;

    final type = data['type'] as String?;
    switch (type) {
      case 'Metaspace':
        return _parseMetaspace(data);
      case 'WhitespaceSplit':
        return const PreTokenizerSpec(
          whitespaceSplit: true,
          useMetaspace: false,
        );
      case 'Sequence':
        final rawPreTokenizers = data['pretokenizers'];
        if (rawPreTokenizers is! List) {
          throw const FormatException(
            'Pre-tokenizer Sequence is missing pretokenizers',
          );
        }
        var whitespaceSplit = false;
        PreTokenizerSpec? metaspace;
        for (final raw in rawPreTokenizers) {
          if (raw is! Map<String, dynamic>) {
            throw const FormatException(
              'Pre-tokenizer operation must be an object',
            );
          }
          final childType = raw['type'] as String?;
          if (childType == 'WhitespaceSplit') {
            if (metaspace != null) {
              throw UnsupportedError(
                'WhitespaceSplit after Metaspace is unsupported',
              );
            }
            whitespaceSplit = true;
          } else if (childType == 'Metaspace') {
            if (metaspace != null) {
              throw UnsupportedError(
                'Multiple Metaspace pre-tokenizers are unsupported',
              );
            }
            metaspace = _parseMetaspace(raw);
          } else {
            throw UnsupportedError(
              'Unsupported Hugging Face pre-tokenizer: $childType',
            );
          }
        }
        if (metaspace == null) {
          return PreTokenizerSpec(
            whitespaceSplit: whitespaceSplit,
            useMetaspace: false,
          );
        }
        return PreTokenizerSpec(
          whitespaceSplit: whitespaceSplit,
          useMetaspace: true,
          addPrefixSpace: metaspace.addPrefixSpace,
          replacement: metaspace.replacement,
          split: metaspace.split,
        );
      default:
        throw UnsupportedError('Unsupported Hugging Face pre-tokenizer: $type');
    }
  }

  static PreTokenizerSpec _parseMetaspace(Map<String, dynamic> data) {
    final replacement = data['replacement'] as String? ?? '\u2581';
    if (replacement.isEmpty) {
      throw const FormatException('Metaspace replacement must not be empty');
    }

    final legacyPrefix = data['add_prefix_space'] as bool?;
    final scheme = data['prepend_scheme'] as String?;
    final addPrefixSpace =
        legacyPrefix ??
        switch (scheme) {
          null || 'always' || 'first' => true,
          'never' => false,
          _ => throw UnsupportedError(
            'Unsupported Metaspace prepend_scheme: $scheme',
          ),
        };

    return PreTokenizerSpec(
      addPrefixSpace: addPrefixSpace,
      replacement: replacement,
      split: data['split'] as bool? ?? true,
    );
  }

  static bool _hasDecoderByteFallback(Map<String, dynamic> data) {
    final type = data['type'] as String?;
    if (type == 'ByteFallback') return true;
    if (type == 'Sequence') {
      final decoders = data['decoders'] as List? ?? [];
      for (final raw in decoders) {
        if (_hasDecoderByteFallback(raw as Map<String, dynamic>)) return true;
      }
    }
    return false;
  }

  static (bool, bool, int, int, int) _parsePostProcessor(
    Map<String, dynamic> data,
    String bosPiece,
    String eosPiece,
  ) {
    final type = data['type'] as String?;
    if (type != 'TemplateProcessing') return (false, false, 1, 1, 1);

    bool addBos = false;
    bool addEos = false;
    var pairEosCount = 0;
    var pairTypeId = 1;
    var pairSpecialTypeId = 1;

    final single = data['single'] as List?;
    if (single != null && single.isNotEmpty) {
      final first = single.first as Map<String, dynamic>;
      if (first.containsKey('SpecialToken')) {
        final st = first['SpecialToken'] as Map<String, dynamic>;
        final id = st['id'] as String?;
        if (id == bosPiece) addBos = true;
      }
      final last = single.last as Map<String, dynamic>;
      if (last.containsKey('SpecialToken')) {
        final st = last['SpecialToken'] as Map<String, dynamic>;
        final id = st['id'] as String?;
        if (id == eosPiece) addEos = true;
      }
    }

    final pair = data['pair'] as List?;
    if (pair != null) {
      for (final raw in pair) {
        if (raw is! Map<String, dynamic>) continue;
        final sequence = raw['Sequence'];
        if (sequence is Map<String, dynamic> && sequence['id'] == 'B') {
          pairTypeId = sequence['type_id'] as int? ?? pairTypeId;
        }
        final special = raw['SpecialToken'];
        if (special is Map<String, dynamic> && special['id'] == eosPiece) {
          pairEosCount++;
          pairSpecialTypeId = special['type_id'] as int? ?? pairSpecialTypeId;
        }
      }
    }

    final finalEosCount = addEos && pairEosCount > 0 ? 1 : 0;
    final eosBetween = pair == null ? 1 : pairEosCount - finalEosCount;
    return (addBos, addEos, eosBetween, pairTypeId, pairSpecialTypeId);
  }

  static void _applyAddedTokens(
    SentencePieceTokenizer tokenizer,
    _HfMetadata meta,
    int baseVocabSize,
  ) {
    final sortedTokens = [...meta.addedTokens]
      ..sort((a, b) => a.id.compareTo(b.id));
    for (final token in sortedTokens) {
      if (token.id < baseVocabSize && tokenizer.vocab.contains(token.content)) {
        continue;
      }
      tokenizer.vocab.addTokenAtId(
        token.content,
        token.id,
        special: token.special,
      );
    }
  }

  static void _applyTokenizerSettings(
    SentencePieceTokenizer tokenizer,
    Map<String, dynamic> data,
  ) {
    final truncation = data['truncation'];
    if (truncation is Map<String, dynamic>) {
      final maxLength = truncation['max_length'];
      if (maxLength is! int || maxLength <= 0) {
        throw const FormatException(
          'Hugging Face truncation is missing a positive max_length',
        );
      }
      final stride = truncation['stride'] as int? ?? 0;
      if (stride != 0) {
        throw UnsupportedError(
          'Hugging Face truncation stride is unsupported: $stride',
        );
      }
      tokenizer.enableTruncation(
        maxLength: maxLength,
        direction: _parseTruncationDirection(truncation['direction']),
        strategy: _parseTruncationStrategy(truncation['strategy']),
      );
    }

    final padding = data['padding'];
    if (padding is Map<String, dynamic>) {
      tokenizer.enablePadding(
        direction: _parsePaddingDirection(padding['direction']),
        length: padding['length'] as int?,
        padToMultipleOf: padding['pad_to_multiple_of'] as int?,
        padTokenId: padding['pad_id'] as int?,
        padToken: padding['pad_token'] as String?,
        padTypeId: padding['pad_type_id'] as int? ?? 0,
      );
    }
  }

  static SpPaddingDirection _parsePaddingDirection(Object? value) {
    return switch (value?.toString().toLowerCase()) {
      'left' => SpPaddingDirection.left,
      null || 'right' => SpPaddingDirection.right,
      final direction => throw UnsupportedError(
        'Unsupported Hugging Face padding direction: $direction',
      ),
    };
  }

  static SpTruncationDirection _parseTruncationDirection(Object? value) {
    return switch (value?.toString().toLowerCase()) {
      'left' => SpTruncationDirection.left,
      null || 'right' => SpTruncationDirection.right,
      final direction => throw UnsupportedError(
        'Unsupported Hugging Face truncation direction: $direction',
      ),
    };
  }

  static TruncationStrategy _parseTruncationStrategy(Object? value) {
    return switch (value?.toString().toLowerCase()) {
      null ||
      'longestfirst' ||
      'longest_first' => TruncationStrategy.longestFirst,
      'onlyfirst' || 'only_first' => TruncationStrategy.onlyFirst,
      'onlysecond' || 'only_second' => TruncationStrategy.onlySecond,
      'donottruncate' || 'do_not_truncate' => TruncationStrategy.doNotTruncate,
      final strategy => throw UnsupportedError(
        'Unsupported Hugging Face truncation strategy: $strategy',
      ),
    };
  }

  static bool _isByteToken(String piece) {
    return piece.length == 6 && piece.startsWith('<0x') && piece.endsWith('>');
  }
}
