import 'dart:convert';
import 'dart:io';

import '../model/model_proto.dart';
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
  final List<_HfAddedToken> addedTokens;
  final Set<String> specialContents;

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
    this.addedTokens = const [],
    this.specialContents = const {},
  });
}

class _HfAddedToken {
  final int id;
  final String content;
  final bool special;

  const _HfAddedToken({
    required this.id,
    required this.content,
    required this.special,
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

    final modelType = modelData['type'] as String?;
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
    );

    _applyAddedTokens(tokenizer, meta, model.vocabSize);

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
      ),
      normalizerSpec: _buildNormalizerSpec(meta),
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
      ),
      normalizerSpec: _buildNormalizerSpec(meta),
    );
  }

  static TrainerSpec _buildTrainerSpec(
    ModelType modelType,
    _HfMetadata meta,
    int vocabSize,
    int unkId,
    bool byteFallback,
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
    );
  }

  static NormalizerSpec _buildNormalizerSpec(_HfMetadata meta) {
    return NormalizerSpec(
      name: 'identity',
      addDummyPrefix: meta.addDummyPrefix,
      removeExtraWhitespaces: meta.removeExtraWhitespaces,
      escapeWhitespaces: meta.escapeWhitespaces,
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

    // Parse normalizer settings.
    bool addDummyPrefix = true;
    bool escapeWhitespaces = true;
    bool removeExtraWhitespaces = false;

    final normalizerData = data['normalizer'] as Map<String, dynamic>?;
    if (normalizerData != null) {
      final parsed = _parseNormalizerFlags(normalizerData);
      addDummyPrefix = parsed.addDummyPrefix;
      escapeWhitespaces = parsed.escapeWhitespaces;
      removeExtraWhitespaces = parsed.removeExtraWhitespaces;
    }

    bool byteFallback = false;
    final decoderData = data['decoder'] as Map<String, dynamic>?;
    if (decoderData != null) {
      byteFallback = _hasDecoderByteFallback(decoderData);
    }

    bool addBosToken = false;
    bool addEosToken = false;
    final postProcessor = data['post_processor'] as Map<String, dynamic>?;
    if (postProcessor != null) {
      final parsed = _parsePostProcessor(postProcessor, bosPiece, eosPiece);
      addBosToken = parsed.$1;
      addEosToken = parsed.$2;
    }

    return _HfMetadata(
      addDummyPrefix: addDummyPrefix,
      escapeWhitespaces: escapeWhitespaces,
      removeExtraWhitespaces: removeExtraWhitespaces,
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
      addedTokens: addedTokens,
      specialContents: specialContents,
    );
  }

  static ({
    bool addDummyPrefix,
    bool escapeWhitespaces,
    bool removeExtraWhitespaces,
  })
  _parseNormalizerFlags(Map<String, dynamic> data) {
    bool addDummyPrefix = false;
    bool escapeWhitespaces = false;
    bool removeExtraWhitespaces = false;

    final type = data['type'] as String?;

    if (type == 'Sequence') {
      final normalizers = data['normalizers'] as List? ?? [];
      for (final raw in normalizers) {
        final n = raw as Map<String, dynamic>;
        final flags = _parseNormalizerFlags(n);
        if (flags.addDummyPrefix) addDummyPrefix = true;
        if (flags.escapeWhitespaces) escapeWhitespaces = true;
        if (flags.removeExtraWhitespaces) removeExtraWhitespaces = true;
      }
    } else if (type == 'Prepend') {
      final prepend = data['prepend'] as String?;
      if (prepend == '\u2581' || prepend == ' ') {
        addDummyPrefix = true;
      }
    } else if (type == 'Replace') {
      final content = data['content'] as String?;
      if (content == '\u2581') {
        escapeWhitespaces = true;
      }
    }

    return (
      addDummyPrefix: addDummyPrefix,
      escapeWhitespaces: escapeWhitespaces,
      removeExtraWhitespaces: removeExtraWhitespaces,
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

  static (bool, bool) _parsePostProcessor(
    Map<String, dynamic> data,
    String bosPiece,
    String eosPiece,
  ) {
    final type = data['type'] as String?;
    if (type != 'TemplateProcessing') return (false, false);

    bool addBos = false;
    bool addEos = false;

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

    return (addBos, addEos);
  }

  static void _applyAddedTokens(
    SentencePieceTokenizer tokenizer,
    _HfMetadata meta,
    int baseVocabSize,
  ) {
    final normalTokens = <String>[];
    for (final token in meta.addedTokens) {
      if (token.id < baseVocabSize && tokenizer.vocab.contains(token.content)) {
        continue;
      }
      if (token.special) {
        tokenizer.vocab.addSpecialToken(token.content);
      } else {
        normalTokens.add(token.content);
      }
    }
    if (normalTokens.isNotEmpty) {
      tokenizer.vocab.addTokens(normalTokens);
    }
  }

  static bool _isByteToken(String piece) {
    return piece.length == 6 && piece.startsWith('<0x') && piece.endsWith('>');
  }
}
