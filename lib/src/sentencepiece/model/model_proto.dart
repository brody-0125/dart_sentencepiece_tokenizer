import 'dart:typed_data';

/// SentencePiece token type. Values match proto enum.
enum PieceType {
  normal(1),
  unknown(2),
  control(3),
  userDefined(4),
  unused(5),
  byte(6);

  final int value;
  const PieceType(this.value);

  static PieceType fromValue(int value) {
    return PieceType.values.firstWhere(
      (t) => t.value == value,
      orElse: () => PieceType.normal,
    );
  }
}

/// SentencePiece model type. Values match proto enum.
enum ModelType {
  unigram(1),
  bpe(2),
  word(3),
  char(4);

  final int value;
  const ModelType(this.value);

  static ModelType fromValue(int value) {
    return ModelType.values.firstWhere(
      (t) => t.value == value,
      orElse: () => ModelType.unigram,
    );
  }
}

/// A single vocabulary piece with score and type.
class SentencePiece {
  final String piece;
  final double score;
  final PieceType type;

  const SentencePiece({
    required this.piece,
    this.score = 0.0,
    this.type = PieceType.normal,
  });

  bool get isNormal => type == PieceType.normal;
  bool get isUnknown => type == PieceType.unknown;
  bool get isControl => type == PieceType.control;
  bool get isUserDefined => type == PieceType.userDefined;
  bool get isByte => type == PieceType.byte;
  bool get isUnused => type == PieceType.unused;

  @override
  String toString() => 'SentencePiece($piece, score: $score, type: $type)';
}

/// Training specification from the model.
///
/// Default token IDs (unkId=0, bosId=1, eosId=2, padId=-1) comply with the
/// google/sentencepiece proto spec (sentencepiece_model.proto). Some models
/// (e.g. Gemma: pad=0, eos=1, bos=2) use different values, which are
/// correctly parsed from the model file at runtime.
class TrainerSpec {
  final ModelType modelType;
  final int vocabSize;
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

  const TrainerSpec({
    this.modelType = ModelType.unigram,
    this.vocabSize = 8000,
    // Default token IDs per google/sentencepiece proto spec.
    this.unkId = 0,
    this.bosId = 1,
    this.eosId = 2,
    this.padId = -1,
    this.unkPiece = '<unk>',
    this.bosPiece = '<s>',
    this.eosPiece = '</s>',
    this.padPiece = '<pad>',
    this.byteFallback = false,
    this.fuseUnk = false,
  });

  bool get isUnigram => modelType == ModelType.unigram;
  bool get isBpe => modelType == ModelType.bpe;

  @override
  String toString() => 'TrainerSpec(type: $modelType, vocab: $vocabSize)';
}

/// Normalization specification from the model.
class NormalizerSpec {
  final String name;
  final Uint8List? precompiledCharsmap;
  final bool addDummyPrefix;
  final bool removeExtraWhitespaces;
  final bool escapeWhitespaces;
  final List<NormalizerOperation> operations;

  const NormalizerSpec({
    this.name = '',
    this.precompiledCharsmap,
    this.addDummyPrefix = true,
    this.removeExtraWhitespaces = true,
    this.escapeWhitespaces = true,
    this.operations = const [],
  });

  @override
  String toString() => 'NormalizerSpec(name: $name)';
}

/// A serialized Hugging Face normalizer operation.
///
/// The operation list is populated only when the JSON pipeline contains a
/// precompiled normalizer. Native SentencePiece models continue to use the
/// flags above, preserving their existing behavior.
class NormalizerOperation {
  final String type;
  final Uint8List? precompiledCharsmap;
  final String? pattern;
  final String? replacement;

  const NormalizerOperation({
    required this.type,
    this.precompiledCharsmap,
    this.pattern,
    this.replacement,
  });
}

/// The supported Hugging Face pre-tokenizer pipeline.
class PreTokenizerSpec {
  final bool whitespaceSplit;
  final bool useMetaspace;
  final bool addPrefixSpace;
  final String replacement;
  final bool split;

  const PreTokenizerSpec({
    this.whitespaceSplit = false,
    this.useMetaspace = true,
    this.addPrefixSpace = true,
    this.replacement = '\u2581',
    this.split = true,
  });
}

/// Complete SentencePiece model structure.
class SentencePieceModel {
  final List<SentencePiece> pieces;
  final TrainerSpec trainerSpec;
  final NormalizerSpec normalizerSpec;
  final NormalizerSpec? denormalizerSpec;
  final PreTokenizerSpec? preTokenizerSpec;

  const SentencePieceModel({
    required this.pieces,
    this.trainerSpec = const TrainerSpec(),
    this.normalizerSpec = const NormalizerSpec(),
    this.denormalizerSpec,
    this.preTokenizerSpec,
  });

  int get vocabSize => pieces.length;

  @override
  String toString() =>
      'SentencePieceModel(pieces: ${pieces.length}, type: ${trainerSpec.modelType})';
}
