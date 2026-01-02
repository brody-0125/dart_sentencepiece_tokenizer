import 'dart:typed_data';

import '../../trie.dart';
import '../model/model_proto.dart';

class SpVocabulary {
  final Map<String, int> _pieceToId;
  final List<String> _idToPiece;
  final Float32List _scores;
  final Uint8List _types;
  final Trie _trie;
  final Int16List? _byteToId;

  final int unkId;
  final int bosId;
  final int eosId;
  final int padId;
  final String unkPiece;
  final String bosPiece;
  final String eosPiece;
  final String padPiece;

  SpVocabulary._({
    required Map<String, int> pieceToId,
    required List<String> idToPiece,
    required Float32List scores,
    required Uint8List types,
    required Trie trie,
    Int16List? byteToId,
    required this.unkId,
    required this.bosId,
    required this.eosId,
    required this.padId,
    required this.unkPiece,
    required this.bosPiece,
    required this.eosPiece,
    required this.padPiece,
  })  : _pieceToId = pieceToId,
        _idToPiece = idToPiece,
        _scores = scores,
        _types = types,
        _trie = trie,
        _byteToId = byteToId;

  factory SpVocabulary.fromModel(SentencePieceModel model) {
    final pieces = model.pieces;
    final spec = model.trainerSpec;
    final size = pieces.length;

    final pieceToId = <String, int>{};
    final idToPiece = List<String>.filled(size, '');
    final scores = Float32List(size);
    final types = Uint8List(size);
    final trie = Trie();

    Int16List? byteToId;
    if (spec.byteFallback) {
      byteToId = Int16List(256);
      for (var i = 0; i < 256; i++) {
        byteToId[i] = -1;
      }
    }

    for (var i = 0; i < size; i++) {
      final piece = pieces[i];
      final text = piece.piece;

      pieceToId[text] = i;
      idToPiece[i] = text;
      scores[i] = piece.score;
      types[i] = piece.type.value;

      // Add to trie for prefix matching
      trie.insert(text, i);

      // Track byte fallback tokens
      if (byteToId != null && piece.isByte) {
        final byteValue = _extractByteValue(text);
        if (byteValue != null) {
          byteToId[byteValue] = i;
        }
      }
    }

    return SpVocabulary._(
      pieceToId: pieceToId,
      idToPiece: idToPiece,
      scores: scores,
      types: types,
      trie: trie,
      byteToId: byteToId,
      unkId: spec.unkId,
      bosId: spec.bosId,
      eosId: spec.eosId,
      padId: spec.padId,
      unkPiece: spec.unkPiece,
      bosPiece: spec.bosPiece,
      eosPiece: spec.eosPiece,
      padPiece: spec.padPiece,
    );
  }

  int get size => _idToPiece.length;

  Trie get trie => _trie;

  bool get hasByteFallback => _byteToId != null;

  int pieceToId(String piece) {
    return _pieceToId[piece] ?? unkId;
  }

  String idToPiece(int id) {
    if (id < 0 || id >= _idToPiece.length) {
      return unkPiece;
    }
    return _idToPiece[id];
  }

  double getScore(int id) {
    if (id < 0 || id >= _scores.length) {
      return double.negativeInfinity;
    }
    return _scores[id];
  }

  PieceType getType(int id) {
    if (id < 0 || id >= _types.length) {
      return PieceType.unknown;
    }
    return PieceType.fromValue(_types[id]);
  }

  bool contains(String piece) => _pieceToId.containsKey(piece);

  bool isUnk(int id) => id == unkId;
  bool isBos(int id) => id == bosId;
  bool isEos(int id) => id == eosId;
  bool isPad(int id) => id == padId;

  bool isSpecialToken(int id) {
    final type = getType(id);
    return type == PieceType.control || type == PieceType.unknown;
  }

  bool isByteToken(int id) {
    return getType(id) == PieceType.byte;
  }

  int? byteTokenToValue(int id) {
    if (!isByteToken(id)) return null;
    return _extractByteValue(idToPiece(id));
  }

  int? valueToByteToken(int byteValue) {
    if (_byteToId == null) return null;
    if (byteValue < 0 || byteValue > 255) return null;
    final id = _byteToId[byteValue];
    return id >= 0 ? id : null;
  }

  TrieMatch? findLongestPrefix(String text, [int startIndex = 0]) {
    return _trie.findLongestPrefix(text, startIndex);
  }

  List<TrieMatch> findAllPrefixes(String text, [int startIndex = 0]) {
    return _trie.findAllPrefixes(text, startIndex);
  }

  List<String> get pieces => List.unmodifiable(_idToPiece);

  Float32List get scores => _scores;

  /// Returns an unmodifiable map of token pieces to their IDs.
  Map<String, int> get vocabularyMap => Map.unmodifiable(_pieceToId);

  static int? _extractByteValue(String piece) {
    // Format: <0xHH>
    if (piece.length != 6) return null;
    if (!piece.startsWith('<0x') || !piece.endsWith('>')) return null;
    return int.tryParse(piece.substring(3, 5), radix: 16);
  }

  @override
  String toString() => 'SpVocabulary(size: $size)';
}
