import 'dart:convert';

import '../vocabulary/sp_vocabulary.dart';
import 'tokenization_algorithm.dart';

class BpeAlgorithm implements TokenizationAlgorithm {
  final SpVocabulary vocab;
  final bool byteFallback;

  BpeAlgorithm({required this.vocab, this.byteFallback = false});

  @override
  List<int> tokenize(String text) {
    if (text.isEmpty) return [];

    // Initialize symbols from text
    var symbols = _initializeSymbols(text);
    if (symbols.isEmpty) return [];

    // Iteratively merge highest-scoring pairs
    while (symbols.length > 1) {
      final bestMerge = _findBestMerge(symbols);
      if (bestMerge == null) break;

      final (idx, _) = bestMerge;
      symbols = _applyMerge(symbols, idx);
    }

    // Convert symbols to IDs
    return symbols.map((s) => vocab.pieceToId(s)).toList();
  }

  List<String> _initializeSymbols(String text) {
    final symbols = <String>[];

    for (final rune in text.runes) {
      final char = String.fromCharCode(rune);

      // Check if character exists in vocabulary
      if (vocab.contains(char)) {
        symbols.add(char);
      } else if (byteFallback) {
        // Encode as UTF-8 bytes and use byte tokens
        final bytes = utf8.encode(char);
        for (final byte in bytes) {
          final byteTokenId = vocab.valueToByteToken(byte);
          if (byteTokenId != null) {
            symbols.add(vocab.idToPiece(byteTokenId));
          } else {
            // Fallback to UNK if byte token not found
            symbols.add(vocab.unkPiece);
          }
        }
      } else {
        symbols.add(char);
      }
    }

    return symbols;
  }

  (int, double)? _findBestMerge(List<String> symbols) {
    int bestIdx = -1;
    double bestScore = double.negativeInfinity;

    for (var i = 0; i < symbols.length - 1; i++) {
      final merged = symbols[i] + symbols[i + 1];
      if (vocab.contains(merged)) {
        final id = vocab.pieceToId(merged);
        final score = vocab.getScore(id);
        if (score > bestScore) {
          bestScore = score;
          bestIdx = i;
        }
      }
    }

    if (bestIdx < 0) return null;
    return (bestIdx, bestScore);
  }

  List<String> _applyMerge(List<String> symbols, int idx) {
    final merged = symbols[idx] + symbols[idx + 1];
    return [
      ...symbols.sublist(0, idx),
      merged,
      ...symbols.sublist(idx + 2),
    ];
  }
}
