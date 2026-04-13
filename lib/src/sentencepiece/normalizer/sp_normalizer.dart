import 'dart:typed_data';

import '../model/model_proto.dart';
import 'precompiled_charsmap.dart';

/// SentencePiece whitespace replacement character.
const String kSpaceSymbol = '\u2581'; // ▁ (Lower One Eighth Block)

class SpNormalizer {
  final bool addDummyPrefix;
  final bool removeExtraWhitespaces;
  final bool escapeWhitespaces;
  final String normalizerName;
  final PrecompiledCharsmap? _charsmap;

  SpNormalizer({
    this.addDummyPrefix = true,
    this.removeExtraWhitespaces = true,
    this.escapeWhitespaces = true,
    this.normalizerName = '',
    PrecompiledCharsmap? charsmap,
  }) : _charsmap = charsmap;

  factory SpNormalizer.fromSpec(NormalizerSpec spec) {
    PrecompiledCharsmap? charsmap;
    final charsmapBytes = spec.precompiledCharsmap;
    if (charsmapBytes != null && charsmapBytes.isNotEmpty) {
      charsmap = PrecompiledCharsmap.fromBytes(charsmapBytes);
    }
    return SpNormalizer(
      addDummyPrefix: spec.addDummyPrefix,
      removeExtraWhitespaces: spec.removeExtraWhitespaces,
      escapeWhitespaces: spec.escapeWhitespaces,
      normalizerName: spec.name,
      charsmap: charsmap,
    );
  }

  /// Whether this normalizer has a precompiled charsmap.
  bool get hasCharsmap => _charsmap != null;

  /// Precompiled charsmap bytes for serialization.
  ///
  /// Reconstructed on demand from the parsed trie via [PrecompiledCharsmap.toBytes],
  /// avoiding duplicate storage of both raw bytes and parsed structures.
  Uint8List? get precompiledCharsmapBytes => _charsmap?.toBytes();

  String normalize(String text) {
    if (text.isEmpty) return text;

    var result = text;

    // Step 1: Apply precompiled charsmap normalization (NFKC-like)
    if (_charsmap case final charsmap?) {
      result = charsmap.normalize(result);
    }

    // Step 2: Remove extra whitespaces
    if (removeExtraWhitespaces) {
      result = _collapseWhitespaces(result);
    }

    // Step 3: Add dummy prefix (space at beginning)
    if (addDummyPrefix &&
        result.isNotEmpty &&
        !_isWhitespace(result.codeUnitAt(0))) {
      result = ' $result';
    }

    // Step 4: Escape whitespaces (replace space with ▁)
    if (escapeWhitespaces) {
      result = result.replaceAll(' ', kSpaceSymbol);
    }

    return result;
  }

  String denormalize(String text) {
    if (text.isEmpty) return text;

    var result = text;

    // Step 1: Unescape whitespaces (replace ▁ with space)
    if (escapeWhitespaces) {
      result = result.replaceAll(kSpaceSymbol, ' ');
    }

    // Step 2: Remove dummy prefix
    if (addDummyPrefix && result.startsWith(' ')) {
      result = result.substring(1);
    }

    return result;
  }

  String _collapseWhitespaces(String text) {
    final buffer = StringBuffer();
    var prevWasWhitespace = false;

    for (var i = 0; i < text.length; i++) {
      final codeUnit = text.codeUnitAt(i);

      if (_isWhitespace(codeUnit)) {
        if (!prevWasWhitespace) {
          buffer.write(' ');
          prevWasWhitespace = true;
        }
      } else {
        buffer.writeCharCode(codeUnit);
        prevWasWhitespace = false;
      }
    }

    // Trim leading/trailing whitespace
    var result = buffer.toString();
    if (result.startsWith(' ')) result = result.substring(1);
    if (result.endsWith(' ')) result = result.substring(0, result.length - 1);

    return result;
  }

  bool _isWhitespace(int codeUnit) {
    return codeUnit == 0x0009 || // Tab
        codeUnit == 0x000A || // LF
        codeUnit == 0x000B || // VT
        codeUnit == 0x000C || // FF
        codeUnit == 0x000D || // CR
        codeUnit == 0x0020 || // Space
        codeUnit == 0x00A0 || // NBSP
        codeUnit == 0x1680 || // Ogham space
        (codeUnit >= 0x2000 && codeUnit <= 0x200A) || // Various spaces
        codeUnit == 0x202F || // Narrow NBSP
        codeUnit == 0x205F || // Medium mathematical space
        codeUnit == 0x3000; // Ideographic space
  }

  @override
  String toString() =>
      'SpNormalizer('
      'addDummyPrefix: $addDummyPrefix, '
      'removeExtraWhitespaces: $removeExtraWhitespaces, '
      'escapeWhitespaces: $escapeWhitespaces, '
      'normalizerName: $normalizerName, '
      'hasCharsmap: $hasCharsmap)';
}
