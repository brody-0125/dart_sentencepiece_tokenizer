import 'dart:typed_data';

import '../model/model_proto.dart';
import 'precompiled_charsmap.dart';

/// SentencePiece whitespace replacement character.
const String kSpaceSymbol = '\u2581'; // ▁ (Lower One Eighth Block)

class SpNormalizer {
  final bool addDummyPrefix;
  final bool removeExtraWhitespaces;
  final bool escapeWhitespaces;
  final PrecompiledCharsMap? _charsmap;

  SpNormalizer({
    this.addDummyPrefix = true,
    this.removeExtraWhitespaces = true,
    this.escapeWhitespaces = true,
    PrecompiledCharsMap? precompiledCharsmap,
  }) : _charsmap = precompiledCharsmap;

  factory SpNormalizer.fromSpec(NormalizerSpec spec) {
    PrecompiledCharsMap? charsmap;
    if (spec.precompiledCharsmap != null &&
        spec.precompiledCharsmap!.isNotEmpty) {
      charsmap = PrecompiledCharsMap(spec.precompiledCharsmap!);
    }
    return SpNormalizer(
      addDummyPrefix: spec.addDummyPrefix,
      removeExtraWhitespaces: spec.removeExtraWhitespaces,
      escapeWhitespaces: spec.escapeWhitespaces,
      precompiledCharsmap: charsmap,
    );
  }

  /// Returns the raw precompiled charsmap bytes for serialization.
  Uint8List? get precompiledCharsmapBytes => _charsmap?.rawBytes;

  String normalize(String text) {
    if (text.isEmpty) return text;

    var result = text;

    // Step 0: Apply precompiled charsmap normalization
    if (_charsmap != null) {
      result = _charsmap.normalize(result);
    }

    // Step 1: Remove extra whitespaces
    if (removeExtraWhitespaces) {
      result = _collapseWhitespaces(result);
    }

    // Step 2: Add dummy prefix (space at beginning)
    if (addDummyPrefix &&
        result.isNotEmpty &&
        !_isWhitespace(result.codeUnitAt(0))) {
      result = ' $result';
    }

    // Step 3: Escape whitespaces (replace space with ▁)
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
      'escapeWhitespaces: $escapeWhitespaces)';
}
