import 'dart:convert';
import 'dart:typed_data';

/// Decoder for SentencePiece's precompiled character normalization map.
///
/// The precompiled charsmap is a binary blob containing a DARTS
/// (Double-ARray Trie System) trie and a normalized string pool.
/// It encodes string-to-string mappings used for NFKC-like normalization.
///
/// Binary format:
/// - Bytes 0-3: trie blob size (little-endian uint32)
/// - Bytes 4..(4+trieBlobSize-1): DARTS double-array trie (uint32 units)
/// - Remaining bytes: null-terminated UTF-8 normalized string pool
class PrecompiledCharsmap {
  final Uint32List _array;
  final Uint8List _normalized;

  PrecompiledCharsmap._(this._array, this._normalized);

  /// Parse a precompiled charsmap from its binary representation.
  factory PrecompiledCharsmap.fromBytes(Uint8List data) {
    if (data.length < 4) {
      throw FormatException(
        'Precompiled charsmap too short: ${data.length} bytes',
      );
    }

    final byteData = ByteData.sublistView(data);
    final trieBlobSize = byteData.getUint32(0, Endian.little);

    if (4 + trieBlobSize > data.length) {
      throw FormatException(
        'Trie blob size ($trieBlobSize) exceeds data length (${data.length - 4})',
      );
    }

    if (trieBlobSize % 4 != 0) {
      throw FormatException(
        'Trie blob size ($trieBlobSize) is not a multiple of 4',
      );
    }

    // Parse DARTS double-array (little-endian uint32 units)
    final numUnits = trieBlobSize ~/ 4;
    final array = Uint32List(numUnits);
    final trieByteData = ByteData.sublistView(data, 4, 4 + trieBlobSize);
    for (var i = 0; i < numUnits; i++) {
      array[i] = trieByteData.getUint32(i * 4, Endian.little);
    }

    // Normalized string pool (null-terminated UTF-8 strings)
    final normalized = Uint8List.sublistView(data, 4 + trieBlobSize);

    return PrecompiledCharsmap._(array, normalized);
  }

  /// Apply the precompiled charsmap normalization to [input].
  ///
  /// Uses leftmost-longest matching via the DARTS trie.
  /// Characters not matched by the trie pass through unchanged.
  /// Invalid UTF-8 bytes are replaced with U+FFFD.
  String normalize(String input) {
    if (input.isEmpty || _array.isEmpty) return input;

    final inputBytes = utf8.encode(input);
    final output = BytesBuilder(copy: false);

    var pos = 0;
    while (pos < inputBytes.length) {
      final match = _longestPrefixMatch(inputBytes, pos);

      if (match != null) {
        final (matchLength, normalizedOffset) = match;
        _appendNormalizedString(output, normalizedOffset);
        pos += matchLength;
      } else {
        // No match: copy one UTF-8 character unchanged
        final charLen = _utf8CharLength(inputBytes[pos]);
        if (charLen > 0 && pos + charLen <= inputBytes.length) {
          for (var i = 0; i < charLen; i++) {
            output.addByte(inputBytes[pos + i]);
          }
          pos += charLen;
        } else {
          // Invalid UTF-8: emit U+FFFD replacement character
          output.add(const [0xEF, 0xBF, 0xBD]);
          pos++;
        }
      }
    }

    return utf8.decode(output.takeBytes(), allowMalformed: true);
  }

  /// Find the longest prefix match starting at [start] in [input].
  ///
  /// Returns (matchLength, normalizedStringOffset) or null if no match.
  (int, int)? _longestPrefixMatch(Uint8List input, int start) {
    final arrayLen = _array.length;
    var nodePos = 0;
    var unit = _array[0];
    nodePos ^= _offset(unit);

    var longestLength = 0;
    var longestValue = 0;

    for (var i = start; i < input.length; i++) {
      final c = input[i];

      nodePos ^= c;
      if (nodePos >= arrayLen) break;

      unit = _array[nodePos];
      if (_label(unit) != c) break;

      nodePos ^= _offset(unit);

      if (_hasLeaf(unit)) {
        if (nodePos >= arrayLen) break;
        longestValue = _value(_array[nodePos]);
        longestLength = i - start + 1;
      }
    }

    return longestLength > 0 ? (longestLength, longestValue) : null;
  }

  /// Append the null-terminated normalized string at [offset] to [output].
  void _appendNormalizedString(BytesBuilder output, int offset) {
    var end = offset;
    while (end < _normalized.length && _normalized[end] != 0) {
      end++;
    }
    if (end > offset) {
      output.add(Uint8List.sublistView(_normalized, offset, end));
    }
  }

  // ---- DARTS double-array unit field accessors ----

  /// Whether this unit has a leaf (value) node at its base offset.
  bool _hasLeaf(int unit) => ((unit >> 8) & 1) == 1;

  /// Extract the value from a leaf unit (bits 0-30).
  int _value(int unit) => unit & 0x7FFFFFFF;

  /// Extract the label from a unit (bit 31 + bits 0-7).
  /// For internal nodes, bit 31 is 0 so this returns the byte label (0-255).
  /// For leaf units, bit 31 is set, preventing false label matches.
  int _label(int unit) => unit & 0x800000FF;

  /// Extract the offset (base address) from a unit.
  /// Uses a two-level encoding: bit 9 selects between 22-bit and 30-bit offsets.
  int _offset(int unit) {
    return (unit >> 10) << ((unit & (1 << 9)) >> 6);
  }

  /// Returns the byte length of a UTF-8 character starting with [byte].
  /// Returns 0 for invalid lead bytes.
  static int _utf8CharLength(int byte) {
    if (byte < 0x80) return 1;
    if ((byte & 0xE0) == 0xC0) return 2;
    if ((byte & 0xF0) == 0xE0) return 3;
    if ((byte & 0xF8) == 0xF0) return 4;
    return 0;
  }
}
