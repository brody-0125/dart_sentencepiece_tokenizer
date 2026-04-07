import 'dart:convert';
import 'dart:typed_data';

/// Decodes and applies precompiled character mappings from SentencePiece models.
///
/// The binary format consists of:
/// 1. 4 bytes (LE uint32): trie blob size
/// 2. [trieBlobSize] bytes: Double-Array Trie (DARTS format)
/// 3. Remaining bytes: null-terminated UTF-8 replacement strings
///
/// The DARTS trie operates on raw UTF-8 bytes and maps character sequences
/// to replacement strings for normalization (typically NFKC).
class PrecompiledCharsMap {
  final ByteData _trieData;
  final int _trieNodeCount;
  final Uint8List _normalizedPool;

  /// The raw binary data, retained for serialization across isolates.
  final Uint8List rawBytes;

  PrecompiledCharsMap._(
    this._trieData,
    this._trieNodeCount,
    this._normalizedPool,
    this.rawBytes,
  );

  factory PrecompiledCharsMap(Uint8List data) {
    if (data.length < 4) {
      throw ArgumentError(
        'precompiled_charsmap data too short: ${data.length}',
      );
    }

    final byteData = ByteData.sublistView(data);
    final trieBlobSize = byteData.getUint32(0, Endian.little);

    if (4 + trieBlobSize > data.length) {
      throw ArgumentError(
        'Invalid trie blob size: $trieBlobSize (data length: ${data.length})',
      );
    }

    final trieData = ByteData.sublistView(data, 4, 4 + trieBlobSize);
    final trieNodeCount = trieBlobSize ~/ 8;
    final normalizedPool = Uint8List.sublistView(data, 4 + trieBlobSize);

    return PrecompiledCharsMap._(trieData, trieNodeCount, normalizedPool, data);
  }

  /// Returns the base value for node at [index]. Caller must ensure validity.
  int _base(int index) =>
      _trieData.getInt32(index * 8, Endian.little);

  /// Returns the check value for node at [index]. Caller must ensure validity.
  int _check(int index) =>
      _trieData.getInt32(index * 8 + 4, Endian.little);

  bool _isValidNode(int index) => index >= 0 && index < _trieNodeCount;

  /// Returns a view of the null-terminated UTF-8 bytes at [offset] in the pool.
  Uint8List _readPoolBytes(int offset) {
    if (offset < 0 || offset >= _normalizedPool.length) {
      return Uint8List(0);
    }
    var end = offset;
    while (end < _normalizedPool.length && _normalizedPool[end] != 0) {
      end++;
    }
    return Uint8List.sublistView(_normalizedPool, offset, end);
  }

  /// Finds the longest matching prefix in the trie starting at [start] in
  /// the given UTF-8 [bytes].
  ///
  /// Returns `(poolOffset, matchLength)` or `null` if no match.
  (int, int)? _findLongestMatch(Uint8List bytes, int start) {
    var node = 0;
    int? lastPoolOffset;
    var lastMatchLength = 0;

    for (var i = start; i < bytes.length; i++) {
      final next = _base(node) + bytes[i] + 1;
      if (!_isValidNode(next) || _check(next) != node) break;

      node = next;

      // Terminal check: in SentencePiece's DARTS, base[node] points to the
      // terminal node. A negative base at the terminal is a leaf whose
      // absolute value encodes the string pool offset.
      final nodeBase = _base(node);
      if (_isValidNode(nodeBase) && _check(nodeBase) == node) {
        final termBase = _base(nodeBase);
        if (termBase < 0) {
          lastPoolOffset = -termBase - 1;
          lastMatchLength = i - start + 1;
        }
      }
    }

    if (lastPoolOffset != null) {
      return (lastPoolOffset, lastMatchLength);
    }
    return null;
  }

  /// Returns the number of bytes for a UTF-8 character from its leading byte.
  static int _utf8CharLength(int byte) {
    if (byte < 0x80) return 1;
    if (byte < 0xC0) return 1; // continuation byte as leading byte
    if (byte < 0xE0) return 2;
    if (byte < 0xF0) return 3;
    return 4;
  }

  /// Applies character mapping normalization to [input].
  String normalize(String input) {
    if (input.isEmpty) return input;

    final inputBytes = utf8.encode(input);
    final result = BytesBuilder(copy: false);
    var i = 0;

    while (i < inputBytes.length) {
      final match = _findLongestMatch(inputBytes, i);
      if (match != null) {
        final (poolOffset, matchLength) = match;
        // Add pool bytes directly — avoids decode/re-encode round-trip.
        result.add(_readPoolBytes(poolOffset));
        i += matchLength;
      } else {
        // No match: copy the current UTF-8 character as-is.
        final charLen = _utf8CharLength(inputBytes[i]);
        if (charLen == 1) {
          result.addByte(inputBytes[i]);
          i += 1;
        } else {
          final end = (i + charLen).clamp(0, inputBytes.length);
          result.add(Uint8List.sublistView(inputBytes, i, end));
          i = end;
        }
      }
    }

    return utf8.decode(result.toBytes());
  }
}
