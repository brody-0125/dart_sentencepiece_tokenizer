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

  /// Parses a precompiled_charsmap binary blob.
  factory PrecompiledCharsMap(Uint8List data) {
    if (data.length < 4) {
      throw ArgumentError('precompiled_charsmap data too short: ${data.length}');
    }

    final byteData = ByteData.sublistView(data);
    final trieBlobSize = byteData.getUint32(0, Endian.little);

    if (4 + trieBlobSize > data.length) {
      throw ArgumentError(
        'Invalid trie blob size: $trieBlobSize (data length: ${data.length})',
      );
    }

    // Each DARTS node is a pair of int32 values (base, check) = 8 bytes.
    final trieData = ByteData.sublistView(data, 4, 4 + trieBlobSize);
    final trieNodeCount = trieBlobSize ~/ 8;

    // The remaining bytes are the normalized string pool.
    final normalizedPool = Uint8List.sublistView(data, 4 + trieBlobSize);

    return PrecompiledCharsMap._(trieData, trieNodeCount, normalizedPool, data);
  }

  /// Returns the base value for a DARTS trie node at [index].
  int _base(int index) {
    final offset = index * 8;
    if (offset < 0 || offset + 4 > _trieData.lengthInBytes) return 0;
    return _trieData.getInt32(offset, Endian.little);
  }

  /// Returns the check value for a DARTS trie node at [index].
  int _check(int index) {
    final offset = index * 8 + 4;
    if (offset < 0 || offset + 4 > _trieData.lengthInBytes) return -1;
    return _trieData.getInt32(offset, Endian.little);
  }

  /// Whether a node index is within the trie bounds.
  bool _isValidNode(int index) => index >= 0 && index < _trieNodeCount;

  /// Reads a null-terminated UTF-8 string from the normalized pool at [offset].
  String _readPoolString(int offset) {
    if (offset < 0 || offset >= _normalizedPool.length) return '';
    var end = offset;
    while (end < _normalizedPool.length && _normalizedPool[end] != 0) {
      end++;
    }
    return utf8.decode(_normalizedPool.sublist(offset, end));
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
      final byte = bytes[i];

      // DARTS transition: next = base[node] + byte + 1
      final next = _base(node) + byte + 1;
      if (!_isValidNode(next) || _check(next) != node) break;

      node = next;

      // Check for terminal: follow the "end of key" transition (value 0).
      // In DARTS, terminal node = base[node] + 0 + 1 = base[node] + 1
      // But the standard convention uses a separate check:
      // termNode = base[node]; if base[termNode] is negative, it's a leaf.
      // Actually in SentencePiece's DARTS: the terminal check uses offset 0.
      final termNode = _base(node);
      if (_isValidNode(termNode) && _check(termNode) == node) {
        final termBase = _base(termNode);
        if (termBase < 0) {
          // Leaf node: |base| is offset into the normalized string pool.
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

  /// Returns the number of bytes for a UTF-8 character starting with
  /// the given leading [byte].
  static int _utf8CharLength(int byte) {
    if (byte < 0x80) return 1;
    if (byte < 0xC0) return 1; // continuation byte (shouldn't be leading)
    if (byte < 0xE0) return 2;
    if (byte < 0xF0) return 3;
    return 4;
  }

  /// Applies character mapping normalization to [input].
  ///
  /// For each position in the UTF-8 encoded input, attempts to find the
  /// longest match in the trie. Matched sequences are replaced with the
  /// corresponding string from the pool. Unmatched characters pass through.
  String normalize(String input) {
    if (input.isEmpty) return input;

    final inputBytes = utf8.encode(input);
    final result = BytesBuilder(copy: false);
    var i = 0;

    while (i < inputBytes.length) {
      final match = _findLongestMatch(inputBytes, i);
      if (match != null) {
        final (poolOffset, matchLength) = match;
        final replacement = _readPoolString(poolOffset);
        result.add(utf8.encode(replacement));
        i += matchLength;
      } else {
        // No match: copy the current UTF-8 character as-is.
        final charLen = _utf8CharLength(inputBytes[i]);
        final end = (i + charLen).clamp(0, inputBytes.length);
        result.add(inputBytes.sublist(i, end));
        i = end;
      }
    }

    return utf8.decode(result.toBytes());
  }
}
