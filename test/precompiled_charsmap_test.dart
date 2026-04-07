import 'dart:convert';
import 'dart:typed_data';

import 'package:dart_sentencepiece_tokenizer/src/sentencepiece/model/model_proto.dart';
import 'package:dart_sentencepiece_tokenizer/src/sentencepiece/normalizer/precompiled_charsmap.dart';
import 'package:dart_sentencepiece_tokenizer/src/sentencepiece/normalizer/sp_normalizer.dart';
import 'package:test/test.dart';

void main() {
  group('PrecompiledCharsmap', () {
    group('binary parsing', () {
      test('parses a minimal valid blob', () {
        final blob = _buildCharsmapBlob({'A': 'a'});
        final charsmap = PrecompiledCharsmap.fromBytes(blob);
        expect(charsmap.normalize('A'), equals('a'));
      });

      test('throws on too-short data', () {
        expect(
          () => PrecompiledCharsmap.fromBytes(Uint8List.fromList([0, 0])),
          throwsFormatException,
        );
      });

      test('throws on trie size exceeding data', () {
        // 4 bytes saying trie is 1000 bytes, but only 4 bytes total
        final data = ByteData(4);
        data.setUint32(0, 1000, Endian.little);
        expect(
          () => PrecompiledCharsmap.fromBytes(
            data.buffer.asUint8List(),
          ),
          throwsFormatException,
        );
      });

      test('throws on trie size not multiple of 4', () {
        final blob = ByteData(4 + 7);
        blob.setUint32(0, 7, Endian.little);
        expect(
          () => PrecompiledCharsmap.fromBytes(
            blob.buffer.asUint8List(),
          ),
          throwsFormatException,
        );
      });
    });

    group('single byte mapping', () {
      test('maps ASCII character', () {
        final blob = _buildCharsmapBlob({'A': 'a'});
        final charsmap = PrecompiledCharsmap.fromBytes(blob);
        expect(charsmap.normalize('A'), equals('a'));
      });

      test('passes through unmapped characters', () {
        final blob = _buildCharsmapBlob({'A': 'a'});
        final charsmap = PrecompiledCharsmap.fromBytes(blob);
        expect(charsmap.normalize('B'), equals('B'));
      });

      test('maps mixed text with passthrough', () {
        final blob = _buildCharsmapBlob({'A': 'a', 'B': 'b'});
        final charsmap = PrecompiledCharsmap.fromBytes(blob);
        expect(charsmap.normalize('ABCA'), equals('abCa'));
      });
    });

    group('multi-byte UTF-8 mapping', () {
      test('maps fullwidth ASCII to halfwidth', () {
        // Ａ (U+FF21) → A
        final blob = _buildCharsmapBlob({'\uFF21': 'A'});
        final charsmap = PrecompiledCharsmap.fromBytes(blob);
        expect(charsmap.normalize('\uFF21'), equals('A'));
      });

      test('maps ligature to component characters', () {
        // ﬁ (U+FB01) → fi
        final blob = _buildCharsmapBlob({'\uFB01': 'fi'});
        final charsmap = PrecompiledCharsmap.fromBytes(blob);
        expect(charsmap.normalize('\uFB01'), equals('fi'));
      });

      test('maps fullwidth digits', () {
        // １ (U+FF11) → 1, ２ (U+FF12) → 2
        final blob = _buildCharsmapBlob({'\uFF11': '1', '\uFF12': '2'});
        final charsmap = PrecompiledCharsmap.fromBytes(blob);
        expect(charsmap.normalize('\uFF11\uFF12'), equals('12'));
      });

      test('maps halfwidth katakana to fullwidth', () {
        // ｱ (U+FF71) → ア (U+30A2)
        final blob = _buildCharsmapBlob({'\uFF71': '\u30A2'});
        final charsmap = PrecompiledCharsmap.fromBytes(blob);
        expect(charsmap.normalize('\uFF71'), equals('\u30A2'));
      });
    });

    group('deletion mapping', () {
      test('deletes mapped characters (maps to empty string)', () {
        // Map zero-width joiner to empty string (deletion)
        final blob = _buildCharsmapBlob({'\u200D': ''});
        final charsmap = PrecompiledCharsmap.fromBytes(blob);
        expect(charsmap.normalize('a\u200Db'), equals('ab'));
      });
    });

    group('multiple mappings', () {
      test('applies multiple independent mappings', () {
        final blob = _buildCharsmapBlob({
          '\uFF21': 'A', // Ａ → A
          '\uFF22': 'B', // Ｂ → B
          '\uFB01': 'fi', // ﬁ → fi
        });
        final charsmap = PrecompiledCharsmap.fromBytes(blob);
        expect(
          charsmap.normalize('\uFF21\uFB01\uFF22'),
          equals('AfiB'),
        );
      });

      test('handles mixed mapped and unmapped characters', () {
        final blob = _buildCharsmapBlob({
          '\uFF21': 'A',
          '\uFF22': 'B',
        });
        final charsmap = PrecompiledCharsmap.fromBytes(blob);
        expect(
          charsmap.normalize('hello \uFF21 world \uFF22'),
          equals('hello A world B'),
        );
      });
    });

    group('edge cases', () {
      test('empty input returns empty string', () {
        final blob = _buildCharsmapBlob({'A': 'a'});
        final charsmap = PrecompiledCharsmap.fromBytes(blob);
        expect(charsmap.normalize(''), equals(''));
      });

      test('all characters unmapped passes through unchanged', () {
        final blob = _buildCharsmapBlob({'X': 'x'});
        final charsmap = PrecompiledCharsmap.fromBytes(blob);
        expect(
          charsmap.normalize('hello world'),
          equals('hello world'),
        );
      });

      test('handles Korean text passthrough', () {
        final blob = _buildCharsmapBlob({'A': 'a'});
        final charsmap = PrecompiledCharsmap.fromBytes(blob);
        expect(
          charsmap.normalize('안녕하세요'),
          equals('안녕하세요'),
        );
      });

      test('handles CJK text passthrough', () {
        final blob = _buildCharsmapBlob({'A': 'a'});
        final charsmap = PrecompiledCharsmap.fromBytes(blob);
        expect(
          charsmap.normalize('你好世界'),
          equals('你好世界'),
        );
      });

      test('handles emoji passthrough', () {
        final blob = _buildCharsmapBlob({'A': 'a'});
        final charsmap = PrecompiledCharsmap.fromBytes(blob);
        expect(
          charsmap.normalize('Hello 👋 World'),
          equals('Hello 👋 World'),
        );
      });
    });
  });

  group('SpNormalizer with charsmap', () {
    test('applies charsmap before whitespace processing', () {
      final blob = _buildCharsmapBlob({'\uFF21': 'A'});
      final charsmap = PrecompiledCharsmap.fromBytes(blob);
      final normalizer = SpNormalizer(
        addDummyPrefix: true,
        removeExtraWhitespaces: true,
        escapeWhitespaces: true,
        charsmap: charsmap,
      );
      // Input: Ａ → charsmap → A → dummy prefix → " A" → escape → "▁A"
      expect(normalizer.normalize('\uFF21'), equals('\u2581A'));
    });

    test('fromSpec with precompiled charsmap', () {
      final blob = _buildCharsmapBlob({'\uFF21': 'A'});
      final spec = _createNormalizerSpec(
        name: 'nmt_nfkc',
        precompiledCharsmap: blob,
      );
      final normalizer = SpNormalizer.fromSpec(spec);
      expect(normalizer.hasCharsmap, isTrue);
      expect(normalizer.normalizerName, equals('nmt_nfkc'));
      expect(normalizer.normalize('\uFF21'), equals('\u2581A'));
    });

    test('fromSpec without precompiled charsmap', () {
      final spec = _createNormalizerSpec(name: 'identity');
      final normalizer = SpNormalizer.fromSpec(spec);
      expect(normalizer.hasCharsmap, isFalse);
      expect(normalizer.normalizerName, equals('identity'));
    });

    test('denormalize is unaffected by charsmap', () {
      final blob = _buildCharsmapBlob({'\uFF21': 'A'});
      final charsmap = PrecompiledCharsmap.fromBytes(blob);
      final normalizer = SpNormalizer(
        addDummyPrefix: true,
        escapeWhitespaces: true,
        charsmap: charsmap,
      );
      expect(normalizer.denormalize('\u2581hello'), equals('hello'));
    });

    test('preserves precompiled charsmap bytes for serialization', () {
      final blob = _buildCharsmapBlob({'\uFF21': 'A'});
      final spec = _createNormalizerSpec(
        name: 'nmt_nfkc',
        precompiledCharsmap: blob,
      );
      final normalizer = SpNormalizer.fromSpec(spec);
      expect(normalizer.precompiledCharsmapBytes, isNotNull);
      expect(normalizer.precompiledCharsmapBytes, equals(blob));
    });
  });

  group('backward compatibility', () {
    test('SpNormalizer without charsmap behaves identically', () {
      final normalizer = SpNormalizer(
        addDummyPrefix: true,
        removeExtraWhitespaces: true,
        escapeWhitespaces: true,
      );
      expect(normalizer.hasCharsmap, isFalse);
      expect(normalizer.normalize('hello world'), equals('\u2581hello\u2581world'));
      expect(normalizer.denormalize('\u2581hello\u2581world'), equals('hello world'));
    });
  });
}

// ---- Test Helpers ----

/// Build a precompiled charsmap binary blob from string-to-string mappings.
Uint8List _buildCharsmapBlob(Map<String, String> mappings) {
  // Convert string keys to UTF-8 byte sequences
  final entries = <List<int>, String>{};
  for (final entry in mappings.entries) {
    entries[utf8.encode(entry.key)] = entry.value;
  }

  // Build normalized string table
  final normalizedBytes = BytesBuilder();
  final offsets = <String, int>{};
  for (final entry in mappings.entries) {
    offsets[entry.key] = normalizedBytes.length;
    normalizedBytes.add(utf8.encode(entry.value));
    normalizedBytes.addByte(0); // null terminator
  }

  // Build DARTS double-array trie
  final trieEntries = <List<int>, int>{};
  for (final entry in mappings.entries) {
    trieEntries[utf8.encode(entry.key)] = offsets[entry.key]!;
  }
  final arrayData = _buildDartsArray(trieEntries);

  // Assemble blob: [4-byte trie size][trie data][normalized strings]
  final trieBlobSize = arrayData.length * 4;
  final blob = BytesBuilder();

  // Write trie blob size (little-endian uint32)
  final sizeData = ByteData(4);
  sizeData.setUint32(0, trieBlobSize, Endian.little);
  blob.add(sizeData.buffer.asUint8List());

  // Write trie array data (little-endian uint32 per unit)
  for (final unit in arrayData) {
    final unitData = ByteData(4);
    unitData.setUint32(0, unit, Endian.little);
    blob.add(unitData.buffer.asUint8List());
  }

  // Write normalized string pool
  blob.add(normalizedBytes.toBytes());

  return blob.toBytes();
}

/// Build a DARTS double-array from key-value entries.
/// Keys are UTF-8 byte sequences, values are offsets into the string table.
Uint32List _buildDartsArray(Map<List<int>, int> entries) {
  // Build prefix trie
  final root = _TrieNode();
  for (final entry in entries.entries) {
    var node = root;
    for (final b in entry.key) {
      node = node.children.putIfAbsent(b, () => _TrieNode());
    }
    node.value = entry.value;
  }

  // Convert to double-array
  final array = List<int>.filled(1024, 0);
  final used = List<bool>.filled(1024, false);
  // Track used effective bases to prevent cross-node collisions.
  // In DARTS, each node's effective base must be unique to avoid
  // false positive matches during traversal.
  final usedBases = <int>{};

  void ensureSize(int size) {
    while (array.length < size) {
      array.add(0);
      used.add(false);
    }
  }

  void buildNode(_TrieNode node, int nodePos, int label) {
    final childLabels = node.children.keys.toList()..sort();
    final needLeaf = node.value != null;

    // Find an effectiveBase that:
    // 1. Is not already used as another node's effective base
    // 2. Has no position conflicts for leaf and children
    var effectiveBase = 1;
    outer:
    while (true) {
      ensureSize(effectiveBase + 256);
      // Must be a unique base
      if (usedBases.contains(effectiveBase)) {
        effectiveBase++;
        continue;
      }
      // Check leaf position
      if (needLeaf && used[effectiveBase]) {
        effectiveBase++;
        continue;
      }
      // Check child positions
      for (final cl in childLabels) {
        final pos = effectiveBase ^ cl;
        ensureSize(pos + 1);
        if (used[pos]) {
          effectiveBase++;
          continue outer;
        }
      }
      break;
    }
    usedBases.add(effectiveBase);

    // offset = nodePos ^ effectiveBase (so that nodePos ^ offset = effectiveBase)
    final offset = nodePos ^ effectiveBase;

    // Encode unit: [offset bits 10-31][bit 9: offset mode][bit 8: has_leaf][bits 0-7: label]
    var unit = (offset << 10) | (label & 0xFF);
    if (needLeaf) unit |= 1 << 8;

    ensureSize(nodePos + 1);
    array[nodePos] = unit;
    used[nodePos] = true;

    // Place leaf value unit
    if (needLeaf) {
      ensureSize(effectiveBase + 1);
      array[effectiveBase] = 0x80000000 | node.value!;
      used[effectiveBase] = true;
    }

    // Reserve child positions before recursing
    for (final cl in childLabels) {
      final pos = effectiveBase ^ cl;
      ensureSize(pos + 1);
      used[pos] = true;
    }

    // Build children
    for (final cl in childLabels) {
      final childPos = effectiveBase ^ cl;
      buildNode(node.children[cl]!, childPos, cl);
    }
  }

  buildNode(root, 0, 0);

  // Trim trailing unused entries
  var lastUsed = array.length - 1;
  while (lastUsed > 0 && !used[lastUsed]) {
    lastUsed--;
  }

  return Uint32List.fromList(array.sublist(0, lastUsed + 1));
}

class _TrieNode {
  final children = <int, _TrieNode>{};
  int? value;
}

/// Create a NormalizerSpec for testing.
NormalizerSpec _createNormalizerSpec({
  String name = '',
  Uint8List? precompiledCharsmap,
  bool addDummyPrefix = true,
  bool removeExtraWhitespaces = true,
  bool escapeWhitespaces = true,
}) {
  return NormalizerSpec(
    name: name,
    precompiledCharsmap: precompiledCharsmap,
    addDummyPrefix: addDummyPrefix,
    removeExtraWhitespaces: removeExtraWhitespaces,
    escapeWhitespaces: escapeWhitespaces,
  );
}
