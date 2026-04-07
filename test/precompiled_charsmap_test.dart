import 'dart:convert';
import 'dart:typed_data';

import 'package:dart_sentencepiece_tokenizer/src/sentencepiece/model/model_proto.dart';
import 'package:dart_sentencepiece_tokenizer/src/sentencepiece/normalizer/precompiled_charsmap.dart';
import 'package:dart_sentencepiece_tokenizer/src/sentencepiece/normalizer/sp_normalizer.dart';
import 'package:test/test.dart';

/// Builds a synthetic precompiled_charsmap binary blob for testing.
///
/// The DARTS Double-Array Trie format:
/// - Each node is 8 bytes: int32 base (LE) + int32 check (LE)
/// - Root node is at index 0
/// - Transition: next = base[node] + byte + 1, valid if check[next] == node
/// - Terminal: termNode = base[node], valid if check[termNode] == node
///   and base[termNode] < 0 => poolOffset = -(base[termNode]) - 1
class DartsTrieBuilder {
  // (base, check) pairs indexed by node ID
  final List<int> _base = [];
  final List<int> _check = [];
  final BytesBuilder _pool = BytesBuilder();
  int _nodeCount = 0;

  DartsTrieBuilder() {
    // Allocate root node (index 0)
    _allocateNode();
    _base[0] = 1; // root base; transitions start at base + byte + 1
    _check[0] = -1; // root has no parent
  }

  int _allocateNode() {
    final id = _nodeCount++;
    _base.add(0);
    _check.add(-1);
    return id;
  }

  /// Ensures node at [index] exists, allocating up to that index if needed.
  void _ensureNode(int index) {
    while (_nodeCount <= index) {
      _allocateNode();
    }
  }

  /// Adds a mapping: [utf8Key] bytes -> [replacement] string.
  void addMapping(List<int> utf8Key, String replacement) {
    var node = 0;

    // Walk or create nodes for each byte
    for (final byte in utf8Key) {
      final next = _base[node] + byte + 1;
      _ensureNode(next);

      if (_check[next] == -1) {
        // Unclaimed node, assign it
        _check[next] = node;
        // Set a default base for the new node that won't collide
        // Use a high offset to avoid collisions with other transitions
        _base[next] = _nodeCount + 256;
        _ensureNode(_base[next] + 256); // ensure space
      } else if (_check[next] != node) {
        throw StateError(
          'DARTS collision at node $next: check=${_check[next]} != $node. '
          'This simple builder cannot resolve collisions.',
        );
      }

      node = next;
    }

    // Create terminal node: termNode = base[node], check[termNode] == node
    final termNode = _base[node];
    _ensureNode(termNode);
    _check[termNode] = node;

    // Leaf: base[termNode] = -(poolOffset) - 1
    final poolOffset = _pool.length;
    _pool.add(utf8.encode(replacement));
    _pool.addByte(0); // null terminator
    _base[termNode] = -poolOffset - 1;
  }

  /// Builds the final binary blob.
  Uint8List build() {
    // Trie blob: nodeCount * 8 bytes
    final trieBlobSize = _nodeCount * 8;
    final trieBlob = ByteData(trieBlobSize);
    for (var i = 0; i < _nodeCount; i++) {
      trieBlob.setInt32(i * 8, _base[i], Endian.little);
      trieBlob.setInt32(i * 8 + 4, _check[i], Endian.little);
    }

    final poolBytes = _pool.toBytes();

    // Final blob: 4 bytes (trieBlobSize) + trieBlob + poolBytes
    final result = ByteData(4 + trieBlobSize + poolBytes.length);
    result.setUint32(0, trieBlobSize, Endian.little);

    final resultBytes = result.buffer.asUint8List();
    resultBytes.setRange(4, 4 + trieBlobSize, trieBlob.buffer.asUint8List());
    resultBytes.setRange(
      4 + trieBlobSize,
      4 + trieBlobSize + poolBytes.length,
      poolBytes,
    );

    return resultBytes;
  }
}

void main() {
  group('PrecompiledCharsMap', () {
    group('parsing', () {
      test('throws on data too short', () {
        expect(
          () => PrecompiledCharsMap(Uint8List.fromList([1, 2])),
          throwsArgumentError,
        );
      });

      test('throws on invalid trie blob size', () {
        // Claim 1000 bytes of trie but only provide 8 bytes total
        final data = ByteData(8);
        data.setUint32(0, 1000, Endian.little);
        expect(
          () => PrecompiledCharsMap(data.buffer.asUint8List()),
          throwsArgumentError,
        );
      });

      test('parses valid blob without error', () {
        final builder = DartsTrieBuilder();
        builder.addMapping([0x41], 'a'); // 'A' -> 'a'
        final data = builder.build();
        expect(() => PrecompiledCharsMap(data), returnsNormally);
      });
    });

    group('normalize', () {
      test('empty input returns empty string', () {
        final builder = DartsTrieBuilder();
        builder.addMapping([0x41], 'a');
        final charsmap = PrecompiledCharsMap(builder.build());

        expect(charsmap.normalize(''), equals(''));
      });

      test('single ASCII byte mapping', () {
        final builder = DartsTrieBuilder();
        builder.addMapping([0x41], 'x'); // 'A' (0x41) -> 'x'
        final charsmap = PrecompiledCharsMap(builder.build());

        expect(charsmap.normalize('A'), equals('x'));
      });

      test('unmapped characters pass through', () {
        final builder = DartsTrieBuilder();
        builder.addMapping([0x41], 'x'); // Only 'A' is mapped
        final charsmap = PrecompiledCharsMap(builder.build());

        expect(charsmap.normalize('B'), equals('B'));
        expect(charsmap.normalize('hello'), equals('hello'));
      });

      test('mixed mapped and unmapped characters', () {
        final builder = DartsTrieBuilder();
        builder.addMapping([0x41], 'x'); // 'A' -> 'x'
        final charsmap = PrecompiledCharsMap(builder.build());

        expect(charsmap.normalize('hAllo'), equals('hxllo'));
      });

      test('multiple occurrences of mapped character', () {
        final builder = DartsTrieBuilder();
        builder.addMapping([0x41], 'x'); // 'A' -> 'x'
        final charsmap = PrecompiledCharsMap(builder.build());

        expect(charsmap.normalize('ABA'), equals('xBx'));
      });

      test('mapping to multi-character replacement', () {
        final builder = DartsTrieBuilder();
        builder.addMapping([0x41], 'abc'); // 'A' -> 'abc'
        final charsmap = PrecompiledCharsMap(builder.build());

        expect(charsmap.normalize('A'), equals('abc'));
      });

      test('mapping to empty replacement', () {
        final builder = DartsTrieBuilder();
        builder.addMapping([0x41], ''); // 'A' -> '' (delete)
        final charsmap = PrecompiledCharsMap(builder.build());

        expect(charsmap.normalize('ABA'), equals('B'));
      });

      test('multi-byte UTF-8 source unmapped passes through', () {
        final builder = DartsTrieBuilder();
        builder.addMapping([0x41], 'x'); // Only ASCII 'A' mapped
        final charsmap = PrecompiledCharsMap(builder.build());

        // Korean character '한' is 3 bytes in UTF-8
        expect(charsmap.normalize('한'), equals('한'));
        // 4-byte emoji
        expect(charsmap.normalize('😀'), equals('😀'));
      });

      test('multi-byte UTF-8 source mapped', () {
        final builder = DartsTrieBuilder();
        // Map 'é' (U+00E9) = UTF-8: [0xC3, 0xA9] -> 'e'
        builder.addMapping([0xC3, 0xA9], 'e');
        final charsmap = PrecompiledCharsMap(builder.build());

        expect(charsmap.normalize('café'), equals('cafe'));
      });
    });

    group('rawBytes', () {
      test('preserves original bytes for serialization', () {
        final builder = DartsTrieBuilder();
        builder.addMapping([0x41], 'x');
        final data = builder.build();
        final charsmap = PrecompiledCharsMap(data);

        expect(charsmap.rawBytes, equals(data));
      });
    });
  });

  group('SpNormalizer with precompiledCharsmap', () {
    test('no charsmap behaves identically to before', () {
      final normalizer = SpNormalizer();
      final result = normalizer.normalize('hello world');

      expect(result, contains(kSpaceSymbol));
      expect(result, startsWith(kSpaceSymbol));
    });

    test('charsmap applied before whitespace normalization', () {
      final builder = DartsTrieBuilder();
      builder.addMapping([0x41], ' '); // 'A' -> space
      final charsmap = PrecompiledCharsMap(builder.build());

      final normalizer = SpNormalizer(precompiledCharsmap: charsmap);
      final result = normalizer.normalize('hAllo');

      // 'A' -> ' ', then whitespace collapsing/escaping runs
      // Result: normalize("h llo") -> "▁h▁llo"
      expect(result, equals('${kSpaceSymbol}h${kSpaceSymbol}llo'));
    });

    test('fromSpec creates normalizer with charsmap', () {
      final builder = DartsTrieBuilder();
      builder.addMapping([0x41], 'x');
      final data = builder.build();

      final spec = NormalizerSpec(
        precompiledCharsmap: data,
        addDummyPrefix: false,
        escapeWhitespaces: false,
        removeExtraWhitespaces: false,
      );

      final normalizer = SpNormalizer.fromSpec(spec);
      expect(normalizer.normalize('A'), equals('x'));
    });

    test('fromSpec without charsmap works normally', () {
      const spec = NormalizerSpec(
        addDummyPrefix: false,
        escapeWhitespaces: false,
        removeExtraWhitespaces: false,
      );

      final normalizer = SpNormalizer.fromSpec(spec);
      expect(normalizer.normalize('hello'), equals('hello'));
    });

    test('precompiledCharsmapBytes returns raw bytes', () {
      final builder = DartsTrieBuilder();
      builder.addMapping([0x41], 'x');
      final data = builder.build();

      final spec = NormalizerSpec(precompiledCharsmap: data);
      final normalizer = SpNormalizer.fromSpec(spec);

      expect(normalizer.precompiledCharsmapBytes, equals(data));
    });

    test('precompiledCharsmapBytes returns null when no charsmap', () {
      final normalizer = SpNormalizer();
      expect(normalizer.precompiledCharsmapBytes, isNull);
    });
  });
}
