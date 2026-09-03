import 'dart:typed_data';

import 'package:dart_sentencepiece_tokenizer/src/sentencepiece/model/model_proto.dart';
import 'package:dart_sentencepiece_tokenizer/src/sentencepiece/normalizer/precompiled_charsmap.dart';
import 'package:dart_sentencepiece_tokenizer/src/sentencepiece/normalizer/sp_normalizer.dart';
import 'package:test/test.dart';

import 'precompiled_charsmap_builder.dart';

void main() {
  group('PrecompiledCharsmap', () {
    group('binary parsing', () {
      test('parses a minimal valid blob', () {
        final blob = buildPrecompiledCharsmapBlob({'A': 'a'});
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
          () => PrecompiledCharsmap.fromBytes(data.buffer.asUint8List()),
          throwsFormatException,
        );
      });

      test('throws on trie size not multiple of 4', () {
        final blob = ByteData(4 + 7);
        blob.setUint32(0, 7, Endian.little);
        expect(
          () => PrecompiledCharsmap.fromBytes(blob.buffer.asUint8List()),
          throwsFormatException,
        );
      });
    });

    group('single byte mapping', () {
      test('maps ASCII character', () {
        final blob = buildPrecompiledCharsmapBlob({'A': 'a'});
        final charsmap = PrecompiledCharsmap.fromBytes(blob);
        expect(charsmap.normalize('A'), equals('a'));
      });

      test('passes through unmapped characters', () {
        final blob = buildPrecompiledCharsmapBlob({'A': 'a'});
        final charsmap = PrecompiledCharsmap.fromBytes(blob);
        expect(charsmap.normalize('B'), equals('B'));
      });

      test('maps mixed text with passthrough', () {
        final blob = buildPrecompiledCharsmapBlob({'A': 'a', 'B': 'b'});
        final charsmap = PrecompiledCharsmap.fromBytes(blob);
        expect(charsmap.normalize('ABCA'), equals('abCa'));
      });
    });

    group('multi-byte UTF-8 mapping', () {
      test('maps fullwidth ASCII to halfwidth', () {
        // Ａ (U+FF21) → A
        final blob = buildPrecompiledCharsmapBlob({'\uFF21': 'A'});
        final charsmap = PrecompiledCharsmap.fromBytes(blob);
        expect(charsmap.normalize('\uFF21'), equals('A'));
      });

      test('maps ligature to component characters', () {
        // ﬁ (U+FB01) → fi
        final blob = buildPrecompiledCharsmapBlob({'\uFB01': 'fi'});
        final charsmap = PrecompiledCharsmap.fromBytes(blob);
        expect(charsmap.normalize('\uFB01'), equals('fi'));
      });

      test('maps fullwidth digits', () {
        // １ (U+FF11) → 1, ２ (U+FF12) → 2
        final blob = buildPrecompiledCharsmapBlob({
          '\uFF11': '1',
          '\uFF12': '2',
        });
        final charsmap = PrecompiledCharsmap.fromBytes(blob);
        expect(charsmap.normalize('\uFF11\uFF12'), equals('12'));
      });

      test('maps halfwidth katakana to fullwidth', () {
        // ｱ (U+FF71) → ア (U+30A2)
        final blob = buildPrecompiledCharsmapBlob({'\uFF71': '\u30A2'});
        final charsmap = PrecompiledCharsmap.fromBytes(blob);
        expect(charsmap.normalize('\uFF71'), equals('\u30A2'));
      });
    });

    group('deletion mapping', () {
      test('deletes mapped characters (maps to empty string)', () {
        // Map zero-width joiner to empty string (deletion)
        final blob = buildPrecompiledCharsmapBlob({'\u200D': ''});
        final charsmap = PrecompiledCharsmap.fromBytes(blob);
        expect(charsmap.normalize('a\u200Db'), equals('ab'));
      });
    });

    group('multiple mappings', () {
      test('applies multiple independent mappings', () {
        final blob = buildPrecompiledCharsmapBlob({
          '\uFF21': 'A', // Ａ → A
          '\uFF22': 'B', // Ｂ → B
          '\uFB01': 'fi', // ﬁ → fi
        });
        final charsmap = PrecompiledCharsmap.fromBytes(blob);
        expect(charsmap.normalize('\uFF21\uFB01\uFF22'), equals('AfiB'));
      });

      test('handles mixed mapped and unmapped characters', () {
        final blob = buildPrecompiledCharsmapBlob({
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

    group('prefix-sharing keys', () {
      test('longest match wins when keys share a prefix', () {
        // Both "A" and "AB" are mapped; "AB" should win for "AB" input
        final blob = buildPrecompiledCharsmapBlob({'A': '1', 'AB': '2'});
        final charsmap = PrecompiledCharsmap.fromBytes(blob);
        expect(charsmap.normalize('AB'), equals('2'));
      });

      test('shorter key matches when longer prefix is absent', () {
        final blob = buildPrecompiledCharsmapBlob({'A': '1', 'AB': '2'});
        final charsmap = PrecompiledCharsmap.fromBytes(blob);
        expect(charsmap.normalize('AC'), equals('1C'));
      });

      test('multiple prefix-sharing keys in sequence', () {
        final blob = buildPrecompiledCharsmapBlob({
          'A': '1',
          'AB': '2',
          'ABC': '3',
        });
        final charsmap = PrecompiledCharsmap.fromBytes(blob);
        // ABC→3, AB→2 (longest for ABD), D→passthrough
        expect(charsmap.normalize('ABCABD'), equals('32D'));
      });
    });

    group('edge cases', () {
      test('empty input returns empty string', () {
        final blob = buildPrecompiledCharsmapBlob({'A': 'a'});
        final charsmap = PrecompiledCharsmap.fromBytes(blob);
        expect(charsmap.normalize(''), equals(''));
      });

      test('all characters unmapped passes through unchanged', () {
        final blob = buildPrecompiledCharsmapBlob({'X': 'x'});
        final charsmap = PrecompiledCharsmap.fromBytes(blob);
        expect(charsmap.normalize('hello world'), equals('hello world'));
      });

      test('handles Korean text passthrough', () {
        final blob = buildPrecompiledCharsmapBlob({'A': 'a'});
        final charsmap = PrecompiledCharsmap.fromBytes(blob);
        expect(charsmap.normalize('안녕하세요'), equals('안녕하세요'));
      });

      test('handles CJK text passthrough', () {
        final blob = buildPrecompiledCharsmapBlob({'A': 'a'});
        final charsmap = PrecompiledCharsmap.fromBytes(blob);
        expect(charsmap.normalize('你好世界'), equals('你好世界'));
      });

      test('handles emoji passthrough', () {
        final blob = buildPrecompiledCharsmapBlob({'A': 'a'});
        final charsmap = PrecompiledCharsmap.fromBytes(blob);
        expect(charsmap.normalize('Hello 👋 World'), equals('Hello 👋 World'));
      });
    });
  });

  group('SpNormalizer with charsmap', () {
    test('applies charsmap before whitespace processing', () {
      final blob = buildPrecompiledCharsmapBlob({'\uFF21': 'A'});
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
      final blob = buildPrecompiledCharsmapBlob({'\uFF21': 'A'});
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
      final blob = buildPrecompiledCharsmapBlob({'\uFF21': 'A'});
      final charsmap = PrecompiledCharsmap.fromBytes(blob);
      final normalizer = SpNormalizer(
        addDummyPrefix: true,
        escapeWhitespaces: true,
        charsmap: charsmap,
      );
      expect(normalizer.denormalize('\u2581hello'), equals('hello'));
    });

    test('preserves precompiled charsmap bytes for serialization', () {
      final blob = buildPrecompiledCharsmapBlob({'\uFF21': 'A'});
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
      expect(
        normalizer.normalize('hello world'),
        equals('\u2581hello\u2581world'),
      );
      expect(
        normalizer.denormalize('\u2581hello\u2581world'),
        equals('hello world'),
      );
    });
  });
}

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
