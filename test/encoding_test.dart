import 'dart:typed_data';

import 'package:test/test.dart';
import 'package:dart_sentencepiece_tokenizer/dart_sentencepiece_tokenizer.dart';

void main() {
  group('Encoding sequenceIds', () {
    test('sequenceIds returns consistent results on multiple calls', () {
      final encoding = Encoding(
        tokens: ['<s>', 'hello', 'world'],
        ids: Int32List.fromList([1, 100, 200]),
        typeIds: Uint8List.fromList([0, 0, 0]),
        attentionMask: Uint8List.fromList([1, 1, 1]),
        specialTokensMask: Uint8List.fromList([1, 0, 0]),
        offsets: [(0, 0), (0, 5), (6, 11)],
        wordIds: [null, 0, 1],
      );

      final first = encoding.sequenceIds;
      final second = encoding.sequenceIds;

      // Should return identical (cached) object
      expect(identical(first, second), isTrue);
      expect(first, [null, 0, 0]);
    });

    test('sequenceIds not cached when explicitly provided', () {
      final seqIds = [null, 0, 1];
      final encoding = Encoding(
        tokens: ['<s>', 'hello', 'world'],
        ids: Int32List.fromList([1, 100, 200]),
        typeIds: Uint8List.fromList([0, 0, 1]),
        attentionMask: Uint8List.fromList([1, 1, 1]),
        specialTokensMask: Uint8List.fromList([1, 0, 0]),
        offsets: [(0, 0), (0, 5), (6, 11)],
        wordIds: [null, 0, 1],
        sequenceIds: seqIds,
      );

      expect(encoding.sequenceIds, same(seqIds));
    });

    test('nSequences uses cached sequenceIds', () {
      final encoding = Encoding(
        tokens: ['a', 'b'],
        ids: Int32List.fromList([1, 2]),
        typeIds: Uint8List.fromList([0, 1]),
        attentionMask: Uint8List.fromList([1, 1]),
        specialTokensMask: Uint8List.fromList([0, 0]),
        offsets: [(0, 1), (2, 3)],
        wordIds: [0, 1],
      );

      // First call computes and caches, second reuses
      expect(encoding.nSequences, 2);
      expect(encoding.nSequences, 2);
    });
  });

  group('Encoding padding', () {
    test('withPadding right-pads correctly', () {
      final encoding = Encoding(
        tokens: ['a', 'b'],
        ids: Int32List.fromList([1, 2]),
        typeIds: Uint8List.fromList([0, 0]),
        attentionMask: Uint8List.fromList([1, 1]),
        specialTokensMask: Uint8List.fromList([0, 0]),
        offsets: [(0, 1), (2, 3)],
        wordIds: [0, 1],
        sequenceIds: [0, 0],
      );

      final padded = encoding.withPadding(targetLength: 5, padTokenId: 99);

      expect(padded.length, 5);
      expect(padded.ids[0], 1);
      expect(padded.ids[1], 2);
      expect(padded.ids[2], 99);
      expect(padded.ids[3], 99);
      expect(padded.ids[4], 99);
      expect(padded.specialTokensMask[0], 0);
      expect(padded.specialTokensMask[1], 0);
      expect(padded.specialTokensMask[2], 1);
      expect(padded.specialTokensMask[3], 1);
      expect(padded.specialTokensMask[4], 1);
    });

    test('withPadding left-pads correctly', () {
      final encoding = Encoding(
        tokens: ['a'],
        ids: Int32List.fromList([1]),
        typeIds: Uint8List.fromList([0]),
        attentionMask: Uint8List.fromList([1]),
        specialTokensMask: Uint8List.fromList([0]),
        offsets: [(0, 1)],
        wordIds: [0],
        sequenceIds: [0],
      );

      final padded = encoding.withPadding(
        targetLength: 3,
        padTokenId: 99,
        padOnRight: false,
      );

      expect(padded.length, 3);
      expect(padded.ids[0], 99);
      expect(padded.ids[1], 99);
      expect(padded.ids[2], 1);
      expect(padded.specialTokensMask[0], 1);
      expect(padded.specialTokensMask[1], 1);
      expect(padded.specialTokensMask[2], 0);
    });

    test('withPadding returns this when already at target length', () {
      final encoding = Encoding(
        tokens: ['a', 'b'],
        ids: Int32List.fromList([1, 2]),
        typeIds: Uint8List.fromList([0, 0]),
        attentionMask: Uint8List.fromList([1, 1]),
        specialTokensMask: Uint8List.fromList([0, 0]),
        offsets: [(0, 1), (2, 3)],
        wordIds: [0, 1],
      );

      final padded = encoding.withPadding(targetLength: 2, padTokenId: 99);
      expect(identical(padded, encoding), isTrue);
    });
  });
}
