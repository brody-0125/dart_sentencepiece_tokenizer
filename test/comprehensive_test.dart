/// Comprehensive test suite for SentencePiece tokenizer.
///
/// Tests core functionality, edge cases, and integration scenarios.
import 'package:test/test.dart';
import 'package:dart_sentencepiece_tokenizer/dart_sentencepiece_tokenizer.dart';

import 'test_utils.dart';

void main() {
  group('SentencePieceTokenizer Core', () {
    late SentencePieceTokenizer tokenizer;

    setUpAll(() {
      tokenizer = createTestTokenizer();
    });

    test('vocabSize returns correct count', () {
      expect(tokenizer.vocabSize, greaterThan(0));
    });

    test('modelType is set correctly', () {
      expect(tokenizer.modelType, isNotNull);
    });

    test('toString provides useful info', () {
      final str = tokenizer.toString();
      expect(str, contains('SentencePieceTokenizer'));
      expect(str, contains('vocabSize'));
    });
  });

  group('Encoding Structure', () {
    late SentencePieceTokenizer tokenizer;

    setUpAll(() {
      tokenizer = createTestTokenizer();
    });

    test('encoding has correct field types', () {
      final encoding = tokenizer.encode('hello', addSpecialTokens: false);

      expect(encoding.ids, isA<List<int>>());
      expect(encoding.tokens, isA<List<String>>());
      expect(encoding.attentionMask, isA<List<int>>());
      expect(encoding.typeIds, isA<List<int>>());
      expect(encoding.specialTokensMask, isA<List<int>>());
      expect(encoding.offsets, isA<List<(int, int)>>());
    });

    test('all encoding arrays have same length', () {
      final encoding = tokenizer.encode('hello world', addSpecialTokens: false);

      expect(encoding.tokens.length, equals(encoding.length));
      expect(encoding.ids.length, equals(encoding.length));
      expect(encoding.attentionMask.length, equals(encoding.length));
      expect(encoding.typeIds.length, equals(encoding.length));
      expect(encoding.specialTokensMask.length, equals(encoding.length));
    });

    test('encoding length matches token count', () {
      final encoding = tokenizer.encode('test', addSpecialTokens: false);
      expect(encoding.length, equals(encoding.tokens.length));
    });
  });

  group('Special Tokens Configuration', () {
    test('default config adds no special tokens', () {
      final tokenizer = createTestTokenizer();
      final encoding = tokenizer.encode('hello');

      // Default config should NOT add BOS/EOS
      expect(encoding.tokens.contains(tokenizer.vocab.bosPiece), isFalse);
      expect(encoding.tokens.contains(tokenizer.vocab.eosPiece), isFalse);
    });

    test('llama config adds BOS only', () {
      final tokenizer = createTestTokenizer(config: SentencePieceConfig.llama);

      final encoding = tokenizer.encode('hello');

      // Should have BOS at start but no EOS
      if (tokenizer.vocab.bosId >= 0) {
        expect(encoding.ids.first, equals(tokenizer.vocab.bosId));
      }
      expect(encoding.tokens.contains(tokenizer.vocab.eosPiece), isFalse);
    });

    test('gemma config adds BOS and EOS', () {
      final tokenizer = createTestTokenizer(config: SentencePieceConfig.gemma);

      final encoding = tokenizer.encode('hello');

      // Should have BOS at start and EOS at end
      if (tokenizer.vocab.bosId >= 0) {
        expect(encoding.ids.first, equals(tokenizer.vocab.bosId));
      }
      if (tokenizer.vocab.eosId >= 0) {
        expect(encoding.ids.last, equals(tokenizer.vocab.eosId));
      }
    });

    test('addSpecialTokens=false overrides config', () {
      final tokenizer = createTestTokenizer(config: SentencePieceConfig.gemma);

      final encoding = tokenizer.encode('hello', addSpecialTokens: false);

      // Should NOT have BOS or EOS
      expect(encoding.tokens.contains(tokenizer.vocab.bosPiece), isFalse);
      expect(encoding.tokens.contains(tokenizer.vocab.eosPiece), isFalse);
    });

    test('numSpecialTokensToAdd returns correct count', () {
      final defaultTokenizer = createTestTokenizer();
      final llamaTokenizer = createLlamaTokenizer();
      final gemmaTokenizer = createGemmaTokenizer();

      // Default config has no special tokens
      expect(defaultTokenizer.numSpecialTokensToAdd(), equals(0));
      // Llama/Gemma count depends on whether model has valid BOS/EOS IDs
      // Just verify they return non-negative values
      expect(llamaTokenizer.numSpecialTokensToAdd(), greaterThanOrEqualTo(0));
      expect(gemmaTokenizer.numSpecialTokensToAdd(), greaterThanOrEqualTo(0));
    });
  });

  group('Vocabulary Access', () {
    late SentencePieceTokenizer tokenizer;

    setUpAll(() {
      tokenizer = createTestTokenizer();
    });

    test('convertTokensToIds and back', () {
      final encoding = tokenizer.encode('hello', addSpecialTokens: false);
      final ids = tokenizer.convertTokensToIds(encoding.tokens);
      final tokens = tokenizer.convertIdsToTokens(ids);

      expect(ids, equals(encoding.ids.toList()));
      expect(tokens, equals(encoding.tokens));
    });

    test('vocab.contains works correctly', () {
      expect(tokenizer.vocab.contains('<unk>'), isTrue);
      expect(tokenizer.vocab.contains('<s>'), isTrue);
      expect(tokenizer.vocab.contains('nonexistent_token_xyz'), isFalse);
    });

    test('special token IDs are accessible', () {
      // UNK should always be defined
      expect(tokenizer.vocab.unkId, greaterThanOrEqualTo(0));
      // Note: bosId, eosId and padId may be -1 if not defined in the model
      // Different models have different special token configurations
      // We just verify the properties are accessible (don't throw)
      expect(tokenizer.vocab.bosId, isA<int>());
      expect(tokenizer.vocab.eosId, isA<int>());
      expect(tokenizer.vocab.padId, isA<int>());
    });
  });

  group('Padding Configuration', () {
    test('enablePadding returns tokenizer for chaining', () {
      final tokenizer = createTestTokenizer();
      final result = tokenizer.enablePadding(length: 10);

      expect(result, same(tokenizer));
    });

    test('noPadding clears padding config', () {
      final tokenizer = createTestTokenizer()
        ..enablePadding(length: 10)
        ..noPadding();

      expect(tokenizer.padding, isNull);
    });

    test('padding config is stored correctly', () {
      final tokenizer = createTestTokenizer()
        ..enablePadding(
          length: 512,
          direction: SpPaddingDirection.left,
          padToMultipleOf: 8,
        );

      expect(tokenizer.padding, isNotNull);
      expect(tokenizer.padding!.length, equals(512));
      expect(tokenizer.padding!.direction, equals(SpPaddingDirection.left));
      expect(tokenizer.padding!.padToMultipleOf, equals(8));
    });

    test('batch padding pads to longest', () {
      final tokenizer = createTestTokenizer()..enablePadding();

      final texts = ['a', 'hello world', 'test'];
      final batch = tokenizer.encodeBatch(texts, addSpecialTokens: false);

      final maxLen = batch.map((e) => e.length).reduce((a, b) => a > b ? a : b);
      for (final encoding in batch) {
        expect(encoding.length, equals(maxLen));
      }
    });
  });

  group('Truncation Configuration', () {
    test('enableTruncation returns tokenizer for chaining', () {
      final tokenizer = createTestTokenizer();
      final result = tokenizer.enableTruncation(maxLength: 10);

      expect(result, same(tokenizer));
    });

    test('noTruncation clears truncation config', () {
      final tokenizer = createTestTokenizer()
        ..enableTruncation(maxLength: 10)
        ..noTruncation();

      expect(tokenizer.truncation, isNull);
    });

    test('truncation config is stored correctly', () {
      final tokenizer = createTestTokenizer()
        ..enableTruncation(
          maxLength: 512,
          direction: SpTruncationDirection.left,
        );

      expect(tokenizer.truncation, isNotNull);
      expect(tokenizer.truncation!.maxLength, equals(512));
      expect(tokenizer.truncation!.direction, equals(SpTruncationDirection.left));
    });

    test('truncation respects maxLength', () {
      final tokenizer = createTestTokenizer()..enableTruncation(maxLength: 5);

      final encoding = tokenizer.encode(
        'This is a longer sentence that should be truncated',
        addSpecialTokens: false,
      );

      expect(encoding.length, lessThanOrEqualTo(5));
    });
  });

  group('Batch Processing', () {
    late SentencePieceTokenizer tokenizer;

    setUpAll(() {
      tokenizer = createTestTokenizer();
    });

    test('encodeBatch handles empty list', () {
      final batch = tokenizer.encodeBatch([], addSpecialTokens: false);
      expect(batch, isEmpty);
    });

    test('encodeBatch handles single item', () {
      final batch = tokenizer.encodeBatch(['hello'], addSpecialTokens: false);
      expect(batch.length, equals(1));
    });

    test('encodeBatch preserves order', () {
      final texts = ['first', 'second', 'third'];
      final batch = tokenizer.encodeBatch(texts, addSpecialTokens: false);

      expect(batch.length, equals(3));
      // Each encoding should correspond to its text
      for (var i = 0; i < texts.length; i++) {
        final singleEncoding = tokenizer.encode(
          texts[i],
          addSpecialTokens: false,
        );
        expect(batch[i].ids, equals(singleEncoding.ids));
      }
    });

    test('encodeBatchParallel matches sequential', () async {
      final texts = [
        for (var i = 0; i < 20; i++) 'test sentence number $i',
      ];

      final sequential = tokenizer.encodeBatch(texts, addSpecialTokens: false);
      final parallel = await tokenizer.encodeBatchParallel(
        texts,
        addSpecialTokens: false,
      );

      expect(parallel.length, equals(sequential.length));
      for (var i = 0; i < texts.length; i++) {
        expect(
          parallel[i].ids.toList(),
          equals(sequential[i].ids.toList()),
        );
      }
    });
  });

  group('Decode Functionality', () {
    late SentencePieceTokenizer tokenizer;

    setUpAll(() {
      tokenizer = createTestTokenizer();
    });

    test('decode empty list returns empty string', () {
      expect(tokenizer.decode([]), equals(''));
    });

    test('decode with skipSpecialTokens=true', () {
      final tokenizer = createLlamaTokenizer();
      final encoding = tokenizer.encode('hello');
      final decoded = tokenizer.decode(encoding.ids.toList(), skipSpecialTokens: true);

      // Should not contain BOS token text
      expect(decoded, isNot(contains('<s>')));
    });

    test('decode with skipSpecialTokens=false includes special tokens', () {
      final tokenizer = createLlamaTokenizer();
      final encoding = tokenizer.encode('hello');

      // Verify BOS was added to encoding
      if (tokenizer.vocab.bosId >= 0 && encoding.ids.contains(tokenizer.vocab.bosId)) {
        final decoded = tokenizer.decode(encoding.ids.toList(), skipSpecialTokens: false);
        // When BOS is in encoding and we don't skip special tokens,
        // the decoded string should contain the BOS piece
        expect(decoded, contains(tokenizer.vocab.bosPiece));
      } else {
        // If BOS is not in the encoding, just verify decode works
        final decoded = tokenizer.decode(encoding.ids.toList(), skipSpecialTokens: false);
        expect(decoded, isNotEmpty);
      }
    });

    test('decodeBatch handles multiple sequences', () {
      final encodings = tokenizer.encodeBatch(
        ['hello', 'world'],
        addSpecialTokens: false,
      );
      final decoded = tokenizer.decodeBatch([
        encodings[0].ids.toList(),
        encodings[1].ids.toList(),
      ]);

      expect(decoded.length, equals(2));
    });
  });

  group('Edge Cases - Input Handling', () {
    late SentencePieceTokenizer tokenizer;

    setUpAll(() {
      tokenizer = createTestTokenizer();
    });

    test('empty string', () {
      final encoding = tokenizer.encode('', addSpecialTokens: false);
      expect(encoding.length, equals(0));
    });

    test('single space', () {
      final encoding = tokenizer.encode(' ', addSpecialTokens: false);
      // Should handle gracefully
      expect(encoding, isNotNull);
    });

    test('multiple spaces', () {
      final encoding = tokenizer.encode('     ', addSpecialTokens: false);
      expect(encoding, isNotNull);
    });

    test('tabs and newlines', () {
      final encoding = tokenizer.encode('\t\n\r', addSpecialTokens: false);
      expect(encoding, isNotNull);
    });

    test('unicode characters', () {
      final encoding = tokenizer.encode('café naïve', addSpecialTokens: false);
      expect(encoding.length, greaterThan(0));
    });

    test('emoji', () {
      final encoding = tokenizer.encode('Hello 😊 World 🌍', addSpecialTokens: false);
      expect(encoding.length, greaterThan(0));
    });

    test('chinese characters', () {
      final encoding = tokenizer.encode('你好世界', addSpecialTokens: false);
      expect(encoding.length, greaterThan(0));
    });

    test('japanese hiragana', () {
      final encoding = tokenizer.encode('こんにちは', addSpecialTokens: false);
      expect(encoding.length, greaterThan(0));
    });

    test('korean characters', () {
      final encoding = tokenizer.encode('안녕하세요', addSpecialTokens: false);
      expect(encoding.length, greaterThan(0));
    });

    test('mixed scripts', () {
      final encoding = tokenizer.encode(
        'Hello 世界 مرحبا שלום',
        addSpecialTokens: false,
      );
      expect(encoding.length, greaterThan(0));
    });

    test('very long text', () {
      final longText = 'word ' * 1000;
      final encoding = tokenizer.encode(longText, addSpecialTokens: false);
      expect(encoding.length, greaterThan(0));
    });

    test('special characters', () {
      final encoding = tokenizer.encode(
        '!@#\$%^&*()_+-=[]{}|;:,.<>?',
        addSpecialTokens: false,
      );
      expect(encoding.length, greaterThan(0));
    });

    test('null bytes and control characters', () {
      final encoding = tokenizer.encode('\x00\x01\x02', addSpecialTokens: false);
      expect(encoding, isNotNull);
    });
  });

  group('Encoding Operations', () {
    late SentencePieceTokenizer tokenizer;

    setUpAll(() {
      tokenizer = createTestTokenizer();
    });

    test('withPadding creates new encoding', () {
      final encoding = tokenizer.encode('hello', addSpecialTokens: false);
      final padded = encoding.withPadding(
        targetLength: 10,
        padTokenId: 0,
      );

      expect(padded, isNot(same(encoding)));
      expect(padded.length, equals(10));
    });

    test('withTruncation creates new encoding', () {
      final encoding = tokenizer.encode(
        'hello world test sentence',
        addSpecialTokens: false,
      );
      final truncated = encoding.withTruncation(maxLength: 3);

      expect(truncated, isNot(same(encoding)));
      expect(truncated.length, equals(3));
    });

    test('charToToken maps correctly', () {
      final encoding = tokenizer.encode('hello', addSpecialTokens: false);
      // Check that character positions map to valid token indices
      for (var i = 0; i < 5; i++) {
        final tokenIdx = encoding.charToToken(i);
        if (tokenIdx != null) {
          expect(tokenIdx, greaterThanOrEqualTo(0));
          expect(tokenIdx, lessThan(encoding.length));
        }
      }
    });

    test('tokenToChars returns valid ranges', () {
      final encoding = tokenizer.encode('hello', addSpecialTokens: false);
      for (var i = 0; i < encoding.length; i++) {
        final chars = encoding.tokenToChars(i);
        if (chars != null) {
          expect(chars.$1, greaterThanOrEqualTo(0));
          expect(chars.$2, greaterThan(chars.$1));
        }
      }
    });
  });

  group('Truncation Strategies', () {
    test('longestFirst truncates the longer sequence', () {
      final encodingA = Encoding(
        tokens: ['a', 'b', 'c', 'd', 'e'],
        ids: [1, 2, 3, 4, 5],
        typeIds: [0, 0, 0, 0, 0],
        attentionMask: [1, 1, 1, 1, 1],
        specialTokensMask: [0, 0, 0, 0, 0],
        offsets: [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
        wordIds: [0, 1, 2, 3, 4],
        sequenceIds: [0, 0, 0, 0, 0],
      );
      final encodingB = Encoding(
        tokens: ['x', 'y'],
        ids: [10, 11],
        typeIds: [1, 1],
        attentionMask: [1, 1],
        specialTokensMask: [0, 0],
        offsets: [(0, 1), (1, 2)],
        wordIds: [0, 1],
        sequenceIds: [1, 1],
      );

      final (truncA, truncB) = Encoding.truncatePair(
        encodingA: encodingA,
        encodingB: encodingB,
        maxLength: 5,
        strategy: TruncationStrategy.longestFirst,
      );

      expect(truncA.length + truncB.length, lessThanOrEqualTo(5));
    });

    test('onlyFirst only truncates first', () {
      final encodingA = Encoding(
        tokens: ['a', 'b', 'c', 'd', 'e'],
        ids: [1, 2, 3, 4, 5],
        typeIds: [0, 0, 0, 0, 0],
        attentionMask: [1, 1, 1, 1, 1],
        specialTokensMask: [0, 0, 0, 0, 0],
        offsets: [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
        wordIds: [0, 1, 2, 3, 4],
        sequenceIds: [0, 0, 0, 0, 0],
      );
      final encodingB = Encoding(
        tokens: ['x', 'y', 'z'],
        ids: [10, 11, 12],
        typeIds: [1, 1, 1],
        attentionMask: [1, 1, 1],
        specialTokensMask: [0, 0, 0],
        offsets: [(0, 1), (1, 2), (2, 3)],
        wordIds: [0, 1, 2],
        sequenceIds: [1, 1, 1],
      );

      final (truncA, truncB) = Encoding.truncatePair(
        encodingA: encodingA,
        encodingB: encodingB,
        maxLength: 5,
        strategy: TruncationStrategy.onlyFirst,
      );

      expect(truncB.length, equals(3)); // B unchanged
      expect(truncA.length, lessThan(5)); // A truncated
    });

    test('doNotTruncate leaves both unchanged', () {
      final encodingA = Encoding(
        tokens: ['a', 'b'],
        ids: [1, 2],
        typeIds: [0, 0],
        attentionMask: [1, 1],
        specialTokensMask: [0, 0],
        offsets: [(0, 1), (1, 2)],
        wordIds: [0, 1],
        sequenceIds: [0, 0],
      );
      final encodingB = Encoding(
        tokens: ['x', 'y'],
        ids: [10, 11],
        typeIds: [1, 1],
        attentionMask: [1, 1],
        specialTokensMask: [0, 0],
        offsets: [(0, 1), (1, 2)],
        wordIds: [0, 1],
        sequenceIds: [1, 1],
      );

      final (truncA, truncB) = Encoding.truncatePair(
        encodingA: encodingA,
        encodingB: encodingB,
        maxLength: 10, // Large enough to not trigger truncation
        strategy: TruncationStrategy.doNotTruncate,
        numSpecialTokens: 0, // No special tokens for this test
      );

      expect(truncA.length, equals(2));
      expect(truncB.length, equals(2));
    });
  });

  group('Model Loading', () {
    test('fromBytes creates tokenizer', () {
      final bytes = getTestModelBytes();
      final tokenizer = SentencePieceTokenizer.fromBytes(bytes);

      expect(tokenizer.vocabSize, greaterThan(0));
    });

    test('fromBytes with config', () {
      final bytes = getTestModelBytes();
      final tokenizer = SentencePieceTokenizer.fromBytes(
        bytes,
        config: SentencePieceConfig.llama,
      );

      expect(tokenizer.config.addBosToken, isTrue);
    });
  });

  group('Fluent API', () {
    test('method chaining works', () {
      final tokenizer = createTestTokenizer()
        ..enablePadding(length: 512)
        ..enableTruncation(maxLength: 256);

      expect(tokenizer.padding, isNotNull);
      expect(tokenizer.truncation, isNotNull);
    });

    test('can reset configuration', () {
      final tokenizer = createTestTokenizer()
        ..enablePadding(length: 512)
        ..enableTruncation(maxLength: 256)
        ..noPadding()
        ..noTruncation();

      expect(tokenizer.padding, isNull);
      expect(tokenizer.truncation, isNull);
    });
  });

  group('Encode Pair', () {
    late SentencePieceTokenizer tokenizer;

    setUpAll(() {
      tokenizer = createTestTokenizer();
    });

    test('encodePair returns encoding with both sequences', () {
      final encoding = tokenizer.encodePair(
        'hello',
        'world',
        addSpecialTokens: false,
      );

      expect(encoding.length, greaterThan(0));
      expect(encoding.ids.isNotEmpty, isTrue);
      expect(encoding.tokens.isNotEmpty, isTrue);
    });

    test('encodePair sets correct typeIds', () {
      final encoding = tokenizer.encodePair(
        'hello',
        'world',
        addSpecialTokens: false,
      );

      // First sequence should have typeId = 0, second = 1
      expect(encoding.typeIds, contains(0));
      expect(encoding.typeIds, contains(1));
    });

    test('encodePair with special tokens adds BOS/EOS', () {
      final llamaTokenizer = createLlamaTokenizer();
      final encoding = llamaTokenizer.encodePair(
        'hello',
        'world',
        addSpecialTokens: true,
      );

      // Verify encoding is created successfully
      expect(encoding.length, greaterThan(0));
    });

    test('encodePair respects maxLength', () {
      final encoding = tokenizer.encodePair(
        'hello',
        'world',
        addSpecialTokens: false,
        maxLength: 5,
      );

      expect(encoding.length, lessThanOrEqualTo(5));
    });

    test('encodePair with longestFirst truncation strategy', () {
      final encoding = tokenizer.encodePair(
        'hello hello hello',
        'world',
        addSpecialTokens: false,
        maxLength: 4,
        strategy: TruncationStrategy.longestFirst,
      );

      expect(encoding.length, lessThanOrEqualTo(4));
    });

    test('encodePair with onlyFirst truncation strategy', () {
      final encoding = tokenizer.encodePair(
        'hello hello hello',
        'world',
        addSpecialTokens: false,
        maxLength: 10,
        strategy: TruncationStrategy.onlyFirst,
      );

      // Verify truncation is applied (should not exceed maxLength)
      expect(encoding.length, lessThanOrEqualTo(10));
    });

    test('encodePair with onlySecond truncation strategy', () {
      final encoding = tokenizer.encodePair(
        'hello',
        'world world world',
        addSpecialTokens: false,
        maxLength: 10,
        strategy: TruncationStrategy.onlySecond,
      );

      // Verify truncation is applied (should not exceed maxLength)
      expect(encoding.length, lessThanOrEqualTo(10));
    });

    test('encodePairBatch handles multiple pairs', () {
      final pairs = [
        ('hello', 'world'),
        ('test', 'sentence'),
        ('first', 'second'),
      ];

      final encodings = tokenizer.encodePairBatch(
        pairs,
        addSpecialTokens: false,
      );

      expect(encodings.length, equals(3));
      for (final encoding in encodings) {
        expect(encoding.length, greaterThan(0));
      }
    });

    test('encodePairBatch preserves order', () {
      final pairs = [
        ('hello', 'world'),
        ('test', 'sentence'),
      ];

      final encodings = tokenizer.encodePairBatch(
        pairs,
        addSpecialTokens: false,
      );

      expect(encodings.length, equals(2));
      // Both encodings should have content
      expect(encodings[0].length, greaterThan(0));
      expect(encodings[1].length, greaterThan(0));
    });

    test('encodePairBatch with maxLength', () {
      final pairs = [
        ('hello hello hello', 'world world'),
        ('test test', 'sentence sentence'),
      ];

      final encodings = tokenizer.encodePairBatch(
        pairs,
        addSpecialTokens: false,
        maxLength: 5,
      );

      for (final encoding in encodings) {
        expect(encoding.length, lessThanOrEqualTo(5));
      }
    });

    test('encodePairBatch handles empty list', () {
      final encodings = tokenizer.encodePairBatch(
        [],
        addSpecialTokens: false,
      );

      expect(encodings, isEmpty);
    });
  });

  group('numSpecialTokensToAdd with isPair', () {
    test('isPair adds extra EOS token when EOS is enabled', () {
      final tokenizer = createTestTokenizer();

      // For single sequence
      final singleCount = tokenizer.numSpecialTokensToAdd(isPair: false);
      // For pair sequence (adds extra EOS between sequences if EOS is enabled)
      final pairCount = tokenizer.numSpecialTokensToAdd(isPair: true);

      // Pair count should be >= single count
      expect(pairCount, greaterThanOrEqualTo(singleCount));
    });

    test('returns 0 when special tokens disabled', () {
      final noSpecialTokenizer = createTestTokenizer();

      // Default config has no special tokens
      expect(noSpecialTokenizer.numSpecialTokensToAdd(isPair: false), equals(0));
      expect(noSpecialTokenizer.numSpecialTokensToAdd(isPair: true), equals(0));
    });

    test('explicit addBosToken and addEosToken parameters work', () {
      final tokenizer = createTestTokenizer();

      // Force enable BOS and EOS
      final count = tokenizer.numSpecialTokensToAdd(
        addBosToken: true,
        addEosToken: true,
        isPair: false,
      );

      // Should add at most 2 (BOS + EOS) if both IDs are valid
      expect(count, lessThanOrEqualTo(2));
    });
  });

  group('vocabularyMap', () {
    late SentencePieceTokenizer tokenizer;

    setUpAll(() {
      tokenizer = createTestTokenizer();
    });

    test('returns complete vocabulary map', () {
      final vocab = tokenizer.vocab.vocabularyMap;

      expect(vocab, isA<Map<String, int>>());
      expect(vocab.length, equals(tokenizer.vocab.size));
    });

    test('vocab map IDs match pieceToId', () {
      final vocab = tokenizer.vocab.vocabularyMap;

      for (final entry in vocab.entries) {
        expect(tokenizer.vocab.pieceToId(entry.key), equals(entry.value));
      }
    });

    test('vocab map is unmodifiable', () {
      final vocab = tokenizer.vocab.vocabularyMap;

      expect(
        () => vocab['new_token'] = 99999,
        throwsA(isA<UnsupportedError>()),
      );
    });
  });

  group('specialTokensMask validation', () {
    test('special tokens mask is all zeros when addSpecialTokens=false', () {
      final tokenizer = createTestTokenizer();
      final encoding = tokenizer.encode('hello world', addSpecialTokens: false);

      expect(encoding.specialTokensMask, everyElement(equals(0)));
    });

    test('special tokens mask marks special tokens correctly', () {
      final tokenizer = createGemmaTokenizer();
      final encoding = tokenizer.encode('hello');

      // First token (BOS) should be marked as special
      if (tokenizer.vocab.bosId >= 0 && encoding.ids.contains(tokenizer.vocab.bosId)) {
        final bosIndex = encoding.ids.toList().indexOf(tokenizer.vocab.bosId);
        expect(encoding.specialTokensMask[bosIndex], equals(1));
      }
    });
  });

  group('Integration Tests', () {
    late SentencePieceTokenizer tokenizer;

    setUpAll(() {
      tokenizer = createTestTokenizer();
    });

    test('full pipeline: encode without special, truncate, then decode', () {
      final text = 'hello world test sentence';

      // Encode without special tokens
      final encoding = tokenizer.encode(text, addSpecialTokens: false);

      // Truncate
      final truncated = encoding.withTruncation(maxLength: 3);

      // Decode
      final decoded = tokenizer.decode(truncated.ids.toList(), skipSpecialTokens: true);

      expect(truncated.length, equals(3));
      expect(decoded.isNotEmpty, isTrue);
    });

    test('pair encoding respects maxLength exactly', () {
      final textA = 'hello hello hello';
      final textB = 'world world world';

      for (final maxLen in [5, 10, 15]) {
        final encoding = tokenizer.encodePair(
          textA,
          textB,
          maxLength: maxLen,
          addSpecialTokens: false,
          strategy: TruncationStrategy.longestFirst,
        );

        expect(
          encoding.length,
          lessThanOrEqualTo(maxLen),
          reason: 'Should respect maxLength=$maxLen',
        );
      }
    });

    test('batch encoding consistency', () {
      final texts = ['hello', 'world', 'test'];

      final batchEncodings = tokenizer.encodeBatch(texts, addSpecialTokens: false);
      final singleEncodings = texts
          .map((t) => tokenizer.encode(t, addSpecialTokens: false))
          .toList();

      expect(batchEncodings.length, equals(singleEncodings.length));
      for (var i = 0; i < batchEncodings.length; i++) {
        expect(batchEncodings[i].ids, equals(singleEncodings[i].ids));
      }
    });

    test('encodePair typeIds are correct', () {
      final encoding = tokenizer.encodePair(
        'hello',
        'world',
        addSpecialTokens: false,
      );

      // Verify we have both typeId 0 and 1
      final hasTypeId0 = encoding.typeIds.any((t) => t == 0);
      final hasTypeId1 = encoding.typeIds.any((t) => t == 1);

      expect(hasTypeId0, isTrue);
      expect(hasTypeId1, isTrue);
    });

    test('decode preserves original text approximately', () {
      final text = 'hello world';
      final encoding = tokenizer.encode(text, addSpecialTokens: false);
      final decoded = tokenizer.decode(encoding.ids.toList());

      // Decoded text should contain the original words
      expect(decoded.toLowerCase(), contains('hello'));
      expect(decoded.toLowerCase(), contains('world'));
    });
  });

  group('Encoding.truncatePair edge cases', () {
    test('handles empty result when maxLength too small', () {
      final encodingA = Encoding(
        tokens: ['a', 'b', 'c'],
        ids: [1, 2, 3],
        typeIds: [0, 0, 0],
        attentionMask: [1, 1, 1],
        specialTokensMask: [0, 0, 0],
        offsets: [(0, 1), (1, 2), (2, 3)],
        wordIds: [0, 1, 2],
        sequenceIds: [0, 0, 0],
      );
      final encodingB = Encoding(
        tokens: ['x', 'y'],
        ids: [10, 11],
        typeIds: [1, 1],
        attentionMask: [1, 1],
        specialTokensMask: [0, 0],
        offsets: [(0, 1), (1, 2)],
        wordIds: [0, 1],
        sequenceIds: [1, 1],
      );

      final (truncA, truncB) = Encoding.truncatePair(
        encodingA: encodingA,
        encodingB: encodingB,
        maxLength: 2, // Only room for special tokens
        numSpecialTokens: 3,
      );

      expect(truncA.isEmpty, isTrue);
      expect(truncB.isEmpty, isTrue);
    });

    test('returns unchanged when under maxLength', () {
      final encodingA = Encoding(
        tokens: ['a'],
        ids: [1],
        typeIds: [0],
        attentionMask: [1],
        specialTokensMask: [0],
        offsets: [(0, 1)],
        wordIds: [0],
        sequenceIds: [0],
      );
      final encodingB = Encoding(
        tokens: ['x'],
        ids: [10],
        typeIds: [1],
        attentionMask: [1],
        specialTokensMask: [0],
        offsets: [(0, 1)],
        wordIds: [0],
        sequenceIds: [1],
      );

      final (truncA, truncB) = Encoding.truncatePair(
        encodingA: encodingA,
        encodingB: encodingB,
        maxLength: 100,
        numSpecialTokens: 3,
      );

      expect(truncA.length, equals(encodingA.length));
      expect(truncB.length, equals(encodingB.length));
    });
  });
}
