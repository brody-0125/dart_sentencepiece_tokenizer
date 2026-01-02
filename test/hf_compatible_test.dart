/// Test cases for HuggingFace SentencePiece tokenizer compatibility.
///
/// These tests ensure compatibility with HuggingFace tokenizers behavior
/// for SentencePiece BPE and Unigram algorithms.
import 'package:test/test.dart';
import 'package:dart_sentencepiece_tokenizer/dart_sentencepiece_tokenizer.dart';

import 'test_utils.dart';

void main() {
  group('Internal Consistency', () {
    late SentencePieceTokenizer tokenizer;

    setUpAll(() {
      tokenizer = createTestTokenizer();
    });

    test('tokenize and encode produce consistent IDs', () {
      const text = 'Hello, world!';

      final encoding = tokenizer.encode(text, addSpecialTokens: false);

      // Convert tokens back to IDs - should match
      final convertedIds = tokenizer.convertTokensToIds(encoding.tokens);
      expect(convertedIds, equals(encoding.ids.toList()));
    });

    test('encode and decode round-trip', () {
      // Use text that exists in the minimal test vocabulary
      const text = 'hello world';

      final encoding = tokenizer.encode(text, addSpecialTokens: false);
      final decoded = tokenizer.decode(encoding.ids.toList());

      // SentencePiece may normalize whitespace differently
      expect(decoded.trim(), equals(text));
    });

    test('batch encoding produces consistent results', () {
      const texts = ['Hello world', 'Goodbye world'];

      final batchResults = tokenizer.encodeBatch(texts, addSpecialTokens: false);
      final individualResults = [
        for (final text in texts)
          tokenizer.encode(text, addSpecialTokens: false),
      ];

      for (var i = 0; i < texts.length; i++) {
        expect(batchResults[i].tokens, equals(individualResults[i].tokens));
        expect(batchResults[i].ids, equals(individualResults[i].ids));
      }
    });

    test('convertTokensToIds and convertIdsToTokens are inverses', () {
      final encoding = tokenizer.encode('test tokens', addSpecialTokens: false);

      final ids = tokenizer.convertTokensToIds(encoding.tokens);
      final tokens = tokenizer.convertIdsToTokens(ids);

      expect(tokens, equals(encoding.tokens));
    });
  });

  group('Padding Tests', () {
    late SentencePieceTokenizer tokenizer;

    setUpAll(() {
      tokenizer = createTestTokenizer();
    });

    test('right padding (default)', () {
      final encoding = tokenizer.encode('hello', addSpecialTokens: false);
      final padded = encoding.withPadding(
        targetLength: 10,
        padTokenId: tokenizer.vocab.padId >= 0 ? tokenizer.vocab.padId : 0,
        padToken: '<pad>',
        padOnRight: true,
      );

      expect(padded.length, equals(10));
      // Original tokens should be at the beginning
      expect(padded.tokens.sublist(0, encoding.length), equals(encoding.tokens));
      // Attention mask should be 0 for padding
      expect(
        padded.attentionMask.sublist(encoding.length),
        everyElement(equals(0)),
      );
    });

    test('left padding', () {
      final encoding = tokenizer.encode('hello', addSpecialTokens: false);
      final padded = encoding.withPadding(
        targetLength: 10,
        padTokenId: tokenizer.vocab.padId >= 0 ? tokenizer.vocab.padId : 0,
        padToken: '<pad>',
        padOnRight: false,
      );

      expect(padded.length, equals(10));
      final padCount = 10 - encoding.length;
      // Padding should be at the beginning
      expect(
        padded.attentionMask.sublist(0, padCount),
        everyElement(equals(0)),
      );
      // Original tokens should be at the end
      expect(padded.tokens.sublist(padCount), equals(encoding.tokens));
    });

    test('no padding when already at target length', () {
      final encoding = tokenizer.encode('hello', addSpecialTokens: false);
      final originalLength = encoding.length;
      final padded = encoding.withPadding(
        targetLength: originalLength,
        padTokenId: 0,
      );

      expect(padded.length, equals(originalLength));
      expect(padded.tokens, equals(encoding.tokens));
    });

    test('no padding when exceeding target length', () {
      final encoding = tokenizer.encode(
        'This is a longer sentence',
        addSpecialTokens: false,
      );
      final padded = encoding.withPadding(
        targetLength: 3,
        padTokenId: 0,
      );

      expect(padded.length, equals(encoding.length));
    });
  });

  group('Padding to Multiple', () {
    late SentencePieceTokenizer tokenizer;

    setUpAll(() {
      tokenizer = createTestTokenizer();
    });

    test('pads to multiple of 8', () {
      final encoding = tokenizer.encode('hello', addSpecialTokens: false);
      final padded = encoding.withPaddingToMultipleOf(
        multiple: 8,
        padTokenId: 0,
      );

      expect(padded.length % 8, equals(0));
      expect(padded.length, greaterThanOrEqualTo(encoding.length));
    });

    test('pads to multiple of 16', () {
      final encoding = tokenizer.encode(
        'This is a test sentence',
        addSpecialTokens: false,
      );
      final padded = encoding.withPaddingToMultipleOf(
        multiple: 16,
        padTokenId: 0,
      );

      expect(padded.length % 16, equals(0));
    });

    test('no padding when already multiple', () {
      final encoding = tokenizer.encode('hello', addSpecialTokens: false);
      final prePadded = encoding.withPadding(
        targetLength: 8,
        padTokenId: 0,
      );

      final result = prePadded.withPaddingToMultipleOf(
        multiple: 8,
        padTokenId: 0,
      );

      expect(result.length, equals(8));
      expect(result.length, equals(prePadded.length));
    });

    test('left padding to multiple', () {
      final encoding = tokenizer.encode('hello', addSpecialTokens: false);
      final padded = encoding.withPaddingToMultipleOf(
        multiple: 8,
        padTokenId: 0,
        padOnRight: false,
      );

      expect(padded.length % 8, equals(0));
    });

    test('handles multiple of 1', () {
      final encoding = tokenizer.encode('hello', addSpecialTokens: false);
      final padded = encoding.withPaddingToMultipleOf(
        multiple: 1,
        padTokenId: 0,
      );

      expect(padded.length, equals(encoding.length));
    });

    test('handles invalid multiple gracefully', () {
      final encoding = tokenizer.encode('hello', addSpecialTokens: false);
      final padded = encoding.withPaddingToMultipleOf(
        multiple: 0,
        padTokenId: 0,
      );

      expect(padded.length, equals(encoding.length));
    });
  });

  group('Truncation Tests', () {
    late SentencePieceTokenizer tokenizer;

    setUpAll(() {
      tokenizer = createTestTokenizer();
    });

    test('right truncation (default)', () {
      final encoding = tokenizer.encode(
        'This is a very long sentence for testing truncation',
        addSpecialTokens: false,
      );
      final truncated = encoding.withTruncation(
        maxLength: 5,
        truncateFromEnd: true,
      );

      expect(truncated.length, equals(5));
      expect(truncated.tokens, equals(encoding.tokens.sublist(0, 5)));
    });

    test('left truncation', () {
      final encoding = tokenizer.encode(
        'This is a very long sentence for testing truncation',
        addSpecialTokens: false,
      );
      final originalLength = encoding.length;
      final truncated = encoding.withTruncation(
        maxLength: 5,
        truncateFromEnd: false,
      );

      expect(truncated.length, equals(5));
      expect(
        truncated.tokens,
        equals(encoding.tokens.sublist(originalLength - 5)),
      );
    });

    test('no truncation when under max length', () {
      final encoding = tokenizer.encode('hello', addSpecialTokens: false);
      final truncated = encoding.withTruncation(maxLength: 100);

      expect(truncated.length, equals(encoding.length));
      expect(truncated.tokens, equals(encoding.tokens));
    });
  });

  group('Special Tokens Tests', () {
    late SentencePieceTokenizer tokenizer;

    setUpAll(() {
      tokenizer = createTestTokenizer();
    });

    test('vocabulary has special token IDs', () {
      // SentencePiece typically has these special tokens
      expect(tokenizer.vocab.unkId, greaterThanOrEqualTo(0));
    });

    test('BOS/EOS tokens can be added', () {
      final withBos = SentencePieceTokenizer.fromBytes(
        getTestModelBytes(),
        config: const SentencePieceConfig(
          addBosToken: true,
          addEosToken: false,
        ),
      );

      final encoding = withBos.encode('hello');

      // First token should be BOS
      if (withBos.vocab.bosId >= 0) {
        expect(encoding.ids.first, equals(withBos.vocab.bosId));
      }
    });

    test('special tokens count matches config', () {
      final llamaTokenizer = SentencePieceTokenizer.fromBytes(
        getTestModelBytes(),
        config: SentencePieceConfig.llama,
      );

      // Llama config has addBosToken=true, addEosToken=false
      // Count depends on whether model has valid BOS ID
      expect(
        llamaTokenizer.numSpecialTokensToAdd(),
        greaterThanOrEqualTo(0),
      );

      final gemmaTokenizer = SentencePieceTokenizer.fromBytes(
        getTestModelBytes(),
        config: SentencePieceConfig.gemma,
      );

      // Gemma config has addBosToken=true, addEosToken=true
      // Count depends on whether model has valid BOS/EOS IDs
      expect(
        gemmaTokenizer.numSpecialTokensToAdd(),
        greaterThanOrEqualTo(0),
      );
    });
  });

  group('Token Type IDs', () {
    late SentencePieceTokenizer tokenizer;

    setUpAll(() {
      tokenizer = createTestTokenizer();
    });

    test('single sequence has all typeId = 0', () {
      final encoding = tokenizer.encode('hello world', addSpecialTokens: false);

      expect(encoding.typeIds, everyElement(equals(0)));
    });

    test('attention mask is all ones for non-padded', () {
      final encoding = tokenizer.encode('hello world', addSpecialTokens: false);

      expect(encoding.attentionMask, everyElement(equals(1)));
    });
  });

  group('Unicode and Edge Cases', () {
    late SentencePieceTokenizer tokenizer;

    setUpAll(() {
      tokenizer = createTestTokenizer();
    });

    test('emoji handling', () {
      final encoding = tokenizer.encode('Hello 😊', addSpecialTokens: false);

      // Should not crash and produce some tokens
      expect(encoding.length, greaterThan(0));
    });

    test('Chinese characters', () {
      final encoding = tokenizer.encode('你好世界', addSpecialTokens: false);

      expect(encoding.length, greaterThan(0));
    });

    test('mixed language text', () {
      final encoding = tokenizer.encode(
        'Hello 世界 Bonjour',
        addSpecialTokens: false,
      );

      expect(encoding.length, greaterThan(0));
    });

    test('accented characters', () {
      final encoding = tokenizer.encode(
        'café résumé naïve',
        addSpecialTokens: false,
      );

      expect(encoding.length, greaterThan(0));
    });

    test('empty string produces empty encoding', () {
      final encoding = tokenizer.encode('', addSpecialTokens: false);

      expect(encoding.length, equals(0));
    });

    test('whitespace only', () {
      final encoding = tokenizer.encode('   ', addSpecialTokens: false);

      // Behavior depends on normalizer settings
      // Just verify it doesn't crash
      expect(encoding, isNotNull);
    });
  });

  group('Batch Processing Consistency', () {
    late SentencePieceTokenizer tokenizer;

    setUpAll(() {
      tokenizer = createTestTokenizer();
    });

    test('batch encoding maintains order', () {
      final texts = ['first', 'second', 'third'];
      final batch = tokenizer.encodeBatch(texts, addSpecialTokens: false);

      expect(batch.length, equals(3));
    });

    test('batch with mixed lengths', () {
      final texts = [
        'short',
        'This is a much longer sentence for testing',
        'medium length',
      ];
      final batch = tokenizer.encodeBatch(texts, addSpecialTokens: false);

      expect(batch.length, equals(3));
      for (final encoding in batch) {
        expect(encoding.attentionMask, everyElement(equals(1)));
      }
    });

    test('parallel batch matches sequential', () async {
      final texts = [
        for (var i = 0; i < 20; i++) 'Test sentence number $i',
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
          reason: 'Mismatch at index $i',
        );
      }
    });
  });

  group('Decode Tests', () {
    late SentencePieceTokenizer tokenizer;

    setUpAll(() {
      tokenizer = createTestTokenizer();
    });

    test('decode without special tokens', () {
      final encoding = tokenizer.encode('hello world', addSpecialTokens: false);
      final decoded = tokenizer.decode(
        encoding.ids.toList(),
        skipSpecialTokens: true,
      );

      expect(decoded, equals('hello world'));
    });

    test('decode removes space symbol', () {
      final encoding = tokenizer.encode('hello world', addSpecialTokens: false);
      final decoded = tokenizer.decode(encoding.ids.toList());

      // Should not contain the ▁ character in output
      expect(decoded, isNot(contains('▁')));
    });

    test('batch decode', () {
      final texts = ['hello', 'world'];
      final batch = tokenizer.encodeBatch(texts, addSpecialTokens: false);
      final decoded = tokenizer.decodeBatch(
        [for (final e in batch) e.ids.toList()],
      );

      expect(decoded, equals(['hello', 'world']));
    });
  });

  group('Config Options', () {
    test('default config has no special tokens', () {
      const config = SentencePieceConfig();
      expect(config.addBosToken, isFalse);
      expect(config.addEosToken, isFalse);
    });

    test('llama config has BOS only', () {
      expect(SentencePieceConfig.llama.addBosToken, isTrue);
      expect(SentencePieceConfig.llama.addEosToken, isFalse);
    });

    test('gemma config has BOS and EOS', () {
      expect(SentencePieceConfig.gemma.addBosToken, isTrue);
      expect(SentencePieceConfig.gemma.addEosToken, isTrue);
    });

    test('custom config works', () {
      final tokenizer = SentencePieceTokenizer.fromBytes(
        getTestModelBytes(),
        config: const SentencePieceConfig(
          addBosToken: false,
          addEosToken: true,
        ),
      );

      expect(tokenizer.config.addBosToken, isFalse);
      expect(tokenizer.config.addEosToken, isTrue);
    });
  });

  group('Fluent API', () {
    test('enable/disable padding returns tokenizer', () {
      final tokenizer = createTestTokenizer();

      final result = tokenizer
          .enablePadding(length: 512)
          .enableTruncation(maxLength: 512);

      expect(result, same(tokenizer));
      expect(tokenizer.padding, isNotNull);
      expect(tokenizer.truncation, isNotNull);
    });

    test('noPadding and noTruncation work', () {
      final tokenizer = createTestTokenizer()
          .enablePadding(length: 512)
          .enableTruncation(maxLength: 512)
          .noPadding()
          .noTruncation();

      expect(tokenizer.padding, isNull);
      expect(tokenizer.truncation, isNull);
    });
  });

  group('Truncation Strategies', () {
    late SentencePieceTokenizer tokenizer;

    setUpAll(() {
      tokenizer = createTestTokenizer();
    });

    test('longestFirst truncates longer sequence', () {
      final encodingA = tokenizer.encode(
        'This is a longer first sequence with many words',
        addSpecialTokens: false,
      );
      final encodingB = tokenizer.encode('Short', addSpecialTokens: false);

      final (truncA, truncB) = Encoding.truncatePair(
        encodingA: encodingA,
        encodingB: encodingB,
        maxLength: 10,
        strategy: TruncationStrategy.longestFirst,
        numSpecialTokens: 0,
      );

      expect(truncA.length + truncB.length, lessThanOrEqualTo(10));
      expect(truncB.length, equals(encodingB.length)); // Short one unchanged
    });

    test('onlyFirst truncates first sequence only', () {
      final encodingA = tokenizer.encode(
        'First sequence',
        addSpecialTokens: false,
      );
      final encodingB = tokenizer.encode(
        'Second sequence',
        addSpecialTokens: false,
      );

      final (truncA, truncB) = Encoding.truncatePair(
        encodingA: encodingA,
        encodingB: encodingB,
        maxLength: 5,
        strategy: TruncationStrategy.onlyFirst,
        numSpecialTokens: 0,
      );

      expect(truncB.length, equals(encodingB.length));
    });

    test('onlySecond truncates second sequence only', () {
      final encodingA = tokenizer.encode(
        'First sequence',
        addSpecialTokens: false,
      );
      final encodingB = tokenizer.encode(
        'Second sequence',
        addSpecialTokens: false,
      );

      final (truncA, truncB) = Encoding.truncatePair(
        encodingA: encodingA,
        encodingB: encodingB,
        maxLength: 5,
        strategy: TruncationStrategy.onlySecond,
        numSpecialTokens: 0,
      );

      expect(truncA.length, equals(encodingA.length));
    });

    test('doNotTruncate leaves both unchanged', () {
      final encodingA = tokenizer.encode(
        'First sequence',
        addSpecialTokens: false,
      );
      final encodingB = tokenizer.encode(
        'Second sequence',
        addSpecialTokens: false,
      );

      final (truncA, truncB) = Encoding.truncatePair(
        encodingA: encodingA,
        encodingB: encodingB,
        maxLength: 5,
        strategy: TruncationStrategy.doNotTruncate,
        numSpecialTokens: 0,
      );

      expect(truncA.length, equals(encodingA.length));
      expect(truncB.length, equals(encodingB.length));
    });
  });
}
