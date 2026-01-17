/// BPE algorithm optimization tests.
library;

import 'package:test/test.dart';
import 'package:dart_sentencepiece_tokenizer/src/sentencepiece/algorithm/bpe_algorithm.dart';
import 'package:dart_sentencepiece_tokenizer/src/sentencepiece/algorithm/bpe_algorithm_optimized.dart';
import 'package:dart_sentencepiece_tokenizer/src/sentencepiece/vocabulary/sp_vocabulary.dart';

import 'test_utils.dart';

void main() {
  group('BPE Optimized Algorithm', () {
    late SpVocabulary vocab;
    late BpeAlgorithm originalBpe;
    late BpeAlgorithmOptimized optimizedBpe;

    setUpAll(() {
      final tokenizer = createTestTokenizer();
      vocab = tokenizer.vocab;
      originalBpe = BpeAlgorithm(vocab: vocab, byteFallback: false);
      optimizedBpe = BpeAlgorithmOptimized(vocab: vocab, byteFallback: false);
    });

    test('empty string produces empty result', () {
      final original = originalBpe.tokenize('');
      final optimized = optimizedBpe.tokenize('');

      expect(optimized, equals(original));
      expect(optimized, isEmpty);
    });

    test('single character tokenization matches', () {
      const text = 'a';
      final original = originalBpe.tokenize(text);
      final optimized = optimizedBpe.tokenize(text);

      expect(optimized, equals(original));
    });

    test('simple word tokenization matches', () {
      const text = 'hello';
      final original = originalBpe.tokenize(text);
      final optimized = optimizedBpe.tokenize(text);

      expect(optimized, equals(original));
    });

    test('multiple words tokenization matches', () {
      const text = '▁hello▁world';
      final original = originalBpe.tokenize(text);
      final optimized = optimizedBpe.tokenize(text);

      expect(optimized, equals(original));
    });

    test('longer text tokenization matches', () {
      const text = '▁This▁is▁a▁test▁sentence';
      final original = originalBpe.tokenize(text);
      final optimized = optimizedBpe.tokenize(text);

      expect(optimized, equals(original));
    });

    test('repeated text tokenization matches', () {
      const text = '▁test▁test▁test▁test▁test';
      final original = originalBpe.tokenize(text);
      final optimized = optimizedBpe.tokenize(text);

      expect(optimized, equals(original));
    });

    test('mixed characters tokenization matches', () {
      const text = '▁hello123world';
      final original = originalBpe.tokenize(text);
      final optimized = optimizedBpe.tokenize(text);

      expect(optimized, equals(original));
    });

    test('tokenization is deterministic', () {
      const text = '▁hello▁world▁test';

      final result1 = optimizedBpe.tokenize(text);
      final result2 = optimizedBpe.tokenize(text);
      final result3 = optimizedBpe.tokenize(text);

      expect(result1, equals(result2));
      expect(result2, equals(result3));
    });

    test('clearCache does not affect results', () {
      const text = '▁hello▁world';

      final before = optimizedBpe.tokenize(text);
      optimizedBpe.clearCache();
      final after = optimizedBpe.tokenize(text);

      expect(after, equals(before));
    });
  });

  group('BPE Performance Comparison', () {
    late SpVocabulary vocab;
    late BpeAlgorithm originalBpe;
    late BpeAlgorithmOptimized optimizedBpe;

    setUpAll(() {
      final tokenizer = createTestTokenizer();
      vocab = tokenizer.vocab;
      originalBpe = BpeAlgorithm(vocab: vocab, byteFallback: false);
      optimizedBpe = BpeAlgorithmOptimized(vocab: vocab, byteFallback: false);
    });

    test('optimized handles medium text efficiently', () {
      // Generate medium-length text
      final text = '▁${List.generate(100, (i) => 'hello').join('▁')}';

      // Warmup
      originalBpe.tokenize(text);
      optimizedBpe.tokenize(text);

      // Time original
      final swOriginal = Stopwatch()..start();
      for (var i = 0; i < 10; i++) {
        originalBpe.tokenize(text);
      }
      swOriginal.stop();

      // Time optimized
      final swOptimized = Stopwatch()..start();
      for (var i = 0; i < 10; i++) {
        optimizedBpe.tokenize(text);
      }
      swOptimized.stop();

      print('Medium text (100 words):');
      print('  Original:  ${swOriginal.elapsedMicroseconds / 10}μs/call');
      print('  Optimized: ${swOptimized.elapsedMicroseconds / 10}μs/call');

      // Results should match
      expect(
        optimizedBpe.tokenize(text),
        equals(originalBpe.tokenize(text)),
      );
    });

    test('optimized handles repeated patterns', () {
      // Repeated pattern text
      final text = '▁${List.generate(50, (i) => 'test').join('▁')}';

      final original = originalBpe.tokenize(text);
      final optimized = optimizedBpe.tokenize(text);

      expect(optimized, equals(original));
    });
  });
}
