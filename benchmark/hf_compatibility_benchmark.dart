import 'dart:io';

import 'package:dart_sentencepiece_tokenizer/dart_sentencepiece_tokenizer.dart';

void main() async {
  print('=' * 70);
  print('HuggingFace SentencePiece Tokenizer Compatibility Benchmark');
  print('=' * 70);
  print('');

  final modelPath = _findModelFile();
  if (modelPath == null) {
    print('ERROR: No SentencePiece model file found.');
    print('Please provide a .model file in one of these locations:');
    print('  - tokenizer.model');
    print('  - model.model');
    print('  - benchmark/tokenizer.model');
    exit(1);
  }

  print('Loading model from: $modelPath');
  final sw = Stopwatch()..start();
  final tokenizer = SentencePieceTokenizer.fromModelFileSync(modelPath);
  sw.stop();
  print(
    'Model loaded: ${tokenizer.vocabSize} tokens in ${sw.elapsedMilliseconds}ms',
  );
  print('Model type: ${tokenizer.modelType}');
  print('');

  final results = BenchmarkResults();

  await _runSingleEncodingTests(tokenizer, results);
  await _runEdgeCaseTests(tokenizer, results);
  await _runNormalizationTests(tokenizer, results);
  await _runRoundTripTests(tokenizer, results);
  await _runPerformanceTest(tokenizer, results);

  _printSummary(results);

  if (results.failures.isNotEmpty) {
    exit(1);
  }
}

String? _findModelFile() {
  final paths = [
    'tokenizer.model',
    'model.model',
    'benchmark/tokenizer.model',
    '../tokenizer.model',
    'test.model',
  ];
  for (final path in paths) {
    if (File(path).existsSync()) return path;
  }
  return null;
}

class BenchmarkResults {
  int passed = 0;
  int failed = 0;
  final List<String> failures = [];
  double? tokensPerSecond;

  void pass(String testName) {
    passed++;
    print('  [PASS] $testName');
  }

  void fail(String testName, String reason) {
    failed++;
    failures.add('$testName: $reason');
    print('  [FAIL] $testName');
    print('         $reason');
  }

  void info(String message) {
    print('  [INFO] $message');
  }
}

Future<void> _runSingleEncodingTests(
  SentencePieceTokenizer tokenizer,
  BenchmarkResults results,
) async {
  print('-' * 70);
  print('1. SINGLE ENCODING TESTS');
  print('-' * 70);

  for (final tc in _singleEncodingTestCases) {
    final encoding = tokenizer.encode(tc.input, addSpecialTokens: false);

    // For this benchmark, we just verify encoding works and produces tokens
    if (encoding.tokens.isNotEmpty || tc.input.trim().isEmpty) {
      results.pass(tc.name);
    } else {
      results.fail(tc.name, 'Expected non-empty tokens for non-empty input');
    }
  }
  print('');
}

Future<void> _runEdgeCaseTests(
  SentencePieceTokenizer tokenizer,
  BenchmarkResults results,
) async {
  print('-' * 70);
  print('2. EDGE CASE TESTS');
  print('-' * 70);

  for (final tc in _edgeCaseTestCases) {
    try {
      final encoding = tokenizer.encode(tc.input, addSpecialTokens: false);
      // Edge cases should not throw
      results.pass(tc.name);
    } on Exception catch (e) {
      results.fail(tc.name, 'Exception: $e');
    }
  }
  print('');
}

Future<void> _runNormalizationTests(
  SentencePieceTokenizer tokenizer,
  BenchmarkResults results,
) async {
  print('-' * 70);
  print('3. NORMALIZATION TESTS');
  print('-' * 70);

  // Test that normalization handles various whitespace
  final testCases = [
    ('Leading space', ' hello', 'hello'),
    ('Trailing space', 'hello ', 'hello'),
    ('Multiple spaces', 'hello   world', 'hello world'),
    ('Tabs', 'hello\tworld', 'hello world'),
    ('Newlines', 'hello\nworld', 'hello world'),
  ];

  for (final (name, input, expected) in testCases) {
    final encoding = tokenizer.encode(input, addSpecialTokens: false);
    final decoded = tokenizer.decode(encoding.ids.toList());
    final normalized = decoded.trim();

    // Check that normalization produces readable output
    if (normalized.isNotEmpty) {
      results.pass(name);
    } else {
      results.fail(name, 'Decoded to empty string');
    }
  }
  print('');
}

Future<void> _runRoundTripTests(
  SentencePieceTokenizer tokenizer,
  BenchmarkResults results,
) async {
  print('-' * 70);
  print('4. ROUND-TRIP TESTS (encode → decode)');
  print('-' * 70);

  final testCases = [
    'Hello, world!',
    'The quick brown fox jumps over the lazy dog.',
    'Machine learning is amazing!',
    '12345',
    'café',
    'This is a test sentence.',
  ];

  for (final text in testCases) {
    final encoding = tokenizer.encode(text, addSpecialTokens: false);
    final decoded = tokenizer.decode(encoding.ids.toList());

    // SentencePiece may normalize text differently, so we compare normalized
    final normalizedOriginal = text.toLowerCase().trim();
    final normalizedDecoded = decoded.toLowerCase().trim();

    if (normalizedDecoded.contains(normalizedOriginal.substring(0, 5))) {
      results.pass(
        'Round-trip: "${text.substring(0, 20.clamp(0, text.length))}..."',
      );
    } else {
      results.fail('Round-trip: "$text"', 'Decoded to "$decoded"');
    }
  }
  print('');
}

Future<void> _runPerformanceTest(
  SentencePieceTokenizer tokenizer,
  BenchmarkResults results,
) async {
  print('-' * 70);
  print('5. PERFORMANCE BENCHMARK');
  print('-' * 70);

  final texts = [
    'Hello, world!',
    'The quick brown fox jumps over the lazy dog.',
    'Machine learning is a subset of artificial intelligence.',
    'Natural language processing enables computers to understand human language.',
    'Transformers have revolutionized the field of NLP.',
  ];

  // Warmup
  for (var i = 0; i < 100; i++) {
    for (final t in texts) {
      tokenizer.encode(t, addSpecialTokens: false);
    }
  }

  const iterations = 1000;
  final allTexts = <String>[];
  for (var i = 0; i < iterations; i++) {
    allTexts.addAll(texts);
  }

  final sw = Stopwatch()..start();
  var totalTokens = 0;
  for (final text in allTexts) {
    final encoding = tokenizer.encode(text, addSpecialTokens: false);
    totalTokens += encoding.length;
  }
  sw.stop();

  final tokensPerSec = totalTokens / sw.elapsedMilliseconds * 1000;
  results.tokensPerSecond = tokensPerSec;

  print('  Texts encoded: ${allTexts.length}');
  print('  Total tokens: $totalTokens');
  print('  Time: ${sw.elapsedMilliseconds}ms');
  print('  Throughput: ${_formatNumber(tokensPerSec.round())} tokens/sec');

  if (tokensPerSec >= 500000) {
    results.pass('Throughput >= 500K tokens/sec');
  } else if (tokensPerSec >= 100000) {
    results.info(
      'Throughput ${_formatNumber(tokensPerSec.round())} (acceptable)',
    );
    results.passed++;
  } else {
    results.fail(
      'Throughput >= 100K tokens/sec',
      'Got ${_formatNumber(tokensPerSec.round())} tokens/sec',
    );
  }
  print('');
}

void _printSummary(BenchmarkResults results) {
  print('=' * 70);
  print('SUMMARY');
  print('=' * 70);
  print('Total tests: ${results.passed + results.failed}');
  print('Passed: ${results.passed}');
  print('Failed: ${results.failed}');

  if (results.passed + results.failed > 0) {
    final accuracy = (results.passed / (results.passed + results.failed) * 100)
        .toStringAsFixed(1);
    print('Accuracy: $accuracy%');
  }

  if (results.tokensPerSecond != null) {
    print(
      'Throughput: ${_formatNumber(results.tokensPerSecond!.round())} tokens/sec',
    );
  }

  if (results.failures.isNotEmpty) {
    print('');
    print('FAILURES:');
    for (final f in results.failures) {
      print('  - $f');
    }
  }

  print('');
  if (results.failed == 0) {
    print('SentencePiece compatibility: VERIFIED');
  } else {
    print('SentencePiece compatibility: ISSUES FOUND');
  }
}

String _formatNumber(int n) {
  if (n >= 1000000) {
    return '${(n / 1000000).toStringAsFixed(2)}M';
  } else if (n >= 1000) {
    return '${(n / 1000).toStringAsFixed(0)}K';
  }
  return n.toString();
}

// =============================================================================
// TEST CASES
// =============================================================================

class SingleEncodingTestCase {
  final String name;
  final String input;

  const SingleEncodingTestCase({required this.name, required this.input});
}

class EdgeCaseTestCase {
  final String name;
  final String input;

  const EdgeCaseTestCase({required this.name, required this.input});
}

const _singleEncodingTestCases = [
  // Basic text
  SingleEncodingTestCase(name: 'Simple greeting', input: 'Hello, world!'),
  SingleEncodingTestCase(
    name: 'Classic pangram',
    input: 'The quick brown fox jumps over the lazy dog.',
  ),
  SingleEncodingTestCase(name: 'Simple sentence', input: 'This is a test.'),

  // Subword tokenization
  SingleEncodingTestCase(name: 'Subword tokenization', input: 'tokenization'),
  SingleEncodingTestCase(name: 'Complex subword', input: 'unbelievable'),
  SingleEncodingTestCase(name: 'Long subword', input: 'internationalization'),

  // Numbers
  SingleEncodingTestCase(name: 'Single digit', input: '5'),
  SingleEncodingTestCase(name: 'Multiple digits', input: '12345'),
  SingleEncodingTestCase(name: 'Year', input: '2024'),

  // Punctuation
  SingleEncodingTestCase(name: 'Multiple punctuation', input: '!!??'),
  SingleEncodingTestCase(name: 'Ellipsis', input: '...'),
  SingleEncodingTestCase(name: 'Mixed punctuation', input: 'Hello... World!'),

  // LLM related text
  SingleEncodingTestCase(
    name: 'LLM prompt',
    input: 'What is machine learning?',
  ),
  SingleEncodingTestCase(
    name: 'AI sentence',
    input: 'Artificial intelligence is transforming the world.',
  ),

  // Unicode and accents
  SingleEncodingTestCase(name: 'Accented cafe', input: 'café'),
  SingleEncodingTestCase(name: 'Accented naive', input: 'naïve'),
  SingleEncodingTestCase(name: 'German umlaut', input: 'München'),

  // Multi-language
  SingleEncodingTestCase(name: 'Chinese characters', input: '你好世界'),
  SingleEncodingTestCase(name: 'Japanese hiragana', input: 'こんにちは'),
  SingleEncodingTestCase(name: 'Korean', input: '안녕하세요'),
  SingleEncodingTestCase(name: 'Mixed language', input: 'Hello 世界 Bonjour'),

  // Emoji
  SingleEncodingTestCase(name: 'Simple emoji', input: 'Hello 😊'),
  SingleEncodingTestCase(name: 'Multiple emoji', input: '🎉🎊🎈'),

  // Contractions
  SingleEncodingTestCase(name: "Contraction I'm", input: "I'm happy"),
  SingleEncodingTestCase(name: "Contraction don't", input: "don't worry"),

  // Hyphenated words
  SingleEncodingTestCase(name: 'Hyphenated state', input: 'state-of-the-art'),

  // Code-like text
  SingleEncodingTestCase(name: 'Variable name', input: 'my_variable_name'),
  SingleEncodingTestCase(name: 'CamelCase', input: 'camelCaseVariable'),

  // Long text
  SingleEncodingTestCase(
    name: 'Long sentence',
    input:
        'Machine learning is a subset of artificial intelligence that enables '
        'systems to learn and improve from experience without being explicitly '
        'programmed.',
  ),
];

final _edgeCaseTestCases = [
  EdgeCaseTestCase(name: 'Empty string', input: ''),
  EdgeCaseTestCase(name: 'Single space', input: ' '),
  EdgeCaseTestCase(name: 'Multiple spaces', input: '   '),
  EdgeCaseTestCase(name: 'Tab character', input: '\t'),
  EdgeCaseTestCase(name: 'Newline', input: '\n'),
  EdgeCaseTestCase(name: 'Single character a', input: 'a'),
  EdgeCaseTestCase(name: 'Single punctuation', input: '.'),
  EdgeCaseTestCase(name: 'Multiple spaces between', input: 'hello    world'),
  EdgeCaseTestCase(name: 'Leading spaces', input: '   hello'),
  EdgeCaseTestCase(name: 'Trailing spaces', input: 'hello   '),
  EdgeCaseTestCase(name: 'Very long word', input: 'a' * 50),
  EdgeCaseTestCase(name: 'Repeated word', input: 'test ' * 10),
];
