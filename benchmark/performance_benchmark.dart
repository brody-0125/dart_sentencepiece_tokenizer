import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:dart_sentencepiece_tokenizer/dart_sentencepiece_tokenizer.dart';

void main() async {
  print('=' * 70);
  print('dart_sentencepiece_tokenizer Performance Benchmark');
  print('=' * 70);
  print('');

  final modelPath = _findModelFile();
  SentencePieceTokenizer tokenizer;

  if (modelPath != null) {
    print('Loading model from: $modelPath');
    final sw = Stopwatch()..start();
    tokenizer = SentencePieceTokenizer.fromModelFileSync(modelPath);
    sw.stop();
    print(
      'Model loaded: ${tokenizer.vocabSize} tokens in ${sw.elapsedMilliseconds}ms',
    );
    print('Model type: ${tokenizer.modelType}');
  } else {
    print('No model file found. Using minimal test model.');
    tokenizer = _createMinimalTokenizer();
    print('Minimal model created: ${tokenizer.vocabSize} tokens');
  }
  print('');

  await _runSingleEncodingBenchmark(tokenizer);
  await _runBatchEncodingBenchmark(tokenizer);
  await _runParallelBenchmark(tokenizer);
  await _runThroughputBenchmark(tokenizer);
  await _runMemoryBenchmark(tokenizer);
  await _runModelLoadingBenchmark(modelPath);
  await _runJsonSerializationBenchmark(tokenizer);
  await _runTokenAdditionBenchmark();
  await _runTokenizeBenchmark(tokenizer);

  print('=' * 70);
  print('Benchmark Complete');
  print('=' * 70);
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

SentencePieceTokenizer _createMinimalTokenizer() {
  // Create minimal protobuf model for testing
  final builder = _ProtobufBuilder();

  final testPieces = [
    ('<unk>', 0.0, 2),
    ('<s>', 0.0, 3),
    ('</s>', 0.0, 3),
    ('▁', -1.0, 1),
    ('▁hello', -2.0, 1),
    ('▁world', -2.5, 1),
    ('▁the', -1.5, 1),
    ('▁a', -1.8, 1),
    ('▁is', -2.0, 1),
    ('▁test', -2.2, 1),
    ('▁This', -2.3, 1),
    ('▁sentence', -3.0, 1),
    for (var c in 'helloworld'.split('')) (c, -3.5, 1),
  ];

  for (final (piece, score, type) in testPieces) {
    final pieceBuilder = _ProtobufBuilder();
    pieceBuilder.writeString(1, piece);
    pieceBuilder.writeFloat(2, score);
    pieceBuilder.writeVarint(3, type);
    builder.writeBytes(1, pieceBuilder.toBytes());
  }

  final trainerSpec = _ProtobufBuilder();
  trainerSpec.writeVarint(1, 1);
  trainerSpec.writeVarint(3, testPieces.length);
  trainerSpec.writeVarint(40, 0);
  trainerSpec.writeVarint(41, 1);
  trainerSpec.writeVarint(42, 2);
  trainerSpec.writeVarint(43, -1);
  builder.writeBytes(2, trainerSpec.toBytes());

  final normalizerSpec = _ProtobufBuilder();
  normalizerSpec.writeString(1, 'identity');
  normalizerSpec.writeBool(2, true);
  normalizerSpec.writeBool(3, false);
  normalizerSpec.writeBool(4, true);
  builder.writeBytes(3, normalizerSpec.toBytes());

  return SentencePieceTokenizer.fromBytes(builder.toBytes());
}

Future<void> _runSingleEncodingBenchmark(
  SentencePieceTokenizer tokenizer,
) async {
  print('-' * 70);
  print('1. SINGLE ENCODING LATENCY');
  print('-' * 70);

  final testCases = [
    ('Short', 'Hello, world!'),
    ('Medium', 'The quick brown fox jumps over the lazy dog.'),
    (
      'Long',
      'Machine learning is a subset of artificial intelligence that enables '
          'systems to learn and improve from experience without being '
          'explicitly programmed. Deep learning, a subset of machine learning, '
          'uses neural networks with many layers.'
    ),
  ];

  for (final (name, text) in testCases) {
    // Warmup
    for (var i = 0; i < 100; i++) {
      tokenizer.encode(text, addSpecialTokens: false);
    }

    // Benchmark
    const iterations = 1000;
    final sw = Stopwatch()..start();
    for (var i = 0; i < iterations; i++) {
      tokenizer.encode(text, addSpecialTokens: false);
    }
    sw.stop();

    final encoding = tokenizer.encode(text, addSpecialTokens: false);
    final avgUs = (sw.elapsedMicroseconds / iterations).toStringAsFixed(1);
    print('  $name (${encoding.length} tokens): ${avgUs}μs/encode');
  }
  print('');
}

Future<void> _runBatchEncodingBenchmark(
  SentencePieceTokenizer tokenizer,
) async {
  print('-' * 70);
  print('2. BATCH ENCODING (Sequential)');
  print('-' * 70);

  final texts = [
    for (var i = 0; i < 100; i++)
      'This is test sentence number $i for batch encoding benchmark.',
  ];

  // Warmup
  tokenizer.encodeBatch(texts.take(10).toList(), addSpecialTokens: false);

  // Benchmark different batch sizes
  for (final batchSize in [10, 50, 100]) {
    final batch = texts.take(batchSize).toList();

    final sw = Stopwatch()..start();
    final results = tokenizer.encodeBatch(batch, addSpecialTokens: false);
    sw.stop();

    final totalTokens = results.fold<int>(0, (sum, e) => sum + e.length);
    final tokensPerMs = (totalTokens / sw.elapsedMilliseconds).toStringAsFixed(0);
    print(
      '  Batch $batchSize: ${sw.elapsedMilliseconds}ms ($tokensPerMs tokens/ms)',
    );
  }
  print('');
}

Future<void> _runParallelBenchmark(SentencePieceTokenizer tokenizer) async {
  print('-' * 70);
  print('3. PARALLEL vs SEQUENTIAL BATCH ENCODING');
  print('-' * 70);

  final texts = [
    for (var i = 0; i < 100; i++)
      'This is test sentence number $i for parallel encoding benchmark. '
          'Machine learning and deep learning are transforming many industries.',
  ];

  // Warmup
  tokenizer.encodeBatch(texts.take(10).toList(), addSpecialTokens: false);
  await tokenizer.encodeBatchParallel(
    texts.take(10).toList(),
    addSpecialTokens: false,
  );

  for (final batchSize in [16, 32, 64, 100]) {
    final batch = texts.take(batchSize).toList();

    // Sequential
    final swSeq = Stopwatch()..start();
    final seqResults = tokenizer.encodeBatch(batch, addSpecialTokens: false);
    swSeq.stop();

    // Parallel
    final swPar = Stopwatch()..start();
    final parResults = await tokenizer.encodeBatchParallel(
      batch,
      addSpecialTokens: false,
    );
    swPar.stop();

    final speedup = swPar.elapsedMilliseconds > 0
        ? (swSeq.elapsedMilliseconds / swPar.elapsedMilliseconds)
            .toStringAsFixed(2)
        : 'N/A';
    final seqMs = swSeq.elapsedMilliseconds;
    final parMs = swPar.elapsedMilliseconds;

    print('  Batch $batchSize:');
    print('    Sequential: ${seqMs}ms');
    print('    Parallel:   ${parMs}ms (${speedup}x speedup)');

    // Verify results match
    var match = true;
    for (var i = 0; i < seqResults.length; i++) {
      if (seqResults[i].ids.length != parResults[i].ids.length) {
        match = false;
        break;
      }
    }
    if (!match) print('    WARNING: Results mismatch!');
  }
  print('');
}

Future<void> _runThroughputBenchmark(SentencePieceTokenizer tokenizer) async {
  print('-' * 70);
  print('4. THROUGHPUT (tokens/second)');
  print('-' * 70);

  final baseTexts = [
    'Hello world!',
    'The quick brown fox jumps over the lazy dog.',
    'Machine learning is transforming industries.',
    'Natural language processing enables computers to understand human language.',
    'Transformers have revolutionized the field of deep learning and NLP.',
  ];

  final texts = <String>[];
  for (var i = 0; i < 200; i++) {
    texts.add(baseTexts[i % baseTexts.length]);
  }

  // Warmup
  tokenizer.encodeBatch(texts.take(50).toList(), addSpecialTokens: false);

  // Benchmark
  final sw = Stopwatch()..start();
  final results = tokenizer.encodeBatch(texts, addSpecialTokens: false);
  sw.stop();

  final totalTokens = results.fold<int>(0, (sum, e) => sum + e.length);
  final tokensPerSec = (totalTokens / sw.elapsedMilliseconds * 1000).round();

  print('  Texts processed: ${texts.length}');
  print('  Total tokens: $totalTokens');
  print('  Time: ${sw.elapsedMilliseconds}ms');
  print('  Throughput: ${_formatNumber(tokensPerSec)} tokens/sec');
  print('');
}

Future<void> _runMemoryBenchmark(SentencePieceTokenizer tokenizer) async {
  print('-' * 70);
  print('5. MEMORY EFFICIENCY (Typed Arrays)');
  print('-' * 70);

  final text =
      'This is a test sentence for memory benchmark with many tokens to process.';
  final encoding = tokenizer.encode(text, addSpecialTokens: false);

  print('  Encoding length: ${encoding.length} tokens');
  print('');
  print('  Field types (optimized):');
  print('    ids:              ${encoding.ids.runtimeType}');
  print('    typeIds:          ${encoding.typeIds.runtimeType}');
  print('    attentionMask:    ${encoding.attentionMask.runtimeType}');
  print('    specialTokensMask: ${encoding.specialTokensMask.runtimeType}');
  print('');

  // Calculate memory savings
  final numTokens = encoding.length;

  // Before optimization: List<int> uses ~8 bytes per element (boxed int)
  final beforeIds = numTokens * 8;
  final beforeTypeIds = numTokens * 8;
  final beforeAttention = numTokens * 8;
  final beforeSpecial = numTokens * 8;
  final totalBefore =
      beforeIds + beforeTypeIds + beforeAttention + beforeSpecial;

  // After optimization: Int32List (4 bytes) and Uint8List (1 byte)
  final afterIds = numTokens * 4; // Int32List
  final afterTypeIds = numTokens * 1; // Uint8List
  final afterAttention = numTokens * 1; // Uint8List
  final afterSpecial = numTokens * 1; // Uint8List
  final totalAfter = afterIds + afterTypeIds + afterAttention + afterSpecial;

  final savings = ((1 - totalAfter / totalBefore) * 100).toStringAsFixed(1);

  print('  Memory comparison (numeric fields, $numTokens tokens):');
  print('    Before (List<int>):  $totalBefore bytes');
  print('    After (typed arrays): $totalAfter bytes');
  print('    Savings: $savings%');
  print('');

  // Verify typed array types
  final isOptimized = encoding.ids is Int32List &&
      encoding.typeIds is Uint8List &&
      encoding.attentionMask is Uint8List &&
      encoding.specialTokensMask is Uint8List;

  print('  Typed arrays verified: ${isOptimized ? "YES" : "NO"}');
  print('');
}

Future<void> _runModelLoadingBenchmark(String? modelPath) async {
  print('-' * 70);
  print('6. MODEL LOADING');
  print('-' * 70);

  if (modelPath == null) {
    print('  Skipped (no model file available)');
    print('');
    return;
  }

  // Warmup - read file into memory
  final bytes = File(modelPath).readAsBytesSync();

  // Benchmark sync loading from bytes
  const iterations = 10;
  final sw = Stopwatch()..start();
  for (var i = 0; i < iterations; i++) {
    SentencePieceTokenizer.fromBytes(bytes);
  }
  sw.stop();

  final avgMs = (sw.elapsedMilliseconds / iterations).toStringAsFixed(1);
  final fileSize = (bytes.length / 1024).toStringAsFixed(1);

  print('  Model file size: $fileSize KB');
  print('  Average load time: ${avgMs}ms');
  print('');
}

String _formatNumber(int n) {
  if (n >= 1000000) {
    return '${(n / 1000000).toStringAsFixed(2)}M';
  } else if (n >= 1000) {
    return '${(n / 1000).toStringAsFixed(0)}K';
  }
  return n.toString();
}

// ============================================================
// Phase 1 (v0.3.0) Benchmarks
// ============================================================

Future<void> _runJsonSerializationBenchmark(
  SentencePieceTokenizer tokenizer,
) async {
  print('-' * 70);
  print('7. JSON SERIALIZATION (Phase 1 Feature)');
  print('-' * 70);

  // Warmup
  tokenizer.toJson();

  // Benchmark toJson
  const iterations = 100;
  var sw = Stopwatch()..start();
  for (var i = 0; i < iterations; i++) {
    tokenizer.toJson();
  }
  sw.stop();
  final toJsonAvgMs = (sw.elapsedMicroseconds / iterations / 1000).toStringAsFixed(2);
  print('  toJson():           ${toJsonAvgMs}ms/call');

  // Get JSON once for fromJsonString benchmark
  final json = tokenizer.toJson();
  final jsonSizeKb = (json.length / 1024).toStringAsFixed(1);
  print('  JSON size:          $jsonSizeKb KB');

  // Benchmark fromJsonString
  sw = Stopwatch()..start();
  for (var i = 0; i < iterations; i++) {
    TokenizerJsonLoader.fromJsonString(json);
  }
  sw.stop();
  final fromJsonAvgMs = (sw.elapsedMicroseconds / iterations / 1000).toStringAsFixed(2);
  print('  fromJsonString():   ${fromJsonAvgMs}ms/call');

  // Verify round-trip correctness
  final restored = TokenizerJsonLoader.fromJsonString(json);
  final original = tokenizer.encode('hello world', addSpecialTokens: false);
  final restoredEnc = restored.encode('hello world', addSpecialTokens: false);
  final match = original.ids.length == restoredEnc.ids.length;
  print('  Round-trip verified: ${match ? "YES" : "NO"}');
  print('');
}

Future<void> _runTokenAdditionBenchmark() async {
  print('-' * 70);
  print('8. DYNAMIC TOKEN ADDITION (Phase 1 Feature)');
  print('-' * 70);

  final tokenizer = _createMinimalTokenizer();
  final originalSize = tokenizer.vocabSize;

  // Benchmark addTokens
  final tokens = [for (var i = 0; i < 100; i++) '<token_$i>'];

  final sw = Stopwatch()..start();
  final added = tokenizer.addTokens(tokens);
  sw.stop();

  print('  Tokens added:       $added');
  print('  Vocab size before:  $originalSize');
  print('  Vocab size after:   ${tokenizer.vocabSize}');
  print('  Time to add 100:    ${sw.elapsedMicroseconds}μs');
  print('  Per token:          ${(sw.elapsedMicroseconds / added).toStringAsFixed(1)}μs/token');

  // Verify encoding works with added tokens
  final encoding = tokenizer.encode('<token_0> test', addSpecialTokens: false);
  final works = encoding.tokens.contains('<token_0>');
  print('  Added tokens work:  ${works ? "YES" : "NO"}');

  // Benchmark getAddedVocab
  final sw2 = Stopwatch()..start();
  for (var i = 0; i < 1000; i++) {
    tokenizer.getAddedVocab();
  }
  sw2.stop();
  print('  getAddedVocab():    ${(sw2.elapsedMicroseconds / 1000).toStringAsFixed(2)}μs/call');
  print('');
}

Future<void> _runTokenizeBenchmark(SentencePieceTokenizer tokenizer) async {
  print('-' * 70);
  print('9. tokenize() METHOD (Phase 1 Feature)');
  print('-' * 70);

  final testCases = [
    ('Short', 'Hello, world!'),
    ('Medium', 'The quick brown fox jumps over the lazy dog.'),
    (
      'Long',
      'Machine learning is a subset of artificial intelligence that enables '
          'systems to learn and improve from experience without being '
          'explicitly programmed. Deep learning uses neural networks.'
    ),
  ];

  for (final (name, text) in testCases) {
    // Warmup
    for (var i = 0; i < 100; i++) {
      tokenizer.tokenize(text);
    }

    // Benchmark tokenize()
    const iterations = 1000;
    final sw = Stopwatch()..start();
    for (var i = 0; i < iterations; i++) {
      tokenizer.tokenize(text);
    }
    sw.stop();

    final tokens = tokenizer.tokenize(text);
    final avgUs = (sw.elapsedMicroseconds / iterations).toStringAsFixed(1);
    print('  $name (${tokens.length} tokens): ${avgUs}μs/call');
  }

  // Compare with encode().tokens
  const text = 'The quick brown fox jumps over the lazy dog.';

  // tokenize()
  var sw = Stopwatch()..start();
  for (var i = 0; i < 1000; i++) {
    tokenizer.tokenize(text);
  }
  sw.stop();
  final tokenizeTime = sw.elapsedMicroseconds / 1000;

  // encode().tokens
  sw = Stopwatch()..start();
  for (var i = 0; i < 1000; i++) {
    tokenizer.encode(text, addSpecialTokens: false).tokens;
  }
  sw.stop();
  final encodeTime = sw.elapsedMicroseconds / 1000;

  print('');
  print('  tokenize() vs encode().tokens comparison:');
  print('    tokenize():       ${tokenizeTime.toStringAsFixed(1)}μs');
  print('    encode().tokens:  ${encodeTime.toStringAsFixed(1)}μs');
  print('');
}

/// Simple protobuf builder for creating test data.
class _ProtobufBuilder {
  final _buffer = BytesBuilder();

  void writeVarint(int fieldNumber, int value) {
    final tag = (fieldNumber << 3) | 0;
    _writeVarintRaw(tag);
    _writeVarintRaw(value);
  }

  void writeString(int fieldNumber, String value) {
    final tag = (fieldNumber << 3) | 2;
    _writeVarintRaw(tag);
    final bytes = utf8.encode(value); // Use UTF-8 encoding for protobuf
    _writeVarintRaw(bytes.length);
    _buffer.add(bytes);
  }

  void writeFloat(int fieldNumber, double value) {
    final tag = (fieldNumber << 3) | 5;
    _writeVarintRaw(tag);
    final data = ByteData(4);
    data.setFloat32(0, value, Endian.little);
    _buffer.add(data.buffer.asUint8List());
  }

  void writeBool(int fieldNumber, bool value) {
    writeVarint(fieldNumber, value ? 1 : 0);
  }

  void writeBytes(int fieldNumber, Uint8List bytes) {
    final tag = (fieldNumber << 3) | 2;
    _writeVarintRaw(tag);
    _writeVarintRaw(bytes.length);
    _buffer.add(bytes);
  }

  void _writeVarintRaw(int value) {
    var v = value < 0 ? value + (1 << 32) : value;
    while (v >= 0x80) {
      _buffer.addByte((v & 0x7F) | 0x80);
      v >>= 7;
    }
    _buffer.addByte(v);
  }

  Uint8List toBytes() => _buffer.toBytes();
}
