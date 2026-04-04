import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:dart_sentencepiece_tokenizer/dart_sentencepiece_tokenizer.dart';

void main() async {
  print('=' * 70);
  print('Streaming API Benchmark (v1.3.0)');
  print('=' * 70);
  print('');

  final modelPath = _findModelFile();
  SentencePieceTokenizer tokenizer;

  if (modelPath != null) {
    print('Loading model from: $modelPath');
    tokenizer = SentencePieceTokenizer.fromModelFileSync(modelPath);
    print('Model loaded: ${tokenizer.vocabSize} tokens');
  } else {
    print('No model file found. Using minimal test model.');
    tokenizer = _createMinimalTokenizer();
    print('Minimal model created: ${tokenizer.vocabSize} tokens');
  }
  print('');

  await _runTextStreamerBenchmark(tokenizer);
  await _runDecodeVsStreamingBenchmark(tokenizer);
  await _runCallbackOverheadBenchmark(tokenizer);
  await _runWordBoundaryHeuristicsBenchmark(tokenizer);
  await _runMemoryBenchmark(tokenizer);

  print('=' * 70);
  print('Streaming Benchmark Complete');
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

Future<void> _runTextStreamerBenchmark(SentencePieceTokenizer tokenizer) async {
  print('-' * 70);
  print('1. TextStreamer THROUGHPUT');
  print('-' * 70);

  // Generate token sequences of various lengths
  final testCases = [
    ('Short (10 tokens)', _generateTokens(tokenizer, 'hello world test', 10)),
    (
      'Medium (100 tokens)',
      _generateTokens(tokenizer, 'hello world test sentence', 100),
    ),
    (
      'Long (500 tokens)',
      _generateTokens(tokenizer, 'hello world test sentence is', 500),
    ),
  ];

  for (final (name, tokens) in testCases) {
    // Warmup
    for (var i = 0; i < 10; i++) {
      final streamer = tokenizer.createTextStreamer(
        onFinalizedText: (_, {required streamEnd}) {},
      );
      for (final id in tokens) {
        streamer.put(id);
      }
      streamer.end();
    }

    // Benchmark
    const iterations = 1000;
    final sw = Stopwatch()..start();
    for (var i = 0; i < iterations; i++) {
      final streamer = tokenizer.createTextStreamer(
        onFinalizedText: (_, {required streamEnd}) {},
      );
      for (final id in tokens) {
        streamer.put(id);
      }
      streamer.end();
    }
    sw.stop();

    final avgUs = (sw.elapsedMicroseconds / iterations).toStringAsFixed(1);
    final tokensPerSec =
        (tokens.length * iterations / (sw.elapsedMilliseconds / 1000)).round();
    print('  $name: ${avgUs}μs/stream (${_formatNumber(tokensPerSec)} tok/s)');
  }
  print('');
}

Future<void> _runDecodeVsStreamingBenchmark(
  SentencePieceTokenizer tokenizer,
) async {
  print('-' * 70);
  print('2. BATCH decode() vs STREAMING decode');
  print('-' * 70);

  final tokens = _generateTokens(
    tokenizer,
    'hello world test sentence is',
    100,
  );

  // Warmup
  for (var i = 0; i < 10; i++) {
    tokenizer.decode(tokens);
    final streamer = tokenizer.createTextStreamer(
      onFinalizedText: (_, {required streamEnd}) {},
    );
    for (final id in tokens) {
      streamer.put(id);
    }
    streamer.end();
  }

  const iterations = 1000;

  // Batch decode
  var sw = Stopwatch()..start();
  for (var i = 0; i < iterations; i++) {
    tokenizer.decode(tokens);
  }
  sw.stop();
  final batchUs = (sw.elapsedMicroseconds / iterations).toStringAsFixed(1);

  // TextStreamer streaming
  sw = Stopwatch()..start();
  for (var i = 0; i < iterations; i++) {
    final streamer = tokenizer.createTextStreamer(
      onFinalizedText: (_, {required streamEnd}) {},
    );
    for (final id in tokens) {
      streamer.put(id);
    }
    streamer.end();
  }
  sw.stop();
  final streamerUs = (sw.elapsedMicroseconds / iterations).toStringAsFixed(1);

  print('  ${tokens.length} tokens comparison:');
  print('    decode() (batch):       ${batchUs}μs');
  print('    TextStreamer:           ${streamerUs}μs');
  print('');

  // Verify results match
  final batchResult = tokenizer.decode(tokens);
  final streamerResult = StringBuffer();
  final streamer = tokenizer.createTextStreamer(
    onFinalizedText: (text, {required streamEnd}) => streamerResult.write(text),
  );
  for (final id in tokens) {
    streamer.put(id);
  }
  streamer.end();

  final match = batchResult.trim() == streamerResult.toString().trim();
  print('  Results match: ${match ? "YES" : "NO"}');
  print('');
}

Future<void> _runCallbackOverheadBenchmark(
  SentencePieceTokenizer tokenizer,
) async {
  print('-' * 70);
  print('3. CALLBACK OVERHEAD');
  print('-' * 70);

  final tokens = _generateTokens(tokenizer, 'hello world test', 50);

  // No-op callback
  const iterations = 1000;
  var sw = Stopwatch()..start();
  for (var i = 0; i < iterations; i++) {
    final streamer = tokenizer.createTextStreamer(
      onFinalizedText: (_, {required streamEnd}) {},
    );
    for (final id in tokens) {
      streamer.put(id);
    }
    streamer.end();
  }
  sw.stop();
  final noopUs = (sw.elapsedMicroseconds / iterations).toStringAsFixed(1);

  // StringBuffer callback
  sw = Stopwatch()..start();
  for (var i = 0; i < iterations; i++) {
    final buffer = StringBuffer();
    final streamer = tokenizer.createTextStreamer(
      onFinalizedText: (text, {required streamEnd}) => buffer.write(text),
    );
    for (final id in tokens) {
      streamer.put(id);
    }
    streamer.end();
  }
  sw.stop();
  final bufferUs = (sw.elapsedMicroseconds / iterations).toStringAsFixed(1);

  // List<String> callback
  sw = Stopwatch()..start();
  for (var i = 0; i < iterations; i++) {
    final chunks = <String>[];
    final streamer = tokenizer.createTextStreamer(
      onFinalizedText: (text, {required streamEnd}) => chunks.add(text),
    );
    for (final id in tokens) {
      streamer.put(id);
    }
    streamer.end();
  }
  sw.stop();
  final listUs = (sw.elapsedMicroseconds / iterations).toStringAsFixed(1);

  print('  ${tokens.length} tokens, callback comparison:');
  print('    No-op callback:         ${noopUs}μs');
  print('    StringBuffer callback:  ${bufferUs}μs');
  print('    List<String> callback:  ${listUs}μs');
  print('');
}

Future<void> _runWordBoundaryHeuristicsBenchmark(
  SentencePieceTokenizer tokenizer,
) async {
  print('-' * 70);
  print('4. WORD BOUNDARY HEURISTICS');
  print('-' * 70);

  // Count emissions for different text types
  final testCases = [
    ('Latin text', 'hello world this is a test of word boundaries'),
    ('CJK text', '你好世界这是一个测试'),
    ('Mixed', 'hello 世界 this 是 a test'),
  ];

  for (final (name, text) in testCases) {
    final encoding = tokenizer.encode(text, addSpecialTokens: false);
    var emissionCount = 0;

    final streamer = tokenizer.createTextStreamer(
      onFinalizedText: (_, {required streamEnd}) => emissionCount++,
    );
    for (final id in encoding.ids) {
      streamer.put(id);
    }
    streamer.end();

    print(
      '  $name: ${encoding.length} tokens -> $emissionCount emissions (${(emissionCount / encoding.length * 100).toStringAsFixed(0)}% ratio)',
    );
  }
  print('');
}

Future<void> _runMemoryBenchmark(SentencePieceTokenizer tokenizer) async {
  print('-' * 70);
  print('5. STREAMING MEMORY USAGE');
  print('-' * 70);

  // TextStreamer internal state
  print('  TextStreamer internal state:');
  print('    _tokenCache: List<int> (grows with token count)');
  print('    _printLen: int (constant)');
  print('    _promptTokensRemaining: int (constant)');
  print('');

  // Memory estimate
  final tokens = _generateTokens(tokenizer, 'hello world test sentence', 100);

  // TextStreamer caches all tokens
  final textStreamerBytes = tokens.length * 8; // List<int> ≈ 8 bytes per int

  print('  For ${tokens.length} tokens:');
  print('    TextStreamer cache: ~$textStreamerBytes bytes');
  print('');
  print('  Note: TextStreamer re-decodes full sequence on each put()');
  print('        for accurate word boundary detection');
  print('');
}

/// Generate a token sequence by repeating encoding of given text.
List<int> _generateTokens(
  SentencePieceTokenizer tokenizer,
  String text,
  int targetLength,
) {
  final encoding = tokenizer.encode(text, addSpecialTokens: false);
  final tokens = <int>[];
  while (tokens.length < targetLength) {
    tokens.addAll(encoding.ids);
  }
  return tokens.sublist(0, targetLength);
}

String _formatNumber(int n) {
  if (n >= 1000000) {
    return '${(n / 1000000).toStringAsFixed(2)}M';
  } else if (n >= 1000) {
    return '${(n / 1000).toStringAsFixed(0)}K';
  }
  return n.toString();
}

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
    final bytes = utf8.encode(value);
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
