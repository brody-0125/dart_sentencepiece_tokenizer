/// Test utilities for SentencePiece tokenizer tests.
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:dart_sentencepiece_tokenizer/dart_sentencepiece_tokenizer.dart';

/// Path to test model file.
const testModelPath = 'test/fixtures/test.model';

/// Alternative model paths to search.
const alternativeModelPaths = [
  'test.model',
  'tokenizer.model',
  'model.model',
];

/// Cache for test model bytes.
Uint8List? _cachedModelBytes;

/// Get test model bytes, loading from file if available.
///
/// Returns a minimal test model if no model file is found.
Uint8List getTestModelBytes() {
  if (_cachedModelBytes != null) {
    return _cachedModelBytes!;
  }

  // Try to find a model file
  final paths = [testModelPath, ...alternativeModelPaths];
  for (final path in paths) {
    final file = File(path);
    if (file.existsSync()) {
      _cachedModelBytes = file.readAsBytesSync();
      return _cachedModelBytes!;
    }
  }

  // No model file found - create a minimal test model
  // This is a simplified protobuf structure for testing
  _cachedModelBytes = _createMinimalTestModel();
  return _cachedModelBytes!;
}

/// Create a minimal SentencePiece model for testing.
///
/// This creates a valid protobuf structure that can be parsed
/// by the SentencePieceModelLoader.
Uint8List _createMinimalTestModel() {
  // Build a minimal protobuf model
  final builder = _ProtobufBuilder();

  // Add pieces (field 1, repeated)
  final testPieces = [
    ('<unk>', 0.0, 2), // Unknown token
    ('<s>', 0.0, 3), // BOS token
    ('</s>', 0.0, 3), // EOS token
    ('▁', -1.0, 1), // Space token
    ('▁hello', -2.0, 1),
    ('▁world', -2.5, 1),
    ('▁the', -1.5, 1),
    ('▁a', -1.8, 1),
    ('▁is', -2.0, 1),
    ('▁test', -2.2, 1),
    ('▁This', -2.3, 1),
    ('▁sentence', -3.0, 1),
    ('h', -3.5, 1),
    ('e', -3.5, 1),
    ('l', -3.5, 1),
    ('o', -3.5, 1),
    ('w', -3.5, 1),
    ('r', -3.5, 1),
    ('d', -3.5, 1),
    ('t', -3.5, 1),
    ('s', -3.5, 1),
    ('n', -3.5, 1),
    ('c', -3.5, 1),
    ('i', -3.5, 1),
    ('▁first', -2.8, 1),
    ('▁second', -2.9, 1),
    ('▁short', -3.0, 1),
    ('▁longer', -3.1, 1),
    ('▁many', -3.2, 1),
    ('▁words', -3.3, 1),
    ('▁with', -2.5, 1),
    (',', -2.0, 1),
    ('.', -2.0, 1),
    ('!', -2.5, 1),
    ('?', -2.5, 1),
  ];

  for (final (piece, score, type) in testPieces) {
    final pieceBuilder = _ProtobufBuilder();
    pieceBuilder.writeString(1, piece); // piece field
    pieceBuilder.writeFloat(2, score); // score field
    pieceBuilder.writeVarint(3, type); // type field
    builder.writeBytes(1, pieceBuilder.toBytes());
  }

  // Add trainer_spec (field 2)
  final trainerSpec = _ProtobufBuilder();
  trainerSpec.writeVarint(1, 1); // model_type = unigram
  trainerSpec.writeVarint(3, testPieces.length); // vocab_size
  trainerSpec.writeVarint(40, 0); // unk_id
  trainerSpec.writeVarint(41, 1); // bos_id
  trainerSpec.writeVarint(42, 2); // eos_id
  trainerSpec.writeVarint(43, -1); // pad_id (not set)
  builder.writeBytes(2, trainerSpec.toBytes());

  // Add normalizer_spec (field 3)
  final normalizerSpec = _ProtobufBuilder();
  normalizerSpec.writeString(1, 'identity'); // name
  normalizerSpec.writeBool(2, true); // add_dummy_prefix
  normalizerSpec.writeBool(3, false); // remove_extra_whitespaces
  normalizerSpec.writeBool(4, true); // escape_whitespaces
  builder.writeBytes(3, normalizerSpec.toBytes());

  return builder.toBytes();
}

/// Create a test tokenizer with default configuration.
SentencePieceTokenizer createTestTokenizer({
  SentencePieceConfig config = const SentencePieceConfig(),
}) {
  return SentencePieceTokenizer.fromBytes(
    getTestModelBytes(),
    config: config,
  );
}

/// Create a test tokenizer with Llama configuration.
SentencePieceTokenizer createLlamaTokenizer() {
  return createTestTokenizer(config: SentencePieceConfig.llama);
}

/// Create a test tokenizer with Gemma configuration.
SentencePieceTokenizer createGemmaTokenizer() {
  return createTestTokenizer(config: SentencePieceConfig.gemma);
}

/// Simple protobuf builder for creating test data.
class _ProtobufBuilder {
  final _buffer = BytesBuilder();

  void writeVarint(int fieldNumber, int value) {
    final tag = (fieldNumber << 3) | 0; // wire type 0 = varint
    _writeVarintRaw(tag);
    _writeVarintRaw(value);
  }

  void writeString(int fieldNumber, String value) {
    final tag = (fieldNumber << 3) | 2; // wire type 2 = length-delimited
    _writeVarintRaw(tag);
    final bytes = utf8.encode(value); // Use UTF-8 encoding for protobuf
    _writeVarintRaw(bytes.length);
    _buffer.add(bytes);
  }

  void writeFloat(int fieldNumber, double value) {
    final tag = (fieldNumber << 3) | 5; // wire type 5 = 32-bit
    _writeVarintRaw(tag);
    final data = ByteData(4);
    data.setFloat32(0, value, Endian.little);
    _buffer.add(data.buffer.asUint8List());
  }

  void writeBool(int fieldNumber, bool value) {
    writeVarint(fieldNumber, value ? 1 : 0);
  }

  void writeBytes(int fieldNumber, Uint8List bytes) {
    final tag = (fieldNumber << 3) | 2; // wire type 2 = length-delimited
    _writeVarintRaw(tag);
    _writeVarintRaw(bytes.length);
    _buffer.add(bytes);
  }

  void _writeVarintRaw(int value) {
    // Handle negative numbers as unsigned
    var v = value < 0 ? value + (1 << 32) : value;
    while (v >= 0x80) {
      _buffer.addByte((v & 0x7F) | 0x80);
      v >>= 7;
    }
    _buffer.addByte(v);
  }

  Uint8List toBytes() => _buffer.toBytes();
}
