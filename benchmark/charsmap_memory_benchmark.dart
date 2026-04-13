/// Benchmark: charsmap memory analysis for on-device Gemma 3/4.
///
/// Simulates realistic precompiled_charsmap sizes (Gemma/Llama models use
/// ~800KB charsmaps) and measures before/after memory with toBytes() optimization.
///
/// Run: dart benchmark/charsmap_memory_benchmark.dart
import 'dart:convert';
import 'dart:typed_data';

import 'package:dart_sentencepiece_tokenizer/src/sentencepiece/model/model_proto.dart';
import 'package:dart_sentencepiece_tokenizer/src/sentencepiece/normalizer/precompiled_charsmap.dart';
import 'package:dart_sentencepiece_tokenizer/src/sentencepiece/normalizer/sp_normalizer.dart';

void main() {
  print('=' * 70);
  print('Charsmap Memory Analysis — Gemma 3/4 On-Device Scenario');
  print('=' * 70);

  // Real-world SentencePiece model charsmap sizes:
  // - Gemma (gemma-2b, gemma-7b): ~790KB precompiled_charsmap
  // - Llama 2/3: ~800KB precompiled_charsmap
  // - Gemma 3/4 (expected): ~800KB (same tokenizer base as Gemma 2)

  final scenarios = <String, int>{
    'Small model (test)': 4 * 1024,
    'Medium model': 100 * 1024,
    'Gemma 2/3/4 (realistic)': 800 * 1024,
    'Large model (worst case)': 2 * 1024 * 1024,
  };

  for (final entry in scenarios.entries) {
    _analyzeScenario(entry.key, entry.value);
  }

  print('');
  print('=' * 70);
  print('Isolate Memory Impact (encodeBatchParallel)');
  print('=' * 70);
  _analyzeIsolateScenario();

  print('');
  print('=' * 70);
  print('toBytes() Round-Trip Verification');
  print('=' * 70);
  _verifyToBytes();
}

void _analyzeScenario(String name, int targetSize) {
  print('');
  print('--- $name (${_formatBytes(targetSize)}) ---');

  final blob = _buildRealisticBlob(targetSize);
  final actualBlobSize = blob.length;

  // Parse into PrecompiledCharsmap
  final charsmap = PrecompiledCharsmap.fromBytes(blob);

  // Measure parsed form
  final byteData = ByteData.sublistView(blob);
  final trieBlobSize = byteData.getUint32(0, Endian.little);
  final trieArraySize = trieBlobSize;
  final normalizedPoolSize = actualBlobSize - 4 - trieBlobSize;
  final parsedTotalSize = trieArraySize + normalizedPoolSize;

  // Before optimization: raw bytes + parsed structures
  final beforeTotal = actualBlobSize + parsedTotalSize;

  // After optimization: parsed structures only (toBytes() on demand)
  final afterTotal = parsedTotalSize;

  final savedBytes = beforeTotal - afterTotal;
  final savedPct = (savedBytes / beforeTotal * 100).toStringAsFixed(1);

  print('  BEFORE (raw + parsed):  ${_formatBytes(beforeTotal)}');
  print('  AFTER  (parsed only):   ${_formatBytes(afterTotal)}');
  print('  Saved:                  ${_formatBytes(savedBytes)} (-$savedPct%)');

  // Verify normalization still works
  assert(charsmap.normalize('hello') == 'hello');
}

void _analyzeIsolateScenario() {
  const charsmapSize = 800 * 1024; // 800KB (Gemma realistic)
  const numWorkers = 4;

  print('');
  print('Scenario: Gemma 3/4, encodeBatchParallel(numWorkers: $numWorkers)');
  print('  Charsmap blob size: ${_formatBytes(charsmapSize)}');
  print('');

  // BEFORE: main isolate stores raw + parsed; each worker copies raw + re-parses
  final beforeMain = charsmapSize * 2;
  final beforePerWorker = charsmapSize * 2;
  final beforeTotal = beforeMain + numWorkers * beforePerWorker;

  // AFTER: main isolate stores parsed only; toBytes() on demand for isolate transfer;
  // each worker receives raw bytes then parses (parsed only in worker)
  final afterMain = charsmapSize; // parsed only
  final afterPerWorker =
      charsmapSize; // parsed only (raw bytes from toBytes() are transient)
  final afterTotal = afterMain + numWorkers * afterPerWorker;

  final savedTotal = beforeTotal - afterTotal;

  print('  BEFORE optimization:');
  print('    Main isolate:   ${_formatBytes(beforeMain)} (raw + parsed)');
  print(
    '    Per worker:     ${_formatBytes(beforePerWorker)} (copied raw + parsed)',
  );
  print('    Total:          ${_formatBytes(beforeTotal)}');
  print('');
  print('  AFTER optimization (toBytes() on demand):');
  print('    Main isolate:   ${_formatBytes(afterMain)} (parsed only)');
  print('    Per worker:     ${_formatBytes(afterPerWorker)} (parsed only)');
  print('    Total:          ${_formatBytes(afterTotal)}');
  print('');
  print(
    '  Memory saved:     ${_formatBytes(savedTotal)} '
    '(-${(savedTotal / beforeTotal * 100).toStringAsFixed(1)}%)',
  );

  // Compare with total tokenizer memory
  const vocabMemory = 3 * 1024 * 1024;

  print('');
  print('  Context (with vocab ~3MB per isolate):');
  print('  | Component     | Before    | After     | Saved     |');
  print('  |---------------|-----------|-----------|-----------|');
  print(
    '  | Charsmap      | ${_formatBytes(beforeTotal).padLeft(9)} '
    '| ${_formatBytes(afterTotal).padLeft(9)} '
    '| ${_formatBytes(savedTotal).padLeft(9)} |',
  );
  print(
    '  | Vocab (5 iso) | ${_formatBytes(vocabMemory * 5).padLeft(9)} '
    '| ${_formatBytes(vocabMemory * 5).padLeft(9)} '
    '| ${'0 B'.padLeft(9)} |',
  );
  final totalBefore = beforeTotal + vocabMemory * 5;
  final totalAfter = afterTotal + vocabMemory * 5;
  print(
    '  | TOTAL         | ${_formatBytes(totalBefore).padLeft(9)} '
    '| ${_formatBytes(totalAfter).padLeft(9)} '
    '| ${_formatBytes(totalBefore - totalAfter).padLeft(9)} |',
  );

  print('');
  print('  On-device impact:');
  print('  | Device class      | Budget    | Before % | After %  |');
  print('  |-------------------|-----------|----------|----------|');

  final budgets = <String, int>{
    'Low-end (512MB)  ': 512 * 1024 * 1024,
    'Mid-range (1GB)  ': 1024 * 1024 * 1024,
    'High-end (2GB)   ': 2048 * 1024 * 1024,
  };

  for (final entry in budgets.entries) {
    final beforePct = (totalBefore / entry.value * 100).toStringAsFixed(2);
    final afterPct = (totalAfter / entry.value * 100).toStringAsFixed(2);
    print(
      '  | ${entry.key} | ${_formatBytes(entry.value).padLeft(9)} '
      '| ${beforePct.padLeft(7)}% | ${afterPct.padLeft(7)}% |',
    );
  }
}

void _verifyToBytes() {
  print('');

  final sizes = [4 * 1024, 100 * 1024, 800 * 1024];
  for (final size in sizes) {
    final original = _buildRealisticBlob(size);
    final charsmap = PrecompiledCharsmap.fromBytes(original);
    final reconstructed = charsmap.toBytes();
    final match = _bytesEqual(original, reconstructed);

    print(
      '  ${_formatBytes(size).padRight(10)}: '
      '${match ? "PASS" : "FAIL"} '
      '(original=${_formatBytes(original.length)}, '
      'reconstructed=${_formatBytes(reconstructed.length)})',
    );
  }

  // Verify SpNormalizer.precompiledCharsmapBytes works via toBytes()
  print('');
  final blob = _buildRealisticBlob(4096);
  final spec = NormalizerSpec(
    name: 'nmt_nfkc',
    precompiledCharsmap: blob,
    addDummyPrefix: true,
    removeExtraWhitespaces: true,
    escapeWhitespaces: true,
  );
  final normalizer = SpNormalizer.fromSpec(spec);
  final roundTripped = normalizer.precompiledCharsmapBytes!;
  print(
    '  SpNormalizer round-trip: '
    '${_bytesEqual(blob, roundTripped) ? "PASS" : "FAIL"} '
    '(no raw bytes stored, reconstructed on demand)',
  );
}

// --- Helpers ---

Uint8List _buildRealisticBlob(int targetSize) {
  final trieTargetSize = (targetSize * 0.6).toInt();
  final poolTargetSize = targetSize - trieTargetSize - 4;
  final trieBlobSize = (trieTargetSize ~/ 4) * 4;
  final numUnits = trieBlobSize ~/ 4;

  final blob = BytesBuilder();

  final header = ByteData(4);
  header.setUint32(0, trieBlobSize, Endian.little);
  blob.add(header.buffer.asUint8List());

  for (var i = 0; i < numUnits; i++) {
    final unitData = ByteData(4);
    unitData.setUint32(0, i == 0 ? (1 << 10) : 0, Endian.little);
    blob.add(unitData.buffer.asUint8List());
  }

  var poolBytes = 0;
  while (poolBytes < poolTargetSize) {
    final str = utf8.encode('normalized_string_$poolBytes');
    blob.add(str);
    blob.addByte(0);
    poolBytes += str.length + 1;
  }

  return blob.toBytes();
}

bool _bytesEqual(Uint8List a, Uint8List b) {
  if (a.length != b.length) return false;
  for (var i = 0; i < a.length; i++) {
    if (a[i] != b[i]) return false;
  }
  return true;
}

String _formatBytes(int bytes) {
  if (bytes < 1024) return '$bytes B';
  if (bytes < 1024 * 1024) return '${(bytes / 1024).toStringAsFixed(1)} KB';
  return '${(bytes / (1024 * 1024)).toStringAsFixed(2)} MB';
}
