import 'dart:convert';
import 'dart:typed_data';

/// Build a precompiled charsmap binary blob from string-to-string mappings.
Uint8List buildPrecompiledCharsmapBlob(Map<String, String> mappings) {
  final normalizedBytes = BytesBuilder();
  final offsets = <String, int>{};
  for (final entry in mappings.entries) {
    offsets[entry.key] = normalizedBytes.length;
    normalizedBytes.add(utf8.encode(entry.value));
    normalizedBytes.addByte(0);
  }

  final trieEntries = <List<int>, int>{};
  for (final entry in mappings.entries) {
    trieEntries[utf8.encode(entry.key)] = offsets[entry.key]!;
  }
  final arrayData = _buildDartsArray(trieEntries);

  final blob = BytesBuilder();
  final sizeData = ByteData(4)
    ..setUint32(0, arrayData.length * 4, Endian.little);
  blob.add(sizeData.buffer.asUint8List());

  for (final unit in arrayData) {
    final unitData = ByteData(4)..setUint32(0, unit, Endian.little);
    blob.add(unitData.buffer.asUint8List());
  }
  blob.add(normalizedBytes.toBytes());
  return blob.toBytes();
}

Uint32List _buildDartsArray(Map<List<int>, int> entries) {
  final root = _TrieNode();
  for (final entry in entries.entries) {
    var node = root;
    for (final byte in entry.key) {
      node = node.children.putIfAbsent(byte, _TrieNode.new);
    }
    node.value = entry.value;
  }

  final array = List<int>.filled(1024, 0);
  final used = List<bool>.filled(1024, false);
  final usedBases = <int>{};

  void ensureSize(int size) {
    while (array.length < size) {
      array.add(0);
      used.add(false);
    }
  }

  void buildNode(_TrieNode node, int nodePos, int label) {
    final childLabels = node.children.keys.toList()..sort();
    final needLeaf = node.value != null;
    var effectiveBase = 1;

    outer:
    while (true) {
      ensureSize(effectiveBase + 256);
      if (usedBases.contains(effectiveBase)) {
        effectiveBase++;
        continue;
      }
      if (needLeaf && used[effectiveBase]) {
        effectiveBase++;
        continue;
      }
      for (final childLabel in childLabels) {
        final position = effectiveBase ^ childLabel;
        ensureSize(position + 1);
        if (used[position]) {
          effectiveBase++;
          continue outer;
        }
      }
      break;
    }

    usedBases.add(effectiveBase);
    final offset = nodePos ^ effectiveBase;
    var unit = (offset << 10) | (label & 0xFF);
    if (needLeaf) unit |= 1 << 8;
    ensureSize(nodePos + 1);
    array[nodePos] = unit;
    used[nodePos] = true;

    if (needLeaf) {
      ensureSize(effectiveBase + 1);
      array[effectiveBase] = 0x80000000 | node.value!;
      used[effectiveBase] = true;
    }

    for (final childLabel in childLabels) {
      final position = effectiveBase ^ childLabel;
      ensureSize(position + 1);
      used[position] = true;
    }
    for (final childLabel in childLabels) {
      final childPos = effectiveBase ^ childLabel;
      buildNode(node.children[childLabel]!, childPos, childLabel);
    }
  }

  buildNode(root, 0, 0);

  var lastUsed = array.length - 1;
  while (lastUsed > 0 && !used[lastUsed]) {
    lastUsed--;
  }
  return Uint32List.fromList(array.sublist(0, lastUsed + 1));
}

class _TrieNode {
  final children = <int, _TrieNode>{};
  int? value;
}
