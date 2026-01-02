import 'dart:typed_data';

import 'package:test/test.dart';
import 'package:dart_sentencepiece_tokenizer/dart_sentencepiece_tokenizer.dart';
import 'package:dart_sentencepiece_tokenizer/src/sentencepiece/model/model_proto.dart';
import 'package:dart_sentencepiece_tokenizer/src/sentencepiece/model/protobuf_reader.dart';
import 'package:dart_sentencepiece_tokenizer/src/sentencepiece/normalizer/sp_normalizer.dart';
import 'package:dart_sentencepiece_tokenizer/src/sentencepiece/vocabulary/sp_vocabulary.dart';
import 'package:dart_sentencepiece_tokenizer/src/sentencepiece/algorithm/bpe_algorithm.dart';
import 'package:dart_sentencepiece_tokenizer/src/sentencepiece/algorithm/unigram_algorithm.dart';

void main() {
  group('ProtobufReader', () {
    test('readVarint for small numbers', () {
      // Varint encoding of 1 is just 0x01
      final reader = ProtobufReader(Uint8List.fromList([0x01]));
      expect(reader.readVarint(), equals(1));
    });

    test('readVarint for larger numbers', () {
      // Varint encoding of 300 is 0xAC 0x02
      final reader = ProtobufReader(Uint8List.fromList([0xAC, 0x02]));
      expect(reader.readVarint(), equals(300));
    });

    test('readTag', () {
      // Field 1, wire type 0 (varint) = (1 << 3) | 0 = 8 = 0x08
      final reader = ProtobufReader(Uint8List.fromList([0x08]));
      final tag = reader.readTag();
      expect(tag, isNotNull);
      expect(tag!.$1, equals(1)); // field number
      expect(tag.$2, equals(0)); // wire type (varint)
    });

    test('readString', () {
      // Length-delimited string "test"
      // Length = 4, followed by "test"
      final reader = ProtobufReader(
        Uint8List.fromList([4, 116, 101, 115, 116]), // 4 + "test"
      );
      expect(reader.readString(), equals('test'));
    });

    test('readFloat', () {
      // Float 1.0 in little-endian
      final reader = ProtobufReader(
        Uint8List.fromList([0x00, 0x00, 0x80, 0x3F]),
      );
      expect(reader.readFloat(), closeTo(1.0, 0.0001));
    });

    test('readBool', () {
      final readerTrue = ProtobufReader(Uint8List.fromList([0x01]));
      expect(readerTrue.readBool(), isTrue);

      final readerFalse = ProtobufReader(Uint8List.fromList([0x00]));
      expect(readerFalse.readBool(), isFalse);
    });
  });

  group('SpNormalizer', () {
    test('normalize with default settings', () {
      const normalizer = SpNormalizer();

      // Should add dummy prefix and escape whitespaces
      final result = normalizer.normalize('hello world');

      expect(result, contains(kSpaceSymbol));
      expect(result, startsWith(kSpaceSymbol)); // dummy prefix
    });

    test('normalize without dummy prefix', () {
      const normalizer = SpNormalizer(addDummyPrefix: false);

      final result = normalizer.normalize('hello world');

      expect(result, isNot(startsWith(kSpaceSymbol + kSpaceSymbol)));
    });

    test('remove extra whitespaces', () {
      const normalizer = SpNormalizer(removeExtraWhitespaces: true);

      final result = normalizer.normalize('hello   world');

      // Should collapse multiple spaces to one
      expect(result.contains('$kSpaceSymbol$kSpaceSymbol'), isFalse);
    });

    test('preserve whitespaces when not escaping', () {
      const normalizer = SpNormalizer(escapeWhitespaces: false);

      final result = normalizer.normalize('hello');

      expect(result, isNot(contains(kSpaceSymbol)));
    });

    test('denormalize reverses normalize', () {
      const normalizer = SpNormalizer();

      final normalized = normalizer.normalize('hello world');
      final denormalized = normalizer.denormalize(normalized);

      expect(denormalized, equals('hello world'));
    });

    test('empty string', () {
      const normalizer = SpNormalizer();

      expect(normalizer.normalize(''), equals(''));
      expect(normalizer.denormalize(''), equals(''));
    });
  });

  group('PieceType', () {
    test('fromValue', () {
      expect(PieceType.fromValue(1), equals(PieceType.normal));
      expect(PieceType.fromValue(2), equals(PieceType.unknown));
      expect(PieceType.fromValue(3), equals(PieceType.control));
      expect(PieceType.fromValue(4), equals(PieceType.userDefined));
      expect(PieceType.fromValue(5), equals(PieceType.unused));
      expect(PieceType.fromValue(6), equals(PieceType.byte));
    });

    test('value', () {
      expect(PieceType.normal.value, equals(1));
      expect(PieceType.unknown.value, equals(2));
      expect(PieceType.control.value, equals(3));
    });
  });

  group('ModelType', () {
    test('fromValue', () {
      expect(ModelType.fromValue(1), equals(ModelType.unigram));
      expect(ModelType.fromValue(2), equals(ModelType.bpe));
      expect(ModelType.fromValue(3), equals(ModelType.word));
      expect(ModelType.fromValue(4), equals(ModelType.char));
    });
  });

  group('SpVocabulary', () {
    late SpVocabulary vocab;

    setUp(() {
      final model = SentencePieceModel(
        pieces: [
          const SentencePiece(piece: '<unk>', score: 0, type: PieceType.unknown),
          const SentencePiece(piece: '<s>', score: 0, type: PieceType.control),
          const SentencePiece(piece: '</s>', score: 0, type: PieceType.control),
          const SentencePiece(piece: '▁hello', score: -1.0, type: PieceType.normal),
          const SentencePiece(piece: '▁world', score: -1.5, type: PieceType.normal),
          const SentencePiece(piece: '▁', score: -2.0, type: PieceType.normal),
          const SentencePiece(piece: 'hello', score: -3.0, type: PieceType.normal),
        ],
        trainerSpec: const TrainerSpec(
          modelType: ModelType.unigram,
          unkId: 0,
          bosId: 1,
          eosId: 2,
          padId: -1,
        ),
        normalizerSpec: const NormalizerSpec(),
      );
      vocab = SpVocabulary.fromModel(model);
    });

    test('size', () {
      expect(vocab.size, equals(7));
    });

    test('pieceToId', () {
      expect(vocab.pieceToId('▁hello'), equals(3));
      expect(vocab.pieceToId('▁world'), equals(4));
      expect(vocab.pieceToId('unknown'), equals(0)); // returns unkId
    });

    test('idToPiece', () {
      expect(vocab.idToPiece(3), equals('▁hello'));
      expect(vocab.idToPiece(4), equals('▁world'));
    });

    test('getScore', () {
      expect(vocab.getScore(3), closeTo(-1.0, 0.0001));
      expect(vocab.getScore(4), closeTo(-1.5, 0.0001));
    });

    test('special token IDs', () {
      expect(vocab.unkId, equals(0));
      expect(vocab.bosId, equals(1));
      expect(vocab.eosId, equals(2));
    });

    test('isSpecialToken', () {
      expect(vocab.isSpecialToken(0), isTrue); // unk
      expect(vocab.isSpecialToken(1), isTrue); // bos
      expect(vocab.isSpecialToken(2), isTrue); // eos
      expect(vocab.isSpecialToken(3), isFalse); // regular token
    });

    test('contains', () {
      expect(vocab.contains('▁hello'), isTrue);
      expect(vocab.contains('nonexistent'), isFalse);
    });

    test('findAllPrefixes', () {
      final matches = vocab.findAllPrefixes('▁hello world');
      expect(matches, isNotEmpty);
      expect(matches.any((m) => m.token == '▁'), isTrue);
    });
  });

  group('BpeAlgorithm', () {
    late SpVocabulary vocab;
    late BpeAlgorithm algorithm;

    setUp(() {
      final model = SentencePieceModel(
        pieces: [
          const SentencePiece(piece: '<unk>', score: 0, type: PieceType.unknown),
          const SentencePiece(piece: 'a', score: -1.0, type: PieceType.normal),
          const SentencePiece(piece: 'b', score: -1.0, type: PieceType.normal),
          const SentencePiece(piece: 'ab', score: -0.5, type: PieceType.normal),
          const SentencePiece(piece: 'abc', score: -0.3, type: PieceType.normal),
          const SentencePiece(piece: 'c', score: -1.0, type: PieceType.normal),
        ],
        trainerSpec: const TrainerSpec(modelType: ModelType.bpe),
        normalizerSpec: const NormalizerSpec(),
      );
      vocab = SpVocabulary.fromModel(model);
      algorithm = BpeAlgorithm(vocab: vocab);
    });

    test('empty string returns empty list', () {
      expect(algorithm.tokenize(''), isEmpty);
    });

    test('single character tokenization', () {
      final ids = algorithm.tokenize('a');
      expect(ids, equals([1])); // 'a' has id 1
    });

    test('merge to longer token', () {
      final ids = algorithm.tokenize('ab');
      // Should merge 'a' + 'b' to 'ab' since 'ab' has higher score
      expect(ids, equals([3])); // 'ab' has id 3
    });

    test('merge to longest token', () {
      final ids = algorithm.tokenize('abc');
      // Should merge to 'abc' since it has the highest score
      expect(ids, equals([4])); // 'abc' has id 4
    });
  });

  group('UnigramAlgorithm', () {
    late SpVocabulary vocab;
    late UnigramAlgorithm algorithm;

    setUp(() {
      final model = SentencePieceModel(
        pieces: [
          const SentencePiece(piece: '<unk>', score: -100, type: PieceType.unknown),
          const SentencePiece(piece: 'a', score: -1.0, type: PieceType.normal),
          const SentencePiece(piece: 'b', score: -1.0, type: PieceType.normal),
          const SentencePiece(piece: 'ab', score: -0.5, type: PieceType.normal),
          const SentencePiece(piece: 'abc', score: -0.3, type: PieceType.normal),
          const SentencePiece(piece: 'c', score: -1.0, type: PieceType.normal),
        ],
        trainerSpec: const TrainerSpec(modelType: ModelType.unigram),
        normalizerSpec: const NormalizerSpec(),
      );
      vocab = SpVocabulary.fromModel(model);
      algorithm = UnigramAlgorithm(vocab: vocab);
    });

    test('empty string returns empty list', () {
      expect(algorithm.tokenize(''), isEmpty);
    });

    test('single character tokenization', () {
      final ids = algorithm.tokenize('a');
      expect(ids, equals([1])); // 'a' has id 1
    });

    test('viterbi finds optimal segmentation', () {
      final ids = algorithm.tokenize('abc');
      // Viterbi should find 'abc' as optimal since it has highest score
      expect(ids, equals([4])); // 'abc' has id 4
    });

    test('multiple tokens when no single match', () {
      final ids = algorithm.tokenize('abab');
      // Should segment as 'ab' + 'ab'
      expect(ids, equals([3, 3])); // 'ab' has id 3
    });
  });

  group('SentencePieceConfig', () {
    test('default config', () {
      const config = SentencePieceConfig();
      expect(config.addBosToken, isFalse);
      expect(config.addEosToken, isFalse);
    });

    test('gemma config', () {
      expect(SentencePieceConfig.gemma.addBosToken, isTrue);
      expect(SentencePieceConfig.gemma.addEosToken, isTrue);
    });

    test('llama config', () {
      expect(SentencePieceConfig.llama.addBosToken, isTrue);
      expect(SentencePieceConfig.llama.addEosToken, isFalse);
    });
  });
}
