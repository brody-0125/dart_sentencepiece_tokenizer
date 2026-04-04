/// Phase 1 (v0.3.0) feature tests.
library;

import 'dart:io';

import 'package:test/test.dart';
import 'package:dart_sentencepiece_tokenizer/dart_sentencepiece_tokenizer.dart';

import 'test_utils.dart';

void main() {
  group('JSON Serialization', () {
    late SentencePieceTokenizer tokenizer;

    setUpAll(() {
      tokenizer = createTestTokenizer();
    });

    test('toJson produces valid JSON', () {
      final json = tokenizer.toJson();

      expect(json, isNotEmpty);
      expect(json, contains('"version"'));
      expect(json, contains('"model_type"'));
      expect(json, contains('"vocab"'));
      expect(json, contains('"special_tokens"'));
      expect(json, contains('"normalizer"'));
      expect(json, contains('"config"'));
    });

    test('toJson with pretty formatting', () {
      final json = tokenizer.toJson(pretty: true);

      expect(json, contains('\n'));
      expect(json, contains('  '));
    });

    test('round-trip JSON serialization preserves encoding', () {
      final originalJson = tokenizer.toJson();
      final restored = TokenizerJsonLoader.fromJsonString(originalJson);

      const text = 'hello world';
      final originalEncoding = tokenizer.encode(text, addSpecialTokens: false);
      final restoredEncoding = restored.encode(text, addSpecialTokens: false);

      expect(
        restoredEncoding.ids.toList(),
        equals(originalEncoding.ids.toList()),
      );
      expect(restoredEncoding.tokens, equals(originalEncoding.tokens));
    });

    test('round-trip preserves vocab size', () {
      final json = tokenizer.toJson();
      final restored = TokenizerJsonLoader.fromJsonString(json);

      expect(restored.vocabSize, equals(tokenizer.vocabSize));
    });

    test('round-trip preserves special token IDs', () {
      final json = tokenizer.toJson();
      final restored = TokenizerJsonLoader.fromJsonString(json);

      expect(restored.vocab.unkId, equals(tokenizer.vocab.unkId));
      expect(restored.vocab.bosId, equals(tokenizer.vocab.bosId));
      expect(restored.vocab.eosId, equals(tokenizer.vocab.eosId));
    });

    test('config override during load', () {
      final json = tokenizer.toJson();
      final restored = TokenizerJsonLoader.fromJsonString(
        json,
        config: SentencePieceConfig.gemma,
      );

      expect(restored.config.addBosToken, isTrue);
      expect(restored.config.addEosToken, isTrue);
    });

    test('saveToJsonSync and fromJsonFileSync work', () {
      final tempFile = File('${Directory.systemTemp.path}/test_tokenizer.json');
      try {
        tokenizer.saveToJsonSync(tempFile.path);
        expect(tempFile.existsSync(), isTrue);

        final restored = TokenizerJsonLoader.fromJsonFileSync(tempFile.path);
        expect(restored.vocabSize, equals(tokenizer.vocabSize));
      } finally {
        if (tempFile.existsSync()) {
          tempFile.deleteSync();
        }
      }
    });

    test('saveToJson and fromJsonFile work async', () async {
      final tempFile = File(
        '${Directory.systemTemp.path}/test_tokenizer_async.json',
      );
      try {
        await tokenizer.saveToJson(tempFile.path);
        expect(tempFile.existsSync(), isTrue);

        final restored = await TokenizerJsonLoader.fromJsonFile(tempFile.path);
        expect(restored.vocabSize, equals(tokenizer.vocabSize));
      } finally {
        if (tempFile.existsSync()) {
          tempFile.deleteSync();
        }
      }
    });
  });

  group('Token Management', () {
    late SentencePieceTokenizer tokenizer;

    setUp(() {
      tokenizer = createTestTokenizer();
    });

    test('addTokens adds new tokens', () {
      final originalSize = tokenizer.vocabSize;
      final added = tokenizer.addTokens(['<custom1>', '<custom2>']);

      expect(added, equals(2));
      expect(tokenizer.vocabSize, equals(originalSize + 2));
    });

    test('addTokens skips duplicates', () {
      tokenizer.addTokens(['<custom>']);
      final sizeAfterFirst = tokenizer.vocabSize;

      final added = tokenizer.addTokens(['<custom>', '<new>']);

      expect(added, equals(1)); // Only <new> was added
      expect(tokenizer.vocabSize, equals(sizeAfterFirst + 1));
    });

    test('addTokens skips existing vocab tokens', () {
      // '▁hello' should already exist in test vocab
      final originalSize = tokenizer.vocabSize;
      final added = tokenizer.addTokens(['▁hello']);

      expect(added, equals(0));
      expect(tokenizer.vocabSize, equals(originalSize));
    });

    test('added tokens are recognized during encoding', () {
      tokenizer.addTokens(['<CUSTOM>']);

      final encoding = tokenizer.encode(
        'test <CUSTOM> text',
        addSpecialTokens: false,
      );

      expect(encoding.tokens, contains('<CUSTOM>'));
    });

    test('addSpecialTokens adds special tokens', () {
      final originalSize = tokenizer.vocabSize;
      final added = tokenizer.addSpecialTokens({'mask_token': '<mask>'});

      expect(added, equals(1));
      expect(tokenizer.vocabSize, equals(originalSize + 1));
      expect(
        tokenizer.vocab.isSpecialToken(tokenizer.vocab.pieceToId('<mask>')),
        isTrue,
      );
    });

    test('addSpecialTokens updates pad_token reference', () {
      tokenizer.addSpecialTokens({'pad_token': '<PAD>'});

      final padId = tokenizer.vocab.padId;
      expect(padId, greaterThanOrEqualTo(0));
      expect(tokenizer.vocab.padPiece, equals('<PAD>'));
    });

    test('getAddedVocab returns only added tokens', () {
      tokenizer.addTokens(['<added1>', '<added2>']);

      final addedVocab = tokenizer.getAddedVocab();

      expect(addedVocab.length, equals(2));
      expect(addedVocab.keys, containsAll(['<added1>', '<added2>']));
    });

    test('isAddedToken identifies added tokens', () {
      tokenizer.addTokens(['<mytoken>']);

      expect(tokenizer.isAddedToken('<mytoken>'), isTrue);
      expect(tokenizer.isAddedToken('hello'), isFalse);
    });

    test('getVocab returns full vocabulary', () {
      final vocab = tokenizer.getVocab();

      expect(vocab.length, equals(tokenizer.vocabSize));
    });

    test('getVocab without added tokens excludes them', () {
      final originalSize = tokenizer.vocabSize;
      tokenizer.addTokens(['<extra>']);

      final vocabWithAdded = tokenizer.getVocab(withAddedTokens: true);
      final vocabWithoutAdded = tokenizer.getVocab(withAddedTokens: false);

      expect(vocabWithAdded.length, equals(originalSize + 1));
      expect(vocabWithoutAdded.length, equals(originalSize));
      expect(vocabWithoutAdded.containsKey('<extra>'), isFalse);
    });
  });

  group('tokenize() Method', () {
    late SentencePieceTokenizer tokenizer;

    setUpAll(() {
      tokenizer = createTestTokenizer();
    });

    test('tokenize returns token strings', () {
      final tokens = tokenizer.tokenize('hello world');

      expect(tokens, isA<List<String>>());
      expect(tokens, isNotEmpty);
    });

    test('tokenize matches encode tokens', () {
      const text = 'hello world';
      final tokens = tokenizer.tokenize(text);
      final encoding = tokenizer.encode(text, addSpecialTokens: false);

      expect(tokens, equals(encoding.tokens));
    });

    test('tokenize returns empty for empty string', () {
      final tokens = tokenizer.tokenize('');

      expect(tokens, isEmpty);
    });

    test('tokenize throws for too long input', () {
      final longText = 'a' * 600000;

      expect(() => tokenizer.tokenize(longText), throwsArgumentError);
    });

    test('tokenizeBatch processes multiple texts', () {
      final texts = ['hello', 'world', 'test'];
      final results = tokenizer.tokenizeBatch(texts);

      expect(results.length, equals(3));
      for (var i = 0; i < texts.length; i++) {
        expect(results[i], equals(tokenizer.tokenize(texts[i])));
      }
    });
  });

  group('Vocabulary Access', () {
    late SentencePieceTokenizer tokenizer;

    setUpAll(() {
      tokenizer = createTestTokenizer();
    });

    test('vocabSize returns correct size', () {
      expect(tokenizer.vocabSize, greaterThan(0));
      expect(tokenizer.vocabSize, equals(tokenizer.vocab.size));
    });

    test('vocab.vocabularyMap is accessible', () {
      final vocabMap = tokenizer.vocab.vocabularyMap;

      expect(vocabMap, isNotEmpty);
      expect(vocabMap.length, equals(tokenizer.vocabSize));
    });

    test('convertTokensToIds and convertIdsToTokens are consistent', () {
      final encoding = tokenizer.encode('test', addSpecialTokens: false);
      final tokens = encoding.tokens;

      final ids = tokenizer.convertTokensToIds(tokens);
      final backToTokens = tokenizer.convertIdsToTokens(ids);

      expect(ids, equals(encoding.ids.toList()));
      expect(backToTokens, equals(tokens));
    });
  });

  group('JSON Serialization with Added Tokens', () {
    test('added tokens are preserved in JSON round-trip', () {
      final tokenizer = createTestTokenizer();
      tokenizer.addTokens(['<custom1>', '<custom2>']);
      tokenizer.addSpecialTokens({'mask_token': '<mask>'});

      final json = tokenizer.toJson();
      final restored = TokenizerJsonLoader.fromJsonString(json);

      expect(restored.vocabSize, equals(tokenizer.vocabSize));

      // Verify tokens work
      final encoding = restored.encode(
        'test <custom1>',
        addSpecialTokens: false,
      );
      expect(encoding.tokens, contains('<custom1>'));
    });
  });
}
