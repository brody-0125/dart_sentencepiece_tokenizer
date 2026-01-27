/// v1.2.1 improvement tests.
library;

import 'dart:convert';

import 'package:test/test.dart';
import 'package:dart_sentencepiece_tokenizer/dart_sentencepiece_tokenizer.dart';

import 'test_utils.dart';

void main() {
  group('Batch addTokens optimization', () {
    late SentencePieceTokenizer tokenizer;

    setUp(() {
      tokenizer = createTestTokenizer();
    });

    test('adding many tokens at once works correctly', () {
      final tokens = List.generate(100, (i) => '<batch_$i>');
      final originalSize = tokenizer.vocabSize;

      final added = tokenizer.addTokens(tokens);

      expect(added, equals(100));
      expect(tokenizer.vocabSize, equals(originalSize + 100));

      // Verify each token is accessible
      for (var i = 0; i < 100; i++) {
        final token = '<batch_$i>';
        expect(tokenizer.vocab.contains(token), isTrue);
        expect(tokenizer.isAddedToken(token), isTrue);
      }
    });

    test('batch add with mixed duplicates works correctly', () {
      tokenizer.addTokens(['<existing>']);
      final sizeAfterFirst = tokenizer.vocabSize;

      final tokens = ['<existing>', '<new1>', '<new2>', '<existing>'];
      final added = tokenizer.addTokens(tokens);

      expect(added, equals(2));
      expect(tokenizer.vocabSize, equals(sizeAfterFirst + 2));
    });

    test('batch add with all duplicates returns zero', () {
      tokenizer.addTokens(['<a>', '<b>']);
      final size = tokenizer.vocabSize;

      final added = tokenizer.addTokens(['<a>', '<b>']);

      expect(added, equals(0));
      expect(tokenizer.vocabSize, equals(size));
    });

    test('added tokens encode correctly after batch add', () {
      tokenizer.addTokens(['<TAG1>', '<TAG2>', '<TAG3>']);

      final encoding = tokenizer.encode(
        'test <TAG1> and <TAG2>',
        addSpecialTokens: false,
      );

      expect(encoding.tokens, contains('<TAG1>'));
      expect(encoding.tokens, contains('<TAG2>'));
    });
  });

  group('BPE merge optimization', () {
    test('BPE tokenization produces consistent results', () {
      final tokenizer = createTestTokenizer();

      // Run multiple times to verify determinism
      const text = 'hello world test';
      final first = tokenizer.encode(text, addSpecialTokens: false);
      final second = tokenizer.encode(text, addSpecialTokens: false);

      expect(first.ids.toList(), equals(second.ids.toList()));
      expect(first.tokens, equals(second.tokens));
    });
  });

  group('encode() input validation', () {
    late SentencePieceTokenizer tokenizer;

    setUpAll(() {
      tokenizer = createTestTokenizer();
    });

    test('encode throws for too long input', () {
      final longText = 'a' * 600000;

      expect(
        () => tokenizer.encode(longText),
        throwsArgumentError,
      );
    });

    test('encode accepts input at max length', () {
      // Should not throw for exactly 500000 chars
      final text = 'a' * 500000;

      expect(
        () => tokenizer.encode(text),
        returnsNormally,
      );
    });

    test('encodePair throws for too long first text', () {
      final longText = 'a' * 600000;

      expect(
        () => tokenizer.encodePair(longText, 'short'),
        throwsArgumentError,
      );
    });

    test('encodePair throws for too long second text', () {
      final longText = 'a' * 600000;

      expect(
        () => tokenizer.encodePair('short', longText),
        throwsArgumentError,
      );
    });
  });

  group('JSON deserialization error handling', () {
    test('missing version throws FormatException', () {
      final json = jsonEncode({'model_type': 'unigram'});

      expect(
        () => TokenizerJsonLoader.fromJsonString(json),
        throwsFormatException,
      );
    });

    test('missing model_type throws FormatException', () {
      final json = jsonEncode({'version': '1.0'});

      expect(
        () => TokenizerJsonLoader.fromJsonString(json),
        throwsFormatException,
      );
    });

    test('missing vocab throws FormatException', () {
      final json = jsonEncode({
        'version': '1.0',
        'model_type': 'unigram',
      });

      expect(
        () => TokenizerJsonLoader.fromJsonString(json),
        throwsFormatException,
      );
    });

    test('missing vocab fields throws FormatException', () {
      final json = jsonEncode({
        'version': '1.0',
        'model_type': 'unigram',
        'vocab': {'pieces': ['a']},
      });

      expect(
        () => TokenizerJsonLoader.fromJsonString(json),
        throwsFormatException,
      );
    });

    test('inconsistent vocab array lengths throws FormatException', () {
      final json = jsonEncode({
        'version': '1.0',
        'model_type': 'unigram',
        'vocab': {
          'pieces': ['a', 'b'],
          'scores': [1.0],
          'types': [1, 2],
        },
        'special_tokens': {
          'unk': {'id': 0, 'piece': '<unk>'},
          'bos': {'id': 1, 'piece': '<s>'},
          'eos': {'id': 2, 'piece': '</s>'},
          'pad': {'id': -1, 'piece': '<pad>'},
        },
      });

      expect(
        () => TokenizerJsonLoader.fromJsonString(json),
        throwsFormatException,
      );
    });

    test('missing special_tokens throws FormatException', () {
      final json = jsonEncode({
        'version': '1.0',
        'model_type': 'unigram',
        'vocab': {
          'pieces': ['a'],
          'scores': [1.0],
          'types': [1],
        },
      });

      expect(
        () => TokenizerJsonLoader.fromJsonString(json),
        throwsFormatException,
      );
    });

    test('missing individual special token throws FormatException', () {
      final json = jsonEncode({
        'version': '1.0',
        'model_type': 'unigram',
        'vocab': {
          'pieces': ['a'],
          'scores': [1.0],
          'types': [1],
        },
        'special_tokens': {
          'unk': {'id': 0, 'piece': '<unk>'},
          'bos': {'id': 1, 'piece': '<s>'},
          // missing eos and pad
        },
      });

      expect(
        () => TokenizerJsonLoader.fromJsonString(json),
        throwsFormatException,
      );
    });
  });
}
