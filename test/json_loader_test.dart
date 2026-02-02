import 'dart:convert';

import 'package:test/test.dart';
import 'package:dart_sentencepiece_tokenizer/dart_sentencepiece_tokenizer.dart';

void main() {
  group('TokenizerJsonLoader error handling', () {
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
