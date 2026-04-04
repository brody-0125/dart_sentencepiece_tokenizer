import 'dart:convert';

import 'package:test/test.dart';
import 'package:dart_sentencepiece_tokenizer/dart_sentencepiece_tokenizer.dart';

const _defaultAddedTokens = [
  {
    'id': 0,
    'content': '<unk>',
    'single_word': false,
    'lstrip': false,
    'rstrip': false,
    'normalized': false,
    'special': true,
  },
  {
    'id': 1,
    'content': '<s>',
    'single_word': false,
    'lstrip': false,
    'rstrip': false,
    'normalized': false,
    'special': true,
  },
  {
    'id': 2,
    'content': '</s>',
    'single_word': false,
    'lstrip': false,
    'rstrip': false,
    'normalized': false,
    'special': true,
  },
];

const _defaultNormalizer = {
  'type': 'Sequence',
  'normalizers': [
    {'type': 'Prepend', 'prepend': '\u2581'},
    {
      'type': 'Replace',
      'pattern': {'Regex': ' '},
      'content': '\u2581',
    },
  ],
};

const _defaultDecoder = {
  'type': 'Sequence',
  'decoders': [
    {
      'type': 'Replace',
      'pattern': {'String': '\u2581'},
      'content': ' ',
    },
    {'type': 'ByteFallback'},
    {'type': 'Strip', 'content': ' ', 'start': 1, 'stop': 0},
  ],
};

/// Helper to build a minimal HF tokenizer.json map with shared defaults.
Map<String, dynamic> _buildHfJson({
  required Map<String, dynamic> model,
  List<Map<String, dynamic>>? addedTokens,
  Map<String, dynamic>? normalizer,
  Map<String, dynamic>? decoder,
  Map<String, dynamic>? postProcessor,
}) {
  return {
    'version': '1.0',
    'truncation': null,
    'padding': null,
    'added_tokens': addedTokens ?? _defaultAddedTokens,
    'normalizer': normalizer ?? _defaultNormalizer,
    'pre_tokenizer': null,
    'post_processor': postProcessor,
    'decoder': decoder ?? _defaultDecoder,
    'model': model,
  };
}

Map<String, dynamic> _buildHfUnigramJson({
  List<List<dynamic>>? vocab,
  int unkId = 0,
  List<Map<String, dynamic>>? addedTokens,
  Map<String, dynamic>? normalizer,
  Map<String, dynamic>? decoder,
  Map<String, dynamic>? postProcessor,
}) {
  return _buildHfJson(
    addedTokens: addedTokens,
    normalizer: normalizer,
    decoder: decoder,
    postProcessor: postProcessor,
    model: {
      'type': 'Unigram',
      'unk_id': unkId,
      'vocab':
          vocab ??
          [
            ['<unk>', 0.0],
            ['<s>', 0.0],
            ['</s>', 0.0],
            ['\u2581', -1.0],
            ['\u2581Hello', -2.5],
            ['Hello', -3.0],
            ['\u2581world', -2.5],
            ['world', -3.5],
            ['o', -4.0],
            ['l', -4.0],
            ['H', -4.5],
            ['e', -4.0],
            ['r', -4.5],
            ['d', -4.5],
            ['w', -4.5],
            ['<0x00>', -100.0],
            ['<0x01>', -100.0],
            ['<0x02>', -100.0],
          ],
    },
  );
}

Map<String, dynamic> _buildHfBpeJson({
  Map<String, int>? vocab,
  List<String>? merges,
  bool byteFallback = true,
  List<Map<String, dynamic>>? addedTokens,
  Map<String, dynamic>? normalizer,
  Map<String, dynamic>? decoder,
  Map<String, dynamic>? postProcessor,
}) {
  return _buildHfJson(
    addedTokens: addedTokens,
    normalizer: normalizer,
    decoder: decoder,
    postProcessor: postProcessor,
    model: {
      'type': 'BPE',
      'dropout': null,
      'unk_token': '<unk>',
      'continuing_subword_prefix': null,
      'end_of_word_suffix': null,
      'fuse_unk': false,
      'byte_fallback': byteFallback,
      'vocab':
          vocab ??
          {
            '<unk>': 0,
            '<s>': 1,
            '</s>': 2,
            '\u2581': 3,
            'H': 4,
            'e': 5,
            'l': 6,
            'o': 7,
            'He': 8,
            'll': 9,
            'Hell': 10,
            'Hello': 11,
            '\u2581Hello': 12,
          },
      'merges': merges ?? ['H e', 'l l', 'He ll', 'Hell o', '\u2581 Hello'],
    },
  );
}

void main() {
  group('HuggingFaceTokenizerLoader', () {
    group('Unigram model parsing', () {
      test('loads minimal Unigram tokenizer', () {
        final json = jsonEncode(_buildHfUnigramJson());
        final tokenizer = HuggingFaceTokenizerLoader.fromJsonString(json);

        expect(tokenizer.vocab.size, 18);
        expect(tokenizer.modelType, ModelType.unigram);
      });

      test('correctly identifies special tokens', () {
        final json = jsonEncode(_buildHfUnigramJson());
        final tokenizer = HuggingFaceTokenizerLoader.fromJsonString(json);

        expect(tokenizer.vocab.unkId, 0);
        expect(tokenizer.vocab.bosId, 1);
        expect(tokenizer.vocab.eosId, 2);
        expect(tokenizer.vocab.unkPiece, '<unk>');
        expect(tokenizer.vocab.bosPiece, '<s>');
        expect(tokenizer.vocab.eosPiece, '</s>');
      });

      test('correctly identifies byte tokens', () {
        final json = jsonEncode(_buildHfUnigramJson());
        final tokenizer = HuggingFaceTokenizerLoader.fromJsonString(json);

        expect(tokenizer.vocab.isByteToken(15), isTrue); // <0x00>
        expect(tokenizer.vocab.isByteToken(16), isTrue); // <0x01>
        expect(tokenizer.vocab.isByteToken(0), isFalse); // <unk>
      });

      test('preserves vocabulary scores', () {
        final json = jsonEncode(_buildHfUnigramJson());
        final tokenizer = HuggingFaceTokenizerLoader.fromJsonString(json);

        // Check a normal token score.
        final helloId = tokenizer.vocab.pieceToId('\u2581Hello');
        expect(tokenizer.vocab.getScore(helloId), closeTo(-2.5, 0.01));
      });

      test('detects byte_fallback from decoder', () {
        final json = jsonEncode(_buildHfUnigramJson());
        final tokenizer = HuggingFaceTokenizerLoader.fromJsonString(json);

        expect(tokenizer.vocab.hasByteFallback, isTrue);
      });

      test('no byte_fallback when decoder has no ByteFallback', () {
        final json = jsonEncode(
          _buildHfUnigramJson(
            decoder: {
              'type': 'Sequence',
              'decoders': [
                {
                  'type': 'Replace',
                  'pattern': {'String': '\u2581'},
                  'content': ' ',
                },
              ],
            },
          ),
        );
        final tokenizer = HuggingFaceTokenizerLoader.fromJsonString(json);

        expect(tokenizer.vocab.hasByteFallback, isFalse);
      });

      test('respects explicit config override', () {
        final json = jsonEncode(_buildHfUnigramJson());
        final tokenizer = HuggingFaceTokenizerLoader.fromJsonString(
          json,
          config: SentencePieceConfig.gemma,
        );

        expect(tokenizer.config.addBosToken, isTrue);
        expect(tokenizer.config.addEosToken, isTrue);
      });
    });

    group('BPE model parsing', () {
      test('loads minimal BPE tokenizer', () {
        final json = jsonEncode(_buildHfBpeJson());
        final tokenizer = HuggingFaceTokenizerLoader.fromJsonString(json);

        expect(tokenizer.vocab.size, 13);
        expect(tokenizer.modelType, ModelType.bpe);
      });

      test('assigns scores based on merge order', () {
        final json = jsonEncode(_buildHfBpeJson());
        final tokenizer = HuggingFaceTokenizerLoader.fromJsonString(json);

        // "He" is merge index 0, should have score -0.0
        final heId = tokenizer.vocab.pieceToId('He');
        final heScore = tokenizer.vocab.getScore(heId);

        // "ll" is merge index 1, should have score -1.0
        final llId = tokenizer.vocab.pieceToId('ll');
        final llScore = tokenizer.vocab.getScore(llId);

        // Earlier merges should have higher scores.
        expect(heScore, greaterThan(llScore));
      });

      test('base tokens get lowest scores', () {
        final json = jsonEncode(_buildHfBpeJson());
        final tokenizer = HuggingFaceTokenizerLoader.fromJsonString(json);

        // 'H' is a base character, not a merge result.
        final hId = tokenizer.vocab.pieceToId('H');
        final hScore = tokenizer.vocab.getScore(hId);

        // 'He' is a merge result.
        final heId = tokenizer.vocab.pieceToId('He');
        final heScore = tokenizer.vocab.getScore(heId);

        expect(hScore, lessThan(heScore));
      });

      test('correctly detects byte_fallback from model data', () {
        final json = jsonEncode(_buildHfBpeJson(byteFallback: true));
        final tokenizer = HuggingFaceTokenizerLoader.fromJsonString(json);

        expect(tokenizer.vocab.hasByteFallback, isTrue);
      });

      test('correctly identifies unk token', () {
        final json = jsonEncode(_buildHfBpeJson());
        final tokenizer = HuggingFaceTokenizerLoader.fromJsonString(json);

        expect(tokenizer.vocab.unkId, 0);
        expect(tokenizer.vocab.unkPiece, '<unk>');
      });

      test('handles empty merges list', () {
        final json = jsonEncode(
          _buildHfBpeJson(
            vocab: {'<unk>': 0, '<s>': 1, '</s>': 2, 'a': 3, 'b': 4},
            merges: [],
          ),
        );
        final tokenizer = HuggingFaceTokenizerLoader.fromJsonString(json);

        expect(tokenizer.vocab.size, 5);
        expect(tokenizer.modelType, ModelType.bpe);
      });
    });

    group('normalizer inference', () {
      test('detects addDummyPrefix from Prepend normalizer', () {
        final json = jsonEncode(
          _buildHfUnigramJson(
            normalizer: {'type': 'Prepend', 'prepend': '\u2581'},
          ),
        );
        final tokenizer = HuggingFaceTokenizerLoader.fromJsonString(json);

        expect(tokenizer.normalizer.addDummyPrefix, isTrue);
      });

      test('detects escapeWhitespaces from Replace normalizer', () {
        final json = jsonEncode(
          _buildHfUnigramJson(
            normalizer: {
              'type': 'Replace',
              'pattern': {'Regex': ' '},
              'content': '\u2581',
            },
          ),
        );
        final tokenizer = HuggingFaceTokenizerLoader.fromJsonString(json);

        expect(tokenizer.normalizer.escapeWhitespaces, isTrue);
      });

      test('handles null normalizer with defaults', () {
        final json = jsonEncode(_buildHfUnigramJson(normalizer: null));
        // Need to set it to null explicitly in the map.
        final data = jsonDecode(json) as Map<String, dynamic>;
        data['normalizer'] = null;
        final tokenizer = HuggingFaceTokenizerLoader.fromMap(data);

        // Default when null: addDummyPrefix=true, escapeWhitespaces=true
        expect(tokenizer.normalizer.addDummyPrefix, isTrue);
        expect(tokenizer.normalizer.escapeWhitespaces, isTrue);
      });
    });

    group('post_processor config inference', () {
      test('detects addBosToken from TemplateProcessing', () {
        final json = jsonEncode(
          _buildHfUnigramJson(
            postProcessor: {
              'type': 'TemplateProcessing',
              'single': [
                {
                  'SpecialToken': {'id': '<s>', 'type_id': 0},
                },
                {
                  'Sequence': {'id': 'A', 'type_id': 0},
                },
              ],
              'pair': [],
              'special_tokens': {},
            },
          ),
        );
        final tokenizer = HuggingFaceTokenizerLoader.fromJsonString(json);

        expect(tokenizer.config.addBosToken, isTrue);
        expect(tokenizer.config.addEosToken, isFalse);
      });

      test('detects addBosToken and addEosToken', () {
        final json = jsonEncode(
          _buildHfUnigramJson(
            postProcessor: {
              'type': 'TemplateProcessing',
              'single': [
                {
                  'SpecialToken': {'id': '<s>', 'type_id': 0},
                },
                {
                  'Sequence': {'id': 'A', 'type_id': 0},
                },
                {
                  'SpecialToken': {'id': '</s>', 'type_id': 0},
                },
              ],
              'pair': [],
              'special_tokens': {},
            },
          ),
        );
        final tokenizer = HuggingFaceTokenizerLoader.fromJsonString(json);

        expect(tokenizer.config.addBosToken, isTrue);
        expect(tokenizer.config.addEosToken, isTrue);
      });

      test('no special tokens inferred without post_processor', () {
        final json = jsonEncode(_buildHfUnigramJson(postProcessor: null));
        final data = jsonDecode(json) as Map<String, dynamic>;
        data['post_processor'] = null;
        final tokenizer = HuggingFaceTokenizerLoader.fromMap(data);

        expect(tokenizer.config.addBosToken, isFalse);
        expect(tokenizer.config.addEosToken, isFalse);
      });
    });

    group('added_tokens handling', () {
      test('extra added_tokens beyond base vocab are added', () {
        final json = jsonEncode(
          _buildHfUnigramJson(
            addedTokens: [
              {'id': 0, 'content': '<unk>', 'special': true},
              {'id': 1, 'content': '<s>', 'special': true},
              {'id': 2, 'content': '</s>', 'special': true},
              {'id': 99999, 'content': '<mask>', 'special': true},
            ],
          ),
        );
        final tokenizer = HuggingFaceTokenizerLoader.fromJsonString(json);

        // <mask> should be added as a special token.
        expect(tokenizer.vocab.contains('<mask>'), isTrue);
      });
    });

    group('auto-detection via TokenizerJsonLoader', () {
      test('HF format is auto-detected', () {
        final json = jsonEncode(_buildHfUnigramJson());
        final tokenizer = TokenizerJsonLoader.fromJsonString(json);

        expect(tokenizer.modelType, ModelType.unigram);
        expect(tokenizer.vocab.size, 18);
      });

      test('isHuggingFaceFormat returns true for HF format', () {
        final data = _buildHfUnigramJson();
        expect(TokenizerJsonLoader.isHuggingFaceFormat(data), isTrue);
      });

      test('isHuggingFaceFormat returns false for custom format', () {
        final data = {
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
            'eos': {'id': 2, 'piece': '</s>'},
            'pad': {'id': -1, 'piece': '<pad>'},
          },
          'normalizer': {
            'add_dummy_prefix': true,
            'remove_extra_whitespaces': true,
            'escape_whitespaces': true,
          },
        };
        expect(TokenizerJsonLoader.isHuggingFaceFormat(data), isFalse);
      });
    });

    group('error handling', () {
      test('missing model section throws FormatException', () {
        final json = jsonEncode({'version': '1.0', 'added_tokens': []});

        expect(
          () => HuggingFaceTokenizerLoader.fromJsonString(json),
          throwsFormatException,
        );
      });

      test('missing model type throws FormatException', () {
        final json = jsonEncode({
          'version': '1.0',
          'added_tokens': [],
          'model': {'vocab': []},
        });

        expect(
          () => HuggingFaceTokenizerLoader.fromJsonString(json),
          throwsFormatException,
        );
      });

      test('unsupported model type throws UnsupportedError', () {
        final json = jsonEncode({
          'version': '1.0',
          'added_tokens': [],
          'model': {'type': 'WordPiece', 'vocab': {}},
        });

        expect(
          () => HuggingFaceTokenizerLoader.fromJsonString(json),
          throwsA(isA<UnsupportedError>()),
        );
      });

      test('missing Unigram vocab throws FormatException', () {
        final json = jsonEncode({
          'version': '1.0',
          'added_tokens': [],
          'model': {'type': 'Unigram', 'unk_id': 0},
        });

        expect(
          () => HuggingFaceTokenizerLoader.fromJsonString(json),
          throwsFormatException,
        );
      });

      test('missing BPE vocab throws FormatException', () {
        final json = jsonEncode({
          'version': '1.0',
          'added_tokens': [],
          'model': {'type': 'BPE', 'merges': []},
        });

        expect(
          () => HuggingFaceTokenizerLoader.fromJsonString(json),
          throwsFormatException,
        );
      });
    });
  });
}
