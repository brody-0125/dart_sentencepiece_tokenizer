import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:dart_sentencepiece_tokenizer/dart_sentencepiece_tokenizer.dart';
import 'package:dart_sentencepiece_tokenizer/src/sentencepiece/model/model_proto.dart';
import 'package:test/test.dart';

import 'precompiled_charsmap_builder.dart';
import 'test_utils.dart';

const _emptyNormalizer = {'type': 'Sequence', 'normalizers': <dynamic>[]};

Map<String, dynamic> _xlmTokenizerJson({
  required List<String> pieces,
  required Map<String, dynamic> normalizer,
  Map<String, dynamic>? preTokenizer,
  Map<String, dynamic>? postProcessor,
  List<Map<String, dynamic>>? addedTokens,
}) {
  return {
    'version': '1.0',
    'added_tokens':
        addedTokens ??
        [
          {'id': 0, 'content': '<s>', 'special': true},
          {'id': 1, 'content': '<pad>', 'special': true},
          {'id': 2, 'content': '</s>', 'special': true},
          {'id': 3, 'content': '<unk>', 'special': true},
        ],
    'normalizer': normalizer,
    'pre_tokenizer': preTokenizer,
    'post_processor': postProcessor,
    'decoder': null,
    'model': {
      'type': 'Unigram',
      'unk_id': 3,
      'vocab': [
        for (var i = 0; i < pieces.length; i++) [pieces[i], -i.toDouble()],
      ],
    },
  };
}

Map<String, dynamic> _xlmTemplateProcessor() {
  return {
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
    'special_tokens': {
      '<s>': [0, '<s>'],
      '</s>': [2, '</s>'],
    },
  };
}

void _expectUnsupportedSplit({
  required Object pattern,
  required String behavior,
  required bool invert,
}) {
  final json = _xlmTokenizerJson(
    pieces: ['<s>', '<pad>', '</s>', '<unk>', 'a'],
    normalizer: _emptyNormalizer,
    preTokenizer: {
      'type': 'Split',
      'pattern': pattern,
      'behavior': behavior,
      'invert': invert,
    },
  );

  expect(
    () => HuggingFaceTokenizerLoader.fromMap(json),
    throwsA(isA<UnsupportedError>()),
  );
}

void main() {
  group('tokenizer.json acceptance coverage', () {
    test(
      'extracts a direct Precompiled normalizer and applies its charsmap',
      () {
        final charsmap = buildPrecompiledCharsmapBlob({'A': 'a'});
        final json = _xlmTokenizerJson(
          pieces: ['<s>', '<pad>', '</s>', '<unk>', 'a', 'A'],
          normalizer: {
            'type': 'Precompiled',
            'precompiled_charsmap': base64Encode(charsmap),
          },
        );

        final tokenizer = HuggingFaceTokenizerLoader.fromMap(json);

        expect(tokenizer.tokenize('A'), ['a']);
        expect(tokenizer.convertTokensToIds(tokenizer.tokenize('A')), [4]);

        final restored = TokenizerJsonLoader.fromJsonString(tokenizer.toJson());
        expect(restored.tokenize('A'), ['a']);
      },
    );

    test('extracts a nested Precompiled normalizer from Sequence', () {
      final charsmap = buildPrecompiledCharsmapBlob({'A': 'a'});
      final json = _xlmTokenizerJson(
        pieces: ['<s>', '<pad>', '</s>', '<unk>', 'a', 'A'],
        normalizer: {
          'type': 'Sequence',
          'normalizers': [
            {
              'type': 'Precompiled',
              'precompiled_charsmap': base64Encode(charsmap),
            },
          ],
        },
      );

      final tokenizer = HuggingFaceTokenizerLoader.fromMap(json);

      expect(tokenizer.tokenize('A'), ['a']);
      expect(tokenizer.convertTokensToIds(tokenizer.tokenize('A')), [4]);
    });

    test('auto-detects a type-less Unigram tokenizer.json', () {
      final charsmap = buildPrecompiledCharsmapBlob({'A': 'a'});
      final json = _xlmTokenizerJson(
        pieces: ['<s>', '<pad>', '</s>', '<unk>', 'a'],
        normalizer: {
          'type': 'Precompiled',
          'precompiled_charsmap': base64Encode(charsmap),
        },
      );
      (json['model'] as Map<String, dynamic>).remove('type');

      final tokenizer = TokenizerJsonLoader.fromJsonString(jsonEncode(json));

      expect(tokenizer.tokenize('A'), ['a']);
    });

    test('preserves normalizer order: Precompiled then Replace', () {
      final charsmap = buildPrecompiledCharsmapBlob({'A': 'a'});
      final json = _xlmTokenizerJson(
        pieces: ['<s>', '<pad>', '</s>', '<unk>', 'b', 'a', 'A'],
        normalizer: {
          'type': 'Sequence',
          'normalizers': [
            {
              'type': 'Precompiled',
              'precompiled_charsmap': base64Encode(charsmap),
            },
            {
              'type': 'Replace',
              'pattern': {'Regex': 'a'},
              'content': 'b',
            },
          ],
        },
      );

      final tokenizer = HuggingFaceTokenizerLoader.fromMap(json);

      expect(tokenizer.tokenize('A'), ['b']);
      expect(tokenizer.convertTokensToIds(tokenizer.tokenize('A')), [4]);
    });

    test('applies a direct Metaspace pre-tokenizer with prefix space', () {
      final json = _xlmTokenizerJson(
        pieces: ['<s>', '<pad>', '</s>', '<unk>', '▁hello', 'hello'],
        normalizer: _emptyNormalizer,
        preTokenizer: {
          'type': 'Metaspace',
          'replacement': '▁',
          'add_prefix_space': true,
        },
      );

      final tokenizer = HuggingFaceTokenizerLoader.fromMap(json);

      expect(tokenizer.tokenize('hello'), ['▁hello']);
      expect(tokenizer.convertTokensToIds(tokenizer.tokenize('hello')), [4]);
    });

    test('supports current Metaspace prepend_scheme serialization', () {
      final json = _xlmTokenizerJson(
        pieces: ['<s>', '<pad>', '</s>', '<unk>', 'hello', '▁world'],
        normalizer: _emptyNormalizer,
        preTokenizer: {
          'type': 'Metaspace',
          'replacement': '▁',
          'prepend_scheme': 'never',
          'split': true,
        },
      );

      final tokenizer = HuggingFaceTokenizerLoader.fromMap(json);

      expect(tokenizer.tokenize('hello world'), ['hello', '▁world']);
    });

    test('applies WhitespaceSplit then Metaspace to all whitespace', () {
      final json = _xlmTokenizerJson(
        pieces: ['<s>', '<pad>', '</s>', '<unk>', '▁hello', '▁world', '▁again'],
        normalizer: _emptyNormalizer,
        preTokenizer: {
          'type': 'Sequence',
          'pretokenizers': [
            {'type': 'WhitespaceSplit'},
            {'type': 'Metaspace', 'replacement': '▁', 'add_prefix_space': true},
          ],
        },
      );

      final tokenizer = HuggingFaceTokenizerLoader.fromMap(json);

      expect(
        tokenizer.convertTokensToIds(tokenizer.tokenize('hello\tworld\nagain')),
        [4, 5, 6],
      );
    });

    test('preserves the JSON pipeline in parallel batch encoding', () async {
      final charsmap = buildPrecompiledCharsmapBlob({'A': 'a'});
      final json = _xlmTokenizerJson(
        pieces: ['<s>', '<pad>', '</s>', '<unk>', '▁a'],
        normalizer: {
          'type': 'Precompiled',
          'precompiled_charsmap': base64Encode(charsmap),
        },
        preTokenizer: {
          'type': 'Metaspace',
          'replacement': '▁',
          'add_prefix_space': true,
        },
      );

      final tokenizer = HuggingFaceTokenizerLoader.fromMap(json);
      final texts = List<String>.filled(8, 'A');
      final sequential = tokenizer.encodeBatch(texts, addSpecialTokens: false);
      final parallel = await tokenizer.encodeBatchParallel(
        texts,
        addSpecialTokens: false,
        numWorkers: 2,
      );

      expect(
        parallel.map((encoding) => encoding.ids.toList()).toList(),
        sequential.map((encoding) => encoding.ids.toList()).toList(),
      );
    });

    test('fuses consecutive Unigram OOV characters into one unknown token', () {
      final json = _xlmTokenizerJson(
        pieces: ['<s>', '<pad>', '</s>', '<unk>', 'a'],
        normalizer: _emptyNormalizer,
      );

      final tokenizer = HuggingFaceTokenizerLoader.fromMap(json);

      expect(tokenizer.encode('xyz', addSpecialTokens: false).ids, [3]);
    });

    test(
      'adds XLM-R special tokens from TemplateProcessing with exact IDs',
      () {
        final json = _xlmTokenizerJson(
          pieces: ['<s>', '<pad>', '</s>', '<unk>', 'hello'],
          normalizer: _emptyNormalizer,
          postProcessor: _xlmTemplateProcessor(),
        );

        final tokenizer = HuggingFaceTokenizerLoader.fromMap(json);
        final encoding = tokenizer.encode('hello');

        expect(encoding.ids, [0, 4, 2]);
        expect(encoding.specialTokensMask, [1, 0, 1]);
      },
    );

    test('uses pair template EOS count and type IDs', () {
      final postProcessor = _xlmTemplateProcessor()
        ..['pair'] = [
          {
            'SpecialToken': {'id': '<s>', 'type_id': 0},
          },
          {
            'Sequence': {'id': 'A', 'type_id': 0},
          },
          {
            'SpecialToken': {'id': '</s>', 'type_id': 0},
          },
          {
            'SpecialToken': {'id': '</s>', 'type_id': 0},
          },
          {
            'Sequence': {'id': 'B', 'type_id': 0},
          },
          {
            'SpecialToken': {'id': '</s>', 'type_id': 0},
          },
        ];
      final tokenizer = HuggingFaceTokenizerLoader.fromMap(
        _xlmTokenizerJson(
          pieces: ['<s>', '<pad>', '</s>', '<unk>', 'hello'],
          normalizer: _emptyNormalizer,
          postProcessor: postProcessor,
        ),
      );

      final encoding = tokenizer.encodePair('hello', 'hello');

      expect(encoding.ids, [0, 4, 2, 2, 4, 2]);
      expect(encoding.typeIds, [0, 0, 0, 0, 0, 0]);
      expect(encoding.specialTokensMask, [1, 0, 1, 1, 0, 1]);

      final restored = TokenizerJsonLoader.fromJsonString(tokenizer.toJson());
      final restoredPair = restored.encodePair('hello', 'hello');
      expect(restoredPair.ids, encoding.ids);
      expect(restoredPair.typeIds, encoding.typeIds);
    });

    test('keeps pair sequence and special-token type IDs independent', () {
      final postProcessor = _xlmTemplateProcessor()
        ..['pair'] = [
          {
            'SpecialToken': {'id': '<s>', 'type_id': 0},
          },
          {
            'Sequence': {'id': 'A', 'type_id': 0},
          },
          {
            'SpecialToken': {'id': '</s>', 'type_id': 0},
          },
          {
            'SpecialToken': {'id': '</s>', 'type_id': 0},
          },
          {
            'Sequence': {'id': 'B', 'type_id': 1},
          },
          {
            'SpecialToken': {'id': '</s>', 'type_id': 0},
          },
        ];

      final tokenizer = HuggingFaceTokenizerLoader.fromMap(
        _xlmTokenizerJson(
          pieces: ['<s>', '<pad>', '</s>', '<unk>', 'hello'],
          normalizer: _emptyNormalizer,
          postProcessor: postProcessor,
        ),
      );

      expect(tokenizer.encodePair('hello', 'hello').typeIds, [
        0,
        0,
        0,
        0,
        1,
        0,
      ]);
    });

    test('recognizes added special tokens before model normalization', () {
      final tokenizer = HuggingFaceTokenizerLoader.fromMap(
        _xlmTokenizerJson(
          pieces: ['<s>', '<pad>', '</s>', '<unk>', 'hello'],
          normalizer: _emptyNormalizer,
          postProcessor: _xlmTemplateProcessor(),
        ),
      );

      final encoding = tokenizer.encode('<s>');

      expect(encoding.ids, [0, 0, 2]);
      expect(encoding.tokens, ['<s>', '<s>', '</s>']);
      expect(encoding.offsets, [(0, 0), (0, 3), (0, 0)]);
      expect(encoding.specialTokensMask, [1, 0, 1]);
    });

    test(
      'preserves declared IDs for added tokens regardless of JSON order',
      () {
        final tokenizer = HuggingFaceTokenizerLoader.fromMap(
          _xlmTokenizerJson(
            pieces: ['<unk>', 'a'],
            normalizer: _emptyNormalizer,
            addedTokens: [
              {'id': 3, 'content': '<foo>', 'special': false},
              {'id': 2, 'content': '<bar>', 'special': true},
            ],
          ),
        );

        expect(tokenizer.encode('<foo>', addSpecialTokens: false).ids, [3]);
        expect(tokenizer.encode('<bar>', addSpecialTokens: false).ids, [2]);
      },
    );

    test('honors normalized AddedToken matching', () {
      final charsmap = buildPrecompiledCharsmapBlob({'Ａ': 'A'});
      final tokenizer = HuggingFaceTokenizerLoader.fromMap(
        _xlmTokenizerJson(
          pieces: ['<unk>', 'a'],
          normalizer: {
            'type': 'Precompiled',
            'precompiled_charsmap': base64Encode(charsmap),
          },
          addedTokens: [
            {'id': 5, 'content': 'Ａ', 'special': false, 'normalized': true},
          ],
        ),
      );

      expect(tokenizer.encode('A', addSpecialTokens: false).ids, [5]);
      expect(tokenizer.encode('Ａ', addSpecialTokens: false).ids, [5]);
    });

    test(
      'loads tokenizer.json truncation and batch padding settings',
      () async {
        final json =
            _xlmTokenizerJson(
                pieces: ['<s>', '<pad>', '</s>', '<unk>', 'a'],
                normalizer: _emptyNormalizer,
                postProcessor: _xlmTemplateProcessor(),
              )
              ..['truncation'] = {
                'max_length': 4,
                'stride': 0,
                'strategy': 'LongestFirst',
              }
              ..['padding'] = {
                'strategy': 'BatchLongest',
                'direction': 'Right',
                'pad_to_multiple_of': null,
                'pad_id': 1,
                'pad_type_id': 0,
                'pad_token': '<pad>',
              };

        final tokenizer = HuggingFaceTokenizerLoader.fromMap(json);

        expect(tokenizer.truncation!.maxLength, 4);
        expect(tokenizer.encode('aaaaa').ids.toList(), [0, 4, 4, 2]);
        final batch = tokenizer.encodeBatch(['a', 'aa']);
        expect(batch.map((encoding) => encoding.ids.toList()).toList(), [
          [0, 4, 2, 1],
          [0, 4, 4, 2],
        ]);

        final truncatedBatch = tokenizer.encodeBatch(['a', 'aaaaa']);
        expect(
          truncatedBatch.map((encoding) => encoding.ids.toList()).toList(),
          [
            [0, 4, 2, 1],
            [0, 4, 4, 2],
          ],
        );

        final parallelBatch = await tokenizer.encodeBatchParallel(
          List.filled(8, 'aaaaa'),
        );
        expect(
          parallelBatch.map((encoding) => encoding.ids.toList()).toList(),
          List.filled(8, [0, 4, 4, 2]),
        );
      },
    );

    test('fails on malformed Precompiled base64 instead of ignoring it', () {
      final json = _xlmTokenizerJson(
        pieces: ['<s>', '<pad>', '</s>', '<unk>', 'a'],
        normalizer: {
          'type': 'Precompiled',
          'precompiled_charsmap': 'not-valid-base64',
        },
      );

      expect(
        () => HuggingFaceTokenizerLoader.fromMap(json),
        throwsA(isA<FormatException>()),
      );
    });

    test('rejects Split with a Regex pattern', () {
      _expectUnsupportedSplit(
        pattern: {'Regex': r'\s+'},
        behavior: 'MergedWithPrevious',
        invert: false,
      );
    });

    test('rejects Split with Isolated behavior', () {
      _expectUnsupportedSplit(
        pattern: {'String': ' '},
        behavior: 'Isolated',
        invert: false,
      );
    });

    test('rejects Split with Removed behavior', () {
      _expectUnsupportedSplit(
        pattern: {'String': ' '},
        behavior: 'Removed',
        invert: false,
      );
    });

    test('rejects Split with invert enabled', () {
      _expectUnsupportedSplit(
        pattern: {'String': ' '},
        behavior: 'MergedWithPrevious',
        invert: true,
      );
    });

    test('fails on a malformed decoded Precompiled charsmap', () {
      final json = _xlmTokenizerJson(
        pieces: ['<s>', '<pad>', '</s>', '<unk>', 'a'],
        normalizer: {
          'type': 'Sequence',
          'normalizers': [
            {
              'type': 'Precompiled',
              'precompiled_charsmap': base64Encode(
                Uint8List.fromList([0, 0, 0]),
              ),
            },
          ],
        },
      );

      expect(
        () => HuggingFaceTokenizerLoader.fromMap(json),
        throwsA(isA<FormatException>()),
      );
    });
  });

  group('existing model regression coverage', () {
    test('native model remains loadable and deterministic', () {
      final tokenizer = createTestTokenizer();
      final first = tokenizer.encode('hello world', addSpecialTokens: false);
      final second = tokenizer.encode('hello world', addSpecialTokens: false);

      expect(tokenizer.modelType, ModelType.unigram);
      expect(first.ids, isNotEmpty);
      expect(first.ids, second.ids);
    });

    test('Llama and Gemma special-token configurations remain unchanged', () {
      final llamaTokenizer = createLlamaTokenizer();
      final gemmaTokenizer = createGemmaTokenizer();
      final llama = llamaTokenizer.encode('hello');
      final gemma = gemmaTokenizer.encode('hello');

      expect(llama.ids.first, llamaTokenizer.vocab.bosId);
      expect(llama.specialTokensMask.first, 1);
      expect(gemma.ids.first, gemmaTokenizer.vocab.bosId);
      expect(gemma.ids.last, gemmaTokenizer.vocab.eosId);
      expect(gemma.specialTokensMask.last, 1);
    });

    test('BPE model tokenization remains stable', () {
      const model = SentencePieceModel(
        pieces: [
          SentencePiece(piece: '<unk>', type: PieceType.unknown),
          SentencePiece(piece: 'a', score: -1),
          SentencePiece(piece: 'b', score: -1),
          SentencePiece(piece: 'ab', score: -0.5),
        ],
        trainerSpec: TrainerSpec(
          modelType: ModelType.bpe,
          vocabSize: 4,
          unkId: 0,
          bosId: -1,
          eosId: -1,
          padId: -1,
        ),
        normalizerSpec: NormalizerSpec(
          addDummyPrefix: false,
          removeExtraWhitespaces: false,
          escapeWhitespaces: false,
        ),
      );

      final tokenizer = SentencePieceTokenizer.fromModel(model);

      expect(tokenizer.encode('ab', addSpecialTokens: false).ids, [3]);
    });
  });

  test('matches the real MiniLM tokenizer.json golden IDs when provided', () {
    final path = Platform.environment['MINILM_TOKENIZER_JSON'];
    if (path == null || !File(path).existsSync()) {
      markTestSkipped(
        'Set MINILM_TOKENIZER_JSON to run the offline real-model parity check',
      );
      return;
    }

    final tokenizer = HuggingFaceTokenizerLoader.fromJsonFileSync(path);
    final single = tokenizer.encode('Hello world');
    expect(single.ids, [0, 35378, 8999, 2]);
    expect(single.offsets, [(0, 0), (0, 5), (6, 11), (0, 0)]);
    final pair = tokenizer.encodePair('Hello world', 'Goodbye');
    expect(pair.ids, [0, 35378, 8999, 2, 2, 18621, 1272, 13, 2]);
    expect(pair.typeIds, [0, 0, 0, 0, 0, 0, 0, 0, 0]);
    final batch = tokenizer.encodeBatch(['Hello world', 'Hello world again']);
    expect(batch[0].ids.toList(), [0, 35378, 8999, 2, 1]);
    expect(batch[1].ids.toList(), [0, 35378, 8999, 13438, 2]);
    expect(tokenizer.encode('Ｈｅｌｌｏ　ｗｏｒｌｄ').ids, [0, 35378, 8999, 2]);
  });
}
