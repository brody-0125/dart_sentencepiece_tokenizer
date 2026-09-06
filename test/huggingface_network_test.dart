import 'dart:convert';
import 'dart:io';

import 'package:crypto/crypto.dart';
import 'package:dart_sentencepiece_tokenizer/dart_sentencepiece_tokenizer.dart';
import 'package:test/test.dart';

const _runNetworkTests = 'RUN_HF_NETWORK_TESTS';

class _HfCase {
  const _HfCase({
    required this.input,
    required this.ids,
    this.tokens,
    this.typeIds,
    this.attentionMask,
    this.specialTokensMask,
    this.offsets,
    this.wordIds,
    this.sequenceIds,
    this.decoded,
  });

  final String input;
  final List<int> ids;
  final List<String>? tokens;
  final List<int>? typeIds;
  final List<int>? attentionMask;
  final List<int>? specialTokensMask;
  final List<(int, int)>? offsets;
  final List<int?>? wordIds;
  final List<int?>? sequenceIds;
  final String? decoded;
}

class _HfPairCase {
  const _HfPairCase({
    required this.firstInput,
    required this.secondInput,
    required this.expected,
  });

  final String firstInput;
  final String secondInput;
  final _HfCase expected;
}

class _HfFixture {
  const _HfFixture({
    required this.name,
    required this.repository,
    required this.revision,
    required this.sha256,
    required this.cases,
    this.pairCases,
    this.batchCases,
  });

  final String name;
  final String repository;
  final String revision;
  final String sha256;
  final List<_HfCase> cases;
  final List<_HfPairCase>? pairCases;
  final List<_HfCase>? batchCases;

  String get url =>
      'https://huggingface.co/$repository/resolve/$revision/tokenizer.json';
}

// Revisions and hashes are intentionally fixed. Golden values were generated
// with Hugging Face tokenizers 0.23.2. The list is curated by serialized
// pipeline shape, following Swift Transformers and goSentencePiece.
final _fixtures = <_HfFixture>[
  const _HfFixture(
    name: 'XLM-R',
    repository: 'FacebookAI/xlm-roberta-base',
    revision: 'e73636d4f797dec63c3081bb6ed5c7b0bb3f2089',
    sha256: 'a898ea75433890f6610f4e470b8ebeb0c21dce5c8dd61f892eb09eb5919d2e2c',
    cases: [
      _HfCase(
        input: 'Hello world',
        ids: [0, 35378, 8999, 2],
        tokens: ['<s>', '▁Hello', '▁world', '</s>'],
        typeIds: [0, 0, 0, 0],
        attentionMask: [1, 1, 1, 1],
        specialTokensMask: [1, 0, 0, 1],
        offsets: [(0, 0), (0, 5), (6, 11), (0, 0)],
        wordIds: [null, 0, 1, null],
        sequenceIds: [null, 0, 0, null],
        decoded: 'Hello world',
      ),
      _HfCase(
        input: 'Hello\tworld\nagain',
        ids: [0, 35378, 8999, 13438, 2],
        offsets: [(0, 0), (0, 5), (6, 11), (12, 17), (0, 0)],
        wordIds: [null, 0, 1, 2, null],
        decoded: 'Hello world again',
      ),
      _HfCase(
        input: 'Hello   world',
        ids: [0, 35378, 8999, 2],
        offsets: [(0, 0), (0, 5), (8, 13), (0, 0)],
        decoded: 'Hello world',
      ),
      _HfCase(
        input: '<s>',
        ids: [0, 0, 2],
        tokens: ['<s>', '<s>', '</s>'],
        typeIds: [0, 0, 0],
        attentionMask: [1, 1, 1],
        specialTokensMask: [1, 0, 1],
        offsets: [(0, 0), (0, 3), (0, 0)],
        decoded: '',
      ),
    ],
    pairCases: [
      _HfPairCase(
        firstInput: 'Hello world',
        secondInput: 'Goodbye',
        expected: _HfCase(
          input: 'Hello world',
          ids: [0, 35378, 8999, 2, 2, 18621, 1272, 13, 2],
          tokens: [
            '<s>',
            '▁Hello',
            '▁world',
            '</s>',
            '</s>',
            '▁Good',
            'by',
            'e',
            '</s>',
          ],
          typeIds: [0, 0, 0, 0, 0, 0, 0, 0, 0],
          attentionMask: [1, 1, 1, 1, 1, 1, 1, 1, 1],
          specialTokensMask: [1, 0, 0, 1, 1, 0, 0, 0, 1],
          wordIds: [null, 0, 1, null, null, 0, 0, 0, null],
          sequenceIds: [null, 0, 0, null, null, 1, 1, 1, null],
          decoded: 'Hello world Goodbye',
        ),
      ),
    ],
  ),
  const _HfFixture(
    name: 'multilingual E5',
    repository: 'intfloat/multilingual-e5-small',
    revision: '614241f622f53c4eeff9890bdc4f31cfecc418b3',
    sha256: '0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39',
    cases: [
      _HfCase(
        input: 'Hello world',
        ids: [0, 35378, 8999, 2],
        tokens: ['<s>', '▁Hello', '▁world', '</s>'],
        typeIds: [0, 0, 0, 0],
        attentionMask: [1, 1, 1, 1],
        specialTokensMask: [1, 0, 0, 1],
        offsets: [(0, 0), (0, 5), (5, 11), (0, 0)],
        wordIds: [null, 0, 1, null],
        sequenceIds: [null, 0, 0, null],
        decoded: 'Hello world',
      ),
      _HfCase(
        input: 'Hello\tworld\nagain',
        ids: [0, 35378, 8999, 13438, 2],
        offsets: [(0, 0), (0, 5), (5, 11), (11, 17), (0, 0)],
        wordIds: [null, 0, 1, 2, null],
        decoded: 'Hello world again',
      ),
      _HfCase(
        input: 'Hello   world',
        ids: [0, 35378, 8999, 2],
        offsets: [(0, 0), (0, 5), (7, 13), (0, 0)],
        decoded: 'Hello world',
      ),
    ],
  ),
  const _HfFixture(
    name: 'multilingual MiniLM',
    repository: 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    revision: 'e8f8c211226b894fcb81acc59f3b34ba3efd5f42',
    sha256: '2c3387be76557bd40970cec13153b3bbf80407865484b209e655e5e4729076b8',
    cases: [
      _HfCase(
        input: 'Hello world',
        ids: [0, 35378, 8999, 2],
        tokens: ['<s>', '▁Hello', '▁world', '</s>'],
        typeIds: [0, 0, 0, 0],
        attentionMask: [1, 1, 1, 1],
        specialTokensMask: [1, 0, 0, 1],
        offsets: [(0, 0), (0, 5), (6, 11), (0, 0)],
        wordIds: [null, 0, 1, null],
        sequenceIds: [null, 0, 0, null],
        decoded: 'Hello world',
      ),
    ],
    batchCases: [
      _HfCase(
        input: 'Hello world',
        ids: [0, 35378, 8999, 2, 1],
        tokens: ['<s>', '▁Hello', '▁world', '</s>', '<pad>'],
        typeIds: [0, 0, 0, 0, 0],
        attentionMask: [1, 1, 1, 1, 0],
        specialTokensMask: [1, 0, 0, 1, 1],
        offsets: [(0, 0), (0, 5), (6, 11), (0, 0), (0, 0)],
        wordIds: [null, 0, 1, null, null],
        sequenceIds: [null, 0, 0, null, null],
        decoded: 'Hello world',
      ),
      _HfCase(
        input: 'Hello world again',
        ids: [0, 35378, 8999, 13438, 2],
        tokens: ['<s>', '▁Hello', '▁world', '▁again', '</s>'],
        typeIds: [0, 0, 0, 0, 0],
        attentionMask: [1, 1, 1, 1, 1],
        specialTokensMask: [1, 0, 0, 0, 1],
        offsets: [(0, 0), (0, 5), (6, 11), (12, 17), (0, 0)],
        wordIds: [null, 0, 1, 2, null],
        sequenceIds: [null, 0, 0, 0, null],
        decoded: 'Hello world again',
      ),
    ],
  ),
  const _HfFixture(
    name: 'T5',
    repository: 'google-t5/t5-base',
    revision: 'a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1',
    sha256: 'd2acde0d8d71dd30a711834b07781b9c89feaac33fd332f60507699282740066',
    cases: [
      _HfCase(
        input: 'Hello world',
        ids: [8774, 296, 1],
        tokens: ['▁Hello', '▁world', '</s>'],
        typeIds: [0, 0, 0],
        attentionMask: [1, 1, 1],
        specialTokensMask: [0, 0, 1],
        offsets: [(0, 5), (6, 11), (0, 0)],
        wordIds: [0, 1, null],
        sequenceIds: [0, 0, null],
        decoded: 'Hello world',
      ),
      _HfCase(
        input: 'Hello\tworld\nagain',
        ids: [8774, 296, 541, 1],
        offsets: [(0, 5), (6, 11), (12, 17), (0, 0)],
        wordIds: [0, 1, 2, null],
        decoded: 'Hello world again',
      ),
    ],
  ),
  // These fixtures cover the Gemma-family Split form and remain opt-in because
  // they fetch pinned tokenizer.json files from Hugging Face.
  _HfFixture(
    name: 'SigLIP2 / Gemma BPE',
    repository: 'onnx-community/siglip2-base-patch16-224-ONNX',
    revision: 'ba1f3b0843f24bc5417d38e19c37b287d719b2f4',
    sha256: 'cb9140fae3ac5122c972d37adf83e1248471a38147ad76f8215c8872c6fd8322',
    cases: [
      _HfCase(
        input: 'Hello world',
        ids: [4521, 2134, 1, ...List<int>.filled(61, 0)],
        typeIds: List<int>.filled(64, 0),
        attentionMask: [1, 1, 1, ...List<int>.filled(61, 0)],
        specialTokensMask: [0, 0, 1, ...List<int>.filled(61, 1)],
        offsets: [
          (0, 5),
          (5, 11),
          (0, 0),
          ...List<(int, int)>.filled(61, (0, 0)),
        ],
        wordIds: [0, 0, null, ...List<int?>.filled(61, null)],
        sequenceIds: [0, 0, null, ...List<int?>.filled(61, null)],
        decoded: 'Hello world',
      ),
      _HfCase(
        input: 'room 101 at 3:45pm',
        ids: [
          2978,
          235248,
          235274,
          235276,
          235274,
          696,
          235248,
          235304,
          235292,
          235310,
          235308,
          3397,
          1,
          ...List<int>.filled(51, 0),
        ],
        decoded: 'room 101 at 3:45pm',
      ),
      _HfCase(
        input: 'цена 1234,56 руб.',
        ids: [
          74001,
          235248,
          235274,
          235284,
          235304,
          235310,
          235269,
          235308,
          235318,
          22810,
          235265,
          1,
          ...List<int>.filled(52, 0),
        ],
        decoded: 'цена 1234,56 руб.',
      ),
      _HfCase(
        input: 'a\nb\tc',
        ids: [235250, 108, 235268, 226, 235260, 1, ...List<int>.filled(58, 0)],
        decoded: 'a\nb\tc',
      ),
      _HfCase(
        input: 'e-mail: a@b.co',
        ids: [
          235249,
          235290,
          1765,
          235292,
          476,
          235348,
          235268,
          235265,
          528,
          1,
          ...List<int>.filled(54, 0),
        ],
        decoded: 'e-mail: a@b.co',
      ),
      _HfCase(
        input: '2026-09-05',
        ids: [
          235284,
          235276,
          235284,
          235318,
          235290,
          235276,
          235315,
          235290,
          235276,
          235308,
          1,
          ...List<int>.filled(53, 0),
        ],
        decoded: '2026-09-05',
      ),
    ],
  ),
  _HfFixture(
    name: 'Gemma-compatible BPE',
    repository: 'pcuenq/gemma-tokenizer',
    revision: '2d1305b81cafe73519bfa0319d00eb8b79f412a6',
    sha256: '3f289bc05132635a8bc7aca7aa21255efd5e18f3710f43e3cdb96bcd41be4922',
    cases: [
      _HfCase(
        input: 'room 101 at 3:45pm',
        ids: [
          2,
          2978,
          235248,
          235274,
          235276,
          235274,
          696,
          235248,
          235304,
          235292,
          235310,
          235308,
          3397,
        ],
        tokens: [
          '<bos>',
          'room',
          '▁',
          '1',
          '0',
          '1',
          '▁at',
          '▁',
          '3',
          ':',
          '4',
          '5',
          'pm',
        ],
        typeIds: List<int>.filled(13, 0),
        attentionMask: List<int>.filled(13, 1),
        specialTokensMask: [1, ...List<int>.filled(12, 0)],
        offsets: [
          (0, 0),
          (0, 4),
          (4, 5),
          (5, 6),
          (6, 7),
          (7, 8),
          (8, 11),
          (11, 12),
          (12, 13),
          (13, 14),
          (14, 15),
          (15, 16),
          (16, 18),
        ],
        wordIds: [null, ...List<int>.filled(12, 0)],
        sequenceIds: [null, ...List<int?>.filled(12, 0)],
        decoded: 'room 101 at 3:45pm',
      ),
      const _HfCase(
        input: 'цена 1234,56 руб.',
        ids: [
          2,
          74001,
          235248,
          235274,
          235284,
          235304,
          235310,
          235269,
          235308,
          235318,
          22810,
          235265,
        ],
        decoded: 'цена 1234,56 руб.',
      ),
      const _HfCase(
        input: 'a\nb\tc',
        ids: [2, 235250, 108, 235268, 226, 235260],
        decoded: 'a\nb\tc',
      ),
      const _HfCase(
        input: 'e-mail: a@b.co',
        ids: [
          2,
          235249,
          235290,
          1765,
          235292,
          476,
          235348,
          235268,
          235265,
          528,
        ],
        decoded: 'e-mail: a@b.co',
      ),
      const _HfCase(
        input: '2026-09-05',
        ids: [
          2,
          235284,
          235276,
          235284,
          235318,
          235290,
          235276,
          235315,
          235290,
          235276,
          235308,
        ],
        decoded: '2026-09-05',
      ),
    ],
  ),
  const _HfFixture(
    name: 'Llama BPE',
    repository: 'hf-internal-testing/llama-tokenizer',
    revision: 'd02ad6cb9dd2c2296a6332199fa2fdca5938fef0',
    sha256: '8eea70c4866c4f1320ba096fc986ac82038a8374dbe135212ba7628835b4a6f1',
    cases: [
      _HfCase(
        input: 'Hello world',
        ids: [1, 15043, 3186],
        tokens: ['<s>', '▁Hello', '▁world'],
        typeIds: [0, 0, 0],
        attentionMask: [1, 1, 1],
        specialTokensMask: [1, 0, 0],
      ),
    ],
  ),
];

Future<List<int>> _fetchBytes(HttpClient client, String url) async {
  final request = await client
      .getUrl(Uri.parse(url))
      .timeout(const Duration(seconds: 30));
  request.followRedirects = true;
  final response = await request.close().timeout(const Duration(seconds: 60));
  expect(response.statusCode, HttpStatus.ok);
  return response
      .fold<List<int>>(<int>[], (bytes, chunk) {
        bytes.addAll(chunk);
        return bytes;
      })
      .timeout(const Duration(seconds: 60));
}

Future<SentencePieceTokenizer> _loadFixture(_HfFixture fixture) async {
  final client = HttpClient()..connectionTimeout = const Duration(seconds: 30);
  try {
    final bytes = await _fetchBytes(client, fixture.url);
    expect(
      sha256.convert(bytes).toString(),
      fixture.sha256,
      reason: '${fixture.repository}@${fixture.revision}',
    );
    return TokenizerJsonLoader.fromJsonString(utf8.decode(bytes));
  } finally {
    client.close(force: true);
  }
}

void _expectEncoding(
  SentencePieceTokenizer tokenizer,
  Encoding encoding,
  _HfCase expected,
) {
  expect(encoding.ids.toList(), expected.ids, reason: expected.input);
  if (expected.tokens != null) {
    expect(encoding.tokens, expected.tokens, reason: expected.input);
  }
  if (expected.typeIds != null) {
    expect(encoding.typeIds.toList(), expected.typeIds, reason: expected.input);
  }
  if (expected.attentionMask != null) {
    expect(
      encoding.attentionMask.toList(),
      expected.attentionMask,
      reason: expected.input,
    );
  }
  if (expected.specialTokensMask != null) {
    expect(
      encoding.specialTokensMask.toList(),
      expected.specialTokensMask,
      reason: expected.input,
    );
  }
  if (expected.offsets != null) {
    expect(encoding.offsets, expected.offsets, reason: expected.input);
  }
  if (expected.wordIds != null) {
    expect(encoding.wordIds, expected.wordIds, reason: expected.input);
  }
  if (expected.sequenceIds != null) {
    expect(encoding.sequenceIds, expected.sequenceIds, reason: expected.input);
  }
  if (expected.decoded != null) {
    expect(
      tokenizer.decode(encoding.ids.toList()),
      expected.decoded,
      reason: expected.input,
    );
  }
}

void main() {
  for (final fixture in _fixtures) {
    group('pinned Hugging Face fixture: ${fixture.name}', () {
      SentencePieceTokenizer? tokenizer;
      Object? loadError;
      StackTrace? loadStack;

      setUpAll(() async {
        if (Platform.environment[_runNetworkTests] != '1') {
          return;
        }
        try {
          tokenizer = await _loadFixture(fixture);
        } catch (error, stackTrace) {
          loadError = error;
          loadStack = stackTrace;
        }
      });

      bool skipUnavailable() {
        if (Platform.environment[_runNetworkTests] != '1') {
          markTestSkipped(
            'Set $_runNetworkTests=1 to run Hugging Face fixtures',
          );
          return true;
        }
        if (loadError != null) {
          markTestSkipped('Fixture failed to load: $loadError');
          return true;
        }
        return false;
      }

      test('loads fixture', () {
        if (Platform.environment[_runNetworkTests] != '1') {
          markTestSkipped(
            'Set $_runNetworkTests=1 to run Hugging Face fixtures',
          );
          return;
        }
        if (loadError != null) {
          fail('Fixture failed to load: $loadError\n$loadStack');
        }
        expect(tokenizer, isNotNull);
      });

      for (final expected in fixture.cases) {
        test('encodes ${expected.input}', () {
          if (skipUnavailable()) return;
          _expectEncoding(
            tokenizer!,
            tokenizer!.encode(expected.input),
            expected,
          );
        });
      }

      for (final pair in fixture.pairCases ?? const <_HfPairCase>[]) {
        test('encodes pair ${pair.firstInput} / ${pair.secondInput}', () {
          if (skipUnavailable()) return;
          _expectEncoding(
            tokenizer!,
            tokenizer!.encodePair(pair.firstInput, pair.secondInput),
            pair.expected,
          );
        });
      }

      final batchCases = fixture.batchCases;
      if (batchCases != null) {
        for (var i = 0; i < batchCases.length; i++) {
          test('encodes batch item $i: ${batchCases[i].input}', () {
            if (skipUnavailable()) return;
            final batch = tokenizer!.encodeBatch(
              batchCases.map((expected) => expected.input).toList(),
            );
            expect(batch.length, batchCases.length, reason: fixture.name);
            _expectEncoding(tokenizer!, batch[i], batchCases[i]);
          });
        }
      }
    });
  }
}
