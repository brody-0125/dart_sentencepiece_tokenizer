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
    this.offsets,
    this.wordIds,
    this.decoded,
  });

  final String input;
  final List<int> ids;
  final List<(int, int)>? offsets;
  final List<int?>? wordIds;
  final String? decoded;
}

class _HfFixture {
  const _HfFixture({
    required this.name,
    required this.repository,
    required this.revision,
    required this.sha256,
    required this.cases,
    this.batchIds,
  });

  final String name;
  final String repository;
  final String revision;
  final String sha256;
  final List<_HfCase> cases;
  final List<List<int>>? batchIds;

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
        offsets: [(0, 0), (0, 5), (6, 11), (0, 0)],
        wordIds: [null, 0, 1, null],
        decoded: 'Hello world',
      ),
      _HfCase(
        input: 'Hello\tworld\nagain',
        ids: [0, 35378, 8999, 13438, 2],
        offsets: [(0, 0), (0, 5), (6, 11), (12, 17), (0, 0)],
        wordIds: [null, 0, 1, 2, null],
        decoded: 'Hello world again',
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
        offsets: [(0, 0), (0, 5), (5, 11), (0, 0)],
        wordIds: [null, 0, 1, null],
        decoded: 'Hello world',
      ),
      _HfCase(
        input: 'Hello\tworld\nagain',
        ids: [0, 35378, 8999, 13438, 2],
        offsets: [(0, 0), (0, 5), (5, 11), (11, 17), (0, 0)],
        wordIds: [null, 0, 1, 2, null],
        decoded: 'Hello world again',
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
        offsets: [(0, 0), (0, 5), (6, 11), (0, 0)],
        wordIds: [null, 0, 1, null],
        decoded: 'Hello world',
      ),
    ],
    batchIds: [
      [0, 35378, 8999, 2, 1],
      [0, 35378, 8999, 13438, 2],
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
        offsets: [(0, 5), (6, 11), (0, 0)],
        wordIds: [0, 1, null],
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
  // These two fixtures intentionally remain active: until the external
  // narrow Split contribution lands, opt-in verification must expose the
  // compatibility gap instead of hiding it behind a skip.
  _HfFixture(
    name: 'SigLIP2 / Gemma BPE',
    repository: 'onnx-community/siglip2-base-patch16-224-ONNX',
    revision: 'ba1f3b0843f24bc5417d38e19c37b287d719b2f4',
    sha256: 'cb9140fae3ac5122c972d37adf83e1248471a38147ad76f8215c8872c6fd8322',
    cases: [
      _HfCase(
        input: 'Hello world',
        ids: [4521, 2134, 1, ...List<int>.filled(61, 0)],
        decoded: 'Hello world',
      ),
    ],
  ),
  const _HfFixture(
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
        decoded: 'room 101 at 3:45pm',
      ),
    ],
  ),
  const _HfFixture(
    name: 'Llama BPE',
    repository: 'hf-internal-testing/llama-tokenizer',
    revision: 'd02ad6cb9dd2c2296a6332199fa2fdca5938fef0',
    sha256: '8eea70c4866c4f1320ba096fc986ac82038a8374dbe135212ba7628835b4a6f1',
    cases: [
      _HfCase(input: 'Hello world', ids: [1, 15043, 3186]),
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

void _expectCase(SentencePieceTokenizer tokenizer, _HfCase expected) {
  final encoding = tokenizer.encode(expected.input);
  expect(encoding.ids.toList(), expected.ids, reason: expected.input);
  if (expected.offsets != null) {
    expect(encoding.offsets, expected.offsets, reason: expected.input);
  }
  if (expected.wordIds != null) {
    expect(encoding.wordIds, expected.wordIds, reason: expected.input);
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
    test('loads pinned Hugging Face fixture: ${fixture.name}', () async {
      if (Platform.environment[_runNetworkTests] != '1') {
        markTestSkipped('Set $_runNetworkTests=1 to run Hugging Face fixtures');
        return;
      }

      final client = HttpClient()
        ..connectionTimeout = const Duration(seconds: 30);
      try {
        final bytes = await _fetchBytes(client, fixture.url);
        expect(
          sha256.convert(bytes).toString(),
          fixture.sha256,
          reason: '${fixture.repository}@${fixture.revision}',
        );

        final tokenizer = TokenizerJsonLoader.fromJsonString(
          utf8.decode(bytes),
        );
        for (final expected in fixture.cases) {
          _expectCase(tokenizer, expected);
        }

        if (fixture.batchIds != null) {
          final batch = tokenizer.encodeBatch([
            'Hello world',
            'Hello world again',
          ]);
          expect(
            batch.map((encoding) => encoding.ids.toList()).toList(),
            fixture.batchIds,
            reason: fixture.name,
          );
        }
      } finally {
        client.close(force: true);
      }
    });
  }
}
