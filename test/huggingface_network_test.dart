import 'dart:convert';
import 'dart:io';

import 'package:dart_sentencepiece_tokenizer/dart_sentencepiece_tokenizer.dart';
import 'package:test/test.dart';

const _officialXlmTokenizerUrl =
    'https://huggingface.co/FacebookAI/xlm-roberta-base/resolve/main/tokenizer.json';
const _officialE5TokenizerUrl =
    'https://huggingface.co/intfloat/multilingual-e5-small/resolve/main/tokenizer.json';

void main() {
  test(
    'fetches and loads official Hugging Face tokenizer.json files when enabled',
    () async {
      if (Platform.environment['RUN_HF_NETWORK_TESTS'] != '1') {
        markTestSkipped(
          'Set RUN_HF_NETWORK_TESTS=1 to run the Hugging Face fetch test',
        );
        return;
      }

      final client = HttpClient()
        ..connectionTimeout = const Duration(seconds: 30);
      try {
        Future<String> fetch(String url) async {
          final request = await client
              .getUrl(Uri.parse(url))
              .timeout(const Duration(seconds: 30));
          request.followRedirects = true;
          final response = await request.close().timeout(
            const Duration(seconds: 60),
          );
          expect(response.statusCode, HttpStatus.ok);
          return utf8
              .decodeStream(response)
              .timeout(const Duration(seconds: 60));
        }

        final tokenizer = TokenizerJsonLoader.fromJsonString(
          await fetch(_officialXlmTokenizerUrl),
        );

        // Golden IDs produced by Hugging Face tokenizers for xlm-roberta-base.
        final single = tokenizer.encode('Hello world');
        expect(single.ids, [0, 35378, 8999, 2]);
        expect(single.offsets, [(0, 0), (0, 5), (6, 11), (0, 0)]);

        final whitespace = tokenizer.encode('Hello\tworld\nagain');
        expect(whitespace.ids, [0, 35378, 8999, 13438, 2]);
        expect(whitespace.offsets, [(0, 0), (0, 5), (6, 11), (12, 17), (0, 0)]);

        final repeatedWhitespace = tokenizer.encode('Hello   world');
        expect(repeatedWhitespace.ids, [0, 35378, 8999, 2]);
        expect(repeatedWhitespace.offsets, [(0, 0), (0, 5), (8, 13), (0, 0)]);

        final addedSpecial = tokenizer.encode('<s>');
        expect(addedSpecial.ids, [0, 0, 2]);
        expect(addedSpecial.offsets, [(0, 0), (0, 3), (0, 0)]);

        final pair = tokenizer.encodePair('Hello world', 'Goodbye');
        expect(pair.ids, [0, 35378, 8999, 2, 2, 18621, 1272, 13, 2]);
        expect(pair.typeIds, [0, 0, 0, 0, 0, 0, 0, 0, 0]);
        expect(pair.specialTokensMask, [1, 0, 0, 1, 1, 0, 0, 0, 1]);

        // multilingual-e5-small uses the same Unigram model family but a
        // direct Metaspace pre-tokenizer, so its offsets exercise that path.
        final e5 = TokenizerJsonLoader.fromJsonString(
          await fetch(_officialE5TokenizerUrl),
        );
        final e5Single = e5.encode('Hello world');
        expect(e5Single.ids, [0, 35378, 8999, 2]);
        expect(e5Single.offsets, [(0, 0), (0, 5), (5, 11), (0, 0)]);
        expect(e5Single.wordIds, [null, 0, 1, null]);
        final e5RepeatedWhitespace = e5.encode('Hello   world');
        expect(e5RepeatedWhitespace.ids, [0, 35378, 8999, 2]);
        expect(e5RepeatedWhitespace.offsets, [(0, 0), (0, 5), (7, 13), (0, 0)]);
      } finally {
        client.close(force: true);
      }
    },
  );
}
