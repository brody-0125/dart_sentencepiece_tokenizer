import 'dart:convert';
import 'dart:io';

import 'package:dart_sentencepiece_tokenizer/dart_sentencepiece_tokenizer.dart';
import 'package:test/test.dart';

const _officialXlmTokenizerUrl =
    'https://huggingface.co/FacebookAI/xlm-roberta-base/resolve/main/tokenizer.json';

void main() {
  test(
    'fetches and loads the official XLM-R tokenizer.json when enabled',
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
        final request = await client
            .getUrl(Uri.parse(_officialXlmTokenizerUrl))
            .timeout(const Duration(seconds: 30));
        request.followRedirects = true;
        final response = await request.close().timeout(
          const Duration(seconds: 60),
        );
        expect(response.statusCode, HttpStatus.ok);

        final json = await utf8
            .decodeStream(response)
            .timeout(const Duration(seconds: 60));
        final tokenizer = TokenizerJsonLoader.fromJsonString(json);

        // Golden IDs produced by Hugging Face tokenizers for xlm-roberta-base.
        expect(tokenizer.encode('Hello world').ids, [0, 35378, 8999, 2]);
      } finally {
        client.close(force: true);
      }
    },
  );
}
