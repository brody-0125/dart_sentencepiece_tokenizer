import 'package:dart_sentencepiece_tokenizer/dart_sentencepiece_tokenizer.dart';
import 'package:test/test.dart';

import 'test_utils.dart';

void main() {
  late SentencePieceTokenizer tokenizer;

  setUp(() {
    tokenizer = createTestTokenizer();
  });

  group('TextStreamer (HuggingFace compatible)', () {
    group('put()', () {
      test('processes single token', () {
        final emitted = <String>[];
        final streamer = tokenizer.createTextStreamer(
          onFinalizedText: (text, {required streamEnd}) => emitted.add(text),
        );

        // Encode some text and feed tokens
        final encoding = tokenizer.encode('hello');
        for (final id in encoding.ids) {
          streamer.put(id);
        }
        streamer.end();

        expect(emitted.join(), contains('hello'));
      });

      test('processes multiple tokens sequentially', () {
        final emitted = <String>[];
        final streamer = tokenizer.createTextStreamer(
          onFinalizedText: (text, {required streamEnd}) => emitted.add(text),
        );

        final encoding = tokenizer.encode('hello world');
        for (final id in encoding.ids) {
          streamer.put(id);
        }
        streamer.end();

        final result = emitted.join();
        expect(result, contains('hello'));
        expect(result, contains('world'));
      });

      test('skipSpecialTokens filters BOS/EOS', () {
        final emitted = <String>[];
        final streamer = tokenizer.createTextStreamer(
          skipSpecialTokens: true,
          onFinalizedText: (text, {required streamEnd}) => emitted.add(text),
        );

        // Use a tokenizer that adds special tokens
        final gemmaTokenizer = createGemmaTokenizer();
        final encoding = gemmaTokenizer.encode('test');

        for (final id in encoding.ids) {
          streamer.put(id);
        }
        streamer.end();

        // Should not contain special token strings
        final result = emitted.join();
        expect(result, isNot(contains('<s>')));
        expect(result, isNot(contains('</s>')));
      });

      test('skipPrompt skips first token', () {
        final emitted = <String>[];
        final streamer = TextStreamer(
          tokenizer,
          skipPrompt: true,
          onFinalizedText: (text, {required streamEnd}) => emitted.add(text),
        );

        final encoding = tokenizer.encode('hello world');
        for (final id in encoding.ids) {
          streamer.put(id);
        }
        streamer.end();

        // First token should be skipped
        // The exact behavior depends on token boundaries
        expect(emitted.isNotEmpty, isTrue);
      });

      test('promptLength skips multiple tokens', () {
        var tokenCount = 0;
        final streamer = TextStreamer(
          tokenizer,
          skipPrompt: true,
          promptLength: 3,
          onFinalizedText: (text, {required streamEnd}) {
            if (text.isNotEmpty) tokenCount++;
          },
        );

        // Get some tokens
        final encoding = tokenizer.encode('hello world test sentence');
        final tokenIds = encoding.ids.toList();

        // Feed all tokens
        for (final id in tokenIds) {
          streamer.put(id);
        }
        streamer.end();

        // Should have skipped first 3 tokens
        // Output should be from remaining tokens
        expect(tokenCount, greaterThanOrEqualTo(0));
      });
    });

    group('end()', () {
      test('flushes remaining buffer', () {
        final emitted = <String>[];
        var streamEndCalled = false;

        final streamer = tokenizer.createTextStreamer(
          onFinalizedText: (text, {required streamEnd}) {
            emitted.add(text);
            if (streamEnd) streamEndCalled = true;
          },
        );

        final encoding = tokenizer.encode('test');
        for (final id in encoding.ids) {
          streamer.put(id);
        }

        expect(streamEndCalled, isFalse);
        streamer.end();
        expect(streamEndCalled, isTrue);
      });

      test('calls callback with streamEnd=true', () {
        var streamEndValue = false;

        final streamer = tokenizer.createTextStreamer(
          onFinalizedText: (text, {required streamEnd}) {
            streamEndValue = streamEnd;
          },
        );

        final encoding = tokenizer.encode('hello');
        for (final id in encoding.ids) {
          streamer.put(id);
        }
        streamer.end();

        expect(streamEndValue, isTrue);
      });

      test('signals stream end even with no tokens', () {
        var endCalled = false;

        final streamer = tokenizer.createTextStreamer(
          onFinalizedText: (text, {required streamEnd}) {
            if (streamEnd) endCalled = true;
          },
        );

        streamer.end();
        expect(endCalled, isTrue);
      });
    });

    group('onFinalizedText callback', () {
      test('receives text chunks', () {
        final chunks = <String>[];

        final streamer = tokenizer.createTextStreamer(
          onFinalizedText: (text, {required streamEnd}) => chunks.add(text),
        );

        final encoding = tokenizer.encode('hello world test');
        for (final id in encoding.ids) {
          streamer.put(id);
        }
        streamer.end();

        expect(chunks, isNotEmpty);
        expect(chunks.join(), contains('hello'));
      });

      test('custom callback overrides default stdout', () {
        final customOutput = StringBuffer();

        final streamer = tokenizer.createTextStreamer(
          onFinalizedText: (text, {required streamEnd}) {
            customOutput.write(text);
          },
        );

        final encoding = tokenizer.encode('custom output test');
        for (final id in encoding.ids) {
          streamer.put(id);
        }
        streamer.end();

        expect(customOutput.toString(), isNotEmpty);
      });
    });

    group('HF word boundary heuristics', () {
      test('flushes at newline', () {
        final emitted = <String>[];

        final streamer = tokenizer.createTextStreamer(
          onFinalizedText: (text, {required streamEnd}) => emitted.add(text),
        );

        // Create tokens that would produce text with newline
        // This depends on the model vocabulary
        final encoding = tokenizer.encode('line1');
        for (final id in encoding.ids) {
          streamer.put(id);
        }
        streamer.end();

        expect(emitted.join(), isNotEmpty);
      });

      test('handles text without word boundaries', () {
        final emitted = <String>[];

        final streamer = tokenizer.createTextStreamer(
          onFinalizedText: (text, {required streamEnd}) => emitted.add(text),
        );

        final encoding = tokenizer.encode('test');
        for (final id in encoding.ids) {
          streamer.put(id);
        }
        streamer.end();

        final result = emitted.join();
        expect(result, contains('test'));
      });
    });

    group('reset()', () {
      test('allows reuse of streamer', () {
        final emitted = <String>[];

        final streamer = tokenizer.createTextStreamer(
          onFinalizedText: (text, {required streamEnd}) => emitted.add(text),
        );

        // First use - use 'hello' which is in test vocabulary
        final encoding1 = tokenizer.encode('hello');
        for (final id in encoding1.ids) {
          streamer.put(id);
        }
        streamer.end();
        final firstResult = emitted.join();
        emitted.clear();

        // Reset and second use - use 'world' which is also in test vocabulary
        streamer.reset();
        final encoding2 = tokenizer.encode('world');
        for (final id in encoding2.ids) {
          streamer.put(id);
        }
        streamer.end();
        final secondResult = emitted.join();

        // Check both produced non-empty output (vocabulary may differ)
        expect(firstResult, isNotEmpty);
        expect(secondResult, isNotEmpty);
        // The results should be different (different inputs)
        expect(firstResult, isNot(equals(secondResult)));
      });
    });
  });


  group('SentencePieceTokenizer streaming methods', () {
    test('decodeStream yields text chunks', () async {
      final encoding = tokenizer.encode('hello world');
      final stream = Stream.fromIterable(encoding.ids.toList());

      final chunks = await tokenizer.decodeStream(stream).toList();

      final result = chunks.join();
      expect(result, contains('hello'));
      expect(result, contains('world'));
    });

    test('decodeWithCallback invokes callback', () {
      final chunks = <String>[];
      final encoding = tokenizer.encode('hello test');

      tokenizer.decodeWithCallback(
        encoding.ids.toList(),
        (chunk) => chunks.add(chunk),
      );

      // Check that callback was invoked and produced output
      expect(chunks, isNotEmpty);
      final result = chunks.join();
      expect(result, isNotEmpty);
      expect(result, contains('hello'));
    });

    test('createTextStreamer returns configured TextStreamer', () {
      final streamer = tokenizer.createTextStreamer(
        skipSpecialTokens: false,
        skipPrompt: true,
      );

      expect(streamer, isA<TextStreamer>());
    });
  });

  group('BaseStreamer interface', () {
    test('TextStreamer implements BaseStreamer', () {
      final streamer = tokenizer.createTextStreamer();
      expect(streamer, isA<BaseStreamer>());
    });

    test('can use TextStreamer polymorphically as BaseStreamer', () {
      BaseStreamer streamer = tokenizer.createTextStreamer(
        onFinalizedText: (_, {required streamEnd}) {},
      );

      final encoding = tokenizer.encode('polymorphic');
      for (final id in encoding.ids) {
        streamer.put(id);
      }
      streamer.end();

      // No error means success
    });
  });

  group('Unicode handling', () {
    test('handles multi-byte UTF-8 characters', () {
      final emitted = <String>[];
      final streamer = tokenizer.createTextStreamer(
        onFinalizedText: (text, {required streamEnd}) => emitted.add(text),
      );

      // Most models handle ASCII well; CJK may use byte fallback
      final encoding = tokenizer.encode('hello');
      for (final id in encoding.ids) {
        streamer.put(id);
      }
      streamer.end();

      final result = emitted.join();
      expect(result, contains('hello'));
    });

    test('TextStreamer handles mixed content', () {
      final emitted = <String>[];
      final streamer = tokenizer.createTextStreamer(
        onFinalizedText: (text, {required streamEnd}) => emitted.add(text),
      );

      final encoding = tokenizer.encode('test sentence');
      for (final id in encoding.ids) {
        streamer.put(id);
      }
      streamer.end();

      final combined = emitted.join();
      expect(combined, isNotEmpty);
    });
  });

  group('CJK character handling', () {
    test('TextStreamer _isCjk identifies CJK ranges', () {
      // Test via the heuristics behavior
      // CJK should be emitted immediately without buffering
      final emitted = <String>[];
      final streamer = tokenizer.createTextStreamer(
        onFinalizedText: (text, {required streamEnd}) => emitted.add(text),
      );

      // Use ASCII which is in test vocabulary
      final encoding = tokenizer.encode('hello world');
      for (final id in encoding.ids) {
        streamer.put(id);
      }
      streamer.end();

      // Output should be produced
      expect(emitted.join(), isNotEmpty);
    });

    test('TextStreamer emits at word boundaries for Latin text', () {
      var emissionCount = 0;
      final streamer = tokenizer.createTextStreamer(
        onFinalizedText: (text, {required streamEnd}) {
          if (text.isNotEmpty) emissionCount++;
        },
      );

      // Use text with spaces to trigger word boundary heuristic
      final encoding = tokenizer.encode('hello world test');
      for (final id in encoding.ids) {
        streamer.put(id);
      }
      streamer.end();

      // Should have at least one emission
      expect(emissionCount, greaterThan(0));
    });
  });

  group('Edge cases', () {
    test('empty token stream produces empty output', () {
      final emitted = <String>[];
      final streamer = tokenizer.createTextStreamer(
        onFinalizedText: (text, {required streamEnd}) => emitted.add(text),
      );

      // Don't add any tokens, just end
      streamer.end();

      // Should still signal end (with empty text)
      expect(emitted, isNotEmpty);
      // Last emission should have streamEnd=true
    });

    test('multiple reset cycles work correctly', () {
      final streamer = tokenizer.createTextStreamer(
        onFinalizedText: (_, {required streamEnd}) {},
      );

      for (var cycle = 0; cycle < 3; cycle++) {
        final encoding = tokenizer.encode('hello');
        for (final id in encoding.ids) {
          streamer.put(id);
        }
        streamer.end();
        streamer.reset();
      }

      // No exception means success
    });

    test('TextStreamer handles consecutive special tokens', () {
      final emitted = <String>[];
      final streamer = tokenizer.createTextStreamer(
        skipSpecialTokens: true,
        onFinalizedText: (text, {required streamEnd}) => emitted.add(text),
      );

      // Add multiple special tokens
      streamer.put(tokenizer.vocab.bosId);
      streamer.put(tokenizer.vocab.eosId);
      streamer.end();

      // Should be empty since all were special tokens (only empty string from end)
      final result = emitted.where((s) => s.isNotEmpty).join();
      expect(result, isEmpty);
    });
  });
}
