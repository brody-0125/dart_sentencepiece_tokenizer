import 'package:test/test.dart';
import 'package:dart_sentencepiece_tokenizer/dart_sentencepiece_tokenizer.dart';

void main() {
  group('Trie', () {
    test('handles basic ASCII characters', () {
      final trie = Trie();
      trie.insert('abc', 0);

      final match = trie.findLongestPrefix('abcdef');
      expect(match, isNotNull);
      expect(match!.token, 'abc');
      expect(match.end, 3);
    });

    test('handles surrogate pairs in findLongestPrefix', () {
      final trie = Trie();
      // 𝄞 is U+1D11E (Musical Symbol G Clef), requires surrogate pair
      trie.insert('𝄞', 0);
      trie.insert('𝄞x', 1);

      final match = trie.findLongestPrefix('𝄞xyz');
      expect(match, isNotNull);
      expect(match!.token, '𝄞x');
      expect(match.tokenId, 1);
    });

    test('handles surrogate pairs in findAllPrefixes', () {
      final trie = Trie();
      trie.insert('𝄞', 0);
      trie.insert('𝄞x', 1);

      final matches = trie.findAllPrefixes('𝄞xyz');
      expect(matches, hasLength(2));
      expect(matches[0].token, '𝄞');
      expect(matches[1].token, '𝄞x');
    });

    test(
      'findLongestPrefix and findAllPrefixes produce consistent results',
      () {
        final trie = Trie();
        trie.insert('h', 0);
        trie.insert('he', 1);
        trie.insert('hel', 2);
        trie.insert('hello', 3);

        final longest = trie.findLongestPrefix('hello world');
        final all = trie.findAllPrefixes('hello world');

        expect(longest, isNotNull);
        expect(longest!.token, 'hello');
        expect(all.last.token, 'hello');
        expect(all, hasLength(4));
      },
    );
  });
}
