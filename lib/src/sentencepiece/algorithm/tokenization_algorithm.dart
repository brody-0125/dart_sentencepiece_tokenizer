abstract class TokenizationAlgorithm {
  List<int> tokenize(String text);
}

class TokenResult {
  final List<int> ids;
  final List<String> pieces;

  const TokenResult({required this.ids, required this.pieces});
}
