import XCTest
@testable import AnemllCore

/// Needs a directory with tokenizer.json / tokenizer_config.json / config.json, for example
/// a converted model's output directory. Skipped unless ANEMLL_TEST_TOKENIZER_DIR is set:
///
///     ANEMLL_TEST_TOKENIZER_DIR=/path/to/converted_model swift test
final class TokenizerTests: XCTestCase {
    private func tokenizerDir() throws -> String {
        guard let dir = ProcessInfo.processInfo.environment["ANEMLL_TEST_TOKENIZER_DIR"], !dir.isEmpty else {
            throw XCTSkip("Set ANEMLL_TEST_TOKENIZER_DIR to a directory with tokenizer files")
        }
        return dir
    }

    func testTokenizeDetokenizeRoundTrip() async throws {
        let tokenizer = try await Tokenizer(modelPath: try tokenizerDir())

        let text = "Hello, world!"
        let tokens = tokenizer.tokenize(text)
        XCTAssertFalse(tokens.isEmpty)
        XCTAssertEqual(tokenizer.detokenize(tokens), text)
    }

    func testChatTemplateContainsUserMessage() async throws {
        let tokenizer = try await Tokenizer(modelPath: try tokenizerDir())

        let tokens = tokenizer.applyChatTemplate(input: [Tokenizer.ChatMessage.user("Hello")])
        XCTAssertFalse(tokens.isEmpty)
        XCTAssertTrue(tokenizer.decode(tokens: tokens, skipSpecialTokens: false).contains("Hello"))
    }
}
