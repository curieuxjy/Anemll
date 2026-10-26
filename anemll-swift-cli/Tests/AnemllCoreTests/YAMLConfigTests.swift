import XCTest
@testable import AnemllCore

/// YAMLConfig tests that don't need converted models: meta.yaml files are written to a
/// temporary directory, and model "files" are empty directories where a path check needs one.
final class YAMLConfigTests: XCTestCase {
    private var tempDir: URL!

    override func setUpWithError() throws {
        tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("YAMLConfigTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: tempDir)
    }

    // MARK: - Helpers

    private func writeMeta(_ contents: String) throws -> String {
        let url = tempDir.appendingPathComponent("meta.yaml")
        try contents.write(to: url, atomically: true, encoding: .utf8)
        return url.path
    }

    private func touchModel(_ name: String) throws {
        try FileManager.default.createDirectory(
            at: tempDir.appendingPathComponent(name), withIntermediateDirectories: true)
    }

    private var baseDir: String { tempDir.path }

    // MARK: - init(from:)

    func testInitRequiresModelPath() {
        XCTAssertThrowsError(try YAMLConfig(from: "tokenizer_model: /tmp/tok\n")) { error in
            guard case ConfigError.missingField("model_path") = error else {
                return XCTFail("Expected missingField(model_path), got \(error)")
            }
        }
    }

    func testInitRequiresTokenizerModel() {
        XCTAssertThrowsError(try YAMLConfig(from: "model_path: /tmp/model\n")) { error in
            guard case ConfigError.missingField("tokenizer_model") = error else {
                return XCTFail("Expected missingField(tokenizer_model), got \(error)")
            }
        }
    }

    func testInitRejectsNonDictionaryYAML() {
        XCTAssertThrowsError(try YAMLConfig(from: "- a\n- b\n")) { error in
            guard case ConfigError.invalidFormat = error else {
                return XCTFail("Expected invalidFormat, got \(error)")
            }
        }
    }

    func testInitDefaults() throws {
        let config = try YAMLConfig(from: """
            model_path: /tmp/model
            tokenizer_model: /tmp/tok
            """)
        XCTAssertEqual(config.contextLength, 2048)
        XCTAssertEqual(config.stateLength, 2048)
        XCTAssertEqual(config.batchSize, 32)
        XCTAssertEqual(config.lutBits, 4)
        XCTAssertEqual(config.numChunks, 1)
        XCTAssertEqual(config.splitLMHead, 8)
        XCTAssertNil(config.vocabSize)
        XCTAssertNil(config.lmHeadChunkSizes)
        XCTAssertFalse(config.isMonolithic)
        XCTAssertNil(config.monolithicModelPath)
        XCTAssertFalse(config.argmaxInModel)
        XCTAssertNil(config.slidingWindow)
        XCTAssertFalse(config.updateMaskPrefill)
        XCTAssertFalse(config.prefillDynamicSlice)
        XCTAssertEqual(config.modelPrefix, "llama")
        XCTAssertNil(config.recommendedSampling)
        XCTAssertEqual(config.configVersion, "0.3.5")
    }

    func testInitStateLengthDefaultsToContextLength() throws {
        let config = try YAMLConfig(from: """
            model_path: /tmp/model
            tokenizer_model: /tmp/tok
            context_length: 1024
            """)
        XCTAssertEqual(config.stateLength, 1024)

        let explicit = try YAMLConfig(from: """
            model_path: /tmp/model
            tokenizer_model: /tmp/tok
            context_length: 1024
            state_length: 4096
            """)
        XCTAssertEqual(explicit.stateLength, 4096)
    }

    func testInitAddsFirstChunkSuffixForMultiChunkFFN() throws {
        let config = try YAMLConfig(from: """
            model_path: /tmp/model
            tokenizer_model: /tmp/tok
            num_chunks: 2
            ffn_path: /models/qwen_FFN_PF_lut4.mlmodelc
            """)
        XCTAssertEqual(config.ffnPath, "/models/qwen_FFN_PF_lut4_chunk_01of02.mlmodelc")
    }

    func testInitKeepsFFNPathThatAlreadyHasChunkSuffix() throws {
        let config = try YAMLConfig(from: """
            model_path: /tmp/model
            tokenizer_model: /tmp/tok
            num_chunks: 2
            ffn_path: /models/qwen_FFN_PF_lut4_chunk_01of02.mlmodelc
            """)
        XCTAssertEqual(config.ffnPath, "/models/qwen_FFN_PF_lut4_chunk_01of02.mlmodelc")
    }

    func testInitRecommendedSamplingAcceptsStringsAndCamelCaseKeys() throws {
        let config = try YAMLConfig(from: """
            model_path: /tmp/model
            tokenizer_model: /tmp/tok
            recommended_sampling:
              temperature: "0.6"
              topP: 0.95
              topK: "20"
            """)
        let sampling = try XCTUnwrap(config.recommendedSampling)
        XCTAssertTrue(sampling.doSample, "do_sample should default to true")
        XCTAssertEqual(sampling.temperature, 0.6, accuracy: 1e-9)
        XCTAssertEqual(sampling.topP, 0.95, accuracy: 1e-9)
        XCTAssertEqual(sampling.topK, 20)
    }

    func testInitRecommendedSamplingNeedsTemperatureTopPAndTopK() throws {
        let config = try YAMLConfig(from: """
            model_path: /tmp/model
            tokenizer_model: /tmp/tok
            recommended_sampling:
              do_sample: false
              temperature: 0.7
              top_p: 0.9
            """)
        XCTAssertNil(config.recommendedSampling)
    }

    // MARK: - load(from:)

    func testLoadMissingFileThrows() {
        let path = tempDir.appendingPathComponent("does-not-exist.yaml").path
        XCTAssertThrowsError(try YAMLConfig.load(from: path)) { error in
            guard case ConfigError.invalidFormat = error else {
                return XCTFail("Expected invalidFormat, got \(error)")
            }
        }
    }

    func testLoadRequiresModelInfo() throws {
        let path = try writeMeta("parameters:\n  context_length: 512\n")
        XCTAssertThrowsError(try YAMLConfig.load(from: path)) { error in
            guard case ConfigError.missingField("model_info") = error else {
                return XCTFail("Expected missingField(model_info), got \(error)")
            }
        }
    }

    func testLoadRequiresParameters() throws {
        let path = try writeMeta("model_info:\n  name: test\n")
        XCTAssertThrowsError(try YAMLConfig.load(from: path)) { error in
            guard case ConfigError.missingField("model_info.parameters") = error else {
                return XCTFail("Expected missingField(model_info.parameters), got \(error)")
            }
        }
    }

    /// Same layout as the meta.yaml that convert_model.sh writes for a chunked --argmax model.
    func testLoadChunkedMetaWithPredefinedPaths() throws {
        try touchModel("qwen25_FFN_PF_lut4_chunk_01of01.mlmodelc")
        let path = try writeMeta("""
            model_info:
              name: anemll-Qwen-Qwen2.5-0.5B-ctx512
              version: 0.3.5
              architecture: qwen2
              parameters:
                context_length: 512
                batch_size: 64
                lut_embeddings: none
                lut_ffn: 4
                lut_lmhead: 6
                num_chunks: 1
                model_prefix: qwen25
                embeddings: qwen25_embeddings.mlmodelc
                lm_head: qwen25_lm_head_lut6.mlmodelc
                ffn: qwen25_FFN_PF_lut4_chunk_01of01.mlmodelc
                split_lm_head: 16
                argmax_in_model: true
                vocab_size: 151936
                lm_head_chunk_sizes: [9496, 9496, 9496, 9496, 9496, 9496, 9496, 9496, 9496, 9496, 9496, 9496, 9496, 9496, 9496, 9496]
                prefill_dynamic_slice: true
            """)

        let config = try YAMLConfig.load(from: path)
        XCTAssertFalse(config.isMonolithic)
        XCTAssertNil(config.monolithicModelPath)
        XCTAssertEqual(config.embedPath, "\(baseDir)/qwen25_embeddings.mlmodelc")
        XCTAssertEqual(config.lmheadPath, "\(baseDir)/qwen25_lm_head_lut6.mlmodelc")
        XCTAssertEqual(config.ffnPath, "\(baseDir)/qwen25_FFN_PF_lut4_chunk_01of01.mlmodelc")
        XCTAssertEqual(config.modelPath, config.ffnPath)
        XCTAssertEqual(config.tokenizerModel, baseDir)
        XCTAssertEqual(config.contextLength, 512)
        XCTAssertEqual(config.stateLength, 512)
        XCTAssertEqual(config.batchSize, 64)
        XCTAssertEqual(config.numChunks, 1)
        XCTAssertEqual(config.modelPrefix, "qwen25")
        XCTAssertEqual(config.splitLMHead, 16)
        XCTAssertTrue(config.argmaxInModel)
        XCTAssertEqual(config.vocabSize, 151936)
        XCTAssertEqual(config.lmHeadChunkSizes, Array(repeating: 9496, count: 16))
        XCTAssertTrue(config.prefillDynamicSlice)
        XCTAssertFalse(config.updateMaskPrefill)
        XCTAssertNil(config.slidingWindow)
        XCTAssertEqual(config.configVersion, "0.3.5")
    }

    /// Same layout as the meta.yaml that convert_monolith.sh writes.
    func testLoadMonolithicMeta() throws {
        let path = try writeMeta("""
            model_info:
              name: anemll-Qwen-Qwen2.5-0.5B-ctx512-monolithic
              version: 0.3.5
              model_type: monolithic
              parameters:
                context_length: 512
                batch_size: 64
                lut_bits: 6
                lut_per_channel: 4
                model_prefix: qwen25
                monolithic_model: qwen25_monolithic_full_lut6.mlmodelc
                split_lm_head: 16
                argmax_in_model: true
                vocab_size: 151936
                prefill_dynamic_slice: true
                functions:
                  - infer
                  - prefill
            """)

        let config = try YAMLConfig.load(from: path)
        XCTAssertTrue(config.isMonolithic)
        XCTAssertEqual(config.monolithicModelPath, "\(baseDir)/qwen25_monolithic_full_lut6.mlmodelc")
        XCTAssertEqual(config.modelPath, config.monolithicModelPath)
        XCTAssertEqual(config.lutBits, 6)
        XCTAssertTrue(config.argmaxInModel)
        XCTAssertNil(config.lmHeadChunkSizes)
    }

    func testLoadMonolithicFallbackPathUsesLutBits() throws {
        let path = try writeMeta("""
            model_info:
              model_type: monolithic
              parameters:
                context_length: 512
                lut_bits: 4
                model_prefix: llama
            """)
        let config = try YAMLConfig.load(from: path)
        XCTAssertEqual(config.monolithicModelPath, "\(baseDir)/llama_monolithic_full_lut4.mlmodelc")
    }

    func testLoadBuildsPathsFromPrefixAndLutWhenNotPredefined() throws {
        let path = try writeMeta("""
            model_info:
              parameters:
                context_length: 512
                lut_embeddings: none
                lut_ffn: 4
                lut_lmhead: 6
                num_chunks: 2
                model_prefix: llama
            """)
        let config = try YAMLConfig.load(from: path)
        XCTAssertEqual(config.embedPath, "\(baseDir)/llama_embeddings.mlmodelc")
        XCTAssertEqual(config.lmheadPath, "\(baseDir)/llama_lm_head_lut6.mlmodelc")
        XCTAssertEqual(config.ffnPath, "\(baseDir)/llama_FFN_PF_lut4_chunk_01of02.mlmodelc")
    }

    func testLoadSingleChunkPrefersChunkSuffixWhenItExists() throws {
        try touchModel("llama_FFN_PF_lut4_chunk_01of01.mlmodelc")
        let path = try writeMeta("""
            model_info:
              parameters:
                lut_ffn: 4
                num_chunks: 1
                model_prefix: llama
            """)
        let config = try YAMLConfig.load(from: path)
        XCTAssertEqual(config.ffnPath, "\(baseDir)/llama_FFN_PF_lut4_chunk_01of01.mlmodelc")
    }

    func testLoadSingleChunkWithoutChunkSuffixOnDisk() throws {
        let path = try writeMeta("""
            model_info:
              parameters:
                lut_ffn: 4
                num_chunks: 1
                model_prefix: llama
            """)
        let config = try YAMLConfig.load(from: path)
        XCTAssertEqual(config.ffnPath, "\(baseDir)/llama_FFN_PF_lut4.mlmodelc")
    }

    func testLoadPredefinedFFNFallsBackToChunkSuffix() throws {
        try touchModel("llama_FFN_PF_lut4_chunk_01of01.mlmodelc")
        let path = try writeMeta("""
            model_info:
              parameters:
                num_chunks: 1
                model_prefix: llama
                ffn: llama_FFN_PF_lut4.mlmodelc
            """)
        let config = try YAMLConfig.load(from: path)
        XCTAssertEqual(config.ffnPath, "\(baseDir)/llama_FFN_PF_lut4_chunk_01of01.mlmodelc")
    }

    func testLoadGemma3DefaultsSlidingWindowAboveContext512() throws {
        let path = try writeMeta("""
            model_info:
              parameters:
                context_length: 1024
                model_prefix: gemma3
            """)
        XCTAssertEqual(try YAMLConfig.load(from: path).slidingWindow, 512)
    }

    func testLoadGemma3NoSlidingWindowAtContext512() throws {
        let path = try writeMeta("""
            model_info:
              parameters:
                context_length: 512
                model_prefix: gemma3
            """)
        XCTAssertNil(try YAMLConfig.load(from: path).slidingWindow)
    }

    func testLoadExplicitSlidingWindowWins() throws {
        let path = try writeMeta("""
            model_info:
              parameters:
                context_length: 4096
                model_prefix: gemma3_1b
                sliding_window: 1024
            """)
        XCTAssertEqual(try YAMLConfig.load(from: path).slidingWindow, 1024)
    }

    func testLoadPassesThroughRecommendedSampling() throws {
        let path = try writeMeta("""
            model_info:
              parameters:
                model_prefix: qwen
                recommended_sampling:
                  do_sample: true
                  temperature: 0.6
                  top_p: 0.95
                  top_k: 20
            """)
        let sampling = try XCTUnwrap(try YAMLConfig.load(from: path).recommendedSampling)
        XCTAssertTrue(sampling.doSample)
        XCTAssertEqual(sampling.temperature, 0.6, accuracy: 1e-9)
        XCTAssertEqual(sampling.topP, 0.95, accuracy: 1e-9)
        XCTAssertEqual(sampling.topK, 20)
    }
}
