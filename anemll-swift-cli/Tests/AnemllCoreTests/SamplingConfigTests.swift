import XCTest
@testable import AnemllCore

final class SamplingConfigTests: XCTestCase {
    func testGreedyPreset() {
        let config = SamplingConfig.greedy
        XCTAssertFalse(config.doSample)
        XCTAssertEqual(config.temperature, 0.0)
        XCTAssertEqual(config.topK, 0)
        XCTAssertEqual(config.topP, 1.0)
        XCTAssertEqual(config.repetitionPenalty, 1.0)
    }

    func testDefaultSamplingPreset() {
        let config = SamplingConfig.defaultSampling
        XCTAssertTrue(config.doSample)
        XCTAssertEqual(config.temperature, 0.7)
        XCTAssertEqual(config.topK, 50)
        XCTAssertEqual(config.topP, 0.95)
        XCTAssertEqual(config.repetitionPenalty, 1.1)
    }
}
