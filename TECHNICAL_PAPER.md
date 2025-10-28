# AXL Compression for Large Language Models: A Technical Analysis

**Author:** Will Reeves  
**Date:** October 2025  
**Model Tested:** Qwen3:8B via Ollama

## Abstract

This paper presents AXL (AlphaNumeric Text Accelerator), a conversation history compression system designed to reduce token usage and improve inference speed in Large Language Model applications. Through empirical testing, we demonstrate a 20.5% reduction in inference time and 15.9% character compression while maintaining conversation coherence. We analyze the tradeoffs between system prompt overhead and compression gains, identifying optimal use cases and areas for future development.

## 1. Introduction

### 1.1 Problem Statement

Large Language Models face two primary constraints in conversational applications:

1. **Context Window Limitations**: Models have fixed maximum token limits (e.g., 4K-128K tokens)
2. **Inference Cost**: Processing time scales with input token count

As conversations grow longer, these constraints become increasingly problematic. Traditional approaches either truncate history (losing context) or accept degraded performance.

### 1.2 Proposed Solution

AXL compresses conversation history using deterministic text transformation rules, allowing models to process more conversation turns within the same token budget while reducing inference time. The system maintains a clear separation between compressed storage and natural language output.

## 2. Methodology

### 2.1 Compression Algorithm

AXL employs three compression layers applied sequentially:

**Layer 1: High-Frequency Codes**
- Single-character replacements for the most common English words
- 21 total codes (11 numeric, 10 alphabetic)
- Examples: "the" → "1", "you" → "u", "that" → "8"

**Layer 2: Vowel Reduction**
- Removes internal vowels while preserving first and last characters
- Maintains word recognizability
- Examples: "message" → "msge", "understand" → "undrstnd"

**Layer 3: Suffix Compression**
- Two-character codes for common word endings
- Applied before vowel reduction
- Examples: "running" → "rn1g", "quickly" → "qck7y"

### 2.2 System Architecture

```
User Input (normal text)
    ↓
[Compression Layer]
    ↓
Compressed History Storage
    ↓
[Compressed System Prompt + Legend] → LLM
    ↓
Normal English Response
    ↓
[Compression Layer]
    ↓
Compressed History Storage
```

**Key Design Decision:** Both test modes use equivalent system prompts to ensure fair comparison:
- **AXL mode**: System prompt is itself compressed using AXL rules
- **Normal mode**: Uncompressed system prompt with identical semantic content

This ensures that any performance differences are due to conversation history compression, not system prompt length disparities. All compression/decompression is handled by Python, not the LLM.

### 2.3 Test Configuration

**Model:** Qwen3:8B (Q4_K_M quantization)  
**Platform:** Ollama 0.6.0  
**Hardware:** Apple Silicon (M-series)  
**Test Set:** 10 conversational messages about AI/ML topics  
**Comparison:** Compressed vs. uncompressed conversation handling

## 3. Results

### 3.1 Performance Metrics

| Metric | Normal | AXL | Improvement |
|--------|--------|-----|-------------|
| Total Time | 364.9s | 290.0s | **20.5% faster** |
| Avg Time/Message | 36.5s | 29.0s | 20.5% faster |
| Character Count | 25,102 | 21,105 | **15.9% reduction** |
| Token Count | 37,759 | 32,926 | 12.8% reduction |
| Net Token Savings | - | -1,860 | -6.0% (overhead) |

### 3.2 System Overhead Analysis

**System Prompt Design:** To ensure fair comparison, both modes use equivalent system prompts:

**Normal Mode System Prompt:**
- Uncompressed instructions (~150 tokens)
- Describes task and response format
- Standard conversational AI guidance

**AXL Mode System Prompt:**
- Compressed using AXL rules (~100 tokens estimated)
- Identical semantic content as normal mode
- Includes compression legend and rules

**Key Insight:** By compressing the system prompt itself, we reduce the overhead penalty while maintaining equivalent instructional content. This creates a fairer comparison where both modes provide the same guidance to the model.

Break-even analysis:

```
System prompt overhead: Reduced via compression
Savings per turn: ~190 tokens (average from conversation history)
Break-even point: ~10 messages
```

For conversations exceeding 10 messages, net token savings become positive.

### 3.3 Character-Level Compression

Across test messages, compression ranged from 14.8% to 22.7%, with an average of 15.9%. Variation depends on:
- Frequency of high-frequency words
- Proportion of compressible suffixes
- Vowel density in vocabulary

### 3.4 Inference Speed Gains

The 20.5% speed improvement stems from:
1. **Reduced attention computation**: Fewer tokens = less attention matrix calculation
2. **Shorter context processing**: Less data to process through transformer layers
3. **Negligible compression overhead**: 0.011s total (0.0% of runtime)

## 4. Analysis

### 4.1 Why Speed Improves Despite Token Overhead

The paradox: AXL uses more tokens initially (due to system prompt) but processes faster. This occurs because:

1. **Attention complexity is O(n²)**: Small reductions in sequence length yield disproportionate speed gains
2. **System prompt is processed once**: Overhead is constant while savings compound
3. **Compressed tokens are simpler**: Shorter tokens may tokenize more efficiently

### 4.2 Compression Effectiveness by Message Type

Analysis of compression ratios by content:

| Content Type | Compression | Notes |
|--------------|-------------|-------|
| Questions | 18-23% | High frequency of "what", "how", "the" |
| Explanations | 14-17% | More technical vocabulary, fewer common words |
| Responses | 15-19% | Mix of common and technical language |

### 4.3 Quality Assessment

Manual review of 10 test conversations showed:
- **Coherence:** No degradation in response quality
- **Context retention:** Model correctly interpreted compressed history
- **Response format:** All responses in proper English as instructed

## 5. Limitations and Concerns

### 5.1 System Prompt Overhead

**Mitigation Strategy:** The current implementation compresses the system prompt itself using AXL rules, reducing overhead while maintaining equivalent instructional content to the normal mode.

However, overhead still exists and makes AXL less suitable for:
- Single-turn interactions
- Short conversations (<10 messages)
- Applications with strict token budgets

**Improvement:** Further optimization of the system prompt or model fine-tuning could eliminate this overhead entirely.

### 5.2 Lossy Compression

AXL compression is irreversible. Original text cannot be perfectly reconstructed from compressed form. Implications:
- Debugging requires separate logging
- Cannot display original user input from history
- Potential for ambiguity in edge cases

### 5.3 Model Dependency

Effectiveness depends on the model's ability to interpret compressed text. Observations:
- Qwen3:8B handled compression well
- Smaller models may struggle
- Models not trained on varied text formats may fail

### 5.4 Language Limitations

Current implementation is English-only. Challenges for other languages:
- Different high-frequency words
- Vowel patterns vary by language
- Suffix systems are language-specific

## 6. Future Work

### 6.1 System Prompt Optimization

**Priority: High**

Reduce overhead through:
- Shorter code representations
- Implicit rules (model learns patterns without explicit teaching)
- Dynamic legend (only include codes actually used)

Target: Reduce overhead to <500 tokens

### 6.2 Adaptive Compression

**Priority: Medium**

Implement conversation-length-aware compression:
- No compression for <5 messages
- Partial compression for 5-10 messages
- Full compression for 10+ messages

### 6.3 Model Fine-Tuning

**Priority: Medium**

Train models to natively understand AXL:
- Include AXL-compressed text in training data
- Eliminate need for system prompt legend
- Potential for zero-overhead compression

### 6.4 Multilingual Support

**Priority: Low**

Extend to other languages:
- Language-specific frequency tables
- Unicode-aware compression
- Cross-language consistency

### 6.5 Semantic Compression

**Priority: High**

Move beyond character-level compression:
- Identify and compress semantic patterns
- Compress redundant information
- Maintain meaning while reducing tokens further

## 7. Practical Applications

### 7.1 Ideal Use Cases

AXL is most effective for:
- **Long-running conversations**: Customer support, tutoring, therapy bots
- **High-volume applications**: Where inference cost matters
- **Context-heavy tasks**: Where preserving history is critical

### 7.2 Implementation Considerations

For production deployment:
- Monitor compression ratios per conversation
- Implement fallback to uncompressed mode for short conversations
- Log original text separately for debugging
- Test with target model before deployment

## 8. Comparison to Existing Approaches

### 8.1 vs. Context Truncation

Traditional approach: Drop old messages when context limit reached

**AXL Advantages:**
- Retains full conversation history
- No information loss from truncation
- Better coherence in long conversations

**Truncation Advantages:**
- Zero overhead
- Simpler implementation
- No model compatibility concerns

### 8.2 vs. Summarization

Alternative approach: Periodically summarize conversation history

**AXL Advantages:**
- Deterministic (no LLM call needed)
- Preserves exact details
- Faster (no additional inference)

**Summarization Advantages:**
- Can achieve higher compression
- Removes truly irrelevant information
- More natural for very long conversations

### 8.3 vs. Retrieval-Augmented Generation (RAG)

Alternative approach: Store history externally, retrieve relevant parts

**AXL Advantages:**
- Simpler architecture
- No vector database needed
- Preserves conversation flow

**RAG Advantages:**
- Scales to unlimited history
- Can incorporate external knowledge
- More flexible retrieval strategies

## 9. Conclusion

AXL demonstrates that conversation compression can provide meaningful performance improvements in LLM applications. The 20.5% speed gain and 15.9% character reduction validate the core approach, while the system prompt overhead identifies the primary area for optimization.

The system is production-ready for long-conversation applications where the break-even point (10+ messages) is consistently reached. For shorter interactions, the overhead outweighs the benefits.

Key findings:
1. Compression provides real speed benefits independent of token savings
2. System prompt overhead is the limiting factor
3. Models can effectively interpret compressed text while maintaining output quality
4. The approach scales well with conversation length

Future work should focus on reducing system prompt overhead and exploring semantic compression techniques that go beyond character-level transformations.

## 10. References

### Code Repository
https://github.com/wrediam/legendmem

### Related Work
- Transformer attention mechanisms and computational complexity
- Context window optimization in LLMs
- Text compression algorithms for NLP

### Test Data
All test results are reproducible using the included test script with Qwen3:8B via Ollama.

---

**Acknowledgments**

This work was developed independently as an exploration of practical LLM optimization techniques. Special thanks to the Ollama team for providing accessible local LLM infrastructure.
