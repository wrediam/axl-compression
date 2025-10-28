# AXL: AlphaNumeric Text Accelerator

A conversation compression system for Large Language Models that reduces token usage and improves inference speed through aggressive text compression.

## Overview

AXL compresses conversation history using a three-layer approach:
1. High-frequency word codes (single characters)
2. Vowel reduction (preserving first and last letters)
3. Suffix compression (common endings)

The system allows LLMs to process compressed conversation history while responding in normal English, with Python automatically compressing responses before adding them to the conversation chain.

## Performance Results

Tested with Qwen3:8B on 10-message conversations:

**Speed Improvement:**
- Normal: 365 seconds
- AXL: 290 seconds
- **20.5% faster inference**

**Character Reduction:**
- Normal: 25,102 characters
- AXL: 21,105 characters
- **15.9% compression**

**Token Efficiency:**
- Overhead: 1,920 tokens (system prompt)
- Break-even: ~10 messages
- Net savings increase with conversation length

## Installation

```bash
pip install ollama tiktoken
```

Requires Ollama running locally with a model installed:
```bash
ollama pull qwen3:8b
```

## Usage

Run the test suite:
```bash
python3 axl_compression.py
```

The script will:
1. Warm up the model
2. Run 10 test messages with AXL compression
3. Run the same messages without compression
4. Display comparative results

## How It Works

### Compression Rules

**Rule 1: Core Codes**
```
Numbers: 1=the, 2=to, 3=and, 4=for, 5=of, 6=is, 7=it, 8=that, 9=in, 0=not
Letters: a=a, b=be, c=see, h=have, i=I, k=know, r=are, u=you, w=with, y=why
```

**Rule 2: Vowel Reduction**
```
message → msge
understand → undrstnd
people → pple
```

**Rule 3: Suffix Compression**
```
running → rn1g (ing → 1g)
walked → wlk2d (ed → 2d)
quickly → qck7y (ly → 7y)
```

### Example Compression

**Original:**
```
I think that artificial intelligence is going to change everything. What do you think?
```

**AXL Compressed:**
```
i thnk 8 artfcl intllgnce 6 go1g 2 chnge evryth1g . wht do u thnk ?
```

**Reduction:** 22.1%

## System Architecture

The system uses a clear, readable system prompt that teaches the model to interpret AXL compression while responding in normal English. Python handles all compression/decompression automatically.

**Conversation Flow:**
1. User input → compressed → added to history
2. Compressed history sent to LLM with legend
3. LLM responds in normal English
4. Response → compressed → added to history
5. Repeat

## Requirements

- Python 3.8+
- ollama
- tiktoken
- Local Ollama installation with model

## Testing

The test script includes:
- Automated comparison between compressed and normal modes
- Real-time conversation history display
- Token counting and timing metrics
- Character reduction statistics

## Limitations

- System prompt overhead: ~1,920 tokens
- Compression is lossy (cannot perfectly reconstruct original)
- Requires ~10+ messages to achieve net token savings
- Best suited for long conversations

## Future Work

- Reduce system prompt overhead
- Adaptive compression based on conversation length
- Support for multiple languages
- Fine-tuned models that understand AXL natively

## License

MIT

## Author

Will Reeves
