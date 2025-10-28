#!/usr/bin/env python3
"""
Legend Compression System V3 - AXL (AlphaNumeric Text Accelerator)

Aggressive compression system achieving ~50% reduction using:
1. High-frequency single-character codes
2. Vowel reduction
3. Suffix compression

Based on the AXL compression method.
"""

import re
import time
from typing import List, Dict, Tuple
import ollama
import tiktoken


class AXLCompressor:
    """AXL compression system - aggressive alphanumeric compression."""
    
    def __init__(self):
        # Rule 1: Core codes (single character replacements)
        self.number_codes = {
            'the': '1',
            'to': '2',
            'too': '2',
            'two': '2',
            'and': '3',
            'for': '4',
            'of': '5',
            'is': '6',
            'as': '6',
            'it': '7',
            'its': '7',
            'that': '8',
            'in': '9',
            'not': '0',
            'no': '0',
        }
        
        self.letter_codes = {
            'a': 'a',
            'be': 'b',
            'see': 'c',
            'have': 'h',
            'I': 'i',
            'know': 'k',
            'are': 'r',
            'you': 'u',
            'with': 'w',
            'why': 'y',
        }
        
        # Rule 3: Suffix compression
        self.suffix_codes = {
            'ing': '1g',
            'ed': '2d',
            'tion': '5n',
            'sion': '5n',
            'ment': '6t',
            'ly': '7y',
            'able': '8l',
            'ible': '8l',
            'ful': '9l',
        }
        
        # Combine all codes for reverse lookup
        self.all_codes = {**self.number_codes, **self.letter_codes}
        self.reverse_codes = {v: k for k, v in self.all_codes.items()}
        
        # Ultra-minimal system prompt
        self.system_prompt = self._generate_system_prompt()
    
    def _generate_system_prompt(self) -> str:
        """Clear system prompt teaching AXL - readable for the model."""
        # Build compact legend but keep instructions clear
        num_legend = " ".join([f"{v}={k}" for k, v in self.number_codes.items()])
        let_legend = " ".join([f"{v}={k}" for k, v in self.letter_codes.items()])
        sfx_legend = " ".join([f"{v}={k}" for k, v in self.suffix_codes.items()])
        
        return f"""The conversation history below uses AXL compression to save tokens:
- Number codes: {num_legend}
- Letter codes: {let_legend}
- Suffix codes: {sfx_legend}
- Vowels removed from most words (first/last letters kept)

Read the compressed history, understand it, then respond in normal, clear, uncompressed English."""
    
    def _remove_vowels(self, word: str) -> str:
        """Rule 2: Remove internal vowels, keep first and last letters."""
        if len(word) <= 2:
            return word
        
        first = word[0]
        last = word[-1]
        middle = word[1:-1]
        
        # Remove vowels from middle
        middle_compressed = re.sub(r'[aeiouAEIOU]', '', middle)
        
        return first + middle_compressed + last
    
    def _compress_suffix(self, word: str) -> str:
        """Rule 3: Compress common suffixes."""
        word_lower = word.lower()
        
        # Check each suffix (longest first)
        for suffix, code in sorted(self.suffix_codes.items(), key=lambda x: len(x[0]), reverse=True):
            if word_lower.endswith(suffix):
                base = word[:-len(suffix)]
                # Apply vowel reduction to base
                base_compressed = self._remove_vowels(base)
                return base_compressed + code
        
        return None
    
    def compress_text(self, text: str) -> str:
        """Compress text using AXL system."""
        # Split into words while preserving punctuation
        words = re.findall(r'\b\w+\b|[^\w\s]', text)
        compressed_words = []
        
        for word in words:
            if not word.strip() or not word.isalnum():
                # Keep punctuation as-is
                compressed_words.append(word)
                continue
            
            word_lower = word.lower()
            
            # Rule 1: Check core codes first
            if word_lower in self.all_codes:
                compressed_words.append(self.all_codes[word_lower])
            else:
                # Rule 3: Try suffix compression
                suffix_result = self._compress_suffix(word)
                if suffix_result:
                    compressed_words.append(suffix_result)
                else:
                    # Rule 2: Vowel reduction
                    compressed_words.append(self._remove_vowels(word))
        
        return ' '.join(compressed_words)
    
    def decompress_text(self, text: str) -> str:
        """Decompress for human display (approximate)."""
        # This is lossy - we can't perfectly reconstruct
        # Just expand known codes
        result = text
        for code, word in self.reverse_codes.items():
            result = re.sub(r'\b' + re.escape(code) + r'\b', word, result)
        return result


class TokenCounter:
    """Accurate token counting."""
    
    def __init__(self):
        try:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        except:
            self.encoding = None
    
    def count(self, text: str) -> int:
        """Count actual tokens."""
        if self.encoding:
            return len(self.encoding.encode(text))
        else:
            return len(text) // 4


class OllamaInterface:
    """Handle Ollama model interaction."""
    
    def __init__(self):
        self.client = ollama.Client()
    
    def list_models(self) -> List[str]:
        """Get available Ollama models."""
        try:
            result = self.client.list()
            if hasattr(result, 'models'):
                return [model.model for model in result.models]
            return []
        except Exception as e:
            print(f"❌ Error: {e}")
            return []
    
    def generate(self, model: str, system_prompt: str, user_message: str) -> str:
        """Generate response from Ollama."""
        try:
            response = self.client.chat(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ]
            )
            return response['message']['content']
        except Exception as e:
            return f"❌ Error: {e}"


class AXLCompressionSession:
    """Session manager with AXL compression."""
    
    def __init__(self, model: str, use_compression: bool = True):
        self.model = model
        self.use_compression = use_compression
        self.compressor = AXLCompressor()
        self.token_counter = TokenCounter()
        self.ollama = OllamaInterface()
        
        self.history: List[Dict[str, str]] = []
        
        self.stats = {
            'total_tokens_normal': 0,
            'total_tokens_compressed': 0,
            'messages': 0,
            'system_prompt_overhead': 0,
            'total_time': 0.0,
            'compression_time': 0.0,
            'char_count_normal': 0,
            'char_count_compressed': 0
        }
    
    def add_to_history(self, role: str, content: str):
        """Add message to history - always compress before storing."""
        original_content = content
        
        if self.use_compression:
            # Always compress before storing in history
            compressed_content = self.compressor.compress_text(content)
            # Track character reduction
            self.stats['char_count_normal'] += len(original_content)
            self.stats['char_count_compressed'] += len(compressed_content)
            
            self.history.append({
                'role': role,
                'content': compressed_content
            })
        else:
            self.history.append({
                'role': role,
                'content': content
            })
    
    def build_conversation_context(self) -> str:
        """Build conversation history string."""
        context_parts = []
        for msg in self.history:
            role_marker = "U:" if msg['role'] == 'user' else "A:"
            context_parts.append(f"{role_marker} {msg['content']}")
        return "\n".join(context_parts)
    
    def send_message(self, user_message: str, verbose: bool = False, show_context: bool = False) -> Tuple[str, Dict]:
        """Send message and get response with token stats."""
        start_time = time.time()
        
        # Time compression
        compression_start = time.time()
        self.add_to_history('user', user_message)
        conversation_context = self.build_conversation_context()
        compression_time = time.time() - compression_start
        
        # Show conversation context if requested
        if show_context:
            mode = "AXL COMPRESSED" if self.use_compression else "NORMAL"
            print(f"\n{'='*70}")
            print(f"📝 {mode} CONVERSATION HISTORY:")
            print(f"{'='*70}")
            print(conversation_context)
            if self.use_compression:
                char_reduction = (1 - self.stats['char_count_compressed'] / self.stats['char_count_normal']) * 100
                print(f"\n💾 Character reduction: {char_reduction:.1f}%")
            print(f"{'='*70}\n")
        
        if self.use_compression:
            system_prompt = self.compressor.system_prompt
            full_prompt = f"{conversation_context}"
        else:
            system_prompt = "You are a helpful AI assistant."
            full_prompt = conversation_context
        
        # Count tokens
        system_tokens = self.token_counter.count(system_prompt)
        context_tokens = self.token_counter.count(full_prompt)
        total_tokens_this_turn = system_tokens + context_tokens
        
        # Calculate normal (uncompressed) equivalent
        if self.use_compression:
            normal_context = "\n".join([
                f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
                for msg in self.history
            ])
            normal_system = "You are a helpful AI assistant."
            normal_tokens = self.token_counter.count(normal_system + normal_context)
        else:
            normal_tokens = total_tokens_this_turn
        
        if verbose:
            print(f"\n🤖 Sending to {self.model}...")
        
        # Time LLM response
        llm_start = time.time()
        response = self.ollama.generate(self.model, system_prompt, full_prompt)
        llm_time = time.time() - llm_start
        
        # Time adding response to history
        compression_start = time.time()
        self.add_to_history('assistant', response)
        compression_time += time.time() - compression_start
        
        total_time = time.time() - start_time
        
        # Update stats
        self.stats['messages'] += 1
        self.stats['total_tokens_normal'] += normal_tokens
        self.stats['total_tokens_compressed'] += total_tokens_this_turn
        self.stats['system_prompt_overhead'] += system_tokens
        self.stats['total_time'] += total_time
        self.stats['compression_time'] += compression_time
        
        turn_stats = {
            'system_tokens': system_tokens,
            'context_tokens': context_tokens,
            'total_this_turn': total_tokens_this_turn,
            'normal_this_turn': normal_tokens,
            'saved_this_turn': normal_tokens - total_tokens_this_turn,
            'compression_ratio': f"{(1 - total_tokens_this_turn/normal_tokens)*100:.1f}%" if normal_tokens > 0 else "0%",
            'total_time': total_time,
            'llm_time': llm_time,
            'compression_time': compression_time
        }
        
        return response, turn_stats
    
    def get_cumulative_stats(self) -> Dict:
        """Get cumulative statistics."""
        total_saved = self.stats['total_tokens_normal'] - self.stats['total_tokens_compressed']
        savings_pct = (total_saved / self.stats['total_tokens_normal'] * 100) if self.stats['total_tokens_normal'] > 0 else 0
        
        char_reduction = 0
        if self.stats['char_count_normal'] > 0:
            char_reduction = (1 - self.stats['char_count_compressed'] / self.stats['char_count_normal']) * 100
        
        return {
            'messages': self.stats['messages'],
            'total_normal': self.stats['total_tokens_normal'],
            'total_compressed': self.stats['total_tokens_compressed'],
            'total_saved': total_saved,
            'savings_percent': f"{savings_pct:.1f}%",
            'system_overhead': self.stats['system_prompt_overhead'],
            'is_efficient': total_saved > 0,
            'total_time': self.stats['total_time'],
            'compression_time': self.stats['compression_time'],
            'avg_time_per_msg': self.stats['total_time'] / self.stats['messages'] if self.stats['messages'] > 0 else 0,
            'char_reduction': f"{char_reduction:.1f}%",
            'char_count_normal': self.stats['char_count_normal'],
            'char_count_compressed': self.stats['char_count_compressed']
        }


def run_axl_test():
    """Run test with AXL compression."""
    print("\n" + "="*80)
    print("🧪 AXL COMPRESSION TEST - AlphaNumeric Text Accelerator")
    print("="*80)
    
    # Test messages
    test_messages = [
        "I think that artificial intelligence is going to change everything. What do you think?",
        "That being said, could you please explain machine learning in simple terms?",
        "To be honest, I'm curious about neural networks. Let me know how they work.",
        "In other words, for example, how would a large language model process this text?",
        "However, on the other hand, I think that deep learning is complex. You know what I mean?",
        "Can you explain the difference between artificial intelligence and machine learning?",
        "What are the main applications of neural networks in natural language processing?",
        "How do large language models handle context in long conversations?",
        "What is the role of machine learning in modern artificial intelligence systems?",
        "Can you describe how neural networks learn from data?"
    ]
    
    ollama_if = OllamaInterface()
    model = "qwen3:8b"
    
    print(f"📊 Model: {model}")
    print(f"📝 Messages: {len(test_messages)}")
    
    # Warmup
    print(f"\n🔥 Warming up {model}...", end=" ", flush=True)
    warmup_start = time.time()
    ollama_if.generate(model, "You are a helpful assistant.", "Hello, just warming up. Reply with 'Ready'.")
    warmup_time = time.time() - warmup_start
    print(f"✓ ({warmup_time:.2f}s)\n")
    
    # Test with AXL compression
    print("🗜️  Testing WITH AXL compression...")
    compressed_session = AXLCompressionSession(model, use_compression=True)
    
    for i, msg in enumerate(test_messages, 1):
        print(f"  {i}/{len(test_messages)}...", end=" ", flush=True)
        # Show context at messages 3, 6, 9 to see progression
        show_ctx = (i % 3 == 0)
        compressed_session.send_message(msg, show_context=show_ctx)
        print("✓")
    
    # Test without compression
    print("\n📄 Testing WITHOUT compression...")
    normal_session = AXLCompressionSession(model, use_compression=False)
    
    for i, msg in enumerate(test_messages, 1):
        print(f"  {i}/{len(test_messages)}...", end=" ", flush=True)
        # Show context at message 3, 6, 9
        show_ctx = (i % 3 == 0)
        normal_session.send_message(msg, show_context=show_ctx)
        print("✓")
    
    # Results
    compressed_stats = compressed_session.get_cumulative_stats()
    normal_stats = normal_session.get_cumulative_stats()
    
    print("\n" + "="*80)
    print("📊 RESULTS - AXL vs NORMAL")
    print("="*80)
    
    print("\n🔢 TOKEN EFFICIENCY:")
    print(f"  Normal:              {normal_stats['total_normal']} tokens")
    print(f"  AXL Compressed:      {compressed_stats['total_compressed']} tokens")
    print(f"  Tokens saved:        {compressed_stats['total_saved']} tokens")
    print(f"  Token savings:       {compressed_stats['savings_percent']}")
    print(f"  System overhead:     {compressed_stats['system_overhead']} tokens")
    
    print("\n💾 CHARACTER EFFICIENCY:")
    print(f"  Normal chars:        {compressed_stats['char_count_normal']}")
    print(f"  AXL chars:           {compressed_stats['char_count_compressed']}")
    print(f"  Character reduction: {compressed_stats['char_reduction']}")
    
    print("\n⏱️  TIMING COMPARISON:")
    print(f"  Normal total time:        {normal_stats['total_time']:.3f}s")
    print(f"  AXL total time:           {compressed_stats['total_time']:.3f}s")
    print(f"  Time difference:          {compressed_stats['total_time'] - normal_stats['total_time']:.3f}s")
    print(f"  Compression overhead:     {compressed_stats['compression_time']:.3f}s ({compressed_stats['compression_time']/compressed_stats['total_time']*100:.1f}% of total)")
    print(f"  Avg time per msg (normal):     {normal_stats['avg_time_per_msg']:.3f}s")
    print(f"  Avg time per msg (AXL):        {compressed_stats['avg_time_per_msg']:.3f}s")
    
    # Speed comparison
    if compressed_stats['total_time'] < normal_stats['total_time']:
        speedup = ((normal_stats['total_time'] - compressed_stats['total_time']) / normal_stats['total_time'] * 100)
        print(f"  ⚡ AXL is {speedup:.1f}% FASTER")
    else:
        slowdown = ((compressed_stats['total_time'] - normal_stats['total_time']) / normal_stats['total_time'] * 100)
        print(f"  🐌 AXL is {slowdown:.1f}% SLOWER")
    
    print(f"\n{'✅ AXL COMPRESSION WORKS!' if compressed_stats['is_efficient'] else '❌ Still inefficient'}")
    print("="*80)
    
    # Analysis
    if compressed_stats['is_efficient']:
        print("\n💡 SUCCESS: AXL compression is beneficial!")
        print(f"   Token savings: {compressed_stats['total_saved']} tokens")
        print(f"   Character reduction: {compressed_stats['char_reduction']}")
    else:
        print("\n⚠️  INSIGHT: System prompt overhead still dominates")
        overhead = compressed_stats['system_overhead']
        if compressed_stats['total_saved'] < 0:
            savings_per_msg = abs(compressed_stats['total_saved']) / compressed_stats['messages']
            breakeven = int(overhead / savings_per_msg) if savings_per_msg > 0 else 999
            print(f"   Estimated break-even: ~{breakeven} messages")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    try:
        run_axl_test()
    except KeyboardInterrupt:
        print("\n\n👋 Test interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
