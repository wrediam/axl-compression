#!/usr/bin/env python3
"""
AXL Compression Test with OpenRouter Support

Tests AXL compression with modern LLMs via OpenRouter API.
Supports GPT-4, Claude, Gemini, and other frontier models.
"""

import os
import re
import time
import json
from typing import List, Dict, Tuple
import requests

try:
    import tiktoken
except ImportError:
    print("Error: 'tiktoken' package not found.")
    print("Install with: pip install tiktoken")
    exit(1)


class AXLCompressor:
    """AXL compression system - aggressive alphanumeric compression."""
    
    def __init__(self):
        # Rule 1: Core codes
        self.number_codes = {
            'the': '1', 'to': '2', 'too': '2', 'two': '2',
            'and': '3', 'for': '4', 'of': '5', 'is': '6',
            'as': '6', 'it': '7', 'its': '7', 'that': '8',
            'in': '9', 'not': '0', 'no': '0',
        }
        
        self.letter_codes = {
            'a': 'a', 'be': 'b', 'see': 'c', 'have': 'h',
            'I': 'i', 'know': 'k', 'are': 'r', 'you': 'u',
            'with': 'w', 'why': 'y',
        }
        
        # Rule 3: Suffix compression
        self.suffix_codes = {
            'ing': '1g', 'ed': '2d', 'tion': '5n', 'sion': '5n',
            'ment': '6t', 'ly': '7y', 'able': '8l', 'ible': '8l',
            'ful': '9l',
        }
        
        self.all_codes = {**self.number_codes, **self.letter_codes}
        self.system_prompt = self._generate_system_prompt()
    
    def _generate_system_prompt(self) -> str:
        """Clear system prompt teaching AXL."""
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
        first, last, middle = word[0], word[-1], word[1:-1]
        middle_compressed = re.sub(r'[aeiouAEIOU]', '', middle)
        return first + middle_compressed + last
    
    def _compress_suffix(self, word: str) -> str:
        """Rule 3: Compress common suffixes."""
        word_lower = word.lower()
        for suffix, code in sorted(self.suffix_codes.items(), key=lambda x: len(x[0]), reverse=True):
            if word_lower.endswith(suffix):
                base = word[:-len(suffix)]
                base_compressed = self._remove_vowels(base)
                return base_compressed + code
        return None
    
    def compress_text(self, text: str) -> str:
        """Compress text using AXL system."""
        words = re.findall(r'\b\w+\b|[^\w\s]', text)
        compressed_words = []
        
        for word in words:
            if not word.strip() or not word.isalnum():
                compressed_words.append(word)
                continue
            
            word_lower = word.lower()
            
            if word_lower in self.all_codes:
                compressed_words.append(self.all_codes[word_lower])
            else:
                suffix_result = self._compress_suffix(word)
                if suffix_result:
                    compressed_words.append(suffix_result)
                else:
                    compressed_words.append(self._remove_vowels(word))
        
        return ' '.join(compressed_words)


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


class OpenRouterInterface:
    """Handle OpenRouter API interaction."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        if not self.api_key:
            raise ValueError("OpenRouter API key required. Set OPENROUTER_API_KEY environment variable.")
        
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def list_models(self) -> List[str]:
        """Get available models from OpenRouter."""
        return [
            "openai/gpt-4-turbo",
            "openai/gpt-4",
            "openai/gpt-3.5-turbo",
            "anthropic/claude-3.5-sonnet",
            "anthropic/claude-3-opus",
            "anthropic/claude-3-sonnet",
            "google/gemini-pro-1.5",
            "meta-llama/llama-3.1-70b-instruct",
            "meta-llama/llama-3.1-8b-instruct",
        ]
    
    def generate(self, model: str, system_prompt: str, user_message: str) -> str:
        """Generate response from OpenRouter."""
        try:
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ]
            }
            
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            data = response.json()
            return data['choices'][0]['message']['content']
            
        except Exception as e:
            return f"Error: {e}"


class AXLCompressionSession:
    """Session manager with AXL compression."""
    
    def __init__(self, model: str, api_interface, use_compression: bool = True):
        self.model = model
        self.use_compression = use_compression
        self.compressor = AXLCompressor()
        self.token_counter = TokenCounter()
        self.api = api_interface
        
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
            compressed_content = self.compressor.compress_text(content)
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
        
        compression_start = time.time()
        self.add_to_history('user', user_message)
        conversation_context = self.build_conversation_context()
        compression_time = time.time() - compression_start
        
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
        
        system_tokens = self.token_counter.count(system_prompt)
        context_tokens = self.token_counter.count(full_prompt)
        total_tokens_this_turn = system_tokens + context_tokens
        
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
        
        llm_start = time.time()
        response = self.api.generate(self.model, system_prompt, full_prompt)
        llm_time = time.time() - llm_start
        
        compression_start = time.time()
        self.add_to_history('assistant', response)
        compression_time += time.time() - compression_start
        
        total_time = time.time() - start_time
        
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


def run_openrouter_test(model: str, api_key: str = None):
    """Run test with OpenRouter model."""
    print("\n" + "="*80)
    print(f"AXL COMPRESSION TEST - OpenRouter")
    print("="*80)
    
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
    
    api = OpenRouterInterface(api_key)
    
    print(f"Model: {model}")
    print(f"Messages: {len(test_messages)}")
    
    # Warmup
    print(f"\nWarming up {model}...", end=" ", flush=True)
    warmup_start = time.time()
    api.generate(model, "You are a helpful assistant.", "Hello, just warming up. Reply with 'Ready'.")
    warmup_time = time.time() - warmup_start
    print(f"✓ ({warmup_time:.2f}s)\n")
    
    # Test with AXL compression
    print("Testing WITH AXL compression...")
    compressed_session = AXLCompressionSession(model, api, use_compression=True)
    
    for i, msg in enumerate(test_messages, 1):
        print(f"  {i}/{len(test_messages)}...", end=" ", flush=True)
        show_ctx = (i % 3 == 0)
        compressed_session.send_message(msg, show_context=show_ctx)
        print("✓")
    
    # Test without compression
    print("\nTesting WITHOUT compression...")
    normal_session = AXLCompressionSession(model, api, use_compression=False)
    
    for i, msg in enumerate(test_messages, 1):
        print(f"  {i}/{len(test_messages)}...", end=" ", flush=True)
        show_ctx = (i % 3 == 0)
        normal_session.send_message(msg, show_context=show_ctx)
        print("✓")
    
    # Results
    compressed_stats = compressed_session.get_cumulative_stats()
    normal_stats = normal_session.get_cumulative_stats()
    
    print("\n" + "="*80)
    print("RESULTS - AXL vs NORMAL")
    print("="*80)
    
    print("\nTOKEN EFFICIENCY:")
    print(f"  Normal:              {normal_stats['total_normal']} tokens")
    print(f"  AXL Compressed:      {compressed_stats['total_compressed']} tokens")
    print(f"  Tokens saved:        {compressed_stats['total_saved']} tokens")
    print(f"  Token savings:       {compressed_stats['savings_percent']}")
    print(f"  System overhead:     {compressed_stats['system_overhead']} tokens")
    
    print("\nCHARACTER EFFICIENCY:")
    print(f"  Normal chars:        {compressed_stats['char_count_normal']}")
    print(f"  AXL chars:           {compressed_stats['char_count_compressed']}")
    print(f"  Character reduction: {compressed_stats['char_reduction']}")
    
    print("\nTIMING COMPARISON:")
    print(f"  Normal total time:        {normal_stats['total_time']:.3f}s")
    print(f"  AXL total time:           {compressed_stats['total_time']:.3f}s")
    print(f"  Time difference:          {compressed_stats['total_time'] - normal_stats['total_time']:.3f}s")
    print(f"  Compression overhead:     {compressed_stats['compression_time']:.3f}s")
    print(f"  Avg time per msg (normal):     {normal_stats['avg_time_per_msg']:.3f}s")
    print(f"  Avg time per msg (AXL):        {compressed_stats['avg_time_per_msg']:.3f}s")
    
    if compressed_stats['total_time'] < normal_stats['total_time']:
        speedup = ((normal_stats['total_time'] - compressed_stats['total_time']) / normal_stats['total_time'] * 100)
        print(f"  ⚡ AXL is {speedup:.1f}% FASTER")
    else:
        slowdown = ((compressed_stats['total_time'] - normal_stats['total_time']) / normal_stats['total_time'] * 100)
        print(f"  AXL is {slowdown:.1f}% SLOWER")
    
    print(f"\n{'✅ AXL COMPRESSION WORKS!' if compressed_stats['is_efficient'] else 'System prompt overhead dominates at this message count'}")
    print("="*80)


if __name__ == "__main__":
    import sys
    
    # Check for API key
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("Error: OPENROUTER_API_KEY environment variable not set")
        print("\nSet it with:")
        print("  export OPENROUTER_API_KEY='your-key-here'")
        print("\nGet your key at: https://openrouter.ai/keys")
        sys.exit(1)
    
    # List available models
    api = OpenRouterInterface(api_key)
    models = api.list_models()
    
    print("\nAvailable models:")
    for i, model in enumerate(models, 1):
        print(f"  {i}. {model}")
    
    # Get model selection
    if len(sys.argv) > 1:
        model_choice = sys.argv[1]
    else:
        choice = input(f"\nSelect model (1-{len(models)}) or enter model name: ").strip()
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                model_choice = models[idx]
            else:
                print("Invalid selection")
                sys.exit(1)
        except ValueError:
            model_choice = choice
    
    print(f"\nSelected: {model_choice}")
    
    try:
        run_openrouter_test(model_choice, api_key)
    except KeyboardInterrupt:
        print("\n\nTest interrupted")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
