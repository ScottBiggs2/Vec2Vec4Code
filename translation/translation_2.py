import torch
import torch.nn.functional as F
import requests
import json
import os
import subprocess
from transformers import AutoTokenizer, AutoModelForCausalLM
import tempfile
from typing import Optional, Tuple, List
import argparse
# from blocks.embed import CodeEmbedder

try:
    from blocks.embed import CodeEmbedder
except ImportError:
    try:
        import sys
        sys.path.append('.')
        from blocks.embed import CodeEmbedder
        print(f"Loaded CodeVec2Vec Embedder")
    except ImportError:
        print("❌ Could not import CodeVec2Vec Embedder from blocks.embed")
        print("Make sure you're running from the project root directory")
        print("Current working directory:", os.getcwd())
        raise

class CodeTranslator:
    """
    Main class for translating code between Python and C using trained Vec2Vec model
    """
    
    def __init__(self, model_path: str, ollama_url: str = "http://localhost:11434"):
        # Load the trained vec2vec model
        self.model, self.checkpoint = self.load_model(model_path)
        self.embedding_dim = self.checkpoint['embedding_dim']
        
        # Initialize the code embedder (same model used for training)
        self.embedder = CodeEmbedder()

        # Ollama settings for cleanup
        self.ollama_url = ollama_url
        self.ollama_model_name = "deepseek-coder:1.3B"

        print(f"✅ Loaded vec2vec model with {self.embedding_dim}D embeddings")
        print(f"🔗 Using Ollama for cleanup at {self.ollama_url}")
    
    def load_model(self, model_path: str):
        """Load the trained Vec2Vec model"""
        try:
            from blocks.core import CodeVec2Vec
        except ImportError:
            try:
                import sys
                sys.path.append('.')
                from blocks.core import CodeVec2Vec
            except ImportError:
                print("❌ Could not import CodeVec2Vec from blocks.core")
                print("Make sure you're running from the project root directory")
                print("Current working directory:", os.getcwd())
                raise
        
        checkpoint = torch.load(model_path, map_location='cpu')
        
        model = CodeVec2Vec(
            embedding_dim=checkpoint['embedding_dim'],
            latent_dim=checkpoint['latent_dim']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model, checkpoint
    
    def get_embedding(self, code: str) -> torch.Tensor:
        """Get embedding for a code string using tokenizer embeddings"""
        return self.embedder.embed_code([code])
    
    def embedding_to_tokens(self, target_embedding: torch.Tensor, target_lang: str) -> List[int]:
        """
        Convert embedding directly to tokens using vec2vec trained space
        No similarity search - direct token generation from embedding
        """
        
        print(f"🧠 Converting embedding to {target_lang} tokens...")
        
        # Use the trained vec2vec model to decode embedding to tokens
        # This assumes the vec2vec model learned token-level mappings
        
        # For now, let's use a simple approach: find the tokens that would 
        # produce the closest embedding to our target
        with torch.no_grad():
            # Get all possible token embeddings
            all_token_embeddings = self.embedder.embedding_layer.weight
            
            # Find most similar tokens to our target embedding
            # target_embedding is [1, embed_dim], all_token_embeddings is [vocab_size, embed_dim]
            similarities = F.cosine_similarity(
                target_embedding,
                all_token_embeddings,
                dim=1
            )  # [vocab_size]
            
            # Get top tokens
            top_k = 50  # Get more tokens for better sequence building
            top_indices = similarities.topk(top_k).indices.tolist()
            
            # Build a sequence from these top tokens
            # Simple approach: take the most similar tokens and try to form coherent sequence
            
            if target_lang == 'python':
                # Look for Python-relevant tokens
                python_tokens = []
                for idx in top_indices:
                    token_text = self.embedder.tokenizer.decode([idx])
                    if any(keyword in token_text.lower() for keyword in ['def', 'print', 'import', 'for', 'if', 'return']):
                        python_tokens.append(idx)
                
                # Start with a reasonable sequence
                if python_tokens:
                    return python_tokens[:10]  # Return first 10 relevant tokens
                else:
                    return top_indices[:10]    # Fallback to top similar tokens
            
            else:  # target_lang == 'c'
                # Look for C-relevant tokens
                c_tokens = []
                for idx in top_indices:
                    token_text = self.embedder.tokenizer.decode([idx])
                    if any(keyword in token_text.lower() for keyword in ['#include', 'printf', 'int', 'main', 'for', 'if', 'return']):
                        c_tokens.append(idx)
                
                # Start with a reasonable sequence
                if c_tokens:
                    return c_tokens[:10]
                else:
                    return top_indices[:10]
    
    def embedding_to_code(self, target_embedding: torch.Tensor, target_lang: str) -> Tuple[str, float]:
        """
        Convert embedding to code via direct tokenization
        """
        
        # Get tokens from embedding
        tokens = self.embedding_to_tokens(target_embedding, target_lang)
        
        # Decode tokens to code
        raw_code = self.tokens_to_code(tokens)
        
        # Check if we need cleanup
        if self.should_clean_code(raw_code, target_lang):
            cleaned_code = self.clean_with_llm(raw_code, target_lang)
            final_code = cleaned_code if cleaned_code else raw_code
        else:
            final_code = raw_code
        
        # Calculate confidence by checking how close the final code's embedding is to target
        final_embedding = self.get_embedding(final_code)
        confidence = F.cosine_similarity(target_embedding, final_embedding, dim=1).item()
        
        return final_code, confidence
        
    def tokens_to_code(self, tokens: List[int]) -> str:
        """Convert tokens back to code using tokenizer"""
        return self.embedder.tokenizer.decode(tokens, skip_special_tokens=True)
    
    def should_clean_code(self, code: str, target_lang: str) -> bool:
        """Determine if code needs LLM cleaning"""
        # Simple heuristic: if the code is very short or lacks basic keywords, it's likely garbled.
        if len(code.strip()) < 10:
            return True
        if target_lang == 'python':
            # Check if it looks like valid Python
            return not any(keyword in code for keyword in ['def ', 'import ', 'print(', 'for ', 'if '])
        else:  # target_lang == 'c'
            # Check if it looks like valid C
            return not any(keyword in code for keyword in ['#include', 'int main', 'printf(', 'for(', 'if('])
    
    def clean_with_llm(self, raw_code: str, target_lang: str) -> str:
        """Use Ollama to clean up the raw, decoded code string."""
        print("🧹 Cleaning up generated code with Ollama...")
        
        lang_name = "Python" if target_lang == 'python' else "C"
        prompt = f"""
        The following text is raw {target_lang} code. Review and revise it and return only clean, executable {target_lang} code.

Do not add any explanations, comments, or markdown formatting. Only return the code block enclosed in backticks.

Raw code:
'''{raw_code}'''
"""
        
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.ollama_model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 250  # Max tokens to generate
                    }
                }
            )
            
            if response.status_code == 200:
                cleaned = response.json().get('response', '').strip()
                # LLMs sometimes wrap code in markdown, let's strip it.
                if cleaned.startswith(f"```{target_lang}"):
                    cleaned = cleaned.split('\n', 1)[1]
                    if cleaned.endswith("```"):
                        cleaned = cleaned[:-3].strip()
                elif cleaned.startswith("```"):

                    return cleaned if cleaned else raw_code
            else:
                print(f"⚠️  Ollama API error: {response.status_code} - {response.text}")
                return raw_code

        except requests.exceptions.RequestException as e:
            print(f"⚠️  LLM cleaning failed: Could not connect to Ollama at {self.ollama_url}. Error: {e}")
            print("Please ensure Ollama is running and the model is pulled: `ollama pull deepseek-coder:1.3B`")
            return raw_code
        except Exception as e:
            print(f"⚠️  LLM cleaning failed with an unexpected error: {e}")
            return raw_code
    
    def translate_code(self, code: str, source_lang: str, target_lang: str) -> Tuple[str, float]:
        """
        Translate code directly from tokenization A to tokenization B
        
        Pipeline: Code → Tokens → Vec2Vec → Tokens → Code
        """
        
        # Step 1: Code → Tokens
        print(f"🔍 Tokenizing {source_lang} code...")
        source_tokens = self.embedder.tokenizer.encode(code, truncation=True, max_length=512)
        
        # Step 2: Tokens → Embeddings (for vec2vec input)
        # source_embeddings = self.embedder.embedding_layer[source_tokens]  # was embedding_matrix [seq_len, embed_dim]
        source_embedding = self.get_embedding(code)  # Use the same method as training
        # source_embeddings = self.embedder.embedding_layer(torch.tensor(source_tokens))  # [seq_len, embed_dim]
        source_embedding = source_embedding.mean(dim=0).unsqueeze(0)  # [1, embed_dim]
        
        # Step 3: Embedding → Embedding (vec2vec)
        print(f"🔄 Translating {source_lang} → {target_lang} via vec2vec...")
        with torch.no_grad():
            if source_lang == 'python' and target_lang == 'c':
                target_embedding = self.model.translate_py_to_c(source_embedding)
            elif source_lang == 'c' and target_lang == 'python':
                target_embedding = self.model.translate_c_to_py(source_embedding)
            else:
                return "ERROR: Unsupported language pair", 0.0
        
        # Step 4: Embedding → Tokens (direct mapping)
        print(f"🧠 Converting to {target_lang} tokens...")
        target_tokens = self.embedding_to_tokens(target_embedding, target_lang)
        
        # Step 5: Tokens → Code
        raw_code = self.tokens_to_code(target_tokens)
        
        # Step 6: Optional cleanup
        if self.should_clean_code(raw_code, target_lang):
            cleaned_code = self.clean_with_llm(raw_code, target_lang)
            final_code = cleaned_code if cleaned_code else raw_code
        else:
            final_code = raw_code
        
        # Calculate confidence
        final_embedding = self.get_embedding(final_code)
        confidence = F.cosine_similarity(target_embedding, final_embedding, dim=1).item()
        
        print(f"✅ Translation complete! Confidence: {confidence:.4f}")
        return final_code, confidence

def compile_and_run_c(c_code: str) -> str:
    """Compile and run C code, return output"""
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as f:
            f.write(c_code)
            c_file = f.name
        
        exe_file = c_file.replace('.c', '')
        compile_result = subprocess.run(
            ['gcc', c_file, '-o', exe_file],
            capture_output=True,
            text=True
        )
        
        if compile_result.returncode != 0:
            return f"Compilation Error:\n{compile_result.stderr}"
        
        run_result = subprocess.run(
            [exe_file],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        os.unlink(c_file)
        os.unlink(exe_file)
        
        if run_result.returncode != 0:
            return f"Runtime Error:\n{run_result.stderr}"
        
        return run_result.stdout
        
    except Exception as e:
        return f"Error: {str(e)}"

def run_python_code(python_code: str) -> str:
    """Run Python code and return output"""
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(python_code)
            py_file = f.name
        
        result = subprocess.run(
            ['python', py_file],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        os.unlink(py_file)
        
        if result.returncode != 0:
            return f"Error:\n{result.stderr}"
        
        return result.stdout
        
    except Exception as e:
        return f"Error: {str(e)}"

def snowflake_demo(model_path: str):
    """Demo: Generate Python snowflake code, translate to C, and run both"""
    print("🎄 ===== SNOWFLAKE TRANSLATION DEMO ===== 🎄")
    
    translator = CodeTranslator(model_path)
    
    python_snowflake = """
import random

def generate_snowflake(size=5):
    patterns = ['*', '+', 'o', '.', '#']
    for i in range(size):
        spaces = ' ' * (size - i - 1)
        stars = random.choice(patterns) * (2 * i + 1)
        print(spaces + stars)

generate_snowflake(4)
"""
    
    print("🐍 Original Python Code:")
    print("=" * 50)
    print(python_snowflake)
    print("=" * 50)
    
    print("🚀 Running Python Code:")
    python_output = run_python_code(python_snowflake)
    print(python_output)
    
    print("🔄 Translating Python → C using transformers...")
    c_code, confidence = translator.translate_code(python_snowflake, 'python', 'c')
    
    print(f"⚡ Translation Confidence: {confidence:.4f}")
    print("\n🔧 Translated C Code:")
    print("=" * 50)
    print(c_code)
    print("=" * 50)
    
    print("🚀 Running C Code:")
    c_output = compile_and_run_c(c_code)
    print(c_output)
    
    print("📊 Comparison:")
    if confidence > 0.7:
        print("✅ High confidence translation!")
    elif confidence > 0.5:
        print("⚠️  Moderate confidence translation")
    else:
        print("❌ Low confidence - may need better embedding alignment")

def interactive_translator(model_path: str):
    """Interactive code translation interface"""
    print("🔄 Interactive Code Translator (Transformers-based)")
    print("Enter 'quit' to exit, 'demo' for snowflake demo")
    
    translator = CodeTranslator(model_path)
    
    while True:
        print("\n" + "="*50)
        command = input("Enter command (translate/demo/quit): ").strip().lower()
        
        if command == 'quit':
            break
        elif command == 'demo':
            snowflake_demo(model_path)
        elif command == 'translate':
            source_lang = input("Source language (python/c): ").strip().lower()
            target_lang = input("Target language (python/c): ").strip().lower()
            
            if source_lang not in ['python', 'c'] or target_lang not in ['python', 'c']:
                print("❌ Invalid language. Use 'python' or 'c'")
                continue
            
            print("Enter your code (end with '###' on a new line):")
            code_lines = []
            while True:
                line = input()
                if line.strip() == '###':
                    break
                code_lines.append(line)
            
            code = '\n'.join(code_lines)
            
            if not code.strip():
                print("❌ No code entered")
                continue
            
            # Translate using transformers
            translated, confidence = translator.translate_code(code, source_lang, target_lang)
            
            print(f"\n✅ Translation complete! (Confidence: {confidence:.4f})")
            print("Translated code:")
            print("-" * 40)
            print(translated)
            print("-" * 40)
            
            # Run original code
            if source_lang == 'python':
                print("\n🚀 Running original Python code:")
                print(run_python_code(code))
            elif source_lang == 'c':
                print("\n🚀 Running original C code:")
                print(compile_and_run_c(code))

            # Run translated code
            if target_lang == 'python':
                print("\n🚀 Running translated Python code:")
                print(run_python_code(translated))
            elif target_lang == 'c':
                print("\n🚀 Running translated C code:")
                print(compile_and_run_c(translated))
        else:
            print("❌ Unknown command. Use 'translate', 'demo', or 'quit'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Code Translation using Vec2Vec + Transformers")
    parser.add_argument("model_path", help="Path to trained Vec2Vec model (.pth file)")
    parser.add_argument("--demo", action="store_true", help="Run snowflake demo")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"❌ Model file not found: {args.model_path}")
        exit(1)
    
    if args.demo:
        snowflake_demo(args.model_path)
    elif args.interactive:
        interactive_translator(args.model_path)
    else:
        print("Use --demo or --interactive flag")
        parser.print_help()


# python translation/translation_2.py models/python_c_translator_20250709_142739.pth --demo
# python translation/translation_2.py models/python_c_translator_20250709_152907.pth --demo