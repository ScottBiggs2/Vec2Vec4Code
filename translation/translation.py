import torch
import torch.nn.functional as F
import requests
import json
import os
import subprocess
import tempfile
from typing import Optional, Tuple
import argparse

from data.data import PYTHON_SAMPLES, C_SAMPLES
from blocks.core import CodeVec2Vec, Vec2VecTrainer
from saving.saving import save_vec2vec_model, load_vec2vec_model
from demo.demo import demo_vec2vec_code_translation

class CodeTranslator:
    """
    Main class for translating code between Python and C using trained Vec2Vec model
    """
    
    def __init__(self, model_path: str = "models/python_c_translator_20250708_153052.pth",  ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url
        self.model_name = "deepseek-coder:1.3B"
        
        # Load the trained model
        self.model, self.checkpoint = self.load_model(model_path)
        self.embedding_dim = self.checkpoint['embedding_dim']
        
        print(f"✅ Loaded model with {self.embedding_dim}D embeddings")
        print(f"🔗 Connected to Ollama at {ollama_url}")
    
    def load_model(self, model_path: str):
        """Load the trained Vec2Vec model"""
        from main import CodeVec2Vec  # Import your model class
        
        checkpoint = torch.load(model_path, map_location='cpu')
        
        model = CodeVec2Vec(
            embedding_dim=checkpoint['embedding_dim'],
            latent_dim=checkpoint['latent_dim']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model, checkpoint
    
    def get_embedding(self, code: str) -> torch.Tensor:
        """Get embedding for a code string using Ollama"""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/embeddings",
                json={
                    "model": self.model_name,
                    "prompt": code
                }
            )
            
            if response.status_code == 200:
                embedding = response.json()['embedding']
                return torch.tensor([embedding], dtype=torch.float32)
            else:
                raise Exception(f"Ollama API error: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error getting embedding: {e}")
            return None
    
    def find_nearest_code(self, target_embedding: torch.Tensor, code_database: list, embeddings_database: torch.Tensor) -> Tuple[str, float]:
        """Find the nearest code in the database using cosine similarity"""
        with torch.no_grad():
            similarities = F.cosine_similarity(target_embedding, embeddings_database, dim=1)
            best_idx = similarities.argmax().item()
            best_similarity = similarities[best_idx].item()
            
            return code_database[best_idx], best_similarity
    
    def translate_code(self, code: str, source_lang: str, target_lang: str, code_database: dict) -> Tuple[str, float]:
        """
        Translate code from source language to target language
        
        Args:
            code: Source code string
            source_lang: 'python' or 'c'
            target_lang: 'python' or 'c'  
            code_database: Dict with 'python' and 'c' keys containing code examples
        
        Returns:
            Tuple of (translated_code, similarity_score)
        """
        
        # Get embedding for input code
        print(f"🔍 Getting embedding for {source_lang} code...")
        source_embedding = self.get_embedding(code)
        if source_embedding is None:
            return "ERROR: Could not get embedding", 0.0
        
        # Translate embedding using Vec2Vec model
        print(f"🔄 Translating {source_lang} → {target_lang}...")
        with torch.no_grad():
            if source_lang == 'python' and target_lang == 'c':
                translated_embedding = self.model.translate_py_to_c(source_embedding)
            elif source_lang == 'c' and target_lang == 'python':
                translated_embedding = self.model.translate_c_to_py(source_embedding)
            else:
                return "ERROR: Unsupported language pair", 0.0
        
        # Get embeddings for target language database
        print(f"📚 Searching {target_lang} code database...")
        target_codes = code_database[target_lang]
        target_embeddings = []
        
        for target_code in target_codes:
            emb = self.get_embedding(target_code)
            if emb is not None:
                target_embeddings.append(emb)
        
        if not target_embeddings:
            return "ERROR: Could not process target database", 0.0
        
        target_embeddings = torch.cat(target_embeddings, dim=0)
        
        # Find nearest match
        translated_code, similarity = self.find_nearest_code(
            translated_embedding, target_codes, target_embeddings
        )
        
        print(f"✅ Translation complete! Similarity: {similarity:.4f}")
        return translated_code, similarity

def create_snowflake_database():
    """Create a database of Python and C code examples for snowflake generation"""
    
    python_codes = [
        # Basic snowflake patterns
        """
import random

def generate_snowflake(size=5):
    patterns = ['*', '+', 'o', '.', '#']
    for i in range(size):
        spaces = ' ' * (size - i - 1)
        stars = random.choice(patterns) * (2 * i + 1)
        print(spaces + stars)
        
    for i in range(size - 2, -1, -1):
        spaces = ' ' * (size - i - 1)
        stars = random.choice(patterns) * (2 * i + 1)
        print(spaces + stars)

generate_snowflake(6)
""",
        
        # Simple snowflake with fixed pattern
        """
def simple_snowflake():
    pattern = "    *\\n   ***\\n  *****\\n *******\\n*********\\n *******\\n  *****\\n   ***\\n    *"
    print(pattern)

simple_snowflake()
""",
        
        # Randomized snowflake
        """
import random

def random_snowflake():
    size = random.randint(3, 7)
    chars = ['*', '+', 'o', '#', '.']
    
    for i in range(size):
        padding = ' ' * (size - i - 1)
        char = random.choice(chars)
        line = char * (2 * i + 1)
        print(padding + line)

random_snowflake()
""",
        
        # ASCII art snowflake
        """
def ascii_snowflake():
    print("      *      ")
    print("     /|\\     ")
    print("    * | *    ")
    print("   /  |  \\   ")
    print("  *   |   *  ")
    print(" /    |    \\ ")
    print("*     |     *")
    print(" \\    |    / ")
    print("  *   |   *  ")
    print("   \\  |  /   ")
    print("    * | *    ")
    print("     \\|/     ")
    print("      *      ")

ascii_snowflake()
"""
    ]
    
    c_codes = [
        # Basic C snowflake
        """
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void generate_snowflake(int size) {
    char patterns[] = {'*', '+', 'o', '.', '#'};
    srand(time(NULL));
    
    for(int i = 0; i < size; i++) {
        for(int j = 0; j < size - i - 1; j++) {
            printf(" ");
        }
        char pattern = patterns[rand() % 5];
        for(int k = 0; k < 2 * i + 1; k++) {
            printf("%c", pattern);
        }
        printf("\\n");
    }
    
    for(int i = size - 2; i >= 0; i--) {
        for(int j = 0; j < size - i - 1; j++) {
            printf(" ");
        }
        char pattern = patterns[rand() % 5];
        for(int k = 0; k < 2 * i + 1; k++) {
            printf("%c", pattern);
        }
        printf("\\n");
    }
}

int main() {
    generate_snowflake(6);
    return 0;
}
""",
        
        # Simple C snowflake
        """
#include <stdio.h>

void simple_snowflake() {
    printf("    *\\n");
    printf("   ***\\n");
    printf("  *****\\n");
    printf(" *******\\n");
    printf("*********\\n");
    printf(" *******\\n");
    printf("  *****\\n");
    printf("   ***\\n");
    printf("    *\\n");
}

int main() {
    simple_snowflake();
    return 0;
}
""",
        
        # Random C snowflake
        """
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void random_snowflake() {
    srand(time(NULL));
    int size = rand() % 5 + 3;
    char chars[] = {'*', '+', 'o', '#', '.'};
    
    for(int i = 0; i < size; i++) {
        for(int j = 0; j < size - i - 1; j++) {
            printf(" ");
        }
        char ch = chars[rand() % 5];
        for(int k = 0; k < 2 * i + 1; k++) {
            printf("%c", ch);
        }
        printf("\\n");
    }
}

int main() {
    random_snowflake();
    return 0;
}
""",
        
        # ASCII art C snowflake
        """
#include <stdio.h>

void ascii_snowflake() {
    printf("      *      \\n");
    printf("     /|\\     \\n");
    printf("    * | *    \\n");
    printf("   /  |  \\   \\n");
    printf("  *   |   *  \\n");
    printf(" /    |    \\ \\n");
    printf("*     |     *\\n");
    printf(" \\    |    / \\n");
    printf("  *   |   *  \\n");
    printf("   \\  |  /   \\n");
    printf("    * | *    \\n");
    printf("     \\|/     \\n");
    printf("      *      \\n");
}

int main() {
    ascii_snowflake();
    return 0;
}
"""
    ]
    
    return {
        'python': python_codes,
        'c': c_codes
    }

def compile_and_run_c(c_code: str) -> str:
    """Compile and run C code, return output"""
    try:
        # Create temporary files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as f:
            f.write(c_code)
            c_file = f.name
        
        # Compile
        exe_file = c_file.replace('.c', '')
        compile_result = subprocess.run(
            ['gcc', c_file, '-o', exe_file],
            capture_output=True,
            text=True
        )
        
        if compile_result.returncode != 0:
            return f"Compilation Error:\n{compile_result.stderr}"
        
        # Run
        run_result = subprocess.run(
            [exe_file],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        # Cleanup
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
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(python_code)
            py_file = f.name
        
        # Run
        result = subprocess.run(
            ['python', py_file],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        # Cleanup
        os.unlink(py_file)
        
        if result.returncode != 0:
            return f"Error:\n{result.stderr}"
        
        return result.stdout
        
    except Exception as e:
        return f"Error: {str(e)}"

def snowflake_demo(model_path: str):
    """
    Demo: Generate Python snowflake code, translate to C, and run both
    """
    print("🎄 ===== SNOWFLAKE TRANSLATION DEMO ===== 🎄")
    
    # Initialize translator
    translator = CodeTranslator(model_path)
    
    # Create code database
    code_db = create_snowflake_database()
    
    # Original Python snowflake code
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
    
    # Run Python code
    print("🚀 Running Python Code:")
    python_output = run_python_code(python_snowflake)
    print(python_output)
    
    # Translate to C
    print("🔄 Translating Python → C...")
    c_code, similarity = translator.translate_code(
        python_snowflake, 'python', 'c', code_db
    )
    
    print(f"⚡ Translation Similarity: {similarity:.4f}")
    print("\n🔧 Translated C Code:")
    print("=" * 50)
    print(c_code)
    print("=" * 50)
    
    # Run C code
    print("🚀 Running C Code:")
    c_output = compile_and_run_c(c_code)
    print(c_output)
    
    # Compare outputs
    print("📊 Comparison:")
    print(f"Python output length: {len(python_output.strip().split())}")
    print(f"C output length: {len(c_output.strip().split())}")
    
    if similarity > 0.8:
        print("✅ High similarity translation!")
    elif similarity > 0.6:
        print("⚠️  Moderate similarity translation")
    else:
        print("❌ Low similarity - may need more training data")

def interactive_translator(model_path: str):
    """Interactive code translation interface"""
    print("🔄 Interactive Code Translator")
    print("Enter 'quit' to exit, 'demo' for snowflake demo")
    
    translator = CodeTranslator(model_path)
    code_db = create_snowflake_database()
    
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
            
            # Translate
            translated, similarity = translator.translate_code(
                code, source_lang, target_lang, code_db
            )
            
            print(f"\n✅ Translation complete! (Similarity: {similarity:.4f})")
            print("Translated code:")
            print("-" * 40)
            print(translated)
            print("-" * 40)
        else:
            print("❌ Unknown command. Use 'translate', 'demo', or 'quit'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Code Translation using Vec2Vec")
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


# python translator.py models/your_model_file.pth --demo
# python translation/translation.py models/python_c_translator_20
# 250808_153052.pth --demo