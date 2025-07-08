import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import requests
import json
import random
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt

class CodeEmbedder:
    """Embeds code using DeepSeek Coder model via Ollama"""
    
    def __init__(self, model_name="deepseek-coder:1.3B", ollama_url="http://localhost:11434"):
        self.model_name = model_name
        self.ollama_url = ollama_url
        
        # Test connection to Ollama
        try:
            response = requests.get(f"{ollama_url}/api/tags")
            if response.status_code == 200:
                available_models = [model['name'] for model in response.json()['models']]
                if model_name not in available_models:
                    print(f"Model {model_name} not found. Available models: {available_models}")
                    print(f"To pull the model, run: ollama pull {model_name}")
                else:
                    print(f"Successfully connected to Ollama with model {model_name}")
            else:
                raise ConnectionError(f"Could not connect to Ollama at {ollama_url}")
        except Exception as e:
            print(f"Warning: Could not connect to Ollama ({e})")
            print("Make sure Ollama is running: ollama serve")
    
    def embed_code(self, code_strings: List[str]) -> torch.Tensor:
        """Convert code strings to embeddings using Ollama"""
        embeddings = []
        
        for code in code_strings:
            try:
                # Try embeddings endpoint first
                response = requests.post(
                    f"{self.ollama_url}/api/embeddings",
                    json={
                        "model": self.model_name,
                        "prompt": code
                    }
                )
                
                if response.status_code == 200:
                    embedding = response.json()['embedding']
                    embeddings.append(embedding)
                else:
                    # If embeddings endpoint fails, try generate endpoint with special prompt
                    print(f"Embeddings endpoint failed (status {response.status_code}), trying generate endpoint...")
                    
                    response = requests.post(
                        f"{self.ollama_url}/api/generate",
                        json={
                            "model": self.model_name,
                            "prompt": f"[EMBED] {code}",
                            "stream": False
                        }
                    )
                    
                    if response.status_code == 200:
                        # This is a fallback - we'd need to extract features from the response
                        # For now, create a simple hash-based embedding
                        import hashlib
                        hash_obj = hashlib.md5(code.encode())
                        hash_hex = hash_obj.hexdigest()
                        # Convert hex to pseudo-embedding
                        embedding = [float(int(hash_hex[i:i+2], 16)) / 255.0 for i in range(0, min(len(hash_hex), 64), 2)]
                        embedding = embedding + [0.0] * (4096 - len(embedding))  # Pad to 4096
                        embeddings.append(embedding)
                    else:
                        print(f"Both endpoints failed, using zero embedding")
                        embeddings.append([0.0] * 4096)
                    
            except Exception as e:
                print(f"Error processing code snippet: {e}")
                embeddings.append([0.0] * 4096)
        
        return torch.tensor(embeddings, dtype=torch.float32)
    
    def embed_code_batch(self, code_strings: List[str], batch_size: int = 4) -> torch.Tensor:
        """Embed code in batches for better performance"""
        all_embeddings = []
        
        for i in range(0, len(code_strings), batch_size):
            batch = code_strings[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(code_strings)-1)//batch_size + 1}")
            
            batch_embeddings = self.embed_code(batch)
            all_embeddings.append(batch_embeddings)
        
        return torch.cat(all_embeddings, dim=0)