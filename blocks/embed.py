import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import requests
import json
import random
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import subprocess
import tempfile
from typing import List, Tuple
import argparse

class CodeEmbedder:
    """Embeds code using DeepSeek Coder token embeddings (not hidden states)"""
    
    def __init__(self, model_name="deepseek-ai/deepseek-coder-1.3b-base"):
        self.model_name = model_name
        
        print(f"Loading DeepSeek Coder model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()
        
        # Add padding token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Get the embedding and output layers
        self.embedding_layer = self.model.get_input_embeddings()  # embedding matrix
        self.output_layer = self.model.get_output_embeddings()    # lm_head/output projection
        
        print(f"✅ Loaded model with {self.embedding_layer.weight.shape[0]} vocab, {self.embedding_layer.weight.shape[1]}D embeddings")
    
def embed_code(self, code_strings: List[str]) -> torch.Tensor:
    """Convert code strings to embeddings using HIDDEN STATES (not token embeddings)"""
    with torch.no_grad():
        inputs = self.tokenizer(
            code_strings, 
            padding=True, 
            truncation=True, 
            max_length=512,
            return_tensors="pt"
        )
        
        # Get hidden states from the model (4096-dim)
        outputs = self.model(**inputs, output_hidden_states=True)
        # Use mean pooling of last hidden state
        embeddings = outputs.hidden_states[-1].mean(dim=1)
        
    return embeddings
    
    def embedding_to_tokens(self, embedding: torch.Tensor, max_length: int = 50) -> List[int]:
        """
        Convert embedding back to tokens using the model's output layer
        NO similarity search - uses the actual language model head
        """
        
        with torch.no_grad():
            # Use the output layer to convert embedding to logits
            logits = self.output_layer(embedding)  # [batch_size, vocab_size]
            
            # Get the most likely tokens
            if max_length == 1:
                # Single token
                predicted_token = logits.argmax(dim=-1).item()
                return [predicted_token]
            else:
                # Multiple tokens - use greedy decoding
                tokens = []
                current_embedding = embedding.clone()
                
                for _ in range(max_length):
                    # Get logits for current embedding
                    logits = self.output_layer(current_embedding)
                    
                    # Get most likely token
                    next_token = logits.argmax(dim=-1).item()
                    tokens.append(next_token)
                    
                    # Check for end token
                    if next_token == self.tokenizer.eos_token_id:
                        break
                    
                    # Update embedding by feeding token back through embedding layer
                    # This is an approximation - in reality we'd need the full model
                    token_embedding = self.embedding_layer(torch.tensor([next_token]))
                    current_embedding = token_embedding.mean(dim=0, keepdim=True)
                
                return tokens
    
    def tokens_to_code(self, tokens: List[int]) -> str:
        """Convert tokens back to code using tokenizer"""
        return self.tokenizer.decode(tokens, skip_special_tokens=True)
    
    def embed_code_batch(self, code_strings: List[str], batch_size: int = 4) -> torch.Tensor:
        """Embed code in batches for better performance"""
        all_embeddings = []
        
        for i in range(0, len(code_strings), batch_size):
            batch = code_strings[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(code_strings)-1)//batch_size + 1}")
            
            batch_embeddings = self.embed_code(batch)
            all_embeddings.append(batch_embeddings)
        
        return torch.cat(all_embeddings, dim=0)

# class CodeEmbedder:
#     """Embeds code using DeepSeek Coder model and provides embedding-to-token decoding"""
    
#     def __init__(self, model_name="deepseek-ai/deepseek-coder-1.3b-base"):
#         self.model_name = model_name
        
#         print(f"Loading DeepSeek Coder model: {model_name}")
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.model = AutoModelForCausalLM.from_pretrained(model_name)
#         self.model.eval()
        
#         # Add padding token if not exists
#         if self.tokenizer.pad_token is None:
#             self.tokenizer.pad_token = self.tokenizer.eos_token
            
#         # Get the embedding and output layers
#         self.embedding_layer = self.model.get_input_embeddings()  # embedding matrix
#         self.output_layer = self.model.get_output_embeddings()    # lm_head/output projection
        
#         print(f"✅ Loaded model with {self.embedding_layer.weight.shape[0]} vocab, {self.embedding_layer.weight.shape[1]}D embeddings")
    
#     def embed_code(self, code_strings: List[str]) -> torch.Tensor:
#         """Convert code strings to embeddings using transformers hidden states"""
#         with torch.no_grad():
#             inputs = self.tokenizer(
#                 code_strings, 
#                 padding=True, 
#                 truncation=True, 
#                 max_length=512,
#                 return_tensors="pt"
#             )
            
#             # Get hidden states from the model
#             outputs = self.model(**inputs, output_hidden_states=True)
#             # Use mean pooling of last hidden state
#             embeddings = outputs.hidden_states[-1].mean(dim=1)
            
#         return embeddings
    
#     def embedding_to_tokens(self, embedding: torch.Tensor, max_length: int = 50) -> List[int]:
#         """
#         Convert embedding back to tokens using the model's output layer
#         NO similarity search - uses the actual language model head
#         """
        
#         with torch.no_grad():
#             # Use the output layer to convert embedding to logits
#             logits = self.output_layer(embedding)  # [batch_size, vocab_size]
            
#             # Get the most likely tokens
#             if max_length == 1:
#                 # Single token
#                 predicted_token = logits.argmax(dim=-1).item()
#                 return [predicted_token]
#             else:
#                 # Multiple tokens - use greedy decoding
#                 tokens = []
#                 current_embedding = embedding.clone()
                
#                 for _ in range(max_length):
#                     # Get logits for current embedding
#                     logits = self.output_layer(current_embedding)
                    
#                     # Get most likely token
#                     next_token = logits.argmax(dim=-1).item()
#                     tokens.append(next_token)
                    
#                     # Check for end token
#                     if next_token == self.tokenizer.eos_token_id:
#                         break
                    
#                     # Update embedding by feeding token back through embedding layer
#                     # This is an approximation - in reality we'd need the full model
#                     token_embedding = self.embedding_layer(torch.tensor([next_token]))
#                     current_embedding = token_embedding.mean(dim=0, keepdim=True)
                
#                 return tokens
    
#     def tokens_to_code(self, tokens: List[int]) -> str:
#         """Convert tokens back to code using tokenizer"""
#         return self.tokenizer.decode(tokens, skip_special_tokens=True)
    
#     def embed_code_batch(self, code_strings: List[str], batch_size: int = 4) -> torch.Tensor:
#         """Embed code in batches for better performance"""
#         all_embeddings = []
        
#         for i in range(0, len(code_strings), batch_size):
#             batch = code_strings[i:i+batch_size]
#             print(f"Processing batch {i//batch_size + 1}/{(len(code_strings)-1)//batch_size + 1}")
            
#             batch_embeddings = self.embed_code(batch)
#             all_embeddings.append(batch_embeddings)
        
#         return torch.cat(all_embeddings, dim=0)


# class CodeEmbedder:
#     """Embeds code using DeepSeek Coder model via Ollama"""
    
#     def __init__(self, model_name="deepseek-coder:1.3B", ollama_url="http://localhost:11434"):
#         self.model_name = model_name
#         self.ollama_url = ollama_url
        
#         # Test connection to Ollama
#         try:
#             response = requests.get(f"{ollama_url}/api/tags")
#             if response.status_code == 200:
#                 available_models = [model['name'] for model in response.json()['models']]
#                 if model_name not in available_models:
#                     print(f"Model {model_name} not found. Available models: {available_models}")
#                     print(f"To pull the model, run: ollama pull {model_name}")
#                 else:
#                     print(f"Successfully connected to Ollama with model {model_name}")
#             else:
#                 raise ConnectionError(f"Could not connect to Ollama at {ollama_url}")
#         except Exception as e:
#             print(f"Warning: Could not connect to Ollama ({e})")
#             print("Make sure Ollama is running: ollama serve")
    
#     def embed_code(self, code_strings: List[str]) -> torch.Tensor:
#         """Convert code strings to embeddings using Ollama"""
#         embeddings = []
        
#         for code in code_strings:
#             try:
#                 # Try embeddings endpoint first
#                 response = requests.post(
#                     f"{self.ollama_url}/api/embeddings",
#                     json={
#                         "model": self.model_name,
#                         "prompt": code
#                     }
#                 )
                
#                 if response.status_code == 200:
#                     embedding = response.json()['embedding']
#                     embeddings.append(embedding)
#                 else:
#                     # If embeddings endpoint fails, try generate endpoint with special prompt
#                     print(f"Embeddings endpoint failed (status {response.status_code}), trying generate endpoint...")
                    
#                     response = requests.post(
#                         f"{self.ollama_url}/api/generate",
#                         json={
#                             "model": self.model_name,
#                             "prompt": f"[EMBED] {code}",
#                             "stream": False
#                         }
#                     )
                    
#                     if response.status_code == 200:
#                         # This is a fallback - we'd need to extract features from the response
#                         # For now, create a simple hash-based embedding
#                         import hashlib
#                         hash_obj = hashlib.md5(code.encode())
#                         hash_hex = hash_obj.hexdigest()
#                         # Convert hex to pseudo-embedding
#                         embedding = [float(int(hash_hex[i:i+2], 16)) / 255.0 for i in range(0, min(len(hash_hex), 64), 2)]
#                         embedding = embedding + [0.0] * (4096 - len(embedding))  # Pad to 4096
#                         embeddings.append(embedding)
#                     else:
#                         print(f"Both endpoints failed, using zero embedding")
#                         embeddings.append([0.0] * 4096)
                    
#             except Exception as e:
#                 print(f"Error processing code snippet: {e}")
#                 embeddings.append([0.0] * 4096)
        
#         return torch.tensor(embeddings, dtype=torch.float32)
    
#     def embed_code_batch(self, code_strings: List[str], batch_size: int = 4) -> torch.Tensor:
#         """Embed code in batches for better performance"""
#         all_embeddings = []
        
#         for i in range(0, len(code_strings), batch_size):
#             batch = code_strings[i:i+batch_size]
#             print(f"Processing batch {i//batch_size + 1}/{(len(code_strings)-1)//batch_size + 1}")
            
#             batch_embeddings = self.embed_code(batch)
#             all_embeddings.append(batch_embeddings)
        
#         return torch.cat(all_embeddings, dim=0)