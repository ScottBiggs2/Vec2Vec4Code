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

        # Explicitly set the pad_token_id on the model's generation config to avoid warnings
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
            
        # Get the embedding and output layers
        self.embedding_layer = self.model.get_input_embeddings()  # embedding matrix
        self.output_layer = self.model.get_output_embeddings()    # lm_head/output projection
        
        print(f"✅ Loaded model with {self.embedding_layer.weight.shape[0]} vocab, {self.embedding_layer.weight.shape[1]}D embeddings")
    
    # To-do: 
    # Investigate embedding structure Should be 1 program : 1 embedding
    # Add system prompt - DONE
    def embed_code(self, code_strings: List[str], language: str = None) -> torch.Tensor:
        """Convert code strings to embeddings using HIDDEN STATES (not token embeddings)"""
        
        # Prepare inputs with a system prompt to guide the model
        prepared_inputs = []
        for code in code_strings:
            lang_name = "Code"
            if language == 'python':
                lang_name = "Python"
            elif language == 'c':
                lang_name = "C"
            prompt = f"You are a universal code expert. Analyze the following {lang_name} snippet and create a semantic embedding that captures its core logic.\n\nCode:\n```\n{code}\n```"
            prepared_inputs.append(prompt)

        with torch.no_grad():
            inputs = self.tokenizer(
                prepared_inputs,
                padding=True, 
                truncation=True, 
                max_length=512,
                return_tensors="pt"
            )
            
            # Get hidden states from the model (2048-dim for 1.3b-base) (fromerly 4096)
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

    def embed_code_batch(self, code_strings: List[str], batch_size: int = 4, language: str = None) -> torch.Tensor:
        """Embed code in batches for better performance"""
        all_embeddings = []
        
        for i in range(0, len(code_strings), batch_size):
            batch = code_strings[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(code_strings)-1)//batch_size + 1}")

            batch_embeddings = self.embed_code(batch, language=language)
            all_embeddings.append(batch_embeddings)
        
        return torch.cat(all_embeddings, dim=0)

    def embedding_to_code_with_generate(self, embedding: torch.Tensor, max_new_tokens: int = 256, num_beams: int = 5) -> str:
        """
        Convert a single embedding vector back to a code string using beam search.
        This treats the input embedding as the initial state for the generator.
        """
        # The `generate` function expects inputs_embeds to be [batch_size, seq_len, hidden_dim].
        # Our input is [batch_size, hidden_dim]. We'll treat it as a sequence of length 1.
        inputs_embeds = embedding.unsqueeze(1)  # -> [batch_size, 1, hidden_dim]

        # Ensure the embedding is on the same device as the model
        inputs_embeds = inputs_embeds.to(self.model.device)

        # Create an attention mask for the inputs_embeds.
        # It has shape [batch_size, sequence_length], which is [batch_size, 1] here.
        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device)

        with torch.no_grad():
            output_sequences = self.model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1,
            )

        return self.tokenizer.decode(output_sequences[0], skip_special_tokens=True)
