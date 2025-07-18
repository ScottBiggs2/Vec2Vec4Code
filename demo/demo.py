import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from transformers import AutoTokenizer, AutoModel
import random
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt

from data.data import PYTHON_SAMPLES, C_SAMPLES
from blocks.core import CodeVec2Vec, CodeEmbedder, Vec2VecTrainer
from saving.saving import save_vec2vec_model, load_vec2vec_model

def demo_vec2vec_code_translation(PYTHON_SAMPLES, C_SAMPLES, training_epochs = 50):
    """Demonstrate the Vec2Vec code translation MVP"""
    print("=== Vec2Vec Code Translation MVP ===")
    print("Initializing DeepSeek Coder embedder using transformers")
    
    # Initialize the Ollama-based DeepSeek Coder embedder
    try:
        # embedder = CodeEmbedder("deepseek-coder:1.3B")
        embedder = CodeEmbedder()

        print("Successfully connected to transformers!")
        
        print("Generating real embeddings for Python code...")
        py_embeddings = embedder.embed_code_batch(PYTHON_SAMPLES, language='python')
        
        print("Generating real embeddings for C code...")
        c_embeddings = embedder.embed_code_batch(C_SAMPLES, language='c')
        
        embedding_dim = py_embeddings.shape[1]
        print(f"Using real embeddings with dimension: {embedding_dim}")
        
    except Exception as e:
        print(f"Could not use transformers ({e}), falling back to simulated embeddings...")
        
        # Fallback to simulated embeddings
        embedding_dim = 2048  # DeepSeek Coder 1.3B hidden state dimension - fromerly 4096
        py_embeddings = torch.randn(len(PYTHON_SAMPLES), embedding_dim)
        c_embeddings = torch.randn(len(C_SAMPLES), embedding_dim)
        print(f"Using simulated embeddings with dimension: {embedding_dim}")
    
    print(f"Python embeddings shape: {py_embeddings.shape}")
    print(f"C embeddings shape: {c_embeddings.shape}")

    if not py_embeddings.shape == c_embeddings.shape:
        print(f"Trimming embeddings to match")
        min_length = min(py_embeddings.shape[0], c_embeddings.shape[0])
        py_embeddings = py_embeddings[:min_length]
        c_embeddings = c_embeddings[:min_length]
        print(f"Python embeddings shape: {py_embeddings.shape}")
        print(f"C embeddings shape: {c_embeddings.shape}")
    
    # Show some basic embedding statistics
    print(f"Python embedding mean: {py_embeddings.mean():.4f}, std: {py_embeddings.std():.4f}")
    print(f"C embedding mean: {c_embeddings.mean():.4f}, std: {c_embeddings.std():.4f}")
    
    # Check if embeddings are all zeros (failed to get real embeddings)
    if py_embeddings.std() == 0 and c_embeddings.std() == 0:
        print("WARNING: All embeddings are zero - using random initialization instead")
        py_embeddings = torch.randn_like(py_embeddings) * 0.1
        c_embeddings = torch.randn_like(c_embeddings) * 0.1


       # Calculate initial similarity between Python and C embeddings
    initial_similarity = F.cosine_similarity(py_embeddings, c_embeddings, dim=1).mean()
    print(f"Initial cross-language similarity: {initial_similarity:.4f}")
    
    # Initialize Vec2Vec model
    print("\nInitializing Vec2Vec model...")
    model = CodeVec2Vec(embedding_dim, latent_dim=512) # tune for size/depth
    trainer = Vec2VecTrainer(model)
    
    # print("Training Vec2Vec model...")
    # losses_history = []
    
    # Check if embeddings are already well-aligned
    if initial_similarity > 0.8:
        print(f"WARNING: Embeddings are already highly similar ({initial_similarity:.4f})")
        print("This suggests DeepSeek Coder already learned cross-language representations.")
        print("Vec2Vec may not be necessary, but continuing with adjusted hyperparameters...")
        
        # Use more conservative hyperparameters for already-aligned embeddings
        trainer = Vec2VecTrainer(model, learning_rate=1e-3)  # Much lower learning rate
        # training_epochs = 100  # Fewer epochs
        loss_weights = {'rec': 0.5, 'cycle': 0.3, 'vsp': 0.1, 'adv': 0.1}  # Focus on reconstruction
    else:
        trainer = Vec2VecTrainer(model, learning_rate=1e-3)
        # training_epochs = 100
        loss_weights = {'rec': 0.5, 'cycle': 0.3, 'vsp': 0.1, 'adv': 0.1}
    
    print(f"Training for {training_epochs} epochs with loss weights: {loss_weights}")
    print("Training Vec2Vec model...")
    losses_history = []
    
    for epoch in range(training_epochs):
        # Shuffle and batch data
        indices = torch.randperm(min(len(py_embeddings), len(c_embeddings)))
        py_batch = py_embeddings[indices]
        c_batch = c_embeddings[indices]
        
        losses = trainer.train_step(py_batch, c_batch, loss_weights)
        losses_history.append(losses)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Gen Loss: {losses['loss_gen']:.4f}, "
                  f"Disc Loss: {losses['loss_disc']:.4f}, "
                  f"Cycle Loss: {losses['loss_cycle']:.4f}, "
                  f"Rec Loss: {losses['loss_rec']:.4f}")
    
    print("Training complete!")
    
    # Test translation
    print("\n=== Testing Translation ===")
    model.eval()
    
    with torch.no_grad():
        # Translate Python to C
        py_to_c_translated = model.translate_py_to_c(py_embeddings)
        c_to_py_translated = model.translate_c_to_py(c_embeddings)
        
        # Calculate similarities
        original_py_similarity = F.cosine_similarity(py_embeddings[0], py_embeddings[1], dim=0)
        py_to_c_sample_similarity = F.cosine_similarity(py_to_c_translated[0], c_embeddings[0], dim=0)
        
        print(f"Original Python embedding similarity: {original_py_similarity:.4f}")
        print(f"Python->C translation similarity (sample): {py_to_c_sample_similarity:.4f}")
        
        # Test cycle consistency
        py_cycled = model.translate_c_to_py(py_to_c_translated)
        cycle_similarity = F.cosine_similarity(py_embeddings, py_cycled, dim=1).mean()
        print(f"Cycle consistency (Python->C->Python): {cycle_similarity:.4f}")
        
        # Show improvement in cross-language alignment
        py_to_c_final_similarity = F.cosine_similarity(py_to_c_translated, c_embeddings, dim=1).mean()
        c_to_py_final_similarity = F.cosine_similarity(c_to_py_translated, py_embeddings, dim=1).mean()
        final_bidirectional_similarity = (py_to_c_final_similarity + c_to_py_final_similarity) / 2.0
        print(f"Final bidirectional similarity after translation: {final_bidirectional_similarity:.4f}")
        print(f"Improvement: {final_bidirectional_similarity - initial_similarity:.4f}")
    
    # Plot loss curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot([l['loss_gen'] for l in losses_history])
    plt.title('Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 3, 2)
    plt.plot([l['loss_cycle'] for l in losses_history])
    plt.title('Cycle Consistency Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 3, 3)
    plt.plot([l['loss_vsp'] for l in losses_history])
    plt.title('Vector Space Preservation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    plt.show()
    
    print("\n=== Results Analysis ===")
    print(f"Initial similarity: {initial_similarity:.4f}")
    print(f"Final Bidirectional similarity: {final_bidirectional_similarity:.4f}")
    print(f"Change: {final_bidirectional_similarity - initial_similarity:.4f}")
    
    if initial_similarity > 0.8:
        print("\n🎉 CONCLUSION: DeepSeek Coder already provides excellent cross-language alignment!")
        print("This is actually a SUCCESS for DeepSeek, not a failure of vec2vec.")
        print("The model has learned shared representations across programming languages.")
        
        if final_bidirectional_similarity > 0.7:
            print("✅ Vec2vec maintained reasonable alignment during training")
        else:
            print("⚠️  Vec2vec disrupted the existing alignment - consider different hyperparameters")
    else:
        if final_bidirectional_similarity > initial_similarity:
            print("✅ Vec2vec successfully improved cross-language alignment!")
        else:
            print("❌ Vec2vec failed to improve alignment - may need hyperparameter tuning")
    
    print(f"\nCycle consistency: {cycle_similarity:.4f}")
    if cycle_similarity > 0.8:
        print("✅ Excellent cycle consistency - translations are reversible")
    elif cycle_similarity > 0.6:
        print("⚠️  Moderate cycle consistency - some information loss in round-trip")
    else:
        print("❌ Poor cycle consistency - significant information loss")
    
    print("\n=== Example Code Pairs ===")
    print("Note: High similarity scores indicate DeepSeek already learned cross-language patterns!")
    for i, (py_code, c_code) in enumerate(zip(PYTHON_SAMPLES[:3], C_SAMPLES[:3])):
        print(f"\nPair {i+1}:")
        print("Python:", py_code.replace('\n', '\\n'))
        print("C:", c_code.replace('\n', '\\n'))
        
        # Show embedding similarities
        py_emb = py_embeddings[i]
        c_emb = c_embeddings[i]
        original_sim = F.cosine_similarity(py_emb, c_emb, dim=0)
        print(f"Original similarity: {original_sim:.4f}")
        
        with torch.no_grad():
            translated = model.translate_py_to_c(py_emb.unsqueeze(0))
            translated_sim = F.cosine_similarity(translated.squeeze(0), c_emb, dim=0)
            print(f"After translation: {translated_sim:.4f}")
    
    # Save the trained model if it has improved and meets a quality threshold
    # I'm not sure I trust this. Need to review. 

    print("\n💾 Saving trained model...")
    model_path, metadata_path = save_vec2vec_model(
        model=model,
        trainer=trainer,
        embedding_dim=embedding_dim,
        initial_similarity=initial_similarity,
        final_similarity=final_bidirectional_similarity,
        training_epochs=training_epochs,
        loss_weights=loss_weights,
        model_name="python_c_translator"
    )
    print(f"\n💾 Model & metadata saved to {model_path} and {metadata_path}. ")

    # if final_bidirectional_similarity > initial_similarity and final_bidirectional_similarity > 0.75:
    #     print("\n💾 Saving trained model...")
    #     model_path, metadata_path = save_vec2vec_model(
    #         model=model,
    #         trainer=trainer,
    #         embedding_dim=embedding_dim,
    #         initial_similarity=initial_similarity,
    #         final_similarity=final_bidirectional_similarity,
    #         training_epochs=training_epochs,
    #         loss_weights=loss_weights,
    #         model_name="python_c_translator"
    #     )
    # else:
    #     print(f"\nModel did not improve enough to be saved (final similarity: {final_bidirectional_similarity:.4f}).")

    print("\n=== MVP Complete ===")
    
