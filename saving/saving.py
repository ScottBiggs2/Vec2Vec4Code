import os
import torch
import json
from datetime import datetime
from blocks.core import CodeVec2Vec

def save_vec2vec_model(model, trainer, embedding_dim, initial_similarity, final_similarity, 
                      training_epochs, loss_weights, model_name="vec2vec_code_model"):
    """
    Save the trained Vec2Vec model and training metadata
    """
    # Create models directory if it doesn't exist
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Create timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"{model_name}_{timestamp}"
    
    # Save model state dict
    model_path = os.path.join(models_dir, f"{model_filename}.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_gen_state_dict': trainer.optimizer_gen.state_dict(),
        'optimizer_disc_state_dict': trainer.optimizer_disc.state_dict(),
        'embedding_dim': embedding_dim,
        'latent_dim': model.latent_dim,
    }, model_path)
    
    # Save training metadata
    metadata = {
        'model_name': model_filename,
        'timestamp': timestamp,
        'embedding_dim': embedding_dim,
        'latent_dim': model.latent_dim,
        'initial_similarity': float(initial_similarity),
        'final_similarity': float(final_similarity),
        'improvement': float(final_similarity - initial_similarity),
        'training_epochs': training_epochs,
        'loss_weights': loss_weights,
        'learning_rate': trainer.optimizer_gen.param_groups[0]['lr'],
        'model_architecture': {
            'adapters': 'Vec2VecAdapter',
            'backbone': 'Vec2VecBackbone',
            'discriminators': 'Vec2VecDiscriminator'
        }
    }
    
    metadata_path = os.path.join(models_dir, f"{model_filename}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Model saved to: {model_path}")
    print(f"✅ Metadata saved to: {metadata_path}")
    print(f"📊 Final similarity: {final_similarity:.5f}")
    print(f"📈 Improvement: {final_similarity - initial_similarity:+.5f}")
    
    return model_path, metadata_path

def load_vec2vec_model(model_path, device='cpu'):
    """
    Load a saved Vec2Vec model
    """
    checkpoint = torch.load(model_path, map_location=device)
    
    # Recreate model with saved dimensions
    model = CodeVec2Vec(
        embedding_dim=checkpoint['embedding_dim'],
        latent_dim=checkpoint['latent_dim']
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded from: {model_path}")
    print(f"📊 Embedding dim: {checkpoint['embedding_dim']}")
    print(f"🧠 Latent dim: {checkpoint['latent_dim']}")
    
    return model, checkpoint

# Example usage - add this to your demo function after training:
def save_model_example(model, trainer, embedding_dim, initial_similarity, final_similarity, 
                      training_epochs, loss_weights):
    """
    Example of how to save the model after training
    """
    model_path, metadata_path = save_vec2vec_model(
        model=model,
        trainer=trainer,
        embedding_dim=embedding_dim,
        initial_similarity=initial_similarity,
        final_similarity=final_similarity,
        training_epochs=training_epochs,
        loss_weights=loss_weights,
        model_name="python_c_translator"
    )
    
    # Test loading the model
    loaded_model, checkpoint = load_vec2vec_model(model_path)
    
    # Verify it works
    with torch.no_grad():
        # Test with dummy data
        test_embedding = torch.randn(1, embedding_dim)
        translated = loaded_model.translate_py_to_c(test_embedding)
        print(f"✅ Model test successful - translation shape: {translated.shape}")
    
    return model_path

# To use in your demo, add this after the "Success metrics" section:
# """
# # Save the trained model
# if bidirectional_avg >= 0.95:  # Only save if reasonably good
#     save_model_example(
#         model=model,
#         trainer=trainer,
#         embedding_dim=embedding_dim,
#         initial_similarity=initial_similarity,
#         final_similarity=bidirectional_avg,
#         training_epochs=training_epochs,
#         loss_weights=loss_weights
#     )
# """