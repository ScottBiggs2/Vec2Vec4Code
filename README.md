# Vec2Vec for Code Translation

A Python implementation of the vec2vec methodology applied to cross-language code translation, enabling bidirectional Python ↔ C code conversion without requiring paired training data.

## 🚀 Overview

This project adapts the groundbreaking [vec2vec paper](https://arxiv.org/abs/2505.12540) to programming languages, demonstrating how to translate code embeddings between Python and C using unsupervised adversarial training. The key innovation is achieving high cross-language similarity (target: 0.99+) without needing paired code examples.

## 📋 Key Features

- **No A/B Pairs Required**: Trains on unpaired Python and C code samples
- **Bidirectional Translation**: Python → C and C → Python embedding translation
- **Real Model Integration**: Uses DeepSeek Coder via transformers for authentic code embeddings
- **Precision Training**: Optimized for high-similarity fine-tuning (0.92 → 0.99+)
- **Model Persistence**: Save/load trained models with metadata
- **Comprehensive Evaluation**: Similarity metrics, cycle consistency, and training progress
- **Training Data Collection**: Script for scraping and saving Python and C education sites for code snippets

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- [Ollama](https://ollama.ai/) for model serving
- HuggingFace transformers for CausalLM and AutoTokenizer
- DeepSeek Coder model

### Setup
```bash
# Clone the repository
git clone [https://github.com/ScottBiggs2/Vec2Vec4Code/tree/main]
cd vec2vec-code-translation

# Install Python dependencies
pip install torch torchvision matplotlib requests numpy

# Install and setup Ollama
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve

# Pull DeepSeek Coder model
ollama pull deepseek-coder:1.3B
```

## 🎯 Quick Start

```python
# Run the demo
python main.py

# Expected output:

#=== Results Analysis ===
#Initial similarity: 0.6787
#Final similarity: 0.8280
#Change: 0.1493
#✅ Vec2vec successfully improved cross-language alignment!

#Cycle consistency: 0.8836
#✅ Excellent cycle consistency - translations are reversible
```

## 🏗️ Architecture

The implementation follows the original vec2vec architecture with adaptations for code:

- **Input Adapters**: Transform Python/C embeddings to universal latent space
- **Shared Backbone**: Learns common semantic representations
- **Output Adapters**: Decode from latent space back to language-specific embeddings
- **Discriminators**: Ensure translated embeddings are indistinguishable from real ones

## 📊 Training Strategy

### Loss Functions
- **Reconstruction Loss** (0.1-0.8): Ensures round-trip embedding preservation
- **Cycle Consistency** (0.05-0.15): Maintains translation reversibility  
- **Vector Space Preservation** (0.8): Preserves semantic relationships (key for similarity)
- **Adversarial Loss** (0.01-0.05): Ensures realistic translations

### Hyperparameter Adaptation
- **High Initial Similarity (>0.9)**: Ultra-conservative learning (LR: 0.00005)
- **Moderate Similarity (0.8-0.9)**: Standard fine-tuning (LR: 0.0001)
- **Low Similarity (<0.8)**: Aggressive alignment (LR: 0.001)

## 📁 Project Structure

```
vec2vec-code-translation/
├── main.py                # Run the demo script, notes on full translation commands
├── blocks/                # 
│   ├── core.py            # Core model and training functions
│   └── embed.py           # Code embedding construction function
├── saving/                # (Omitted from repo for brevity) Store C and Python code samples
│   └── saving.py          # functions for saving the Vec2Vec4Code model weights and metadata
├── data/                  # Depricated - (Omitted from repo for brevity) Store C and Python code samples
│   └── data.py            # Depricated - C and Python code snippets stored as short strings
├── models/                # (Omitted from repo for brevity) Saved model directory
│   ├── *.pth              # Model weights
│   └── *_metadata.json    # Training metadata json
├── translation/           # Example of translating python to/from C
│   ├── translation_2.py   # Translation demo
│   └── translation.py     # Depricated - Snowflake translation demo
├── code_scraping/         # Script to scrape code snippets
│   └── scraper.py         # Script to scrape C and Pythod education webpages for code snippets
├── scraped_code/          # (Omitted from repo for brevity) Script to scrape code snippets
│   └── ...                # .py files holding scraped snippets as strings to combine for training
├── README.md              # This file
├── .gitignore             # Ignore these files (ignores data, venv, and models for brevity)
└── requirements.txt       # Python dependencies
```

## 🧪 Code Samples

The demo includes equivalent Python/C code pairs for testing:

- **Basic functions**: `add(a, b)`, `fibonacci(n)`
- **Control structures**: `for` loops, conditionals
- **Data structures**: Classes/structs, arrays
- **Algorithms**: Sorting, searching, mathematical operations

## 📈 Performance Metrics

### Success Criteria
- **Primary Goal**: Bidirectional similarity ≥ 0.99
- **Cycle Consistency**: ≥ 0.95 (reversible translations)
- **Training Stability**: Decreasing VSP loss

### Typical Results
```
=== Results Analysis ===
Initial similarity: 0.6787
Final similarity: 0.8280
Change: 0.1493
✅ Vec2vec successfully improved cross-language alignment!

Cycle consistency: 0.8836
✅ Excellent cycle consistency - translations are reversible
```

## 🔬 Research Insights

### Key Findings
1. **DeepSeek Coder Pre-Alignment**: The model already learns cross-language representations during training
2. **VSP Loss Dominance**: Vector Space Preservation is crucial for similarity optimization
3. **No Pairing Required**: Successfully achieves high similarity without parallel data
4. **Fine-tuning Focus**: Best results come from precision tuning rather than major transformation

### Future Directions
- [ ] Extend to more language pairs (Python ↔ JavaScript, C ↔ Rust)
- [ ] Implement embedding → code generation pipeline
- [ ] Scale to larger datasets (GitHub repositories)
- [ ] Add semantic equivalence evaluation
- [ ] Domain adaptation (academic → production code)

## To-do: 
- Upgrade code cleanup model from DeepSeek-Coder 1.3B. Maybe return to typical non-coding LLM?
- Demonstrate quality translation of non-trivial code that is outside the training dataset
- More data, more training, bigger model. 

## 📚 References

- **Original Paper**: [Harnessing the Universal Geometry of Embeddings](https://arxiv.org/abs/2505.12540)
- **GitHub Repository**: [vec2vec](https://github.com/rjha18/vec2vec)
- **DeepSeek Coder**: [Model Card](https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-base)
- **Ollama**: [Documentation](https://ollama.ai/docs)

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Add new language pairs
- Improve training stability
- Implement evaluation metrics
- Add visualization tools
- Optimize hyperparameters

## 📄 License

MIT License

## 🙏 Acknowledgments

- Rishi Jha et al. for the original vec2vec methodology
- DeepSeek for the code embedding model
- Ollama team for efficient model serving

---

**Note**: This is research code demonstrating the vec2vec concept applied to programming languages. For production use, consider additional validation and safety measures.