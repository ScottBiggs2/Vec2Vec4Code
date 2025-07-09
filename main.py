import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel
import random
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt

from data.data import PYTHON_SAMPLES, C_SAMPLES
from blocks.core import CodeVec2Vec, Vec2VecTrainer
from demo.demo import demo_vec2vec_code_translation
# from translation.translation import snowflake_demo, interactive_translator, CodeTranslator
from translation.translation_2 import CodeTranslator
import argparse
import os


if __name__ == "__main__":
    # Load data
    python_samples = PYTHON_SAMPLES
    c_samples = C_SAMPLES

    demo_vec2vec_code_translation()

    # model_path = "models/python_c_translator_20250708_221459.pth"
    # code = """ print(f"Hello World") """

    # translator = CodeTranslator(model_path)
    # translated_code, confidence = translator.translate_code(code, 'python', 'c')
    # print(f"Translated code: {translated_code} \nConfidence: {confidence}")

# Example commands to run the translator:
# python translation/translation.py models/your_model_file.pth --demo
# python translation/translation.py models/python_c_translator_20250708_194151.pth --demo
# python translation/translation_2.py models/python_c_translator_20250708_222644.pth --demo

