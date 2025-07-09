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
from translation.translation import snowflake_demo, interactive_translator, CodeTranslator
import argparse
import os


if __name__ == "__main__":
    # Load data
    python_samples = PYTHON_SAMPLES
    c_samples = C_SAMPLES

    demo_vec2vec_code_translation()
