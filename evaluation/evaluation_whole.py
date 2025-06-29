import torch
from torchvision import transforms as T
from lightly.transforms import SimCLRTransform, DINOTransform, MAETransform, MoCoV2Transform, utils
from datasets import create_dataset
from models import MAEModel
import pytorch_lightning as pl
import os
import copy
import gc
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os
from pathlib import Path
from models import BYOLModel, SimCLRModel
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import eval
import importlib
import models
from eval import *
importlib.reload(models)
importlib.reload(eval)


def test_all_100():
    config=Config(train_datasets=['cifar100'])
    results_df = run_complete_evaluation(config, results_no=1)

    # Wyświetl podsumowanie
    print("=== EVALUATION SUMMARY ===")
    print(f"Total models evaluated: {len(results_df)}")
    print(f"\nTop 5 models (Linear Probing):")
    print(results_df.nlargest(5, 'linear_probing_accuracy')[
        ['method', 'train_dataset', 'test_dataset', 'linear_probing_accuracy']
    ])

    print(f"\nTop 5 models (k-NN):")
    print(results_df.nlargest(5, 'best_knn_accuracy')[
        ['method', 'train_dataset', 'test_dataset', 'best_knn_accuracy', 'best_knn_k']
    ])

def test_mae_imgnet():
    config=Config(train_datasets=['imagenet'], methods=['mae'])
    results_df = run_complete_evaluation(config, results_no=21)

    # Wyświetl podsumowanie
    print("=== EVALUATION SUMMARY ===")
    print(f"Total models evaluated: {len(results_df)}")
    print(f"\nTop 5 models (Linear Probing):")
    print(results_df.nlargest(5, 'linear_probing_accuracy')[
        ['method', 'train_dataset', 'test_dataset', 'linear_probing_accuracy']
    ])

    print(f"\nTop 5 models (k-NN):")
    print(results_df.nlargest(5, 'best_knn_accuracy')[
        ['method', 'train_dataset', 'test_dataset', 'best_knn_accuracy', 'best_knn_k']
    ])

def test_byol_imgnet():
    config=Config(train_datasets=['imagenet'],methods=['byol'])
    results_df = run_complete_evaluation(config, results_no=22)

    # Wyświetl podsumowanie
    print("=== EVALUATION SUMMARY ===")
    print(f"Total models evaluated: {len(results_df)}")
    print(f"\nTop 5 models (Linear Probing):")
    print(results_df.nlargest(5, 'linear_probing_accuracy')[
        ['method', 'train_dataset', 'test_dataset', 'linear_probing_accuracy']
    ])

    print(f"\nTop 5 models (k-NN):")
    print(results_df.nlargest(5, 'best_knn_accuracy')[
        ['method', 'train_dataset', 'test_dataset', 'best_knn_accuracy', 'best_knn_k']
    ])
def test_simclr_imgnet():
    config=Config(train_datasets=['imagenet'],methods=['simclr'])
    results_df = run_complete_evaluation(config, results_no=23)

    # Wyświetl podsumowanie
    print("=== EVALUATION SUMMARY ===")
    print(f"Total models evaluated: {len(results_df)}")
    print(f"\nTop 5 models (Linear Probing):")
    print(results_df.nlargest(5, 'linear_probing_accuracy')[
        ['method', 'train_dataset', 'test_dataset', 'linear_probing_accuracy']
    ])

    print(f"\nTop 5 models (k-NN):")
    print(results_df.nlargest(5, 'best_knn_accuracy')[
        ['method', 'train_dataset', 'test_dataset', 'best_knn_accuracy', 'best_knn_k']
    ])
def test_add_mae_flowers():
    config=Config(train_datasets=['cifar100'],test_datasets=['flowers'],methods=['mae'])
    results_df = run_complete_evaluation(config, results_no=3)

    # Wyświetl podsumowanie
    print("=== EVALUATION SUMMARY ===")
    print(f"Total models evaluated: {len(results_df)}")
    print(f"\nTop 5 models (Linear Probing):")
    print(results_df.nlargest(5, 'linear_probing_accuracy')[
        ['method', 'train_dataset', 'test_dataset', 'linear_probing_accuracy']
    ])

    print(f"\nTop 5 models (k-NN):")
    print(results_df.nlargest(5, 'best_knn_accuracy')[
        ['method', 'train_dataset', 'test_dataset', 'best_knn_accuracy', 'best_knn_k']
    ])

def test_add_mae_pets():
    config=Config(train_datasets=['cifar100'],test_datasets=['pets'],methods=['mae'])
    results_df = run_complete_evaluation(config, results_no=4)

    # Wyświetl podsumowanie
    print("=== EVALUATION SUMMARY ===")
    print(f"Total models evaluated: {len(results_df)}")
    print(f"\nTop 5 models (Linear Probing):")
    print(results_df.nlargest(5, 'linear_probing_accuracy')[
        ['method', 'train_dataset', 'test_dataset', 'linear_probing_accuracy']
    ])

    print(f"\nTop 5 models (k-NN):")
    print(results_df.nlargest(5, 'best_knn_accuracy')[
        ['method', 'train_dataset', 'test_dataset', 'best_knn_accuracy', 'best_knn_k']
    ])

def test_add_mae_planes():
    config=Config(train_datasets=['cifar100'],test_datasets=['aircrafts'],methods=['mae'])
    results_df = run_complete_evaluation(config, results_no=5)

    # Wyświetl podsumowanie
    print("=== EVALUATION SUMMARY ===")
    print(f"Total models evaluated: {len(results_df)}")
    print(f"\nTop 5 models (Linear Probing):")
    print(results_df.nlargest(5, 'linear_probing_accuracy')[
        ['method', 'train_dataset', 'test_dataset', 'linear_probing_accuracy']
    ])

    print(f"\nTop 5 models (k-NN):")
    print(results_df.nlargest(5, 'best_knn_accuracy')[
        ['method', 'train_dataset', 'test_dataset', 'best_knn_accuracy', 'best_knn_k']
    ])


"""if __name__ == "__main__":
    #test_all_100()
    #test_all_imgnet()
    # test_add_mae_flowers()
    # test_add_mae_pets()
    # test_add_mae_planes()"""