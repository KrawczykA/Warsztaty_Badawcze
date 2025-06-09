from copy import deepcopy
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision import  datasets
import pytorch_lightning as pl
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import umap
from tqdm import tqdm
import warnings
import datasets
from models import MAEModel,SimCLRModel,BYOLModel,ClassifierModel
from pathlib import Path
import gc

warnings.filterwarnings('ignore')

# Konfiguracja
class Config:
    def __init__(self, test_datasets= ['cifar10', 'cifar100'], train_datasets= ['cifar100', 'imagenet'], methods=['mae', 'byol', 'simclr']):
        self.checkpoint_base = "./checkpoints"
        self.results_dir = "./results"
        self.figures_dir = "./figures"
        self.batch_size = 128
        self.num_workers = 4
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = 42
        
        # Parametry ewaluacji
        self.knn_neighbors = [5, 10, 20, 50]
        self.tsne_perplexity = 30
        self.umap_neighbors = 15
        self.pca_components = 50
        
        # Datasety
        self.test_datasets = test_datasets
        self.train_datasets = train_datasets
        
        # Modele
        self.ssl_methods = methods
        self.baseline_methods = ['pca', 'tsne', 'umap']

# Funkcje pomocnicze
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_dataset(dataset_name, train=True):
    """Wczytuje dataset: CIFAR10, CIFAR100, ImageNet, Flowers, Pets, Aircrafts"""

    transform = transforms.Compose([
        transforms.Resize(224),  # dla spójności rozmiarów
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    if dataset_name == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(
            root='./data', train=train, download=True, transform=transform
        )

    elif dataset_name == 'cifar100':
        dataset = torchvision.datasets.CIFAR100(
            root='./data', train=train, download=True, transform=transform
        )

    elif dataset_name == 'imagenet':
        path = os.path.join('data', 'imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train') if train else os.path.join('data', 'imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val')
        

    elif dataset_name == 'flowers':
            dataset = torchvision.datasets.Flowers102(
            root='./data', split='train' if train else 'test', download=True, transform=transform
        )

    elif dataset_name == 'pets':
        dataset = torchvision.datasets.OxfordIIITPet(
            root='./data', download=True,
            transform=transform,
            target_types='category'  # 'segmentation' też dostępne
        )
        if train:
            # Można zaimplementować własny podział, bo oficjalnie nie ma splitu
            dataset = torch.utils.data.Subset(dataset, range(0, int(len(dataset)*0.8)))
        else:
            dataset = torch.utils.data.Subset(dataset, range(int(len(dataset)*0.8), len(dataset)))

    elif dataset_name == 'aircrafts':
        dataset = torchvision.datasets.FGVCAircraft(
            root='./data', download=True,
            split='trainval' if train else 'test',
            transform=transform
        )

    else:
        raise ValueError(f"Nieznany dataset: {dataset_name}")


    subset_size = 0.01 if dataset_name == 'imagenet' else 0.1
    all_indices = list(range(len(dataset)))
    
    # losowy podzbiór danych (np. 1% lub 10%)
    sampled_indices, _ = train_test_split(
        all_indices,
        train_size=subset_size,
        random_state=42,
        shuffle=True
    )
    
    # wybrany podzbiór danych
    dataset = torch.utils.data.Subset(dataset, sampled_indices)

    # Krok 2: podział na train/test (90/10)
    indices = list(range(len(dataset)))  # teraz odniesienia do sampled_indices
    train_indices, test_indices = train_test_split(
        indices,
        test_size=0.1,
        random_state=42,
        shuffle=True
    )

    # Krok 3: zwróć odpowiedni podzbiór
    if train:
        return torch.utils.data.Subset(dataset, train_indices)
    else:
        return torch.utils.data.Subset(dataset, test_indices)

def parse_checkpoint_path(checkpoint_path):
    """Ekstraktuje informacje ze ścieżki checkpointa"""
    parts = Path(checkpoint_path).parts

    # Znajdź indeksy kluczowych elementów
    checkpoints_idx = parts.index('checkpoints')
    
    info = {
        'train_dataset': parts[checkpoints_idx + 1],
        'method': parts[checkpoints_idx + 2],
        'version': parts[checkpoints_idx + 3] if len(parts) > checkpoints_idx + 3 else 'unknown',
        'filename': parts[-1]
    }
    version = info['version']
    if version.startswith('pre-'):
        info['pretrained'] = True
    else:
        info['pretrained'] = False
    info['lr'] = float(1/10**int(version.split("-")[1][0]))


    if info['method']=='mae':    

        info['hyperparam'] = float(int(version.split('-')[1][-2:])/100)

    elif info['method']=='byol':
        end=version.split('-')[1]
        if len(end)==3:
            info['hyperparam'] = float(int(version.split('-')[1][-2:])/100)
        else:
            info['hyperparam'] = float(int(version.split('-')[1][-3:])/1000)


    return info

def load_model(checkpoint_path, method, device):
    """Wczytuje model SSL z checkpointa"""
    # Ekstraktuj informacje ze ścieżki
    path_info = parse_checkpoint_path(checkpoint_path)
    print(f"Loading {method} model: {path_info}")
    
    # Wczytaj checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Określ architekturę na podstawie metody i datasetu
    if method == 'mae':
       model=MAEModel(
        lr=path_info['lr'],
        weight_decay=1e-4,
        max_epochs=20,
        backbone_type="pretrained_resnet18" if path_info['pretrained'] else 'random_resnet18', 
        input_dim=3 * 224 * 224, 
        mask_ratio=path_info['hyperparam']
       )
    elif method == 'byol':
        model=BYOLModel(
            lr=path_info['lr'],
            weight_decay=0.0005,
            tau=path_info['hyperparam'],
            max_epochs=20,
            backbone_type="pretrained_resnet18" if path_info['pretrained'] else 'random_resnet18', 


        )
    
    elif method == 'simclr':
        model=SimCLRModel(
            lr=path_info['lr'],
            max_epochs=20,
            backbone_type="pretrained_resnet18" if path_info['pretrained'] else 'random_resnet18', 

        )
    
    
    # Wczytaj wagi
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        # Jeśli checkpoint to bezpośrednio state_dict
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    # Zwróć też informacje o modelu
    model.checkpoint_info = path_info
    
    return model


def extract_features(model, dataloader, device, method):
    """Ekstraktuje cechy z modelu SSL"""
    features = []
    labels = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(dataloader, desc="Extracting features")):
            data = data.to(device)
            
            # Różne metody mogą mieć różne sposoby ekstrakcji cech
            if method == 'mae':

                feat = model.encoder(data)

                if isinstance(feat, tuple):
                    feat = feat[0]

            elif method == 'byol':
                # BYOL używa online encoder
                    feat = model.online_backbone(data)
                    feat = feat.view(feat.size(0), -1)
            elif method == 'simclr':
                # SimCLR używa encoder + projection head (ale bierzemy przed projection)
                feat = model.backbone(data)
                feat = feat.view(feat.size(0), -1)
            
            features.append(feat.cpu())  # Move tensor to CPU first
            labels.append(target.cpu().numpy())  # For target too if it's on GPU

        features = np.concatenate(features, axis=0)
        labels = np.concatenate(labels, axis=0)
    
    return features, labels

# Metody ewaluacji
def linear_probing(dl_train, dl_eval, model_backbone, number_of_classes, 
                   learning_rate=0.001, num_epochs=10, device='cuda'):
    """Linear probing evaluation"""
    model = deepcopy(model_backbone)

    for param in model.parameters():
        param.requires_grad = False

    # Add the linear layer to the Sequential model
    model = torch.nn.Sequential(
        model,
        torch.nn.Flatten(),  # Flatten the output of the pooling layer
        torch.nn.Linear(512, number_of_classes)
    )

    # Prepare the model
    model = ClassifierModel(
        model=model,
        num_classes=number_of_classes,
        lr=learning_rate, 
        max_epochs=num_epochs
    )

    # Prepare the trainer for pytorch lightning
    trainer = pl.Trainer(max_epochs=num_epochs, devices=-1, accelerator=device.type)

    # Train the model
    trainer.fit(model, dl_train, dl_eval)
    #model.eval()
    #predictions=model(dl_eval.to(device))
    # Get predictions on test set 
    # predictions = trainer.predict(model, dl_eval) # TODO: TO JEST ZLE
    # predictions = torch.cat(predictions, dim=0)
    # predictions = torch.argmax(predictions, dim=1).cpu().numpy()
    predictions = None
    
    return model.val_acc[-1].cpu().item(), predictions


def fine_tuning(dl_train, dl_eval, model_backbone, number_of_classes, 
                   learning_rate=0.001, num_epochs=10, device='cuda'):
    """Linear probing evaluation"""
    model = deepcopy(model_backbone)

    for param in model.parameters():
        param.requires_grad = True

    # Add the linear layer to the Sequential model
    model = torch.nn.Sequential(
        model,
        torch.nn.Flatten(),  # Flatten the output of the pooling layer
        torch.nn.Linear(512, number_of_classes)
    )

    # Prepare the model
    model = ClassifierModel(
        model=model,
        num_classes=number_of_classes,
        lr=learning_rate, 
        max_epochs=num_epochs
    )

    # Prepare the trainer for pytorch lightning
    trainer = pl.Trainer(max_epochs=num_epochs, devices=-1, accelerator=device.type)

    # Train the model
    trainer.fit(model, dl_train, dl_eval)
    #model.eval()
    #predictions=model(dl_eval.to(device))
    # Get predictions on test set 
    # predictions = trainer.predict(model, dl_eval) # TODO: TO JEST ZLE
    # predictions = torch.cat(predictions, dim=0)
    # predictions = torch.argmax(predictions, dim=1).cpu().numpy()
    predictions = None
    
    return model.val_acc[-1].cpu().item(), predictions

def knn_evaluation(train_features, train_labels, test_features, test_labels, k_values):
    """k-NN evaluation dla różnych wartości k"""
    results = {}
    
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    test_features_scaled = scaler.transform(test_features)
    
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(train_features_scaled, train_labels)
        predictions = knn.predict(test_features_scaled)
        accuracy = accuracy_score(test_labels, predictions)
        results[k] = accuracy
    
    return results

def baseline_representations(features, labels, method='pca', n_components=50):
    """Tworzy reprezentacje używając PCA, t-SNE lub UMAP"""
    if method == 'pca':
        reducer = PCA(n_components=n_components, random_state=42)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    elif method == 'umap':
        reducer = umap.UMAP(n_components=2, random_state=42)
    
    reduced_features = reducer.fit_transform(features)
    return reduced_features

# Wizualizacje
def plot_confusion_matrix(y_true, y_pred, title, save_path):
    """Rysuje macierz pomyłek"""
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_space(features, labels, title, save_path, method='tsne'):
    """Wizualizuje przestrzeń cech używając t-SNE lub UMAP"""
    if features.shape[1] > 2:
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
        else:
            reducer = umap.UMAP(n_components=2, random_state=42)
        features_2d = reducer.fit_transform(features)
    else:
        features_2d = features
    
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                         c=labels, cmap='tab10', alpha=0.6, s=10)
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_comparison_results(results_df, metric='accuracy', save_path='comparison.png'):
    """Porównuje wyniki różnych metod"""
    plt.figure(figsize=(15, 8))
    
    # Grupowanie danych
    methods = results_df['method'].unique()
    datasets = results_df['test_dataset'].unique()
    
    x = np.arange(len(methods))
    width = 0.35
    
    for i, dataset in enumerate(datasets):
        data = results_df[results_df['test_dataset'] == dataset]
        accuracies = [data[data['method'] == m][metric].values[0] for m in methods]
        plt.bar(x + i*width, accuracies, width, label=dataset)
    
    plt.xlabel('Method')
    plt.ylabel(metric.capitalize())
    plt.title(f'{metric.capitalize()} Comparison Across Methods')
    plt.xticks(x + width/2, methods, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# Główna klasa ewaluacji
class SSLEvaluator:
    def __init__(self, config: Config):
        self.config = config
        self.results = []
        
        # Tworzenie katalogów
        os.makedirs(self.config.results_dir, exist_ok=True)
        os.makedirs(self.config.figures_dir, exist_ok=True)
    
    def evaluate_single_model(self, checkpoint_path, method, train_dataset, test_dataset, 
                            hyperparameter_info=None):
        """Ewaluuje pojedynczy model"""
        print(f"\nEvaluating {method} trained on {train_dataset}, testing on {test_dataset}")
        print(f"Checkpoint: {checkpoint_path}")
        
        # Wczytywanie modelu
        model = load_model(checkpoint_path, method, self.config.device)
        
        # Przygotowanie datasetów
        train_data = get_dataset(test_dataset, train=True)
        test_data = get_dataset(test_dataset, train=False)
        
        train_loader = DataLoader(train_data, batch_size=self.config.batch_size, 
                                shuffle=False, num_workers=self.config.num_workers)
        test_loader = DataLoader(test_data, batch_size=self.config.batch_size, 
                               shuffle=False, num_workers=self.config.num_workers)
        
        # Ekstrakcja cech
        train_features, train_labels = extract_features(model, train_loader, 
                                                       self.config.device, method)
        test_features, test_labels = extract_features(model, test_loader, 
                                                     self.config.device, method)
        if method == 'mae':
            backbone = model.encoder.backbone if hasattr(model.encoder, 'backbone') else model.encoder
        elif method == 'byol':
            backbone = model.online_backbone
        elif method == 'simclr':
            backbone = model.backbone
        else:
            backbone = model.backbone if hasattr(model, 'backbone') else model

        torch.cuda.empty_cache()
        gc.collect()


        fine_tuning_acc, _ = fine_tuning(
            DataLoader(train_data, batch_size=self.config.batch_size, 
                       shuffle=True, num_workers=self.config.num_workers),
            DataLoader(test_data, batch_size=self.config.batch_size, 
                       shuffle=False, num_workers=self.config.num_workers),
            backbone,  # Używamy backbone modelu
            len(np.unique(train_labels)),  # Liczba klas
            learning_rate=model.checkpoint_info['lr'],  # Używamy lr z checkpointa
            num_epochs=5,
            device=self.config.device
        )

        torch.cuda.empty_cache()
        gc.collect()

        # Linear probing
        linear_acc, linear_preds = linear_probing(
            DataLoader(train_data, batch_size=self.config.batch_size, 
                       shuffle=True, num_workers=self.config.num_workers),
            DataLoader(test_data, batch_size=self.config.batch_size, 
                       shuffle=False, num_workers=self.config.num_workers),
            backbone,  # Używamy backbone modelu
            len(np.unique(train_labels)),  # Liczba klas
            learning_rate=model.checkpoint_info['lr'],  # Używamy lr z checkpointa
            num_epochs=5,
            device=self.config.device
        )

        torch.cuda.empty_cache()
        gc.collect()

        # k-NN evaluation
        knn_results = knn_evaluation(train_features, train_labels, 
                                    test_features, test_labels, 
                                    self.config.knn_neighbors)
        
        # Zapisywanie wyników
        result = {
            'method': method,
            'train_dataset': train_dataset,
            'test_dataset': test_dataset,
            'hyperparameters': hyperparameter_info,
            'linear_probing_accuracy': linear_acc,
            'fine_tuning_acc': fine_tuning_acc,
            'knn_results': knn_results,
            'best_knn_accuracy': max(knn_results.values()),
            'best_knn_k': max(knn_results, key=knn_results.get)
        }
        
        # Wizualizacje
        save_prefix = f"{method}_{train_dataset}_{test_dataset}"
        if hyperparameter_info:
            save_prefix += f"_{hyperparameter_info}"
        
        # Confusion matrix
        # plot_confusion_matrix(test_labels, linear_preds, 
        #                     f"Confusion Matrix - {method} on {test_dataset}",
        #                     os.path.join(self.config.figures_dir, f"{save_prefix}_confusion.png"))
        
        # Feature space visualization
        plot_feature_space(test_features[:5000], test_labels[:5000],  # Subset for speed
                         f"Feature Space - {method} on {test_dataset}",
                         os.path.join(self.config.figures_dir, f"{save_prefix}_features.png"))
        
        return result
    
    def evaluate_baselines(self, test_dataset):
        """Ewaluuje baseline methods (PCA, t-SNE, UMAP)"""
        print(f"\nEvaluating baseline methods on {test_dataset}")
        
        # Wczytywanie danych
        train_data = get_dataset(test_dataset, train=True)
        test_data = get_dataset(test_dataset, train=False)
        
        # Konwersja do numpy
        train_features = []
        train_labels = []
        for img, label in train_data:
            train_features.append(img.numpy().flatten())
            train_labels.append(label)
        train_features = np.array(train_features)
        train_labels = np.array(train_labels)
        
        test_features = []
        test_labels = []
        for img, label in test_data:
            test_features.append(img.numpy().flatten())
            test_labels.append(label)
        test_features = np.array(test_features)
        test_labels = np.array(test_labels)
        
        baseline_results = []
        
        # PCA
        pca_train = baseline_representations(train_features, train_labels, 'pca', 
                                           self.config.pca_components)
        pca_test = PCA(n_components=self.config.pca_components, 
                      random_state=42).fit(train_features).transform(test_features)
        
        pca_acc, _ = linear_probing(pca_train, train_labels, pca_test, test_labels)
        pca_knn = knn_evaluation(pca_train, train_labels, pca_test, test_labels, 
                                self.config.knn_neighbors)
        
        baseline_results.append({
            'method': 'PCA',
            'test_dataset': test_dataset,
            'train_dataset': None,  # or 'N/A'
            'hyperparameters': None,
            'linear_probing_accuracy': pca_acc,
            'knn_results': pca_knn,
            'best_knn_accuracy': max(pca_knn.values()),
            'best_knn_k': max(pca_knn, key=pca_knn.get) if len(pca_knn) > 0 else None
        })
        
        return baseline_results

    def run_full_evaluation(self, results_no=1):
        """Uruchamia pełną ewaluację wszystkich modeli"""
        all_results = []

        # Ewaluacja modeli SSL
        for train_dataset in self.config.train_datasets:
            for method in self.config.ssl_methods:
                # Szukanie checkpointów
                method_path = os.path.join(self.config.checkpoint_base, train_dataset, method)
                if not os.path.exists(method_path):
                    print(f"Skipping {method_path} - directory not found")
                    continue

                # Iteracja przez różne wersje (ran, pre-2, pre-3)
                for version_dir in os.listdir(method_path):
                    version_path = os.path.join(method_path, version_dir)
                    if os.path.isdir(version_path):
                        # Znalezienie checkpointa
                        if train_dataset == 'imagenet':
                            checkpoint_files = [f for f in os.listdir(version_path)
                                                if f.endswith('.ckpt') and ('5' in f)]
                        else:
                            checkpoint_files = [f for f in os.listdir(version_path)
                                                if f.endswith('.ckpt') and ('19' in f or '20' in f)]
                        if checkpoint_files:
                            checkpoint_path = os.path.join(version_path, checkpoint_files[0])

                            # Ewaluacja na różnych datasetach testowych
                            for test_dataset in self.config.test_datasets:
                                result = self.evaluate_single_model(
                                    checkpoint_path, method, train_dataset,
                                    test_dataset, version_dir
                                )
                                all_results.append(result)

        # Ewaluacja baseline methods
        """for test_dataset in self.config.test_datasets:
            baseline_results = self.evaluate_baselines(test_dataset)
            all_results.extend(baseline_results)"""

        # Zapisywanie wyników
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(os.path.join(self.config.results_dir, f'evaluation_results_{results_no}.csv'),
                          index=False)

        # Zapisywanie szczegółowych wyników JSON
        with open(os.path.join(self.config.results_dir, f'detailed_results.json_{results_no}'), 'w') as f:
            json.dump(all_results, f, indent=2)

        # Generowanie wykresów porównawczych
        """plot_comparison_results(results_df, 'linear_probing_accuracy',
                              os.path.join(self.config.figures_dir, 'linear_probing_comparison.png'))
        plot_comparison_results(results_df, 'best_knn_accuracy',
                              os.path.join(self.config.figures_dir, 'knn_comparison.png'))"""

        return results_df
    
    def create_summary_report(self, results_df, results_no):
        """Tworzy raport podsumowujący"""
        report = []
        report.append("# SSL Model Evaluation Report\n")
        report.append("=" * 80 + "\n")
        
        # Najlepsze modele
        report.append("\n## Best Performing Models\n")
        best_linear = results_df.loc[results_df['linear_probing_accuracy'].idxmax()]
        report.append(f"Best Linear Probing: {best_linear['method']} "
                     f"(train: {best_linear['train_dataset']}, "
                     f"test: {best_linear['test_dataset']}) "
                     f"- Accuracy: {best_linear['linear_probing_accuracy']:.4f}\n")
        
        best_knn = results_df.loc[results_df['best_knn_accuracy'].idxmax()]
        report.append(f"Best k-NN: {best_knn['method']} "
                     f"(train: {best_knn['train_dataset']}, "
                     f"test: {best_knn['test_dataset']}) "
                     f"- Accuracy: {best_knn['best_knn_accuracy']:.4f} "
                     f"(k={best_knn.get('best_knn_k', 'N/A')})\n")
        
        # Porównanie metod
        report.append("\n## Method Comparison (Average Accuracies)\n")
        method_comparison = results_df.groupby('method')[
            ['linear_probing_accuracy', 'best_knn_accuracy']
        ].mean()
        report.append(method_comparison.to_string())
        
        # Zapisywanie raportu
        with open(os.path.join(self.config.results_dir, f'evaluation_report{results_no}.txt'), 'w') as f:
            f.writelines(report)
        
        print("\n".join(report))

# Funkcje pomocnicze do używania w Jupyter
def evaluate_single(checkpoint_path, method, train_dataset, test_dataset, config=None):
    """Wrapper do ewaluacji pojedynczego modelu"""
    if config is None:
        config = Config()
    
    evaluator = SSLEvaluator(config)
    result = evaluator.evaluate_single_model(checkpoint_path, method, 
                                           train_dataset, test_dataset)
    return result

def run_complete_evaluation(config=None, results_no=1):
    """Wrapper do pełnej ewaluacji"""
    if config is None:
        config = Config()
    
    set_seed(config.seed)
    evaluator = SSLEvaluator(config)
    results_df = evaluator.run_full_evaluation(results_no=results_no)
    evaluator.create_summary_report(results_df, results_no)
    
    return results_df