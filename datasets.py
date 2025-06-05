import torch
import torchvision
from torch.utils.data import DataLoader, Dataset, Sampler
import numpy as np
from typing import Optional, Union, Callable, Iterator, Sized
import warnings


class BootstrapDataLoader(DataLoader):
    """
    Custom DataLoader that extends PyTorch's DataLoader to implement bootstrap sampling.
    Automatically handles bootstrap sampling without needing custom datasets or samplers.
    """

    def __init__(self, dataset: Dataset, batch_size: int = 1,
                 bootstrap_size: Optional[int] = None,
                 bootstrap_method: str = 'epoch_dependent',
                 random_state: Optional[int] = None,
                 **kwargs):
        """
        Args:
            dataset: The dataset to bootstrap from
            batch_size: Batch size for the dataloader
            bootstrap_size: Size of bootstrap sample. If None, uses dataset size
            bootstrap_method: 'fixed', 'epoch_dependent', or 'random'
                - 'fixed': Same bootstrap sample throughout training
                - 'epoch_dependent': Different bootstrap sample each epoch (deterministic)
                - 'random': Completely random bootstrap sample each iteration
            random_state: Random seed for reproducibility
            **kwargs: Additional arguments passed to parent DataLoader
        """

        self.original_dataset = dataset
        self.bootstrap_size = bootstrap_size or len(dataset)
        self.bootstrap_method = bootstrap_method
        self.random_state = random_state
        self.current_epoch = 0

        # Remove conflicting arguments that we'll handle ourselves
        kwargs.pop('sampler', None)
        kwargs.pop('shuffle', None)  # We handle shuffling through bootstrap

        # Create our custom bootstrap sampler
        if bootstrap_method == 'fixed':
            sampler = self._create_fixed_bootstrap_sampler()
        elif bootstrap_method == 'epoch_dependent':
            sampler = self._create_epoch_bootstrap_sampler()
        elif bootstrap_method == 'random':
            sampler = self._create_random_bootstrap_sampler()
        else:
            raise ValueError(f"Unknown bootstrap_method: {bootstrap_method}")

        # Initialize parent DataLoader with our bootstrap sampler
        super().__init__(dataset, batch_size=batch_size, sampler=sampler, **kwargs)

        # Store reference to our sampler for epoch updates
        self.bootstrap_sampler = sampler

    def _create_fixed_bootstrap_sampler(self):
        """Create a sampler with fixed bootstrap indices."""
        class FixedBootstrapSampler(Sampler):
            def __init__(self, dataset_size: int, bootstrap_size: int, random_state: Optional[int]):
                self.dataset_size = dataset_size
                self.bootstrap_size = bootstrap_size

                # Generate fixed bootstrap indices
                if random_state is not None:
                    np.random.seed(random_state)
                self.bootstrap_indices = np.random.choice(
                    dataset_size, size=bootstrap_size, replace=True
                )

            def __iter__(self) -> Iterator[int]:
                return iter(self.bootstrap_indices)

            def __len__(self):
                return self.bootstrap_size

        return FixedBootstrapSampler(len(self.original_dataset), self.bootstrap_size, self.random_state)

    def _create_epoch_bootstrap_sampler(self):
        """Create a sampler that generates different bootstrap samples each epoch."""
        class EpochBootstrapSampler(Sampler):
            def __init__(self, dataset_size: int, bootstrap_size: int, base_seed: int):
                self.dataset_size = dataset_size
                self.bootstrap_size = bootstrap_size
                self.base_seed = base_seed
                self.epoch = 0

            def __iter__(self) -> Iterator[int]:
                # Generate epoch-specific bootstrap sample
                np.random.seed(self.base_seed + self.epoch)
                bootstrap_indices = np.random.choice(
                    self.dataset_size, size=self.bootstrap_size, replace=True
                )
                return iter(bootstrap_indices)

            def __len__(self):
                return self.bootstrap_size

            def set_epoch(self, epoch: int):
                self.epoch = epoch

        return EpochBootstrapSampler(
            len(self.original_dataset),
            self.bootstrap_size,
            self.random_state or 42
        )

    def _create_random_bootstrap_sampler(self):
        """Create a sampler with completely random bootstrap sampling."""
        class RandomBootstrapSampler(Sampler):
            def __init__(self, dataset_size: int, bootstrap_size: int):
                self.dataset_size = dataset_size
                self.bootstrap_size = bootstrap_size

            def __iter__(self) -> Iterator[int]:
                # Generate completely random bootstrap sample each time
                bootstrap_indices = np.random.choice(
                    self.dataset_size, size=self.bootstrap_size, replace=True
                )
                return iter(bootstrap_indices)

            def __len__(self):
                return self.bootstrap_size

        return RandomBootstrapSampler(len(self.original_dataset), self.bootstrap_size)

    def set_epoch(self, epoch: int):
        """
        Set the current epoch. This is automatically called by PyTorch Lightning.
        Only affects epoch_dependent bootstrap method.
        """
        self.current_epoch = epoch
        if hasattr(self.bootstrap_sampler, 'set_epoch'):
            self.bootstrap_sampler.set_epoch(epoch)

    def get_bootstrap_info(self):
        """Get information about the current bootstrap configuration."""
        return {
            'original_size': len(self.original_dataset),
            'bootstrap_size': self.bootstrap_size,
            'bootstrap_method': self.bootstrap_method,
            'random_state': self.random_state,
            'current_epoch': self.current_epoch
        }

    def get_current_bootstrap_indices(self):
        """
        Get the current bootstrap indices (only works for fixed method).
        For other methods, indices are generated on-the-fly.
        """
        if self.bootstrap_method == 'fixed' and hasattr(self.bootstrap_sampler, 'bootstrap_indices'):
            return self.bootstrap_sampler.bootstrap_indices.copy()
        else:
            warnings.warn(f"Cannot get indices for bootstrap_method='{self.bootstrap_method}'")
            return None


class StratifiedBootstrapDataLoader(BootstrapDataLoader):
    """
    Extended Bootstrap DataLoader that supports stratified bootstrap sampling.
    Maintains the class distribution in the bootstrap sample.
    """

    def __init__(self, dataset: Dataset, labels: np.ndarray, batch_size: int = 1,
                 bootstrap_size: Optional[int] = None,
                 bootstrap_method: str = 'epoch_dependent',
                 random_state: Optional[int] = None,
                 **kwargs):
        """
        Args:
            dataset: The dataset to bootstrap from
            labels: Array of labels for stratified sampling
            batch_size: Batch size for the dataloader
            bootstrap_size: Size of bootstrap sample
            bootstrap_method: Bootstrap sampling method
            random_state: Random seed
            **kwargs: Additional DataLoader arguments
        """

        self.labels = np.array(labels)
        self.unique_classes = np.unique(self.labels)

        # Store original init parameters for custom sampler creation
        self._dataset = dataset
        self._bootstrap_size = bootstrap_size or len(dataset)
        self._bootstrap_method = bootstrap_method
        self._random_state = random_state

        #create bootstrap sampler
        if bootstrap_method == 'fixed':
            sampler = self._create_stratified_fixed_sampler()
        elif bootstrap_method == 'epoch_dependent':
            sampler = self._create_stratified_epoch_sampler()
        elif bootstrap_method == 'random':
            sampler = self._create_stratified_random_sampler()
        else:
            raise ValueError(f"Unknown bootstrap_method: {bootstrap_method}")

        #pass it to super constructor
        super(BootstrapDataLoader, self).__init__(dataset, batch_size=batch_size, sampler=sampler, **kwargs)



        # # Replace the sampler
        # self.sampler = sampler
        self.bootstrap_sampler = sampler

    def _create_stratified_fixed_sampler(self):
        """Create stratified bootstrap sampler with fixed indices."""
        class StratifiedFixedBootstrapSampler(Sampler):
            def __init__(self, labels: np.ndarray, bootstrap_size: int, random_state: Optional[int]):
                self.labels = labels
                self.bootstrap_size = bootstrap_size
                self.unique_classes = np.unique(labels)

                if random_state is not None:
                    np.random.seed(random_state)

                # Calculate samples per class (proportional to original distribution)
                class_counts = np.bincount(labels)
                class_proportions = class_counts / len(labels)
                samples_per_class = (class_proportions * bootstrap_size).astype(int)

                # Adjust for rounding errors
                while samples_per_class.sum() < bootstrap_size:
                    # Add to the class with largest proportion
                    max_class = np.argmax(class_proportions)
                    samples_per_class[max_class] += 1
                    class_proportions[max_class] -= 0.01  # Reduce to avoid repeated selection

                # Generate stratified bootstrap indices
                self.bootstrap_indices = []
                for class_idx, n_samples in enumerate(samples_per_class):
                    if n_samples > 0:
                        class_indices = np.where(labels == class_idx)[0]
                        bootstrap_class_indices = np.random.choice(
                            class_indices, size=n_samples, replace=True
                        )
                        self.bootstrap_indices.extend(bootstrap_class_indices)

                self.bootstrap_indices = np.array(self.bootstrap_indices)
                np.random.shuffle(self.bootstrap_indices)  # Shuffle the final indices

            def __iter__(self) -> Iterator[int]:
                return iter(self.bootstrap_indices)

            def __len__(self):
                return len(self.bootstrap_indices)

        return StratifiedFixedBootstrapSampler(self.labels, self._bootstrap_size, self._random_state)

    def _create_stratified_epoch_sampler(self):
        """Create stratified bootstrap sampler that changes each epoch."""
        class StratifiedEpochBootstrapSampler(Sampler):
            def __init__(self, labels: np.ndarray, bootstrap_size: int, base_seed: int):
                self.labels = labels
                self.bootstrap_size = bootstrap_size
                self.base_seed = base_seed
                self.unique_classes = np.unique(labels)
                self.epoch = 0

                # Pre-calculate class information
                self.class_counts = np.bincount(labels)
                self.class_proportions = self.class_counts / len(labels)
                self.samples_per_class = (self.class_proportions * bootstrap_size).astype(int)

                # Adjust for rounding
                while self.samples_per_class.sum() < bootstrap_size:
                    max_class = np.argmax(self.class_proportions)
                    self.samples_per_class[max_class] += 1
                    self.class_proportions[max_class] -= 0.01

            def __iter__(self) -> Iterator[int]:
                np.random.seed(self.base_seed + self.epoch)

                bootstrap_indices = []
                for class_idx, n_samples in enumerate(self.samples_per_class):
                    if n_samples > 0:
                        class_indices = np.where(self.labels == class_idx)[0]
                        bootstrap_class_indices = np.random.choice(
                            class_indices, size=n_samples, replace=True
                        )
                        bootstrap_indices.extend(bootstrap_class_indices)

                bootstrap_indices = np.array(bootstrap_indices)
                np.random.shuffle(bootstrap_indices)
                return iter(bootstrap_indices)

            def __len__(self):
                return self.bootstrap_size

            def set_epoch(self, epoch: int):
                self.epoch = epoch

        return StratifiedEpochBootstrapSampler(self.labels, self._bootstrap_size, self._random_state or 42)

    def _create_stratified_random_sampler(self):
        """Create completely random stratified bootstrap sampler."""
        # Similar implementation to epoch sampler but without epoch dependency
        return self._create_stratified_epoch_sampler()


# Convenience functions for easy usage
def create_bootstrap_dataloader(dataset: Dataset, bootstrap_size: Optional[int] = None,
                                bootstrap_method: str = 'epoch_dependent',
                                batch_size: int = 32, random_state: Optional[int] = None,
                                **kwargs) -> BootstrapDataLoader:
    """
    Factory function to create a bootstrap dataloader.

    Args:
        dataset: PyTorch dataset
        bootstrap_size: Bootstrap sample size (None = dataset size)
        bootstrap_method: 'fixed', 'epoch_dependent', or 'random'
        batch_size: Batch size
        random_state: Random seed
        **kwargs: Additional DataLoader arguments

    Returns:
        BootstrapDataLoader instance
    """
    return BootstrapDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        bootstrap_size=bootstrap_size,
        bootstrap_method=bootstrap_method,
        random_state=random_state,
        **kwargs
    )


def create_stratified_bootstrap_dataloader(dataset: Dataset, labels: np.ndarray,
                                           bootstrap_size: Optional[int] = None,
                                           bootstrap_method: str = 'epoch_dependent',
                                           batch_size: int = 32,
                                           random_state: Optional[int] = None,
                                           **kwargs) -> StratifiedBootstrapDataLoader:
    """
    Factory function to create a stratified bootstrap dataloader.

    Args:
        dataset: PyTorch dataset
        labels: Labels for stratified sampling
        bootstrap_size: Bootstrap sample size
        bootstrap_method: Bootstrap method
        batch_size: Batch size
        random_state: Random seed
        **kwargs: Additional DataLoader arguments

    Returns:
        StratifiedBootstrapDataLoader instance
    """
    return StratifiedBootstrapDataLoader(
        dataset=dataset,
        labels=labels,
        batch_size=batch_size,
        bootstrap_size=bootstrap_size,
        bootstrap_method=bootstrap_method,
        random_state=random_state,
        **kwargs
    )
from sklearn.model_selection import train_test_split

import os
def create_dataset(set_name, SSL_proportion, train_transform, train_full_transform, test_transform, path_to_data, seed=42, download=True):
    if set_name == 'CIFAR10':
        train_ssl_dataset = torchvision.datasets.CIFAR10(root=path_to_data, train=True, transform=train_transform, download=download)
        train_full_dataset = torchvision.datasets.CIFAR10(root=path_to_data, train=True, transform=train_full_transform, download=download)
        test_dataset = torchvision.datasets.CIFAR10(root=path_to_data, train=False, transform=test_transform, download=download)
    elif set_name == 'CIFAR100':
        train_ssl_dataset = torchvision.datasets.CIFAR100(root=path_to_data, train=True, transform=train_transform, download=download)
        train_full_dataset = torchvision.datasets.CIFAR100(root=path_to_data, train=True, transform=train_full_transform, download=download)
        test_dataset = torchvision.datasets.CIFAR100(root=path_to_data, train=False, transform=test_transform, download=download)
    #trochę inna logika dla imageneta trzeba mieć wcześniej dobrze foldery porobione
    else:
        train_path = os.path.join(path_to_data, 'imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train')
        test_path = os.path.join(path_to_data, 'imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val_restructured')
        train_ssl_dataset = torchvision.datasets.ImageFolder(root=train_path, transform=train_transform)
        train_full_dataset = torchvision.datasets.ImageFolder(root=train_path, transform=train_full_transform)
        test_dataset = torchvision.datasets.ImageFolder(root=test_path, transform=test_transform)

    targets = np.array([y for _, y in train_full_dataset])
    SSL_indices, classification_indices = train_test_split(
        np.arange(len(targets)),
         test_size=1-SSL_proportion,
         random_state=seed,
         stratify=targets
    )
    train_dataset = torch.utils.data.Subset(train_full_dataset, classification_indices)
    train_ssl_dataset = torch.utils.data.Subset(train_ssl_dataset, SSL_indices)
    print("Length of entire train dataset: ", len(train_full_dataset))
    print("Length of SSL train dataset: ", len(train_ssl_dataset))
    print("Length of classification train dataset: ", len(train_dataset))
    print("Length of test dataset: ", len(test_dataset))
    return train_full_dataset, train_ssl_dataset, train_dataset, test_dataset, targets
