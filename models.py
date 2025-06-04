import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from lightly.models.modules.heads import SimCLRProjectionHead
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import numpy as np
import copy
import torchvision
import torchmetrics
from typing import Literal
from lightly.loss import NTXentLoss
def load_resnet(number_of_classes: int | None = None) -> torch.nn.Module:
    model = torchvision.models.resnet18(pretrained=False)
    if number_of_classes is not None:
        model.fc = torch.nn.Linear(model.fc.in_features, number_of_classes)
    return model

class SimpleMLP(nn.Module):
    """Simple MLP backbone with approximately 1M parameters."""
    def __init__(self, input_dim: int = 3072, hidden_dims: list =  [128, 128], output_dim: int = 128):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)
        self.fc = nn.Identity()  # For compatibility with ResNet interface
        self.fc.in_features = output_dim

    def forward(self, x):
        # Flatten input if it's an image (B, C, H, W) -> (B, C*H*W)
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        return self.network(x).unsqueeze(-1).unsqueeze(-1)  # Add dummy spatial dimensions

def create_resnet18_backbone(pretrained: bool = True):
    """Create ResNet18 backbone with or without pretrained weights."""
    import torchvision.models as models

    if pretrained:
        # Load pretrained ResNet18
        resnet = models.resnet18(pretrained=True)
    else:
        # Create ResNet18 with random initialization
        resnet = models.resnet18(pretrained=False)
        # Apply custom initialization
        for m in resnet.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    return resnet

def create_backbone(backbone_type: Literal["pretrained_resnet18", "random_resnet18", "simple_mlp"], input_dim):
    # Create backbone based on type
    if backbone_type == "pretrained_resnet18":
        # Option 1: Pretrained ResNet18
        resnet = create_resnet18_backbone(pretrained=True)
        backbone = nn.Sequential(*list(resnet.children())[:-1])
        hidden_dim = resnet.fc.in_features

    elif backbone_type == "random_resnet18":
        # Option 2: ResNet18 architecture with random initialization
        resnet = create_resnet18_backbone(pretrained=False)
        backbone = nn.Sequential(*list(resnet.children())[:-1])
        hidden_dim = resnet.fc.in_features

    elif backbone_type == "simple_mlp":
        # Option 3: Simple MLP with ~1M parameters
        backbone = SimpleMLP(input_dim=input_dim)
        hidden_dim = backbone.fc.in_features

    else:
        raise ValueError(f"Unknown backbone_type: {backbone_type}")

    return backbone, hidden_dim

class ClassifierModel(pl.LightningModule):
    def __init__(self, model: torch.nn.Module, num_classes, lr: float = 6e-2, weight_decay: float = 5e-4, max_epochs: int = 100):
        super().__init__()
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.save_hyperparameters(ignore=["model"])

        # Separate metrics for each phase to avoid state contamination
        self.train_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

        self.train_acc = []
        self.val_acc = []
        self.test_acc = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        preds = torch.argmax(y_hat, dim=1)
        self.train_metric(preds, y)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_metric, prog_bar=True, on_epoch=True, on_step=False)

        return loss

    def on_train_epoch_end(self):
        epoch_acc = self.train_metric.compute()
        self.train_acc.append(epoch_acc)
        self.train_metric.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        preds = torch.argmax(y_hat, dim=1)
        self.val_metric(preds, y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_metric, prog_bar=True)

        return loss

    def on_validation_epoch_end(self):
        epoch_acc = self.val_metric.compute()
        self.val_acc.append(epoch_acc)
        self.val_metric.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        preds = torch.argmax(y_hat, dim=1)
        self.test_metric(preds, y)

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_metric, prog_bar=True)

        return loss

    def on_test_epoch_end(self):
        epoch_acc = self.test_metric.compute()
        self.test_acc.append(epoch_acc)
        self.test_metric.reset()

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.max_epochs)
        return [optim], [scheduler]

"""
---------------------------------------------------------------------------------------------------------------------
BYOL
---------------------------------------------------------------------------------------------------------------------
"""

class BYOLModel(pl.LightningModule):
    def __init__(self,
                 lr: float = 6e-2,
                 weight_decay: float = 5e-4,
                 max_epochs: int = 100,
                 num_clusters: int = 4,
                 tau: float = 0.996,
                 backbone_type: Literal["pretrained_resnet18", "random_resnet18", "simple_mlp"] = "pretrained_resnet18",
                 input_dim: int = 150528):  # For MLP: 224x224x3 = 150528 for Imagenet1K
        super().__init__()

        backbone, hidden_dim = create_backbone(backbone_type, input_dim)

        # Online network (f_theta + g_theta + q_theta)
        self.online_backbone = backbone  # f_theta
        self.online_projection = SimCLRProjectionHead(hidden_dim, hidden_dim, 128)  # g_theta
        self.online_predictor = self._build_predictor(128, 128)  # q_theta (MLP)

        # Target network (f_xi + g_xi) - no predictor needed
        self.target_backbone = copy.deepcopy(backbone)  # f_xi
        self.target_projection = copy.deepcopy(self.online_projection)  # g_xi

        # Disable gradients for target network
        for param in self.target_backbone.parameters():
            param.requires_grad = False
        for param in self.target_projection.parameters():
            param.requires_grad = False

        # Hyperparameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.num_clusters = num_clusters
        self.tau = tau  # EMA coefficient for target network update
        self.save_hyperparameters()

        # Clustering metrics storage (same as SimCLR)
        self.train_embeddings = []
        self.train_labels = []
        self.val_embeddings = []
        self.val_labels = []
        self.test_embeddings = []
        self.test_labels = []

        # Lists to store clustering scores over epochs
        self.train_ari_scores = []
        self.train_nmi_scores = []
        self.val_ari_scores = []
        self.val_nmi_scores = []
        self.test_ari_scores = []
        self.test_nmi_scores = []

    def _build_predictor(self, input_dim: int, output_dim: int) -> nn.Module:
        """Build a simple MLP predictor for the online network."""
        return nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.BatchNorm1d(input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, output_dim)
        )

    def forward(self, x):
        """Forward pass through online network (used for inference/clustering)."""
        h = self.online_backbone(x).flatten(start_dim=1)
        z = self.online_projection(h)
        return z

    def _forward_online(self, x):
        """Complete forward pass through online network (backbone + projection + predictor)."""
        h = self.online_backbone(x).flatten(start_dim=1)  # f_theta(x)
        z = self.online_projection(h)  # g_theta(f_theta(x))
        p = self.online_predictor(z)  # q_theta(g_theta(f_theta(x)))
        return p, z

    def _forward_target(self, x):
        """Forward pass through target network (backbone + projection)."""
        with torch.no_grad():
            h = self.target_backbone(x).flatten(start_dim=1)  # f_xi(x)
            z = self.target_projection(h)  # g_xi(f_xi(x))
        return z

    def _compute_loss(self, p1, p2, z1, z2):
        """Compute BYOL loss (symmetric)."""
        # Normalize predictions and targets
        p1_norm = F.normalize(p1, dim=1)
        p2_norm = F.normalize(p2, dim=1)
        z1_norm = F.normalize(z1, dim=1)
        z2_norm = F.normalize(z2, dim=1)

        # Compute loss: L = 2 - 2 * <p1, z2> - 2 * <p2, z1>
        loss = 2 - 2 * (p1_norm * z2_norm.detach()).sum(dim=1).mean() - 2 * (p2_norm * z1_norm.detach()).sum(dim=1).mean()
        return loss

    def _update_target_network(self):
        """Update target network using exponential moving average."""
        for param_online, param_target in zip(self.online_backbone.parameters(), self.target_backbone.parameters()):
            param_target.data = self.tau * param_target.data + (1 - self.tau) * param_online.data

        for param_online, param_target in zip(self.online_projection.parameters(), self.target_projection.parameters()):
            param_target.data = self.tau * param_target.data + (1 - self.tau) * param_online.data

    def training_step(self, batch, batch_idx):
        # Handle different possible batch structures
        if isinstance(batch, tuple) and len(batch) == 2:
            # Expected format: ((x1, x2), labels)
            (x1, x2), labels = batch
        elif isinstance(batch, list) and len(batch) == 3:
            # Alternative format: [x1, x2, labels]
            x1, x2, labels = batch
        elif isinstance(batch, tuple) and len(batch) == 3:
            # Alternative format: (x1, x2, labels)
            x1, x2, labels = batch
        else:
            # Try to infer structure based on batch content
            if len(batch) == 2:
                # Could be (inputs, labels) where inputs contains both views
                inputs, labels = batch
                if isinstance(inputs, torch.Tensor) and inputs.shape[0] == 2:
                    # Inputs might be a stacked tensor with two views
                    x1, x2 = inputs[0], inputs[1]
                elif isinstance(inputs, (list, tuple)) and len(inputs) == 2:
                    # Inputs might be a list/tuple of two tensors
                    x1, x2 = inputs
                else:
                    # Handle case where inputs is a single view (use it as both views)
                    x1 = x2 = inputs
            else:
                raise ValueError(f"Unexpected batch format: {type(batch)}, {len(batch)}")



        # Forward pass through online network
        p1, z1 = self._forward_online(x1)
        p2, z2 = self._forward_online(x2)

        # Forward pass through target network
        z1_target = self._forward_target(x1)
        z2_target = self._forward_target(x2)

        # Compute BYOL loss
        loss = self._compute_loss(p1, p2, z1_target, z2_target)

        self.log("train_loss", loss, prog_bar=True)

        # Store embeddings for clustering evaluation (use z1 from online network)
        self.train_embeddings.append(z1.detach().cpu())
        self.train_labels.append(labels.cpu())

        torch.cuda.empty_cache()


        return loss

    def on_train_epoch_end(self):
        # Update target network
        self._update_target_network()

        # Clustering evaluation (same as SimCLR)
        if len(self.train_embeddings) > 0:
            embeddings = torch.cat(self.train_embeddings, dim=0).numpy()
            labels = torch.cat(self.train_labels, dim=0).numpy()

            ari_score, nmi_score = self._evaluate_clustering(embeddings, labels)

            self.log("train_ari", ari_score, prog_bar=True)
            self.log("train_nmi", nmi_score, prog_bar=True)
            self.train_ari_scores.append(ari_score)
            self.train_nmi_scores.append(nmi_score)

            self.train_embeddings.clear()
            self.train_labels.clear()

    def validation_step(self, batch, batch_idx):
        # Handle different possible batch structures
        if isinstance(batch, tuple) and len(batch) == 2:
            # Expected format: ((x1, x2), labels)
            (x1, x2), labels = batch
        elif isinstance(batch, list) and len(batch) == 3:
            # Alternative format: [x1, x2, labels]
            x1, x2, labels = batch
        elif isinstance(batch, tuple) and len(batch) == 3:
            # Alternative format: (x1, x2, labels)
            x1, x2, labels = batch
        else:
            # Try to infer structure based on batch content
            if len(batch) == 2:
                # Could be (inputs, labels) where inputs contains both views
                inputs, labels = batch
                if isinstance(inputs, torch.Tensor) and inputs.shape[0] == 2:
                    # Inputs might be a stacked tensor with two views
                    x1, x2 = inputs[0], inputs[1]
                elif isinstance(inputs, (list, tuple)) and len(inputs) == 2:
                    # Inputs might be a list/tuple of two tensors
                    x1, x2 = inputs
                else:
                    # Handle case where inputs is a single view (use it as both views)
                    x1 = x2 = inputs
            else:
                raise ValueError(f"Unexpected batch format: {type(batch)}, {len(batch)}")



        # Forward pass through online network
        p1, z1 = self._forward_online(x1)
        p2, z2 = self._forward_online(x2)

        # Forward pass through target network
        z1_target = self._forward_target(x1)
        z2_target = self._forward_target(x2)

        # Compute BYOL loss
        loss = self._compute_loss(p1, p2, z1_target, z2_target)

        self.log("val_loss", loss, prog_bar=True)

        # Store embeddings for clustering evaluation
        self.val_embeddings.append(z1.detach().cpu())
        self.val_labels.append(labels.cpu())

        return loss

    def on_validation_epoch_end(self):
        if len(self.val_embeddings) > 0:
            embeddings = torch.cat(self.val_embeddings, dim=0).numpy()
            labels = torch.cat(self.val_labels, dim=0).numpy()

            ari_score, nmi_score = self._evaluate_clustering(embeddings, labels)

            self.log("val_ari", ari_score, prog_bar=True)
            self.log("val_nmi", nmi_score, prog_bar=True)
            self.val_ari_scores.append(ari_score)
            self.val_nmi_scores.append(nmi_score)

            self.val_embeddings.clear()
            self.val_labels.clear()

    def test_step(self, batch, batch_idx):
        # Handle different possible batch structures
        if isinstance(batch, tuple) and len(batch) == 2:
            # Expected format: ((x1, x2), labels)
            (x1, x2), labels = batch
        elif isinstance(batch, list) and len(batch) == 3:
            # Alternative format: [x1, x2, labels]
            x1, x2, labels = batch
        elif isinstance(batch, tuple) and len(batch) == 3:
            # Alternative format: (x1, x2, labels)
            x1, x2, labels = batch
        else:
            # Try to infer structure based on batch content
            if len(batch) == 2:
                # Could be (inputs, labels) where inputs contains both views
                inputs, labels = batch
                if isinstance(inputs, torch.Tensor) and inputs.shape[0] == 2:
                    # Inputs might be a stacked tensor with two views
                    x1, x2 = inputs[0], inputs[1]
                elif isinstance(inputs, (list, tuple)) and len(inputs) == 2:
                    # Inputs might be a list/tuple of two tensors
                    x1, x2 = inputs
                else:
                    # Handle case where inputs is a single view (use it as both views)
                    x1 = x2 = inputs
            else:
                raise ValueError(f"Unexpected batch format: {type(batch)}, {len(batch)}")



        # Forward pass through online network
        p1, z1 = self._forward_online(x1)
        p2, z2 = self._forward_online(x2)

        # Forward pass through target network
        z1_target = self._forward_target(x1)
        z2_target = self._forward_target(x2)

        # Compute BYOL loss
        loss = self._compute_loss(p1, p2, z1_target, z2_target)

        self.log("test_loss", loss, prog_bar=True)

        # Store embeddings for clustering evaluation
        self.test_embeddings.append(z1.detach().cpu())
        self.test_labels.append(labels.cpu())

        return loss

    def on_test_epoch_end(self):
        if len(self.test_embeddings) > 0:
            embeddings = torch.cat(self.test_embeddings, dim=0).numpy()
            labels = torch.cat(self.test_labels, dim=0).numpy()

            ari_score, nmi_score = self._evaluate_clustering(embeddings, labels)

            self.log("test_ari", ari_score, prog_bar=True)
            self.log("test_nmi", nmi_score, prog_bar=True)
            self.test_ari_scores.append(ari_score)
            self.test_nmi_scores.append(nmi_score)

            self.test_embeddings.clear()
            self.test_labels.clear()

    def _evaluate_clustering(self, embeddings, true_labels):
        """
        Evaluate clustering quality using K-means clustering.

        Args:
            embeddings: numpy array of shape (n_samples, embedding_dim)
            true_labels: numpy array of shape (n_samples,) with ground truth labels

        Returns:
            tuple: (adjusted_rand_score, normalized_mutual_info_score)
        """
        try:
            kmeans = KMeans(n_clusters=self.num_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)

            ari_score = adjusted_rand_score(true_labels, cluster_labels)
            nmi_score = normalized_mutual_info_score(true_labels, cluster_labels)

            return ari_score, nmi_score

        except Exception as e:
            print(f"Clustering evaluation failed: {e}")
            return 0.0, 0.0

    def configure_optimizers(self):
        # Only optimize online network parameters
        online_params = list(self.online_backbone.parameters()) + \
                        list(self.online_projection.parameters()) + \
                        list(self.online_predictor.parameters())

        optim = torch.optim.AdamW(
            online_params, lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.max_epochs)
        return [optim], [scheduler]

"""
---------------------------------------------------------------------------------------------------------------------
DINO
---------------------------------------------------------------------------------------------------------------------
"""

class DINOModel(pl.LightningModule):
    def __init__(self, lr: float = 5e-4, weight_decay: float = 4e-5, max_epochs: int = 100,
                 num_clusters: int = 4, tau: float = 0.996, student_temp: float = 0.1,
                 teacher_temp: float = 0.04, warmup_teacher_temp: float = 0.04,
                 warmup_teacher_temp_epochs: int = 0, output_dim: int = 128,
                 backbone_type: Literal["pretrained_resnet18", "random_resnet18", "simple_mlp"] = "pretrained_resnet18",
                 input_dim: int = 150528):
        super().__init__()

        backbone, hidden_dim = create_backbone(backbone_type, input_dim)

        # Student network (f + h)
        self.student_backbone = backbone  # f
        self.student_projection = SimCLRProjectionHead(hidden_dim, hidden_dim, output_dim)  # h

        # Teacher network (f + h) - same architecture as student
        self.teacher_backbone = copy.deepcopy(backbone)  # f
        self.teacher_projection = copy.deepcopy(self.student_projection)  # h

        # Disable gradients for teacher network
        for param in self.teacher_backbone.parameters():
            param.requires_grad = False
        for param in self.teacher_projection.parameters():
            param.requires_grad = False

        # Hyperparameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.num_clusters = num_clusters
        self.tau = tau  # EMA coefficient for teacher network update
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.warmup_teacher_temp = warmup_teacher_temp
        self.warmup_teacher_temp_epochs = warmup_teacher_temp_epochs
        self.output_dim = output_dim
        self.save_hyperparameters()

        # Center for teacher outputs (prevents collapse)
        self.register_buffer("center", torch.zeros(1, output_dim))

        # Clustering metrics storage
        self.train_embeddings = []
        self.train_labels = []
        self.val_embeddings = []
        self.val_labels = []
        self.test_embeddings = []
        self.test_labels = []

        # Lists to store clustering scores over epochs
        self.train_ari_scores = []
        self.train_nmi_scores = []
        self.val_ari_scores = []
        self.val_nmi_scores = []
        self.test_ari_scores = []
        self.test_nmi_scores = []

    def forward(self, x):
        """Forward pass through student network (used for inference/clustering)."""
        h = self.student_backbone(x).flatten(start_dim=1)
        z = self.student_projection(h)
        return z

    def _forward_student(self, x):
        """Forward pass through student network."""
        h = self.student_backbone(x).flatten(start_dim=1)
        z = self.student_projection(h)
        return z

    def _forward_teacher(self, x):
        """Forward pass through teacher network."""
        with torch.no_grad():
            h = self.teacher_backbone(x).flatten(start_dim=1)
            z = self.teacher_projection(h)
        return z

    def _get_teacher_temp(self):
        """Get current teacher temperature with warmup schedule."""
        if self.current_epoch < self.warmup_teacher_temp_epochs:
            # Linear warmup from warmup_teacher_temp to teacher_temp
            alpha = self.current_epoch / self.warmup_teacher_temp_epochs
            return self.warmup_teacher_temp + alpha * (self.teacher_temp - self.warmup_teacher_temp)
        return self.teacher_temp

    def _compute_loss(self, student_outputs, teacher_outputs):
        """
        Compute DINO loss between student and teacher outputs.

        Args:
            student_outputs: List of student outputs for different crops
            teacher_outputs: List of teacher outputs for different crops (global crops only)
        """
        total_loss = 0
        n_loss_terms = 0

        current_teacher_temp = self._get_teacher_temp()

        # Teacher outputs (only global crops, softmax with centering)
        teacher_out = torch.cat(teacher_outputs, dim=0)
        teacher_out = F.softmax((teacher_out - self.center) / current_teacher_temp, dim=-1)
        teacher_out = teacher_out.detach()

        # Split teacher outputs back to individual crops
        teacher_crops = torch.chunk(teacher_out, len(teacher_outputs), dim=0)

        # Student outputs (all crops, softmax)
        for i, student_crop in enumerate(student_outputs):
            student_out = F.log_softmax(student_crop / self.student_temp, dim=-1)

            # Compare with all teacher crops
            for j, teacher_crop in enumerate(teacher_crops):
                if i == j:
                    continue  # Don't compare crop with itself

                loss = torch.sum(-teacher_crop * student_out, dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1

        return total_loss / n_loss_terms

    def _update_teacher_network(self):
        """Update teacher network using exponential moving average."""
        for param_student, param_teacher in zip(self.student_backbone.parameters(), self.teacher_backbone.parameters()):
            param_teacher.data = self.tau * param_teacher.data + (1 - self.tau) * param_student.data

        for param_student, param_teacher in zip(self.student_projection.parameters(), self.teacher_projection.parameters()):
            param_teacher.data = self.tau * param_teacher.data + (1 - self.tau) * param_student.data

    def _update_center(self, teacher_outputs):
        """
        Update center used in teacher output centering.
        """
        with torch.no_grad():
            teacher_out = torch.cat(teacher_outputs, dim=0)
            batch_center = torch.mean(teacher_out, dim=0, keepdim=True)
            # EMA update of center
            self.center = self.center * 0.9 + batch_center * 0.1

    def training_step(self, batch, batch_idx):
        # Handle different possible batch structures
        if isinstance(batch, tuple) and len(batch) == 2:
            # Case where batch is (images, labels)
            images, labels = batch
            if isinstance(images, torch.Tensor):
                # If images is a single tensor, use it for both x1 and x2
                x1 = x2 = images
            elif isinstance(images, tuple) and len(images) >= 2:
                # If images is a tuple of tensors, use the first two
                x1, x2 = images[0], images[1]
            else:
                raise ValueError(f"Unexpected images format in batch: {type(images)}")
        elif isinstance(batch, list):
            # Handle list-type batches
            if len(batch) > 0 and isinstance(batch[0], torch.Tensor):
                # If the list contains tensors, use the first one for both inputs
                x1 = x2 = batch[0]
                labels = torch.zeros(x1.size(0), dtype=torch.long, device=x1.device)
            elif len(batch) > 0 and isinstance(batch[0], list) and len(batch[0]) > 0:
                # Handle nested list structure - extract tensors from first sublist
                if isinstance(batch[0][0], torch.Tensor):
                    x1 = x2 = batch[0][0]
                    labels = torch.zeros(x1.size(0), dtype=torch.long, device=x1.device)
                else:
                    # If still not a tensor, try to convert or further inspect
                    raise ValueError(f"Unsupported inner element type in batch: {type(batch[0][0])}")
            else:
                # For more complex list structures, you might need to inspect and handle differently
                raise ValueError(f"Unsupported list structure in batch: {type(batch[0])}")
        else:
            # If batch is a tensor or another type
            x1 = x2 = batch
            labels = torch.zeros(x1.size(0), dtype=torch.long, device=x1.device)

        # Forward pass through student network
        student_out1 = self._forward_student(x1)
        student_out2 = self._forward_student(x2)
        student_outputs = [student_out1, student_out2]

        # Forward pass through teacher network (global crops only)
        teacher_out1 = self._forward_teacher(x1)
        teacher_out2 = self._forward_teacher(x2)
        teacher_outputs = [teacher_out1, teacher_out2]

        # Update center
        self._update_center(teacher_outputs)

        # Compute DINO loss
        loss = self._compute_loss(student_outputs, teacher_outputs)

        self.log("train_loss", loss, prog_bar=True)
        self.log("teacher_temp", self._get_teacher_temp(), prog_bar=True)

        # Store embeddings for clustering evaluation (use student output)
        self.train_embeddings.append(student_out1.detach().cpu())
        self.train_labels.append(labels.cpu())

        torch.cuda.empty_cache()

        return loss


    def on_train_epoch_end(self):
        # Update teacher network
        self._update_teacher_network()

        # Clustering evaluation
        if len(self.train_embeddings) > 0:
            embeddings = torch.cat(self.train_embeddings, dim=0).numpy()
            labels = torch.cat(self.train_labels, dim=0).numpy()

            ari_score, nmi_score = self._evaluate_clustering(embeddings, labels)

            self.log("train_ari", ari_score, prog_bar=True)
            self.log("train_nmi", nmi_score, prog_bar=True)
            self.train_ari_scores.append(ari_score)
            self.train_nmi_scores.append(nmi_score)

            self.train_embeddings.clear()
            self.train_labels.clear()

    def validation_step(self, batch, batch_idx):
        # Handle different possible batch structures
        if isinstance(batch, tuple) and len(batch) == 2:
            # Case where batch is (images, labels)
            images, labels = batch
            if isinstance(images, torch.Tensor):
                # If images is a single tensor, use it for both x1 and x2
                x1 = x2 = images
            elif isinstance(images, tuple) and len(images) >= 2:
                # If images is a tuple of tensors, use the first two
                x1, x2 = images[0], images[1]
            else:
                raise ValueError(f"Unexpected images format in batch: {type(images)}")
        elif isinstance(batch, list):
            # Handle list-type batches
            if len(batch) > 0 and isinstance(batch[0], torch.Tensor):
                # If the list contains tensors, use the first one for both inputs
                x1 = x2 = batch[0]
                labels = torch.zeros(x1.size(0), dtype=torch.long, device=x1.device)
            else:
                # For more complex list structures, you might need to inspect and handle differently
                raise ValueError(f"Unsupported list structure in validation batch: {type(batch[0])}")
        else:
            # If batch is a tensor or another type
            x1 = x2 = batch
            labels = torch.zeros(x1.size(0), dtype=torch.long, device=x1.device)

        # Forward pass through student network
        student_out1 = self._forward_student(x1)
        student_out2 = self._forward_student(x2)
        student_outputs = [student_out1, student_out2]

        # Forward pass through teacher network
        teacher_out1 = self._forward_teacher(x1)
        teacher_out2 = self._forward_teacher(x2)
        teacher_outputs = [teacher_out1, teacher_out2]

        # Compute DINO loss
        loss = self._compute_loss(student_outputs, teacher_outputs)

        self.log("val_loss", loss, prog_bar=True)

        # Store embeddings for clustering evaluation
        self.val_embeddings.append(student_out1.detach().cpu())
        self.val_labels.append(labels.cpu())

        return loss



    # def validation_step(self, batch, batch_idx):
    #     x1, x2 = batch
    #
    #     # Forward pass through student network
    #     student_out1 = self._forward_student(x1)
    #     student_out2 = self._forward_student(x2)
    #     student_outputs = [student_out1, student_out2]
    #
    #     # Forward pass through teacher network
    #     teacher_out1 = self._forward_teacher(x1)
    #     teacher_out2 = self._forward_teacher(x2)
    #     teacher_outputs = [teacher_out1, teacher_out2]
    #
    #     # Compute DINO loss
    #     loss = self._compute_loss(student_outputs, teacher_outputs)
    #
    #     self.log("val_loss", loss, prog_bar=True)
    #
    #     # Store embeddings for clustering evaluation
    #     self.val_embeddings.append(student_out1.detach().cpu())
    #     self.val_labels.append(labels.cpu())
    #
    #     return loss

    def on_validation_epoch_end(self):
        if len(self.val_embeddings) > 0:
            embeddings = torch.cat(self.val_embeddings, dim=0).numpy()
            labels = torch.cat(self.val_labels, dim=0).numpy()

            ari_score, nmi_score = self._evaluate_clustering(embeddings, labels)

            self.log("val_ari", ari_score, prog_bar=True)
            self.log("val_nmi", nmi_score, prog_bar=True)
            self.val_ari_scores.append(ari_score)
            self.val_nmi_scores.append(nmi_score)

            self.val_embeddings.clear()
            self.val_labels.clear()

    def test_step(self, batch, batch_idx):
        (x1, x2), labels = batch

        # Forward pass through student network
        student_out1 = self._forward_student(x1)
        student_out2 = self._forward_student(x2)
        student_outputs = [student_out1, student_out2]

        # Forward pass through teacher network
        teacher_out1 = self._forward_teacher(x1)
        teacher_out2 = self._forward_teacher(x2)
        teacher_outputs = [teacher_out1, teacher_out2]

        # Compute DINO loss
        loss = self._compute_loss(student_outputs, teacher_outputs)

        self.log("test_loss", loss, prog_bar=True)

        # Store embeddings for clustering evaluation
        self.test_embeddings.append(student_out1.detach().cpu())
        self.test_labels.append(labels.cpu())

        return loss

    def on_test_epoch_end(self):
        if len(self.test_embeddings) > 0:
            embeddings = torch.cat(self.test_embeddings, dim=0).numpy()
            labels = torch.cat(self.test_labels, dim=0).numpy()

            ari_score, nmi_score = self._evaluate_clustering(embeddings, labels)

            self.log("test_ari", ari_score, prog_bar=True)
            self.log("test_nmi", nmi_score, prog_bar=True)
            self.test_ari_scores.append(ari_score)
            self.test_nmi_scores.append(nmi_score)

            self.test_embeddings.clear()
            self.test_labels.clear()

    def _evaluate_clustering(self, embeddings, true_labels):
        """
        Evaluate clustering quality using K-means clustering.

        Args:
            embeddings: numpy array of shape (n_samples, embedding_dim)
            true_labels: numpy array of shape (n_samples,) with ground truth labels

        Returns:
            tuple: (adjusted_rand_score, normalized_mutual_info_score)
        """
        try:
            kmeans = KMeans(n_clusters=self.num_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)

            ari_score = adjusted_rand_score(true_labels, cluster_labels)
            nmi_score = normalized_mutual_info_score(true_labels, cluster_labels)

            return ari_score, nmi_score

        except Exception as e:
            print(f"Clustering evaluation failed: {e}")
            return 0.0, 0.0

    def configure_optimizers(self):
        # Only optimize student network parameters
        student_params = list(self.student_backbone.parameters()) + \
                         list(self.student_projection.parameters())

        optim = torch.optim.AdamW(
            student_params, lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.max_epochs)
        return [optim], [scheduler]

"""
---------------------------------------------------------------------------------------------------------------------
SimCLR
---------------------------------------------------------------------------------------------------------------------
"""

class SimCLRModel(pl.LightningModule):
    def __init__(self, lr: float = 6e-2, weight_decay: float = 5e-4, max_epochs: int = 100, num_clusters: int = 4, backbone_type: Literal["pretrained_resnet18", "random_resnet18", "simple_mlp"] = "pretrained_resnet18",
                 input_dim: int = 150528):
        super().__init__()  # Fixed: removed 'self.' prefix
        backbone, hidden_dim = create_backbone(backbone_type, input_dim)
        self.backbone = backbone
        self.projection_head = SimCLRProjectionHead(
            hidden_dim, hidden_dim, 128
        )
        self.criterion = NTXentLoss()
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.num_clusters = num_clusters
        self.save_hyperparameters()  # Fixed: removed ignore parameter since no 'model' attribute

        # Clustering metrics for representation quality evaluation
        # We'll store embeddings and labels for clustering evaluation
        self.train_embeddings = []
        self.train_labels = []
        self.val_embeddings = []
        self.val_labels = []
        self.test_embeddings = []
        self.test_labels = []

        # Lists to store clustering scores over epochs
        self.train_ari_scores = []  # Adjusted Rand Index
        self.train_nmi_scores = []  # Normalized Mutual Information
        self.val_ari_scores = []
        self.val_nmi_scores = []
        self.test_ari_scores = []
        self.test_nmi_scores = []

    def forward(self, x):
        h = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(h)
        return z

    def training_step(self, batch, batch_idx):  # Fixed: added missing colon
        (x1, x2), labels = batch
        z1, z2 = self.forward(x1), self.forward(x2)
        loss = self.criterion(z1, z2)
        self.log("train_loss", loss, prog_bar=True)

        # Store embeddings and labels for clustering evaluation
        # Use z1 (first augmentation) for clustering evaluation
        self.train_embeddings.append(z1.detach().cpu())
        self.train_labels.append(labels.cpu())

        return loss

    def on_train_epoch_end(self):
        if len(self.train_embeddings) > 0:
            # Concatenate all embeddings and labels from this epoch
            embeddings = torch.cat(self.train_embeddings, dim=0).numpy()
            labels = torch.cat(self.train_labels, dim=0).numpy()

            # Perform clustering and evaluate
            ari_score, nmi_score = self._evaluate_clustering(embeddings, labels)

            # Log and store scores
            self.log("train_ari", ari_score, prog_bar=True)
            self.log("train_nmi", nmi_score, prog_bar=True)
            self.train_ari_scores.append(ari_score)
            self.train_nmi_scores.append(nmi_score)

            # Clear embeddings for next epoch
            self.train_embeddings.clear()
            self.train_labels.clear()

    def validation_step(self, batch, batch_idx):
        (x1, x2), labels = batch
        z1, z2 = self.forward(x1), self.forward(x2)
        loss = self.criterion(z1, z2)
        self.log("val_loss", loss, prog_bar=True)

        # Store embeddings and labels for clustering evaluation
        self.val_embeddings.append(z1.detach().cpu())
        self.val_labels.append(labels.cpu())

        return loss

    def on_validation_epoch_end(self):
        if len(self.val_embeddings) > 0:
            # Concatenate all embeddings and labels from this epoch
            embeddings = torch.cat(self.val_embeddings, dim=0).numpy()
            labels = torch.cat(self.val_labels, dim=0).numpy()

            # Perform clustering and evaluate
            ari_score, nmi_score = self._evaluate_clustering(embeddings, labels)

            # Log and store scores
            self.log("val_ari", ari_score, prog_bar=True)
            self.log("val_nmi", nmi_score, prog_bar=True)
            self.val_ari_scores.append(ari_score)
            self.val_nmi_scores.append(nmi_score)

            # Clear embeddings for next epoch
            self.val_embeddings.clear()
            self.val_labels.clear()

    def test_step(self, batch, batch_idx):
        (x1, x2), labels = batch
        z1, z2 = self.forward(x1), self.forward(x2)
        loss = self.criterion(z1, z2)
        self.log("test_loss", loss, prog_bar=True)

        # Store embeddings and labels for clustering evaluation
        self.test_embeddings.append(z1.detach().cpu())
        self.test_labels.append(labels.cpu())

        return loss

    def on_test_epoch_end(self):
        if len(self.test_embeddings) > 0:
            # Concatenate all embeddings and labels from this epoch
            embeddings = torch.cat(self.test_embeddings, dim=0).numpy()
            labels = torch.cat(self.test_labels, dim=0).numpy()

            # Perform clustering and evaluate
            ari_score, nmi_score = self._evaluate_clustering(embeddings, labels)

            # Log and store scores
            self.log("test_ari", ari_score, prog_bar=True)
            self.log("test_nmi", nmi_score, prog_bar=True)
            self.test_ari_scores.append(ari_score)
            self.test_nmi_scores.append(nmi_score)

            # Clear embeddings for next epoch
            self.test_embeddings.clear()
            self.test_labels.clear()

    def _evaluate_clustering(self, embeddings, true_labels):
        """
        Evaluate clustering quality using K-means clustering.

        Args:
            embeddings: numpy array of shape (n_samples, embedding_dim)
            true_labels: numpy array of shape (n_samples,) with ground truth labels

        Returns:
            tuple: (adjusted_rand_score, normalized_mutual_info_score)
        """
        try:
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=self.num_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)

            # Calculate clustering metrics
            ari_score = adjusted_rand_score(true_labels, cluster_labels)
            nmi_score = normalized_mutual_info_score(true_labels, cluster_labels)

            return ari_score, nmi_score

        except Exception as e:
            # Return 0 scores if clustering fails
            print(f"Clustering evaluation failed: {e}")
            return 0.0, 0.0

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.max_epochs)
        return [optim], [scheduler]

"""
---------------------------------------------------------------------------------------------------------------------
MoCO
---------------------------------------------------------------------------------------------------------------------
"""

class MoCoProjectionHead(nn.Module):
    """Projection head for MoCo"""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return F.normalize(self.projection(x), dim=1)


class MoCoModel(pl.LightningModule):
    def __init__(self, lr: float = 6e-2, weight_decay: float = 5e-4, max_epochs: int = 100,
                 num_clusters: int = 4, backbone_type: Literal["pretrained_resnet18", "random_resnet18", "simple_mlp"] = "pretrained_resnet18",
                 input_dim: int = 150528, queue_size: int = 65536, momentum: float = 0.999, temperature: float = 0.07):
        super().__init__()

        # Create encoder networks
        backbone, hidden_dim = create_backbone(backbone_type, input_dim)
        self.encoder_q = backbone  # Query encoder
        self.encoder_k = copy.deepcopy(backbone)  # Key encoder

        # Projection heads
        self.projection_q = MoCoProjectionHead(hidden_dim, hidden_dim, 128)
        self.projection_k = copy.deepcopy(self.projection_q)

        # Initialize key encoder parameters
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        for param_q, param_k in zip(self.projection_q.parameters(), self.projection_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # MoCo parameters
        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature

        # Initialize queue
        self.register_buffer("queue", torch.randn(128, queue_size))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # Training parameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.num_clusters = num_clusters
        self.save_hyperparameters()

        # Clustering evaluation storage
        self.train_embeddings = []
        self.train_labels = []
        self.val_embeddings = []
        self.val_labels = []
        self.test_embeddings = []
        self.test_labels = []

        # Clustering scores
        self.train_ari_scores = []
        self.train_nmi_scores = []
        self.val_ari_scores = []
        self.val_nmi_scores = []
        self.test_ari_scores = []
        self.test_nmi_scores = []

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder"""
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)
        for param_q, param_k in zip(self.projection_q.parameters(), self.projection_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """Update the queue with new keys"""
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        # Replace the keys at ptr (dequeue and enqueue)
        if ptr + batch_size <= self.queue_size:
            self.queue[:, ptr:ptr + batch_size] = keys.T
        else:
            # Handle wrap around
            remaining = self.queue_size - ptr
            self.queue[:, ptr:] = keys[:remaining].T
            self.queue[:, :batch_size - remaining] = keys[remaining:].T

        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

    def forward(self, x):
        """Forward pass for inference (using query encoder)"""
        h = self.encoder_q(x).flatten(start_dim=1)
        z = self.projection_q(h)
        return z

    def training_step(self, batch, batch_idx):
        (x_q, x_k), labels = batch

        # Compute query features
        q = self.forward(x_q)  # queries: NxC

        # Compute key features
        with torch.no_grad():
            self._momentum_update_key_encoder()
            h_k = self.encoder_k(x_k).flatten(start_dim=1)
            k = self.projection_k(h_k)  # keys: NxC

        # Compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.temperature

        # labels: positive key indicators
        labels_contrastive = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # Compute loss
        loss = F.cross_entropy(logits, labels_contrastive)

        # Update queue
        self._dequeue_and_enqueue(k)

        self.log("train_loss", loss, prog_bar=True)

        # Store embeddings for clustering evaluation
        self.train_embeddings.append(q.detach().cpu())
        self.train_labels.append(labels.cpu())

        return loss

    def on_train_epoch_end(self):
        if len(self.train_embeddings) > 0:
            embeddings = torch.cat(self.train_embeddings, dim=0).numpy()
            labels = torch.cat(self.train_labels, dim=0).numpy()

            ari_score, nmi_score = self._evaluate_clustering(embeddings, labels)

            self.log("train_ari", ari_score, prog_bar=True)
            self.log("train_nmi", nmi_score, prog_bar=True)
            self.train_ari_scores.append(ari_score)
            self.train_nmi_scores.append(nmi_score)

            self.train_embeddings.clear()
            self.train_labels.clear()

    def validation_step(self, batch, batch_idx):
        (x_q, x_k), labels = batch

        # Use query encoder for validation
        q = self.forward(x_q)

        # For validation, we can compute a simplified contrastive loss
        # or just use the query representation
        with torch.no_grad():
            h_k = self.encoder_k(x_k).flatten(start_dim=1)
            k = self.projection_k(h_k)

        # Simple similarity-based loss for validation
        loss = 1 - F.cosine_similarity(q, k).mean()

        self.log("val_loss", loss, prog_bar=True)

        self.val_embeddings.append(q.detach().cpu())
        self.val_labels.append(labels.cpu())

        return loss

    def on_validation_epoch_end(self):
        if len(self.val_embeddings) > 0:
            embeddings = torch.cat(self.val_embeddings, dim=0).numpy()
            labels = torch.cat(self.val_labels, dim=0).numpy()

            ari_score, nmi_score = self._evaluate_clustering(embeddings, labels)

            self.log("val_ari", ari_score, prog_bar=True)
            self.log("val_nmi", nmi_score, prog_bar=True)
            self.val_ari_scores.append(ari_score)
            self.val_nmi_scores.append(nmi_score)

            self.val_embeddings.clear()
            self.val_labels.clear()

    def test_step(self, batch, batch_idx):
        (x_q, x_k), labels = batch
        q = self.forward(x_q)

        with torch.no_grad():
            h_k = self.encoder_k(x_k).flatten(start_dim=1)
            k = self.projection_k(h_k)

        loss = 1 - F.cosine_similarity(q, k).mean()

        self.log("test_loss", loss, prog_bar=True)

        self.test_embeddings.append(q.detach().cpu())
        self.test_labels.append(labels.cpu())

        return loss

    def on_test_epoch_end(self):
        if len(self.test_embeddings) > 0:
            embeddings = torch.cat(self.test_embeddings, dim=0).numpy()
            labels = torch.cat(self.test_labels, dim=0).numpy()

            ari_score, nmi_score = self._evaluate_clustering(embeddings, labels)

            self.log("test_ari", ari_score, prog_bar=True)
            self.log("test_nmi", nmi_score, prog_bar=True)
            self.test_ari_scores.append(ari_score)
            self.test_nmi_scores.append(nmi_score)

            self.test_embeddings.clear()
            self.test_labels.clear()

    def _evaluate_clustering(self, embeddings, true_labels):
        """Evaluate clustering quality using K-means clustering"""
        try:
            kmeans = KMeans(n_clusters=self.num_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)

            ari_score = adjusted_rand_score(true_labels, cluster_labels)
            nmi_score = normalized_mutual_info_score(true_labels, cluster_labels)

            return ari_score, nmi_score
        except Exception as e:
            print(f"Clustering evaluation failed: {e}")
            return 0.0, 0.0

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.max_epochs)
        return [optim], [scheduler]

"""
---------------------------------------------------------------------------------------------------------------------
MAE
---------------------------------------------------------------------------------------------------------------------
"""

class MAEEncoder(nn.Module):
    """Simple encoder for MAE"""
    def __init__(self, backbone, embed_dim: int):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = embed_dim

    def forward(self, x, mask_ratio=0.75):
        # For simplicity, we'll apply masking in the feature space
        h = self.backbone(x).flatten(start_dim=1)

        if self.training and mask_ratio > 0:
            # Random masking in feature space
            mask = torch.rand(h.shape, device=h.device) > mask_ratio
            h_masked = h * mask.float()
        else:
            h_masked = h

        return h_masked, h


class MAEDecoder(nn.Module):
    """Simple decoder for MAE"""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.decoder(x)


class MAEModel(pl.LightningModule):
    def __init__(self, lr: float = 6e-2, weight_decay: float = 5e-4, max_epochs: int = 100,
                 num_clusters: int = 4, backbone_type: Literal["pretrained_resnet18", "random_resnet18", "simple_mlp"] = "pretrained_resnet18",
                 input_dim: int = 150528, mask_ratio: float = 0.75):
        super().__init__()

        backbone, hidden_dim = create_backbone(backbone_type, input_dim)

        # MAE components
        self.encoder = MAEEncoder(backbone, hidden_dim)
        self.decoder = MAEDecoder(hidden_dim, hidden_dim, hidden_dim)

        # Parameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.num_clusters = num_clusters
        self.mask_ratio = mask_ratio
        self.save_hyperparameters()

        # Clustering evaluation storage
        self.train_embeddings = []
        self.train_labels = []
        self.val_embeddings = []
        self.val_labels = []
        self.test_embeddings = []
        self.test_labels = []

        # Clustering scores
        self.train_ari_scores = []
        self.train_nmi_scores = []
        self.val_ari_scores = []
        self.val_nmi_scores = []
        self.test_ari_scores = []
        self.test_nmi_scores = []

    def forward(self, x):
        """Forward pass for inference (no masking)"""
        h_masked, h_original = self.encoder(x, mask_ratio=0.0)
        return h_original

    def training_step(self, batch, batch_idx):
        # For MAE, we typically don't need augmented pairs, just the original images
        if isinstance(batch[0], tuple):
            # If we get augmented pairs, just use the first one
            x, labels = batch[0][0], batch[1]
        else:
            x, labels = batch

        # Encode with masking
        h_masked, h_target = self.encoder(x, mask_ratio=self.mask_ratio)

        # Decode
        h_reconstructed = self.decoder(h_masked)

        # Reconstruction loss (MSE between original and reconstructed features)
        loss = F.mse_loss(h_reconstructed, h_target)

        self.log("train_loss", loss, prog_bar=True)

        # Store original embeddings (without masking) for clustering evaluation
        with torch.no_grad():
            h_clean, _ = self.encoder(x, mask_ratio=0.0)
            self.train_embeddings.append(h_clean.detach().cpu())
            self.train_labels.append(labels.cpu())

        return loss

    def on_train_epoch_end(self):
        if len(self.train_embeddings) > 0:
            embeddings = torch.cat(self.train_embeddings, dim=0).numpy()
            labels = torch.cat(self.train_labels, dim=0).numpy()

            ari_score, nmi_score = self._evaluate_clustering(embeddings, labels)

            self.log("train_ari", ari_score, prog_bar=True)
            self.log("train_nmi", nmi_score, prog_bar=True)
            self.train_ari_scores.append(ari_score)
            self.train_nmi_scores.append(nmi_score)

            self.train_embeddings.clear()
            self.train_labels.clear()

    def validation_step(self, batch, batch_idx):
        if isinstance(batch[0], tuple):
            x, labels = batch[0][0], batch[1]
        else:
            x, labels = batch

        h_masked, h_target = self.encoder(x, mask_ratio=self.mask_ratio)
        h_reconstructed = self.decoder(h_masked)
        loss = F.mse_loss(h_reconstructed, h_target)

        self.log("val_loss", loss, prog_bar=True)

        # Store clean embeddings for clustering
        with torch.no_grad():
            h_clean, _ = self.encoder(x, mask_ratio=0.0)
            self.val_embeddings.append(h_clean.detach().cpu())
            self.val_labels.append(labels.cpu())

        return loss

    def on_validation_epoch_end(self):
        if len(self.val_embeddings) > 0:
            embeddings = torch.cat(self.val_embeddings, dim=0).numpy()
            labels = torch.cat(self.val_labels, dim=0).numpy()

            ari_score, nmi_score = self._evaluate_clustering(embeddings, labels)

            self.log("val_ari", ari_score, prog_bar=True)
            self.log("val_nmi", nmi_score, prog_bar=True)
            self.val_ari_scores.append(ari_score)
            self.val_nmi_scores.append(nmi_score)

            self.val_embeddings.clear()
            self.val_labels.clear()

    def test_step(self, batch, batch_idx):
        if isinstance(batch[0], tuple):
            x, labels = batch[0][0], batch[1]
        else:
            x, labels = batch

        h_masked, h_target = self.encoder(x, mask_ratio=self.mask_ratio)
        h_reconstructed = self.decoder(h_masked)
        loss = F.mse_loss(h_reconstructed, h_target)

        self.log("test_loss", loss, prog_bar=True)

        with torch.no_grad():
            h_clean, _ = self.encoder(x, mask_ratio=0.0)
            self.test_embeddings.append(h_clean.detach().cpu())
            self.test_labels.append(labels.cpu())

        return loss

    def on_test_epoch_end(self):
        if len(self.test_embeddings) > 0:
            embeddings = torch.cat(self.test_embeddings, dim=0).numpy()
            labels = torch.cat(self.test_labels, dim=0).numpy()

            ari_score, nmi_score = self._evaluate_clustering(embeddings, labels)

            self.log("test_ari", ari_score, prog_bar=True)
            self.log("test_nmi", nmi_score, prog_bar=True)
            self.test_ari_scores.append(ari_score)
            self.test_nmi_scores.append(nmi_score)

            self.test_embeddings.clear()
            self.test_labels.clear()

    def _evaluate_clustering(self, embeddings, true_labels):
        """Evaluate clustering quality using K-means clustering"""
        try:
            kmeans = KMeans(n_clusters=self.num_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)

            ari_score = adjusted_rand_score(true_labels, cluster_labels)
            nmi_score = normalized_mutual_info_score(true_labels, cluster_labels)

            return ari_score, nmi_score
        except Exception as e:
            print(f"Clustering evaluation failed: {e}")
            return 0.0, 0.0

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.max_epochs)
        return [optim], [scheduler]