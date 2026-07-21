from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import CONFIG


class MLPClassifierNet(nn.Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class LSTMClassifierNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 48) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.20),
            nn.Linear(hidden_size, 24),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(24, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return self.head(output[:, -1, :]).squeeze(-1)


@dataclass
class TrainingHistory:
    train_loss: list[float]
    validation_loss: list[float]
    best_epoch: int


def _set_seed(seed: int) -> None:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_binary_model(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    max_epochs: int | None = None,
    patience: int | None = None,
    batch_size: int | None = None,
    learning_rate: float = 1e-3,
) -> tuple[nn.Module, TrainingHistory]:
    _set_seed(CONFIG.random_state)
    max_epochs = max_epochs or CONFIG.max_epochs
    patience = patience or CONFIG.patience
    batch_size = batch_size or CONFIG.batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    train_loader = DataLoader(
        TensorDataset(x_train_tensor, y_train_tensor),
        batch_size=batch_size,
        shuffle=True,
    )

    x_validation_tensor = torch.tensor(x_validation, dtype=torch.float32, device=device)
    y_validation_tensor = torch.tensor(y_validation, dtype=torch.float32, device=device)

    positive_count = max(float(y_train.sum()), 1.0)
    negative_count = max(float(len(y_train) - y_train.sum()), 1.0)
    pos_weight = torch.tensor(negative_count / positive_count, dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    train_losses: list[float] = []
    validation_losses: list[float] = []

    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        seen = 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += float(loss.item()) * len(x_batch)
            seen += len(x_batch)

        train_loss = running_loss / max(seen, 1)
        model.eval()
        with torch.no_grad():
            validation_logits = model(x_validation_tensor)
            validation_loss = float(criterion(validation_logits, y_validation_tensor).item())

        train_losses.append(train_loss)
        validation_losses.append(validation_loss)

        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is None:
        raise RuntimeError("Neural-network training failed to produce a model state.")

    model.load_state_dict(best_state)
    model = model.to("cpu")
    history = TrainingHistory(train_losses, validation_losses, best_epoch)
    return model, history


def predict_probabilities(model: nn.Module, x: np.ndarray, batch_size: int = 512) -> np.ndarray:
    model.eval()
    data_loader = DataLoader(
        TensorDataset(torch.tensor(x, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=False,
    )
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for (x_batch,) in data_loader:
            logits = model(x_batch)
            probabilities = torch.sigmoid(logits).cpu().numpy()
            outputs.append(probabilities)
    return np.concatenate(outputs)


def create_sequences(
    features: np.ndarray,
    targets: np.ndarray,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sequences: list[np.ndarray] = []
    labels: list[float] = []
    target_indices: list[int] = []

    for target_index in range(sequence_length - 1, len(features)):
        start = target_index - sequence_length + 1
        sequences.append(features[start : target_index + 1])
        labels.append(targets[target_index])
        target_indices.append(target_index)

    return (
        np.asarray(sequences, dtype=np.float32),
        np.asarray(labels, dtype=np.float32),
        np.asarray(target_indices, dtype=int),
    )


def train_fixed_epochs(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    epochs: int,
    batch_size: int | None = None,
    learning_rate: float = 1e-3,
) -> nn.Module:
    """Refit a selected neural architecture on all labeled data for deployment."""
    _set_seed(CONFIG.random_state)
    batch_size = batch_size or CONFIG.batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    x_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(x_tensor, y_tensor),
        batch_size=batch_size,
        shuffle=True,
    )

    positive_count = max(float(y_train.sum()), 1.0)
    negative_count = max(float(len(y_train) - y_train.sum()), 1.0)
    pos_weight = torch.tensor(negative_count / positive_count, dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    for _ in range(max(int(epochs), 1)):
        model.train()
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    return model.to("cpu")
