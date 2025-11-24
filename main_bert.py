import json
import math
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel
import metrics
import utils2
from data_loader import load_test_data, load_train_data


class TimeSeriesDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            "inputs": torch.tensor(self.sequences[idx], dtype=torch.float32),
            "targets": torch.tensor(self.targets[idx], dtype=torch.float32),
        }


class TimeSeriesBERTFineTune(nn.Module):
    def __init__(
        self,
        input_dim=39,
        hidden_size=768,
        seq_len=50,
        output_dim=39,
        freeze_until_layer=12,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # projection
        self.embedding = nn.Linear(input_dim, hidden_size)

        # sinusoidal PE
        pe = self.sinusoidal_positional_embedding(seq_len + 1, hidden_size)
        self.register_buffer("pos_embedding", pe)

        # learnable CLS
        self.cls_token = nn.Parameter(torch.empty(1, 1, hidden_size))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.bert = BertModel.from_pretrained("jackaduma/SecBERT")

        # head
        self.output_layer = nn.Linear(hidden_size, output_dim)
        self.to(self.device)

    def sinusoidal_positional_embedding(self, seq_len, hidden_dim):
        position = torch.arange(0, seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2) * (-math.log(10000.0) / hidden_dim)
        )
        pe = torch.zeros(seq_len, hidden_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x):
        batch_size = x.size(0)

        # Embedding
        x = self.embedding(x)

        # add [CLS]
        cls_expanded = self.cls_token.expand(batch_size, 1, -1)
        x = torch.cat([cls_expanded, x], dim=1)

        # add positional embedding
        x = x + self.pos_embedding[:, : x.size(1)]

        # attention mask
        attention_mask = torch.ones(
            x.size(0), x.size(1), dtype=torch.long, device=self.device
        )

        # forward through bert
        outputs = self.bert(inputs_embeds=x, attention_mask=attention_mask)

        # get [CLS] hidden state
        cls_state = outputs.last_hidden_state[:, 0, :]

        # output layer
        prediction = self.output_layer(cls_state)

        return prediction

    def compute_loss(self, predictions, targets, lambda_smooth=0.1):
        mse_loss = nn.MSELoss()(predictions, targets)
        loss_smoothness = torch.mean((predictions[:, 1:] - predictions[:, :-1]) ** 2)
        total_loss = mse_loss + lambda_smooth * loss_smoothness
        return total_loss

    @torch.no_grad()
    def reconstruction_errors(self, X, history=100, device="cuda", batch_size=512):
        self.eval()
        all_errors = []

        num_samples = len(X)
        start_idx = history + 1
        end_idx = num_samples

        idx = start_idx
        while idx < end_idx:
            batch_end = min(idx + batch_size, end_idx)
            seq_list = []
            for i in range(idx, batch_end):
                seq = X[i - history : i]
                seq_list.append(seq)
            seq_batch = np.stack(seq_list, axis=0)

            seq_batch_torch = torch.tensor(
                seq_batch, dtype=torch.float32, device=device
            )
            preds = self.forward(seq_batch_torch)

            ground_truth = X[idx:batch_end]
            ground_truth_torch = torch.tensor(
                ground_truth, dtype=torch.float32, device=device
            )

            squared_error = (preds - ground_truth_torch) ** 2

            all_errors.append(squared_error.cpu().numpy())
            idx = batch_end

        all_errors = np.concatenate(all_errors, axis=0)
        return all_errors

    def reconstruction_errors_by_idxs(
        self, Xfull, idxs, history, bs=128, device="cuda"
    ):
        self.eval()
        full_errors = np.zeros((len(idxs), Xfull.shape[1]))

        for start in range(0, len(idxs), bs):
            end = min(start + bs, len(idxs))
            batch_idxs = idxs[start:end]

            Xbatch = np.array([Xfull[i - history : i] for i in batch_idxs])
            Ybatch = np.array([Xfull[i + 1] for i in batch_idxs])

            X_tensor = torch.tensor(Xbatch, dtype=torch.float32).to(device)
            Y_tensor = torch.tensor(Ybatch, dtype=torch.float32).to(device)

            preds = self.forward(X_tensor)

            squared_error = (preds - Y_tensor) ** 2

            full_errors[start:end] = squared_error.detach().cpu().numpy()

        return full_errors

    def cached_detect(self, instance_errors, theta, window=1):
        detection = instance_errors > theta

        if window > 1:
            detection = np.convolve(detection, np.ones(window), "same") // window

        return detection

    def save_detection_params(self, best_theta, best_window):
        self.best_theta = best_theta
        self.best_window = best_window
        with open("best_params.json", "w") as f:
            json.dump(
                {
                    "best_theta": best_theta,
                    "best_window": best_window,
                },
                f,
                indent=4,
            )

    def best_cached_detect(self, instance_errors):
        return self.cached_detect(
            instance_errors, theta=self.best_theta, window=self.best_window
        )


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs=10,
    lr=5e-5,
    weight_decay=1e-4,
    use_scheduler=False,
    early_stopping_patience=10,
):
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    model.to(device)

    train_losses, val_losses = [], []
    val_maes, val_r2s, val_mses = [], [], []

    best_val_loss = float("inf")
    no_improve_count = 0

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2, verbose=True
        )
        if use_scheduler
        else None
    )

    for epoch in range(epochs):
        # TRAIN
        model.train()
        total_loss = 0
        for batch in train_loader:
            inputs = batch["inputs"].to(device)
            targets = batch["targets"].to(device)

            outputs = model(inputs)
            loss = model.compute_loss(outputs, targets, lambda_smooth=0.1)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # VALIDATION
        model.eval()
        val_loss = 0
        predictions, true_targets = [], []

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["inputs"].to(device)
                targets = batch["targets"].to(device)
                outputs = model(inputs)

                loss = criterion(outputs, targets)
                val_loss += loss.item()

                predictions.append(outputs.cpu().numpy())
                true_targets.append(targets.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        preds = np.vstack(predictions)
        trues = np.vstack(true_targets)

        mse = mean_squared_error(trues, preds)
        mae = mean_absolute_error(trues, preds)
        r2 = r2_score(trues, preds)
        val_mses.append(mse)
        val_maes.append(mae)
        val_r2s.append(r2)

        if scheduler:
            scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= early_stopping_patience:
                break

    return train_losses, val_losses, val_mses, val_maes, val_r2s


def hyperparameter_search(
    event_detector,
    config,
    Xfull,
    Xtest,
    Ytest,
    dataset_name,
    val_idxs=None,
    test_split=0.7,
):
    history = 100

    Ytest = Ytest.astype(int)
    Xtest_val, Xtest_test, Ytest_val, Ytest_test = utils2.custom_train_test_split(
        dataset_name, Xtest, Ytest, test_size=test_split, shuffle=False
    )

    # Clip the prediction to match prediction window
    Ytest_test = Ytest_test[history + 1 :]
    Ytest_val = Ytest_val[history + 1 :]

    validation_errors = event_detector.reconstruction_errors_by_idxs(
        Xfull, val_idxs, history=history
    )

    test_errors = event_detector.reconstruction_errors(Xtest_val)
    test_instance_errors = test_errors.mean(axis=1)

    grid_config = config.get("grid_search", dict())

    cutoffs = grid_config.get("percentile", [0.95])
    windows = grid_config.get("window", [1])
    eval_metrics = grid_config.get("metrics", ["F1"])

    for metric in eval_metrics:
        negative_metric = metric == "false_positive_rate"
        if negative_metric:
            best_metric = 1
        else:
            best_metric = -1000

        best_percentile = 0
        best_window = 0
        metric_vals = np.zeros((len(cutoffs), len(windows)))
        metric_func = metrics.get("F1")

        for percentile_idx in range(len(cutoffs)):
            percentile = cutoffs[percentile_idx]
            theta = np.quantile(validation_errors.mean(axis=1), percentile)

            for window_idx in range(len(windows)):
                window = windows[window_idx]

                Yhat = event_detector.cached_detect(
                    test_instance_errors, theta=theta, window=window
                )

                Yhat, Ytest_val = utils2.normalize_array_length(Yhat, Ytest_val)
                choice_value = f1_score(Ytest_val, Yhat, average="micro")

                if negative_metric:
                    if choice_value < best_metric:
                        best_metric = choice_value
                        best_percentile = percentile
                        best_window = window
                else:
                    if choice_value > best_metric:
                        best_metric = choice_value
                        best_percentile = percentile
                        best_window = window

                if grid_config.get("save-metric-info", False):
                    metric_vals[percentile_idx, window_idx] = choice_value

        # Final test performance
        final_test_errors = event_detector.reconstruction_errors(Xtest_test)
        final_test_instance_errors = final_test_errors.mean(axis=1)

        best_theta = np.quantile(validation_errors.mean(axis=1), best_percentile)
        event_detector.save_detection_params(
            best_theta=best_theta, best_window=best_window
        )

        final_Yhat = event_detector.best_cached_detect(final_test_instance_errors)
        final_Yhat = final_Yhat[best_window - 1 :].astype(int)

        metric_func = metrics.get(metric)
        final_Yhat, Ytest_test = utils2.normalize_array_length(final_Yhat, Ytest_test)
        final_value = f1_score(Ytest_test, final_Yhat, average="micro")

    return event_detector


if __name__ == "__main__":
    dataset_name = "BATADAL"
    history = 100
    input_dim = 39
    output_dim = 39

    config = {
        "grid_search": {
            "percentile": [
                0.90,
                0.91,
                0.92,
                0.93,
                0.94,
                0.95,
                0.96,
                0.97,
                0.98,
                0.99,
                0.991,
                0.992,
                0.993,
                0.994,
                0.995,
                0.996,
                0.997,
                0.998,
                0.999,
                0.9995,
                0.99995,
            ],
            "window": [1, 3, 4, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "metrics": ["F1"],
            "save-metric-info": True,
            "save-theta": True,
        }
    }

    Xfull, _ = load_train_data(dataset_name)
    Xtest, Ytest, _ = load_test_data(dataset_name)

    train_idxs, val_idxs = utils2.train_val_history_idx_split(Xfull, history)

    train_seq = [Xfull[i - history : i] for i in train_idxs]
    train_tgt = [Xfull[i + 1] for i in train_idxs]

    val_seq = [Xfull[i - history : i] for i in val_idxs]
    val_tgt = [Xfull[i + 1] for i in val_idxs]

    train_ds = TimeSeriesDataset(train_seq, train_tgt)
    val_ds = TimeSeriesDataset(val_seq, val_tgt)
    train_loader = DataLoader(
        train_ds,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        drop_last=False,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        drop_last=False,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimeSeriesBERTFineTune(
        input_dim=input_dim,
        output_dim=output_dim,
        seq_len=history,
        freeze_until_layer=12,
    )

    # Load pretrained if available
    # model.load_state_dict(torch.load("eval_results/bert_sec_finetune.pt"))
    model.to(device)

    # Train
    train_losses, val_losses, val_mses, val_maes, val_r2s = train_model(
        model,
        train_loader,
        val_loader,
        device,
        epochs=30,
        lr=5e-5,
        weight_decay=1e-4,
        use_scheduler=True,
        early_stopping_patience=5,
    )

    # Save model
    torch.save(model.state_dict(), "trained_model.pt")

    # Hyperparameter tuning
    hyperparameter_search(
        model,
        config,
        Xfull,
        Xtest,
        Ytest,
        dataset_name,
        val_idxs=val_idxs,
    )