import argparse
import csv
import json
import math
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import matthews_corrcoef, roc_auc_score, roc_curve
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from v0_data import H5_LABELS, MODIFICATION_NAMES, build_chemical_feature_matrix, load_modification_table


DEFAULT_THRESHOLDS = [
    0.002887,
    0.004897,
    0.001442,
    0.010347,
    0.036834,
    0.028677,
    0.009135,
    0.095019,
    0.001394,
    0.007883,
    0.113931,
    0.125591,
]


class BahdanauAttention(nn.Module):
    def __init__(self, in_features, hidden_units, num_task):
        super().__init__()
        self.W1 = nn.Linear(in_features=in_features, out_features=hidden_units)
        self.W2 = nn.Linear(in_features=in_features, out_features=hidden_units)
        self.V = nn.Linear(in_features=hidden_units, out_features=num_task)

    def forward(self, hidden_states, values):
        hidden_with_time_axis = torch.unsqueeze(hidden_states, dim=1)
        score = self.V(torch.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))
        attention_weights = torch.softmax(score, dim=1)
        values = torch.transpose(values, 1, 2)
        context_vector = torch.matmul(values, attention_weights)
        context_vector = torch.transpose(context_vector, 1, 2)
        return context_vector, attention_weights


class FrozenEmbeddingSeq(nn.Module):
    def __init__(self, embedding_path):
        super().__init__()
        weight_dict = pickle.load(open(embedding_path, "rb"))
        weights = torch.as_tensor(np.stack(list(weight_dict.values())), dtype=torch.float32)
        self.embedding = nn.Embedding(num_embeddings=weights.shape[0], embedding_dim=weights.shape[1])
        self.embedding.weight = nn.Parameter(weights, requires_grad=False)

    def forward(self, x):
        return self.embedding(x.long())


class PaperMultiRM(nn.Module):
    def __init__(self, embedding_path, num_task=12):
        super().__init__()
        self.num_task = num_task
        self.embed = FrozenEmbeddingSeq(embedding_path)
        self.NaiveBiLSTM = nn.LSTM(input_size=300, hidden_size=256, batch_first=True, bidirectional=True)
        self.Attention = BahdanauAttention(in_features=512, hidden_units=100, num_task=num_task)
        for index in range(num_task):
            setattr(
                self,
                f"NaiveFC{index}",
                nn.Sequential(
                    nn.Linear(in_features=512, out_features=128),
                    nn.ReLU(),
                    nn.Dropout(),
                    nn.Linear(in_features=128, out_features=1),
                    nn.Sigmoid(),
                ),
            )

    def forward(self, x):
        context_vector = self.encode_context(x)
        return self.score_context(context_vector)

    def encode_context(self, x):
        x = self.embed(x)
        batch_size = x.size()[0]
        output, (h_n, _) = self.NaiveBiLSTM(x)
        h_n = h_n.view(batch_size, output.size()[-1])
        context_vector, _ = self.Attention(h_n, output)
        return context_vector

    def score_context(self, context_vector):
        outs = []
        for index in range(self.num_task):
            fc_layer = getattr(self, f"NaiveFC{index}")
            y = fc_layer(context_vector[:, index, :])
            outs.append(torch.squeeze(y, dim=-1))
        return outs


class ChemicalQueryAttention(nn.Module):
    def __init__(self, hidden_size=512, chemical_size=512, attention_size=100):
        super().__init__()
        self.W1 = nn.Linear(hidden_size, attention_size)
        self.W2 = nn.Linear(chemical_size, attention_size)
        self.V = nn.Linear(attention_size, 1)

    def forward(self, chemical_states, values):
        value_term = self.W1(values)[:, None, :, :]
        chemical_term = self.W2(chemical_states)[None, :, None, :]
        scores = self.V(torch.tanh(value_term + chemical_term)).squeeze(-1)
        attention_weights = torch.softmax(scores, dim=-1)
        context_vector = torch.einsum("bml,bld->bmd", attention_weights, values)
        return context_vector, attention_weights


class ChemicalInteractionScorer(nn.Module):
    def __init__(self, context_size=512, chemical_size=512, projection_size=256, hidden_size=512, zero_init_output=False):
        super().__init__()
        self.context_projection = nn.Linear(context_size, projection_size)
        self.chemical_context_projection = nn.Linear(context_size, projection_size)
        self.chemical_projection = nn.Linear(chemical_size, projection_size)
        self.film_generator = nn.Linear(chemical_size, context_size * 2)
        input_size = context_size + context_size + chemical_size + projection_size * 3
        self.delta_mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(),
            nn.Linear(hidden_size, 128),
            nn.GELU(),
            nn.Dropout(),
            nn.Linear(128, 1),
        )
        self.hyper_weight = nn.Linear(chemical_size, context_size)
        self.hyper_bias = nn.Linear(chemical_size, 1)
        if zero_init_output:
            nn.init.zeros_(self.delta_mlp[-1].weight)
            nn.init.zeros_(self.delta_mlp[-1].bias)
            nn.init.zeros_(self.hyper_weight.weight)
            nn.init.zeros_(self.hyper_weight.bias)
            nn.init.zeros_(self.hyper_bias.weight)
            nn.init.zeros_(self.hyper_bias.bias)

    def forward(self, sequence_context, chemical_context, chemical_states):
        batch_size, num_task, context_size = sequence_context.shape
        expanded_chemical = chemical_states[None, :, :].expand(batch_size, num_task, chemical_states.shape[-1])
        film = self.film_generator(expanded_chemical).reshape(batch_size, num_task, 2, context_size)
        gamma = film[:, :, 0, :]
        beta = film[:, :, 1, :]
        modulated_context = sequence_context * (1.0 + gamma) + beta

        projected_context = self.context_projection(modulated_context)
        projected_chemical_context = self.chemical_context_projection(chemical_context)
        projected_chemical = self.chemical_projection(expanded_chemical)
        context_chemical_interaction = projected_context * projected_chemical
        attention_chemical_interaction = projected_chemical_context * projected_chemical
        context_attention_interaction = projected_context * projected_chemical_context

        delta_input = torch.cat(
            [
                modulated_context,
                chemical_context,
                expanded_chemical,
                context_chemical_interaction,
                attention_chemical_interaction,
                context_attention_interaction,
            ],
            dim=-1,
        )
        shared_delta = torch.squeeze(self.delta_mlp(delta_input), dim=-1)
        generated_weight = self.hyper_weight(chemical_states)[None, :, :]
        generated_bias = torch.squeeze(self.hyper_bias(chemical_states), dim=-1)[None, :]
        hyper_delta = (modulated_context * generated_weight).sum(dim=-1) / math.sqrt(context_size) + generated_bias
        return shared_delta + hyper_delta


class ChemicalMultiRM(nn.Module):
    def __init__(self, embedding_path, chemical_features, num_task=12, chemical_hidden_size=512):
        super().__init__()
        self.num_task = num_task
        self.embed = FrozenEmbeddingSeq(embedding_path)
        self.NaiveBiLSTM = nn.LSTM(input_size=300, hidden_size=256, batch_first=True, bidirectional=True)
        self.register_buffer("chemical_features", torch.as_tensor(chemical_features, dtype=torch.float32))
        self.chemical_encoder = nn.Sequential(
            nn.Linear(chemical_features.shape[1], chemical_hidden_size),
            nn.LayerNorm(chemical_hidden_size),
            nn.GELU(),
            nn.Dropout(),
            nn.Linear(chemical_hidden_size, 512),
            nn.LayerNorm(512),
            nn.GELU(),
        )
        self.Attention = ChemicalQueryAttention(hidden_size=512, chemical_size=512, attention_size=100)
        self.Scorer = ChemicalInteractionScorer(zero_init_output=False)

    def forward(self, x):
        x = self.embed(x)
        output, _ = self.NaiveBiLSTM(x)
        chemical_states = self.chemical_encoder(self.chemical_features)
        chemical_context, _ = self.Attention(chemical_states, output)
        center_index = output.shape[1] // 2
        center_context = output[:, center_index, :][:, None, :].expand(output.shape[0], self.num_task, output.shape[-1])
        logits = self.Scorer(center_context, chemical_context, chemical_states)
        predictions = torch.sigmoid(logits)
        return [predictions[:, index] for index in range(self.num_task)]


class RmDataset(Dataset):
    def __init__(self, split_data):
        self.x = torch.from_numpy(split_data["x"].copy())
        self.y = torch.from_numpy(split_data["y"].copy())

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def parse_args():
    parser = argparse.ArgumentParser(description="Paper-aligned MultiRM commands")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_original = subparsers.add_parser("train_original")
    add_common_args(train_original)
    add_training_args(train_original, "Results/paper_aligned/original_from_scratch")

    original_lomo = subparsers.add_parser("train_original_lomo")
    add_common_args(original_lomo)
    add_training_args(original_lomo, "Results/paper_aligned/original_lomo")
    original_lomo.add_argument("--heldout_mod", choices=MODIFICATION_NAMES, required=True)

    train = subparsers.add_parser("train_chemical")
    add_common_args(train)
    add_training_args(train, "Results/paper_aligned/chemical")
    train.add_argument("--modifications_path", default="Data/modifications.csv")

    lomo = subparsers.add_parser("train_chemical_lomo")
    add_common_args(lomo)
    add_training_args(lomo, "Results/paper_aligned/chemical_lomo")
    lomo.add_argument("--modifications_path", default="Data/modifications.csv")
    lomo.add_argument("--heldout_mod", choices=MODIFICATION_NAMES, required=True)
    return parser.parse_args()


def add_training_args(parser, default_save_dir):
    parser.add_argument("--save_dir", default=default_save_dir)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_decay", type=float, default=0.8)
    parser.add_argument("--lr_patience", type=int, default=5)
    parser.add_argument(
        "--loss_strategy",
        choices=["weighted_bce", "weighted_bce_ohem", "uncertainty", "paper_ohem_uw"],
        default="paper_ohem_uw",
    )


def add_common_args(parser):
    parser.add_argument("--data_path", default="Data/MultiRM_data.h5")
    parser.add_argument("--embedding_path", default="Embeddings/embeddings_12RM.pkl")
    parser.add_argument("--cache_dir", default="Results/paper_aligned/cache")
    parser.add_argument("--length", type=int, default=51)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_kmer_index(embedding_path):
    embedding_dict = pickle.load(open(embedding_path, "rb"))
    return {kmer: index for index, kmer in enumerate(embedding_dict.keys())}


def read_split_as_kmers(data_path, split_name, length, embedding_path, cache_dir):
    cache_path = Path(cache_dir) / f"{split_name}_{length}bp_3mer.npz"
    if cache_path.exists():
        loaded = np.load(cache_path)
        return {"x": loaded["x"], "y": loaded["y"]}

    kmer_index = load_kmer_index(embedding_path)
    input_frame = pd.read_hdf(data_path, f"{split_name}_in_nucleo")
    output_frame = pd.read_hdf(data_path, f"{split_name}_out")
    assert list(output_frame.columns) == H5_LABELS
    assert input_frame.shape[1] == 1001
    assert length % 2 == 1
    radius = length // 2
    start = 500 - radius
    end = 500 + radius + 1
    sequence_array = input_frame.iloc[:, start:end].to_numpy(dtype=str)
    assert sequence_array.shape[1] == length

    x = np.zeros((sequence_array.shape[0], length - 2), dtype=np.int64)
    for row_index in range(sequence_array.shape[0]):
        sequence = "".join(sequence_array[row_index].tolist()).replace("U", "T")
        for kmer_start in range(length - 2):
            kmer = sequence[kmer_start:kmer_start + 3]
            assert kmer in kmer_index
            x[row_index, kmer_start] = kmer_index[kmer]
    y = output_frame.to_numpy(dtype=np.float32)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, x=x, y=y)
    return {"x": x, "y": y}


def calculate_loss_weights(labels, beta=0.99999):
    positive_counts = labels.sum(axis=0)
    assert np.all(positive_counts > 0.0)
    effective_num = 1.0 - np.power(beta, positive_counts)
    weights = (1.0 - beta) / effective_num
    weights = weights / weights.sum() * labels.shape[1]
    return torch.as_tensor(weights, dtype=torch.float32)


class TaskUncertaintyWeights(nn.Module):
    def __init__(self, num_task):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros((num_task)))


def paper_training_loss(outputs, labels, loss_weights, ohem, uncertainty_weights):
    predictions = torch.stack(outputs, dim=1)
    predictions = torch.clamp(predictions, min=1e-7, max=1.0 - 1e-7)
    bce = -(labels * torch.log(predictions) + (1.0 - labels) * torch.log(1.0 - predictions))
    if uncertainty_weights is None:
        weighted = bce * loss_weights[None, :]
        regularizer = 0.0
    else:
        weighted = bce * torch.exp(-uncertainty_weights.log_vars)[None, :]
        regularizer = uncertainty_weights.log_vars.sum()
    per_sample_loss = weighted.sum(dim=1)
    if ohem:
        keep_count = int(0.7 * labels.shape[0])
        values, _ = torch.topk(per_sample_loss, keep_count)
        return values.sum() + regularizer
    return per_sample_loss.sum() + regularizer


def paper_training_loss_for_indices(outputs, labels, task_indices, loss_weights, ohem, uncertainty_weights):
    selected_outputs = [outputs[index] for index in task_indices]
    selected_labels = labels[:, task_indices]
    return paper_training_loss(selected_outputs, selected_labels, loss_weights, ohem, uncertainty_weights)


def create_uncertainty_weights(loss_strategy, device):
    if loss_strategy in {"uncertainty", "paper_ohem_uw"}:
        return TaskUncertaintyWeights(len(MODIFICATION_NAMES)).to(device)
    assert loss_strategy in {"weighted_bce", "weighted_bce_ohem"}
    return None


def uses_ohem(loss_strategy):
    assert loss_strategy in {"weighted_bce", "weighted_bce_ohem", "uncertainty", "paper_ohem_uw"}
    return loss_strategy in {"weighted_bce_ohem", "paper_ohem_uw"}


def create_uncertainty_weights_for_task_count(loss_strategy, task_count, device):
    if loss_strategy in {"uncertainty", "paper_ohem_uw"}:
        return TaskUncertaintyWeights(task_count).to(device)
    assert loss_strategy in {"weighted_bce", "weighted_bce_ohem"}
    return None


def decay_learning_rate(optimizer, decay):
    for group in optimizer.param_groups:
        group["lr"] *= decay


def current_learning_rate(optimizer):
    return optimizer.param_groups[0]["lr"]


def collect_predictions(model, loader, device):
    model.eval()
    all_probabilities = []
    all_labels = []
    with torch.no_grad():
        for x, labels in loader:
            outputs = model(x.to(device))
            probabilities = torch.stack(outputs, dim=1).cpu().numpy()
            all_probabilities.append(probabilities)
            all_labels.append(labels.numpy())
    return np.concatenate(all_probabilities, axis=0), np.concatenate(all_labels, axis=0)


def evaluate_loss(model, loader, loss_weights, loss_strategy, uncertainty_weights, device):
    model.eval()
    ohem = uses_ohem(loss_strategy)
    total_loss = 0.0
    total_examples = 0
    with torch.no_grad():
        for x, labels in loader:
            labels = labels.to(device)
            outputs = model(x.to(device))
            loss = paper_training_loss(outputs, labels, loss_weights, ohem, uncertainty_weights)
            total_loss += float(loss.item())
            total_examples += labels.shape[0]
    return total_loss / total_examples


def select_gmean_threshold(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    gmeans = np.sqrt(tpr * (1.0 - fpr))
    best_index = int(np.argmax(gmeans))
    return float(thresholds[best_index])


def binary_metrics(labels, scores):
    threshold = select_gmean_threshold(labels, scores)
    predictions = scores >= threshold
    labels_bool = labels.astype(bool)
    tp = int((predictions & labels_bool).sum())
    tn = int(((~predictions) & (~labels_bool)).sum())
    fp = int((predictions & (~labels_bool)).sum())
    fn = int(((~predictions) & labels_bool).sum())
    return {
        "threshold": threshold,
        "sn": tp / (tp + fn),
        "sp": tn / (tn + fp),
        "acc": (tp + tn) / labels.shape[0],
        "mcc": matthews_corrcoef(labels, predictions.astype(np.float32)),
    }


def compute_table(probabilities, labels):
    rows = []
    for index, name in enumerate(MODIFICATION_NAMES):
        positive_indices = np.flatnonzero(labels[:, index] == 1)
        assert positive_indices.shape[0] > 0
        start = int(positive_indices[0])
        block_size = int(positive_indices.shape[0] * 2)
        end = start + block_size
        assert end <= labels.shape[0]
        aucb_labels = labels[start:end, index]
        aucb_scores = probabilities[start:end, index]
        assert int(aucb_labels.sum()) == positive_indices.shape[0]
        assert int((aucb_labels == 0).sum()) == positive_indices.shape[0]
        aucm_labels = labels[:, index]
        aucm_scores = probabilities[:, index]
        metrics = binary_metrics(aucb_labels, aucb_scores)
        row = {
            "Modification": name,
            "Sn": metrics["sn"],
            "Sp": metrics["sp"],
            "Acc": metrics["acc"],
            "MCC": metrics["mcc"],
            "AUCb": roc_auc_score(aucb_labels, aucb_scores),
            "AUCm": roc_auc_score(aucm_labels, aucm_scores),
            "Threshold": metrics["threshold"],
        }
        rows.append(row)

    rows.append(aggregate_row(rows, "Mean", np.mean))
    rows.append(aggregate_row(rows[:-1], "Median", np.median))
    return rows


def compute_single_modification_row(probabilities, labels, modification_index):
    name = MODIFICATION_NAMES[modification_index]
    positive_indices = np.flatnonzero(labels[:, modification_index] == 1)
    assert positive_indices.shape[0] > 0
    start = int(positive_indices[0])
    block_size = int(positive_indices.shape[0] * 2)
    end = start + block_size
    assert end <= labels.shape[0]
    aucb_labels = labels[start:end, modification_index]
    aucb_scores = probabilities[start:end, modification_index]
    assert int(aucb_labels.sum()) == positive_indices.shape[0]
    assert int((aucb_labels == 0).sum()) == positive_indices.shape[0]
    aucm_labels = labels[:, modification_index]
    aucm_scores = probabilities[:, modification_index]
    metrics = binary_metrics(aucb_labels, aucb_scores)
    return {
        "Modification": name,
        "Sn": metrics["sn"],
        "Sp": metrics["sp"],
        "Acc": metrics["acc"],
        "MCC": metrics["mcc"],
        "AUCb": roc_auc_score(aucb_labels, aucb_scores),
        "AUCm": roc_auc_score(aucm_labels, aucm_scores),
        "Threshold": metrics["threshold"],
    }


def compute_seen_modification_table(probabilities, labels, task_indices):
    rows = [compute_single_modification_row(probabilities, labels, index) for index in task_indices]
    rows.append(aggregate_row(rows, "Mean", np.mean))
    rows.append(aggregate_row(rows[:-1], "Median", np.median))
    return rows


def aggregate_row(rows, name, func):
    return {
        "Modification": name,
        "Sn": float(func([row["Sn"] for row in rows])),
        "Sp": float(func([row["Sp"] for row in rows])),
        "Acc": float(func([row["Acc"] for row in rows])),
        "MCC": float(func([row["MCC"] for row in rows])),
        "AUCb": float(func([row["AUCb"] for row in rows])),
        "AUCm": float(func([row["AUCm"] for row in rows])),
        "Threshold": "",
    }


def print_table(rows):
    headers = ["Modification", "Sn", "Sp", "Acc", "MCC", "AUCb", "AUCm"]
    formatted = []
    for row in rows:
        formatted.append({
            "Modification": row["Modification"],
            "Sn": f"{row['Sn']:.4f}",
            "Sp": f"{row['Sp']:.4f}",
            "Acc": f"{row['Acc']:.4f}",
            "MCC": f"{row['MCC']:.4f}",
            "AUCb": f"{row['AUCb']:.4f}",
            "AUCm": f"{row['AUCm']:.4f}",
        })
    widths = {header: max(len(header), max(len(row[header]) for row in formatted)) for header in headers}
    print("  ".join(header.ljust(widths[header]) for header in headers), flush=True)
    print("  ".join("-" * widths[header] for header in headers), flush=True)
    for row in formatted:
        print("  ".join(row[header].ljust(widths[header]) for header in headers), flush=True)


def save_table(rows, save_dir, split_name):
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_dir / f"{split_name}_table4.csv"
    json_path = save_dir / f"{split_name}_summary.json"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {key: rows[-2][key] for key in ["Sn", "Sp", "Acc", "MCC", "AUCb", "AUCm"]}
    with json_path.open("w") as handle:
        json.dump(summary, handle, indent=2)


def save_single_row(row, save_dir, split_name):
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_dir / f"{split_name}_table4.csv"
    json_path = save_dir / f"{split_name}_summary.json"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    summary = {key: row[key] for key in ["Sn", "Sp", "Acc", "MCC", "AUCb", "AUCm"]}
    with json_path.open("w") as handle:
        json.dump(summary, handle, indent=2)


def build_data_loaders(args):
    train_data = read_split_as_kmers(args.data_path, "train", args.length, args.embedding_path, args.cache_dir)
    valid_data = read_split_as_kmers(args.data_path, "valid", args.length, args.embedding_path, args.cache_dir)
    test_data = read_split_as_kmers(args.data_path, "test", args.length, args.embedding_path, args.cache_dir)

    train_loader = DataLoader(
        RmDataset(train_data),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    valid_loader = DataLoader(
        RmDataset(valid_data),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    test_loader = DataLoader(
        RmDataset(test_data),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    return train_data, valid_data, test_data, train_loader, valid_loader, test_loader


def remove_positive_rows(split_data, modification_index):
    keep_mask = split_data["y"][:, modification_index] < 0.5
    assert keep_mask.sum() < keep_mask.shape[0]
    return {
        "x": split_data["x"][keep_mask].copy(),
        "y": split_data["y"][keep_mask].copy(),
    }


def train_and_evaluate(args, model, train_data, valid_loader, test_loader, result_label):
    train_loader = DataLoader(
        RmDataset(train_data),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    loss_weights = calculate_loss_weights(train_data["y"]).to(args.device)
    uncertainty_weights = create_uncertainty_weights(args.loss_strategy, args.device)
    parameters = list(model.parameters())
    if uncertainty_weights is not None:
        parameters += list(uncertainty_weights.parameters())
    optimizer = Adam(parameters, lr=args.lr)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    with (save_dir / "config.json").open("w") as handle:
        json.dump(vars(args), handle, indent=2)

    best_valid_aucb = -1.0
    best_valid_loss = float("inf")
    epochs_since_valid_loss_improvement = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_examples = 0
        for x, labels in train_loader:
            labels = labels.to(args.device)
            outputs = model(x.to(args.device))
            loss = paper_training_loss(outputs, labels, loss_weights, uses_ohem(args.loss_strategy), uncertainty_weights)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += float(loss.item())
            train_examples += labels.shape[0]

        valid_loss = evaluate_loss(model, valid_loader, loss_weights, args.loss_strategy, uncertainty_weights, args.device)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            epochs_since_valid_loss_improvement = 0
        else:
            epochs_since_valid_loss_improvement += 1
            if epochs_since_valid_loss_improvement >= args.lr_patience:
                decay_learning_rate(optimizer, args.lr_decay)
                epochs_since_valid_loss_improvement = 0
        valid_probabilities, valid_labels = collect_predictions(model, valid_loader, args.device)
        valid_rows = compute_table(valid_probabilities, valid_labels)
        valid_mean_aucb = valid_rows[-2]["AUCb"]
        print(
            f"epoch={epoch} train_loss={train_loss / train_examples:.6f} "
            f"valid_loss={valid_loss:.6f} valid_mean_aucb={valid_mean_aucb:.6f} "
            f"lr={current_learning_rate(optimizer):.10f}",
            flush=True,
        )
        if valid_mean_aucb > best_valid_aucb:
            best_valid_aucb = valid_mean_aucb
            torch.save(model.state_dict(), save_dir / "best_model.pt")
            save_table(valid_rows, save_dir, "valid")

    model.load_state_dict(torch.load(save_dir / "best_model.pt", map_location=args.device))
    test_probabilities, test_labels = collect_predictions(model, test_loader, args.device)
    test_rows = compute_table(test_probabilities, test_labels)
    save_table(test_rows, save_dir, "test")
    print(result_label, flush=True)
    print_table(test_rows)


def run_train_original(args):
    set_seed(args.seed)
    train_data, _, _, _, valid_loader, test_loader = build_data_loaders(args)
    model = PaperMultiRM(args.embedding_path).to(args.device)
    train_and_evaluate(args, model, train_data, valid_loader, test_loader, "Paper-aligned MultiRM from-scratch test results")


def run_train_original_lomo(args):
    set_seed(args.seed)
    heldout_index = MODIFICATION_NAMES.index(args.heldout_mod)
    task_indices = [index for index in range(len(MODIFICATION_NAMES)) if index != heldout_index]

    train_data = read_split_as_kmers(args.data_path, "train", args.length, args.embedding_path, args.cache_dir)
    valid_data = read_split_as_kmers(args.data_path, "valid", args.length, args.embedding_path, args.cache_dir)
    test_data = read_split_as_kmers(args.data_path, "test", args.length, args.embedding_path, args.cache_dir)

    train_data = remove_positive_rows(train_data, heldout_index)
    valid_data_for_selection = remove_positive_rows(valid_data, heldout_index)

    train_loader = DataLoader(
        RmDataset(train_data),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    valid_loader = DataLoader(
        RmDataset(valid_data_for_selection),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    test_loader = DataLoader(
        RmDataset(test_data),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )

    model = PaperMultiRM(args.embedding_path).to(args.device)
    train_lomo_model(
        args,
        model,
        train_data,
        valid_loader,
        test_loader,
        task_indices,
        heldout_index,
        "Original MultiRM LOMO baseline result",
    )


def run_train_chemical(args):
    set_seed(args.seed)
    train_data, _, _, _, valid_loader, test_loader = build_data_loaders(args)
    modification_table = load_modification_table(args.modifications_path)
    chemical_features = build_chemical_feature_matrix(modification_table)
    model = ChemicalMultiRM(args.embedding_path, chemical_features).to(args.device)
    train_and_evaluate(args, model, train_data, valid_loader, test_loader, "Chemical shared-scorer all-modification test results")


def run_train_chemical_lomo(args):
    set_seed(args.seed)
    heldout_index = MODIFICATION_NAMES.index(args.heldout_mod)
    task_indices = [index for index in range(len(MODIFICATION_NAMES)) if index != heldout_index]

    train_data = read_split_as_kmers(args.data_path, "train", args.length, args.embedding_path, args.cache_dir)
    valid_data = read_split_as_kmers(args.data_path, "valid", args.length, args.embedding_path, args.cache_dir)
    test_data = read_split_as_kmers(args.data_path, "test", args.length, args.embedding_path, args.cache_dir)

    train_data = remove_positive_rows(train_data, heldout_index)
    valid_data_for_selection = remove_positive_rows(valid_data, heldout_index)

    train_loader = DataLoader(
        RmDataset(train_data),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    valid_loader = DataLoader(
        RmDataset(valid_data_for_selection),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    test_loader = DataLoader(
        RmDataset(test_data),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )

    modification_table = load_modification_table(args.modifications_path)
    chemical_features = build_chemical_feature_matrix(modification_table)
    model = ChemicalMultiRM(args.embedding_path, chemical_features).to(args.device)

    train_lomo_model(
        args,
        model,
        train_data,
        valid_loader,
        test_loader,
        task_indices,
        heldout_index,
        "Chemical shared-scorer LOMO result",
    )


def train_lomo_model(args, model, train_data, valid_loader, test_loader, task_indices, heldout_index, result_label):
    train_loader = DataLoader(
        RmDataset(train_data),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    loss_weights = calculate_loss_weights(train_data["y"][:, task_indices]).to(args.device)
    uncertainty_weights = create_uncertainty_weights_for_task_count(args.loss_strategy, len(task_indices), args.device)
    parameters = list(model.parameters())
    if uncertainty_weights is not None:
        parameters += list(uncertainty_weights.parameters())
    optimizer = Adam(parameters, lr=args.lr)

    save_dir = Path(args.save_dir) / MODIFICATION_NAMES[heldout_index]
    save_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["heldout_index"] = heldout_index
    config["train_task_indices"] = task_indices
    config["train_task_names"] = [MODIFICATION_NAMES[index] for index in task_indices]
    with (save_dir / "config.json").open("w") as handle:
        json.dump(config, handle, indent=2)

    best_valid_aucb = -1.0
    best_valid_loss = float("inf")
    epochs_since_valid_loss_improvement = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_examples = 0
        for x, labels in train_loader:
            labels = labels.to(args.device)
            outputs = model(x.to(args.device))
            loss = paper_training_loss_for_indices(
                outputs,
                labels,
                task_indices,
                loss_weights,
                uses_ohem(args.loss_strategy),
                uncertainty_weights,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += float(loss.item())
            train_examples += labels.shape[0]

        valid_loss = evaluate_lomo_loss(model, valid_loader, task_indices, loss_weights, args.loss_strategy, uncertainty_weights, args.device)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            epochs_since_valid_loss_improvement = 0
        else:
            epochs_since_valid_loss_improvement += 1
            if epochs_since_valid_loss_improvement >= args.lr_patience:
                decay_learning_rate(optimizer, args.lr_decay)
                epochs_since_valid_loss_improvement = 0

        valid_probabilities, valid_labels = collect_predictions(model, valid_loader, args.device)
        valid_rows = compute_seen_modification_table(valid_probabilities, valid_labels, task_indices)
        valid_mean_aucb = valid_rows[-2]["AUCb"]
        print(
            f"epoch={epoch} heldout={MODIFICATION_NAMES[heldout_index]} train_loss={train_loss / train_examples:.6f} "
            f"valid_loss={valid_loss:.6f} seen_valid_mean_aucb={valid_mean_aucb:.6f} "
            f"lr={current_learning_rate(optimizer):.10f}",
            flush=True,
        )
        if valid_mean_aucb > best_valid_aucb:
            best_valid_aucb = valid_mean_aucb
            torch.save(model.state_dict(), save_dir / "best_model.pt")
            save_table(valid_rows, save_dir, "valid_seen")

    model.load_state_dict(torch.load(save_dir / "best_model.pt", map_location=args.device))
    test_probabilities, test_labels = collect_predictions(model, test_loader, args.device)
    heldout_row = compute_single_modification_row(test_probabilities, test_labels, heldout_index)
    save_single_row(heldout_row, save_dir, "test_heldout")
    print(f"{result_label}: {MODIFICATION_NAMES[heldout_index]}", flush=True)
    print_table([heldout_row])


def evaluate_lomo_loss(model, loader, task_indices, loss_weights, loss_strategy, uncertainty_weights, device):
    model.eval()
    total_loss = 0.0
    total_examples = 0
    with torch.no_grad():
        for x, labels in loader:
            labels = labels.to(device)
            outputs = model(x.to(device))
            loss = paper_training_loss_for_indices(
                outputs,
                labels,
                task_indices,
                loss_weights,
                uses_ohem(loss_strategy),
                uncertainty_weights,
            )
            total_loss += float(loss.item())
            total_examples += labels.shape[0]
    return total_loss / total_examples


def main():
    args = parse_args()
    if args.command == "train_original":
        run_train_original(args)
    elif args.command == "train_original_lomo":
        run_train_original_lomo(args)
    elif args.command == "train_chemical":
        run_train_chemical(args)
    elif args.command == "train_chemical_lomo":
        run_train_chemical_lomo(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
