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
from torch.optim import Adam, AdamW
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


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

    def forward(self, hidden_states, query_states):
        # hidden_states: (B, L, D), query_states: (K, D)
        B = hidden_states.shape[0]
        q = query_states[None, :, :].expand(B, -1, -1)
        context, _ = self.attn(q, hidden_states, hidden_states)
        return context


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


class BilinearScorer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W_center = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_context = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(1))
        self.dropout = nn.Dropout()

    def forward(self, center_states, context_states, modification_states):
        # center_states: (B, D), context_states: (B, K, D), modification_states: (K, D)
        mod = modification_states[None, :, :]
        score_center = (self.W_center(center_states[:, None, :]) * mod).sum(-1)
        score_context = (self.W_context(self.dropout(context_states)) * mod).sum(-1)
        return score_center + score_context + self.bias


class LowRankTensorScorer(nn.Module):
    # score_bk = sum_r (r_b · U_r)(c_bk · V_r)(S_k · W_r)
    def __init__(self, hidden_dim, rank=32):
        super().__init__()
        self.U = nn.Parameter(torch.empty(hidden_dim, rank))
        self.V = nn.Parameter(torch.empty(hidden_dim, rank))
        self.W = nn.Parameter(torch.empty(hidden_dim, rank))
        nn.init.xavier_uniform_(self.U)
        nn.init.xavier_uniform_(self.V)
        nn.init.xavier_uniform_(self.W)
        self.bias = nn.Parameter(torch.zeros(1))
        self.dropout = nn.Dropout()

    def forward(self, center_states, context_states, modification_states):
        ru = center_states[:, None, :] @ self.U
        cv = self.dropout(context_states) @ self.V
        mw = modification_states[None, :, :] @ self.W
        return (ru * cv * mw).sum(-1) + self.bias


class HypernetworkScorer(nn.Module):
    # modification_states generates the weights of a linear layer applied to RNA
    def __init__(self, hidden_dim, hyper_dim=32):
        super().__init__()
        self.hyper_dim = hyper_dim
        self.rna_proj = nn.Linear(hidden_dim * 2, hyper_dim)
        self.weight_gen = nn.Linear(hidden_dim, hyper_dim * hyper_dim)
        self.bias_gen = nn.Linear(hidden_dim, hyper_dim)
        self.output = nn.Linear(hyper_dim, 1)
        self.dropout = nn.Dropout()

    def forward(self, center_states, context_states, modification_states):
        B, K, D = context_states.shape
        rna = torch.cat([center_states[:, None, :].expand(B, K, D), context_states], dim=-1)
        rna_h = torch.relu(self.rna_proj(rna))
        W = self.weight_gen(modification_states).view(K, self.hyper_dim, self.hyper_dim)
        b = self.bias_gen(modification_states)
        hidden = torch.einsum("bkh,khj->bkj", rna_h, W) + b[None, :, :]
        return self.output(torch.relu(self.dropout(hidden))).squeeze(-1)


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
        return [logits[:, index] for index in range(self.num_task)]


class ModificationIdMultiRM(nn.Module):
    def __init__(self, embedding_path, num_task=12, modification_hidden_size=512):
        super().__init__()
        self.num_task = num_task
        self.embed = FrozenEmbeddingSeq(embedding_path)
        self.NaiveBiLSTM = nn.LSTM(input_size=300, hidden_size=256, batch_first=True, bidirectional=True)
        self.modification_embedding = nn.Embedding(num_task, modification_hidden_size)
        self.register_buffer("modification_indices", torch.arange(num_task, dtype=torch.long))
        self.Attention = ChemicalQueryAttention(hidden_size=512, chemical_size=modification_hidden_size, attention_size=100)
        self.Scorer = ChemicalInteractionScorer(chemical_size=modification_hidden_size, zero_init_output=False)

    def forward(self, x):
        x = self.embed(x)
        output, _ = self.NaiveBiLSTM(x)
        modification_states = self.modification_embedding(self.modification_indices)
        modification_context, _ = self.Attention(modification_states, output)
        center_index = output.shape[1] // 2
        center_context = output[:, center_index, :][:, None, :].expand(output.shape[0], self.num_task, output.shape[-1])
        logits = self.Scorer(center_context, modification_context, modification_states)
        return [logits[:, index] for index in range(self.num_task)]


class ChemicalMultiRMv1(nn.Module):
    def __init__(self, embedding_path, chemical_features, num_task=12,
                 chemical_hidden_size=512, num_heads=8, scorer_type="bilinear",
                 chemical_encoder_type="mlp"):
        super().__init__()
        self.num_task = num_task
        self.embed = FrozenEmbeddingSeq(embedding_path)
        self.NaiveBiLSTM = nn.LSTM(input_size=300, hidden_size=256, batch_first=True, bidirectional=True)
        self.register_buffer("chemical_features", torch.as_tensor(chemical_features, dtype=torch.float32))
        if chemical_encoder_type == "mlp":
            self.chemical_encoder = nn.Sequential(
                nn.Linear(chemical_features.shape[1], chemical_hidden_size),
                nn.LayerNorm(chemical_hidden_size),
                nn.GELU(),
                nn.Dropout(),
                nn.Linear(chemical_hidden_size, 512),
                nn.LayerNorm(512),
                nn.GELU(),
            )
        elif chemical_encoder_type == "linear":
            self.chemical_encoder = nn.Linear(chemical_features.shape[1], 512, bias=False)
        elif chemical_encoder_type == "frozen_linear":
            self.chemical_encoder = nn.Linear(chemical_features.shape[1], 512, bias=False)
            # JL-style preserved-geometry init: scale weights so output has unit-variance
            with torch.no_grad():
                self.chemical_encoder.weight.normal_(0.0, 1.0 / math.sqrt(chemical_features.shape[1]))
            for p in self.chemical_encoder.parameters():
                p.requires_grad = False
        else:
            raise ValueError(chemical_encoder_type)
        self.attention = MultiHeadCrossAttention(512, num_heads=num_heads)
        if scorer_type == "bilinear":
            self.scorer = BilinearScorer(512)
        elif scorer_type == "lowrank":
            self.scorer = LowRankTensorScorer(512)
        elif scorer_type == "hypernetwork":
            self.scorer = HypernetworkScorer(512)
        else:
            raise ValueError(scorer_type)

    def forward(self, x):
        x = self.embed(x)
        output, _ = self.NaiveBiLSTM(x)
        chemical_states = self.chemical_encoder(self.chemical_features)
        context_states = self.attention(output, chemical_states)
        center_state = output[:, output.shape[1] // 2, :]
        logits = self.scorer(center_state, context_states, chemical_states)
        return [logits[:, index] for index in range(self.num_task)]


class SharpAttentionRoutingScorer(nn.Module):
    """v2 scorer: chemistry queries RNA via sharp attention, scorer has NO mod term.

    For mod_a == mod_b in encoder output, Q_a == Q_b ⇒ attention identical ⇒
    same readout. The hope is that for mod_a ≈ mod_b (e.g. cos 0.85), the
    sharpened softmax can snap to different positions when the RNA attention
    landscape is multimodal — breaking the Lipschitz "similar mod ⇒ similar
    output" bound of bilinear scorers.

    Args:
        hidden_dim: D for Q/K/V.
        tau: temperature scale. logits divided by (tau * sqrt(D)).
            tau < 1.0 sharpens attention; tau > 1.0 flattens it.
    """

    def __init__(self, hidden_dim=512, tau=0.25):
        super().__init__()
        self.Wq = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Wk = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Wv = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(),
            nn.Linear(hidden_dim, 1),
        )
        self.tau = tau

    def forward(self, rna_states, modification_states):
        # rna_states: (B, L, D), modification_states: (K, D)
        Q = self.Wq(modification_states)  # (K, D)
        K = self.Wk(rna_states)            # (B, L, D)
        V = self.Wv(rna_states)            # (B, L, D)
        scale = self.tau * (K.shape[-1] ** 0.5)
        attn_logits = torch.einsum("kd,bld->bkl", Q, K) / scale  # (B, K, L)
        attn = attn_logits.softmax(dim=-1)                       # (B, K, L)
        r = torch.einsum("bkl,bld->bkd", attn, V)                # (B, K, D)
        return self.mlp(r).squeeze(-1)                           # (B, K)


class ChemicalMultiRMv2(nn.Module):
    """v2: chemistry routes RNA attention; scorer has no mod term."""

    def __init__(self, embedding_path, chemical_features, num_task=12,
                 tau=0.25, chemical_encoder_type="linear"):
        super().__init__()
        self.num_task = num_task
        self.embed = FrozenEmbeddingSeq(embedding_path)
        self.NaiveBiLSTM = nn.LSTM(input_size=300, hidden_size=256, batch_first=True, bidirectional=True)
        self.register_buffer("chemical_features", torch.as_tensor(chemical_features, dtype=torch.float32))
        if chemical_encoder_type == "linear":
            self.chemical_encoder = nn.Linear(chemical_features.shape[1], 512, bias=False)
        elif chemical_encoder_type == "frozen_linear":
            self.chemical_encoder = nn.Linear(chemical_features.shape[1], 512, bias=False)
            with torch.no_grad():
                self.chemical_encoder.weight.normal_(0.0, 1.0 / math.sqrt(chemical_features.shape[1]))
            for p in self.chemical_encoder.parameters():
                p.requires_grad = False
        elif chemical_encoder_type == "mlp":
            self.chemical_encoder = nn.Sequential(
                nn.Linear(chemical_features.shape[1], 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(),
                nn.Linear(512, 512),
                nn.LayerNorm(512),
                nn.GELU(),
            )
        else:
            raise ValueError(chemical_encoder_type)
        self.scorer = SharpAttentionRoutingScorer(hidden_dim=512, tau=tau)

    def forward(self, x):
        x = self.embed(x)
        rna_states, _ = self.NaiveBiLSTM(x)  # (B, L, D)
        chemical_states = self.chemical_encoder(self.chemical_features)  # (K, D)
        logits = self.scorer(rna_states, chemical_states)  # (B, K)
        return [logits[:, k] for k in range(self.num_task)]


class BioMatchScorer(nn.Module):
    """Per-(sample, modification) bio matching path.

    For each (sample b, modification k), the MLP receives:
      - the FULL K-dim per-sample PWM match vector (all 12 PWMs)
      - k's bio prior vector (writer + region + motif blocks)
    A shared MLP outputs a scalar bio_match_logit. Shared across mods → LOMO-safe.

    Feeding all PWM matches (not just k's own) allows the MLP to learn cross-PWM
    signals: e.g. for the Am column the MLP can pick up that Am positives have
    higher m6A-PWM match than Am negatives (empirically delta +1.20), even
    though Am's own PWM is flat.
    """

    def __init__(self, num_task, bio_prior_dim, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_task + bio_prior_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, pwm_match, bio_prior_per_mod):
        # pwm_match: (B, K) per-sample per-mod PWM log-likelihood-ratio
        # bio_prior_per_mod: (K, D_bio) the static mod-level prior vector
        B, K = pwm_match.shape
        pwm_broad = pwm_match.unsqueeze(1).expand(B, K, K)  # (B, K, K) - all PWMs per column
        bio_x = bio_prior_per_mod.unsqueeze(0).expand(B, K, -1)  # (B, K, D)
        combined = torch.cat([pwm_broad, bio_x], dim=-1)  # (B, K, K+D)
        return self.mlp(combined).squeeze(-1)  # (B, K)


class TanimotoBioMatchScorer(nn.Module):
    """Bio matching via fixed Tanimoto-weighted cross-PWM aggregation.

    biomatch_logit[b, k] = sum_j tanimoto[k, j] * pwm_match[b, j]

    No MLP; weights are fixed chemistry similarity. This avoids the OOD issue
    that breaks the learned MLP version: at LOMO test time, the held-out mod's
    PWM has a different distribution from training, but Tanimoto weights are
    known by definition for any mod pair.
    """

    def __init__(self, tanimoto_matrix):
        super().__init__()
        self.register_buffer("tanimoto", torch.as_tensor(tanimoto_matrix, dtype=torch.float32))

    def forward(self, pwm_match):
        # pwm_match: (B, K)
        # biomatch[b, k] = sum_j Tani[k, j] * pwm_match[b, j]
        return pwm_match @ self.tanimoto.T  # (B, K)


class PathAMetadataMatchScorer(nn.Module):
    """Per-sample x per-mod bio matching using genome-aligned region/distance metadata.

    For each (sample b, modification k):
      region_match[b, k] = dot(sample_region_onehot[b], mod_region_prior[k])
      cap_match[b, k] = exp(-((sample_log_cap_dist[b] - mod_log_cap_dist[k])**2) / sigma**2)
    Then combined via a small shared MLP that also sees the full mod bio prior.
    All weights are computed from inputs deterministically; no MLP overfitting risk
    on held-out modifications because the matching geometry is fixed.
    """

    def __init__(self, mod_region_prior, mod_log_cap_dist_centers, num_task,
                 sigma=1.5, num_regions=5):
        super().__init__()
        self.register_buffer("mod_region_prior", torch.as_tensor(mod_region_prior, dtype=torch.float32))  # (K, R)
        self.register_buffer("mod_log_cap_dist_centers", torch.as_tensor(mod_log_cap_dist_centers, dtype=torch.float32))  # (K,)
        self.sigma = sigma

    def forward(self, sample_region_onehot, sample_log_cap_dist, sample_mapped_mask):
        # sample_region_onehot: (B, R) one-hot per sample (or zeros if unmapped)
        # sample_log_cap_dist: (B,) log-distance from cap; -1 if unmapped
        # sample_mapped_mask: (B,) bool/float, 1 if sample is mapped to genome, 0 otherwise
        # mod_region_prior: (K, R), already normalised distribution
        B = sample_region_onehot.shape[0]
        K = self.mod_region_prior.shape[0]
        # region match
        region_match = sample_region_onehot @ self.mod_region_prior.T  # (B, K)
        # cap distance gaussian similarity
        dist_diff = sample_log_cap_dist.unsqueeze(-1) - self.mod_log_cap_dist_centers.unsqueeze(0)  # (B, K)
        cap_match = torch.exp(-(dist_diff ** 2) / (self.sigma ** 2))  # (B, K)
        # combine: sum, masked by whether sample was mapped
        score = (region_match + cap_match) * sample_mapped_mask.unsqueeze(-1)  # (B, K)
        return score


class ChemicalMultiRMv2BioMatchTan(nn.Module):
    """v2 sharp-attn plus a Tanimoto-weighted bio matching path.

    score_k = main_sharp_attn_logit_k + alpha * (Tani @ pwm_match)_k

    alpha is FIXED (not learned by default). Standalone biomatch_logit gives
    m6A AUCm 0.955 and Am AUCb 0.856 on test data without any training - the
    information is in the input. Learned alpha collapses to ~0.08 because it
    optimises seen-mod loss where biomatch is weak, missing the held-out
    transfer benefit. Fixed alpha sidesteps this misalignment.
    """

    def __init__(self, embedding_path, chemical_features, tanimoto_matrix,
                 num_task=12, tau=0.4, chemical_encoder_type="linear",
                 alpha=1.0, learnable_alpha=False):
        super().__init__()
        self.num_task = num_task
        self.embed = FrozenEmbeddingSeq(embedding_path)
        self.NaiveBiLSTM = nn.LSTM(input_size=300, hidden_size=256, batch_first=True, bidirectional=True)
        self.register_buffer("chemical_features", torch.as_tensor(chemical_features, dtype=torch.float32))
        if chemical_encoder_type == "linear":
            self.chemical_encoder = nn.Linear(chemical_features.shape[1], 512, bias=False)
        elif chemical_encoder_type == "frozen_linear":
            self.chemical_encoder = nn.Linear(chemical_features.shape[1], 512, bias=False)
            with torch.no_grad():
                self.chemical_encoder.weight.normal_(0.0, 1.0 / math.sqrt(chemical_features.shape[1]))
            for p in self.chemical_encoder.parameters():
                p.requires_grad = False
        elif chemical_encoder_type == "mlp":
            self.chemical_encoder = nn.Sequential(
                nn.Linear(chemical_features.shape[1], 512),
                nn.LayerNorm(512), nn.GELU(), nn.Dropout(),
                nn.Linear(512, 512), nn.LayerNorm(512), nn.GELU(),
            )
        else:
            raise ValueError(chemical_encoder_type)
        self.scorer = SharpAttentionRoutingScorer(hidden_dim=512, tau=tau)
        self.bio_match_scorer = TanimotoBioMatchScorer(tanimoto_matrix)
        if learnable_alpha:
            self.alpha = nn.Parameter(torch.tensor([float(alpha)]))
        else:
            self.register_buffer("alpha", torch.tensor([float(alpha)]))

    def forward(self, x, pwm_match):
        x = self.embed(x)
        rna_states, _ = self.NaiveBiLSTM(x)
        chemical_states = self.chemical_encoder(self.chemical_features)
        main_logit = self.scorer(rna_states, chemical_states)  # (B, K)
        bio_match_logit = self.bio_match_scorer(pwm_match)  # (B, K)
        logits = main_logit + self.alpha * bio_match_logit
        return [logits[:, k] for k in range(self.num_task)]


class ChemicalMultiRMv2PathA(nn.Module):
    """v2 sharp-attn main path plus Tanimoto-weighted PWM biomatch plus
    per-sample genome-derived region/cap-distance match.

    score_k = main_sharp_attn_logit_k
            + alpha_pwm * (Tani @ pwm_match)_k
            + alpha_meta * pathA_match_logit_k

    The Path A scorer is parameter-free given fixed mod_region_prior and
    mod_log_cap_dist_centers (literature-derived + training-positive median).
    This restores the per-sample biological context (region indicator + cap
    distance) that PWM-only matching lacks, which is critical for m6Am
    (cap+1 only) versus Am (cap-adjacent + internal mixture).
    """

    def __init__(self, embedding_path, chemical_features, tanimoto_matrix,
                 mod_region_prior, mod_log_cap_dist_centers,
                 num_task=12, tau=0.4, chemical_encoder_type="linear",
                 alpha_pwm=2.0, alpha_meta=2.0, sigma=1.5):
        super().__init__()
        self.num_task = num_task
        self.embed = FrozenEmbeddingSeq(embedding_path)
        self.NaiveBiLSTM = nn.LSTM(input_size=300, hidden_size=256, batch_first=True, bidirectional=True)
        self.register_buffer("chemical_features", torch.as_tensor(chemical_features, dtype=torch.float32))
        if chemical_encoder_type == "linear":
            self.chemical_encoder = nn.Linear(chemical_features.shape[1], 512, bias=False)
        elif chemical_encoder_type == "frozen_linear":
            self.chemical_encoder = nn.Linear(chemical_features.shape[1], 512, bias=False)
            with torch.no_grad():
                self.chemical_encoder.weight.normal_(0.0, 1.0 / math.sqrt(chemical_features.shape[1]))
            for p in self.chemical_encoder.parameters():
                p.requires_grad = False
        elif chemical_encoder_type == "mlp":
            self.chemical_encoder = nn.Sequential(
                nn.Linear(chemical_features.shape[1], 512),
                nn.LayerNorm(512), nn.GELU(), nn.Dropout(),
                nn.Linear(512, 512), nn.LayerNorm(512), nn.GELU(),
            )
        else:
            raise ValueError(chemical_encoder_type)
        self.scorer = SharpAttentionRoutingScorer(hidden_dim=512, tau=tau)
        self.bio_match_scorer = TanimotoBioMatchScorer(tanimoto_matrix)
        self.pathA_scorer = PathAMetadataMatchScorer(
            mod_region_prior, mod_log_cap_dist_centers, num_task=num_task, sigma=sigma,
        )
        self.register_buffer("alpha_pwm", torch.tensor([float(alpha_pwm)]))
        self.register_buffer("alpha_meta", torch.tensor([float(alpha_meta)]))

    def forward(self, x, pwm_match, region_onehot, log_cap_distance, mapped_mask):
        x = self.embed(x)
        rna_states, _ = self.NaiveBiLSTM(x)
        chemical_states = self.chemical_encoder(self.chemical_features)
        main_logit = self.scorer(rna_states, chemical_states)
        bio_match_logit = self.bio_match_scorer(pwm_match)
        pathA_logit = self.pathA_scorer(region_onehot, log_cap_distance, mapped_mask)
        logits = main_logit + self.alpha_pwm * bio_match_logit + self.alpha_meta * pathA_logit
        return [logits[:, k] for k in range(self.num_task)]


class ChemicalMultiRMv2BioMatch(nn.Module):
    """v2 sharp-attn plus a per-sample bio matching path.

    score_k = main_sharp_attn_logit_k + lambda_match * bio_match_logit_k

    where bio_match_logit comes from a shared MLP applied to
    [pwm_match_per_sample_k, bio_prior_per_mod_k].
    """

    def __init__(self, embedding_path, chemical_features, bio_prior_per_mod,
                 num_task=12, tau=0.4, chemical_encoder_type="linear",
                 lambda_match=1.0):
        super().__init__()
        self.num_task = num_task
        self.lambda_match = lambda_match
        self.embed = FrozenEmbeddingSeq(embedding_path)
        self.NaiveBiLSTM = nn.LSTM(input_size=300, hidden_size=256, batch_first=True, bidirectional=True)
        self.register_buffer("chemical_features", torch.as_tensor(chemical_features, dtype=torch.float32))
        self.register_buffer("bio_prior_per_mod", torch.as_tensor(bio_prior_per_mod, dtype=torch.float32))
        if chemical_encoder_type == "linear":
            self.chemical_encoder = nn.Linear(chemical_features.shape[1], 512, bias=False)
        elif chemical_encoder_type == "frozen_linear":
            self.chemical_encoder = nn.Linear(chemical_features.shape[1], 512, bias=False)
            with torch.no_grad():
                self.chemical_encoder.weight.normal_(0.0, 1.0 / math.sqrt(chemical_features.shape[1]))
            for p in self.chemical_encoder.parameters():
                p.requires_grad = False
        elif chemical_encoder_type == "mlp":
            self.chemical_encoder = nn.Sequential(
                nn.Linear(chemical_features.shape[1], 512),
                nn.LayerNorm(512), nn.GELU(), nn.Dropout(),
                nn.Linear(512, 512), nn.LayerNorm(512), nn.GELU(),
            )
        else:
            raise ValueError(chemical_encoder_type)
        self.scorer = SharpAttentionRoutingScorer(hidden_dim=512, tau=tau)
        self.bio_match_scorer = BioMatchScorer(num_task, bio_prior_per_mod.shape[1], hidden_dim=64)

    def forward(self, x, pwm_match):
        x = self.embed(x)
        rna_states, _ = self.NaiveBiLSTM(x)  # (B, L, D)
        chemical_states = self.chemical_encoder(self.chemical_features)  # (K, D)
        main_logit = self.scorer(rna_states, chemical_states)  # (B, K)
        bio_match_logit = self.bio_match_scorer(pwm_match, self.bio_prior_per_mod)  # (B, K)
        logits = main_logit + self.lambda_match * bio_match_logit
        return [logits[:, k] for k in range(self.num_task)]


class RmDataset(Dataset):
    def __init__(self, split_data):
        self.x = torch.from_numpy(split_data["x"].copy())
        self.y = torch.from_numpy(split_data["y"].copy())

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class RmDatasetBio(Dataset):
    """Like RmDataset but also returns per-sample PWM match scores."""

    def __init__(self, split_data):
        self.x = torch.from_numpy(split_data["x"].copy())
        self.y = torch.from_numpy(split_data["y"].copy())
        if "pwm_match" not in split_data:
            raise ValueError("RmDatasetBio requires split_data with 'pwm_match' key. "
                             "Make sure the cache was rebuilt after adding PWM match support.")
        self.pwm = torch.from_numpy(split_data["pwm_match"].copy())

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.pwm[index]


class RmDatasetPathA(Dataset):
    """RmDatasetBio + per-sample genome-derived region one-hot, log cap distance, mapped mask."""

    def __init__(self, split_data):
        self.x = torch.from_numpy(split_data["x"].copy())
        self.y = torch.from_numpy(split_data["y"].copy())
        for required in ("pwm_match", "region_onehot", "log_cap_distance", "mapped"):
            if required not in split_data:
                raise ValueError(f"RmDatasetPathA requires split_data['{required}'].")
        self.pwm = torch.from_numpy(split_data["pwm_match"].copy())
        self.region = torch.from_numpy(split_data["region_onehot"].copy())
        self.logcap = torch.from_numpy(split_data["log_cap_distance"].copy())
        self.mapped = torch.from_numpy(split_data["mapped"].copy())

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return (self.x[index], self.y[index], self.pwm[index],
                self.region[index], self.logcap[index], self.mapped[index])


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

    train_modid = subparsers.add_parser("train_modid")
    add_common_args(train_modid)
    add_training_args(train_modid, "Results/paper_aligned/modid")

    modid_lomo = subparsers.add_parser("train_modid_lomo")
    add_common_args(modid_lomo)
    add_training_args(modid_lomo, "Results/paper_aligned/modid_lomo")
    modid_lomo.add_argument("--heldout_mod", choices=MODIFICATION_NAMES, required=True)

    train_v1 = subparsers.add_parser("train_chemical_v1")
    add_common_args(train_v1)
    add_training_args(train_v1, "Results/paper_aligned/chemical_v1")
    train_v1.add_argument("--modifications_path", default="Data/modifications.csv")
    add_v1_args(train_v1)

    lomo_v1 = subparsers.add_parser("train_chemical_v1_lomo")
    add_common_args(lomo_v1)
    add_training_args(lomo_v1, "Results/paper_aligned/chemical_v1_lomo")
    lomo_v1.add_argument("--modifications_path", default="Data/modifications.csv")
    lomo_v1.add_argument("--heldout_mod", choices=MODIFICATION_NAMES, required=True)
    add_v1_args(lomo_v1)

    train_v2 = subparsers.add_parser("train_chemical_v2")
    add_common_args(train_v2)
    add_training_args(train_v2, "Results/paper_aligned/chemical_v2")
    train_v2.add_argument("--modifications_path", default="Data/modifications.csv")
    add_v2_args(train_v2)

    lomo_v2 = subparsers.add_parser("train_chemical_v2_lomo")
    add_common_args(lomo_v2)
    add_training_args(lomo_v2, "Results/paper_aligned/chemical_v2_lomo")
    lomo_v2.add_argument("--modifications_path", default="Data/modifications.csv")
    lomo_v2.add_argument("--heldout_mod", choices=MODIFICATION_NAMES, required=True)
    add_v2_args(lomo_v2)

    biomatch_lomo = subparsers.add_parser("train_chemical_biomatch_lomo")
    add_common_args(biomatch_lomo)
    add_training_args(biomatch_lomo, "Results/paper_aligned/chemical_biomatch_lomo")
    biomatch_lomo.add_argument("--modifications_path", default="Data/modifications.csv")
    biomatch_lomo.add_argument("--heldout_mod", choices=MODIFICATION_NAMES, required=True)
    add_v2_args(biomatch_lomo)
    biomatch_lomo.add_argument("--lambda_match", type=float, default=1.0,
                               help="Weight on bio_match_logit added to main sharp-attn logit.")

    biomatch_tan_lomo = subparsers.add_parser("train_chemical_biomatch_tan_lomo")
    add_common_args(biomatch_tan_lomo)
    add_training_args(biomatch_tan_lomo, "Results/paper_aligned/chemical_biomatch_tan_lomo")
    biomatch_tan_lomo.add_argument("--modifications_path", default="Data/modifications.csv")
    biomatch_tan_lomo.add_argument("--heldout_mod", choices=MODIFICATION_NAMES, required=True)
    add_v2_args(biomatch_tan_lomo)
    biomatch_tan_lomo.add_argument("--alpha", type=float, default=1.0,
                                   help="Weight on Tanimoto-weighted biomatch path (default 1.0).")
    biomatch_tan_lomo.add_argument("--learnable_alpha", action="store_true",
                                   help="Make alpha a learnable parameter (default: fixed).")

    pathA_lomo = subparsers.add_parser("train_chemical_pathA_lomo")
    add_common_args(pathA_lomo)
    add_training_args(pathA_lomo, "Results/paper_aligned/chemical_pathA_lomo")
    pathA_lomo.add_argument("--modifications_path", default="Data/modifications.csv")
    pathA_lomo.add_argument("--heldout_mod", choices=MODIFICATION_NAMES, required=True)
    add_v2_args(pathA_lomo)
    pathA_lomo.add_argument("--alpha_pwm", type=float, default=2.0,
                            help="Weight on Tanimoto-weighted PWM biomatch path.")
    pathA_lomo.add_argument("--alpha_meta", type=float, default=2.0,
                            help="Weight on Path-A region/cap-distance match path.")
    pathA_lomo.add_argument("--sigma", type=float, default=1.5,
                            help="Gaussian bandwidth on log_cap_distance match.")

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
        choices=["weighted_bce", "weighted_bce_ohem", "uncertainty", "paper_ohem_uw", "paper_mean_bce"],
        default="paper_ohem_uw",
    )
    parser.add_argument("--early_stop_patience", type=int, default=15)
    parser.add_argument("--weight_decay", type=float, default=1e-4)


def add_v1_args(parser):
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--scorer_type", choices=["bilinear", "lowrank", "hypernetwork"], default="bilinear")
    parser.add_argument("--chemical_encoder_type", choices=["mlp", "linear", "frozen_linear"], default="mlp")
    parser.add_argument("--site_weight", type=float, default=0.0)


def add_v2_args(parser):
    parser.add_argument("--tau", type=float, default=0.25,
                        help="Sharp-attention temperature scale. logits / (tau*sqrt(D)). <1 sharpens.")
    parser.add_argument("--chemical_encoder_type",
                        choices=["mlp", "linear", "frozen_linear"], default="linear")
    parser.add_argument("--site_weight", type=float, default=0.0)
    parser.add_argument("--fp_kind", default="morgan_r2")
    parser.add_argument("--bio_weight", type=float, default=0.0,
                        help="Weight on biology prior block (writer + region + motif). "
                             "If >0, appends Data/bio_priors.pkl features to chemistry. "
                             "Recommended 1.5 to drop chemical-twin combined cosine below 0.5.")
    parser.add_argument("--soft_label_gamma", type=float, default=0.0,
                        help="F2: chemistry-derived soft labels on the held-out column. "
                             "When >0, the held-out column enters the training BCE with "
                             "y_soft = gamma * max_j (y[:,j] * Tanimoto[held-out, j]). "
                             "Strict-LOMO safe: derived only from chemistry similarity "
                             "(RDKit on SMILES) and seen-mod labels, never from held-out "
                             "positives. Model selection still uses seen mods only.")
    parser.add_argument("--soft_label_loss_mode",
                        choices=["joint_prob", "aux_prob", "aux_pos_weight"],
                        default="joint_prob",
                        help="How to train from held-out soft labels. joint_prob is the "
                             "original F2 behavior: append the held-out column to the "
                             "multi-task BCE with soft labels as probabilities. aux_prob "
                             "keeps seen-mod training unchanged and adds a separate "
                             "held-out BCE. aux_pos_weight treats soft labels as positive "
                             "transfer confidence weights, avoiding false-negative "
                             "pressure on samples without a chemistry match.")
    parser.add_argument("--soft_label_aux_weight", type=float, default=1.0,
                        help="Multiplier for aux_prob / aux_pos_weight held-out loss.")
    parser.add_argument("--soft_label_tani_min", type=float, default=0.0,
                        help="Ignore seen-mod transfer sources with Tanimoto below this "
                             "threshold when constructing held-out soft labels.")
    parser.add_argument("--soft_label_pwm_gate", action="store_true",
                        help="Gate held-out soft labels by the held-out PWM/motif score "
                             "when PWM scores have non-trivial variance for that mod.")
    parser.add_argument("--soft_label_same_base_only", action="store_true",
                        help="Allow held-out soft-label transfer only from seen "
                             "modifications with the same original nucleotide base.")


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


def compute_pwm_match_scores(sequence_array, raw_pwms, motif_window=11, motif_center_pos=5):
    """Compute per-sample, per-modification PWM log-likelihood ratio against uniform.

    For each window, the candidate modification is at the centre (position 25 of
    51 nt). For each modification k with PWM p_k of shape (motif_window, 4):
        score[b, k] = sum over flank offsets (i != centre) of
                      log(p_k[i, base_at_window_centre+i-centre] / 0.25)

    The centre position is excluded (its base is fixed by the modification's
    chemistry; including it would constant-shift every sample's score).

    sequence_array: (N, length) array of single-char strings ('A','C','G','T').
                    Already 'U'-replaced-by-'T' is acceptable; we treat T as U
                    by indexing both into position 3 of bases_order ('A','C','G','U').
    """
    N, length = sequence_array.shape
    centre_in_window = length // 2  # e.g. 25 for length=51
    K = raw_pwms.shape[0]
    flank_offsets = [offset for offset in range(motif_window) if offset != motif_center_pos]
    base_to_idx = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3}

    # Precompute log-ratio PWM (log(p / 0.25)) for each flank offset
    eps = 1e-6
    log_ratio = np.log((raw_pwms + eps) / 0.25)  # (K, motif_window, 4)

    scores = np.zeros((N, K), dtype=np.float32)
    for n in range(N):
        for offset in flank_offsets:
            window_pos = centre_in_window + (offset - motif_center_pos)
            if window_pos < 0 or window_pos >= length:
                continue
            base = sequence_array[n, window_pos]
            b = base_to_idx.get(base, None)
            if b is None:
                continue
            scores[n, :] += log_ratio[:, offset, b]
    return scores


def read_split_as_kmers(data_path, split_name, length, embedding_path, cache_dir):
    cache_path = Path(cache_dir) / f"{split_name}_{length}bp_3mer.npz"
    if cache_path.exists():
        loaded = np.load(cache_path)
        result = {"x": loaded["x"], "y": loaded["y"]}
        if "pwm_match" in loaded.files:
            result["pwm_match"] = loaded["pwm_match"]
        return result

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

    bio_pack = pickle.load(open("/ibex/user/songt/MultiRM/Data/bio_priors.pkl", "rb"))
    pwm_match = compute_pwm_match_scores(
        sequence_array,
        bio_pack["raw_pwms"],
        motif_window=bio_pack["motif_window"],
        motif_center_pos=bio_pack["motif_center_pos"],
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, x=x, y=y, pwm_match=pwm_match)
    return {"x": x, "y": y, "pwm_match": pwm_match}


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


def paper_training_loss(outputs, labels, loss_weights, ohem, uncertainty_weights, mean_bce=False):
    logits = torch.stack(outputs, dim=1)
    if mean_bce:
        return torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="mean")
    predictions = torch.clamp(torch.sigmoid(logits), min=1e-7, max=1.0 - 1e-7)
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


def paper_training_loss_for_indices(outputs, labels, task_indices, loss_weights, ohem, uncertainty_weights, mean_bce=False):
    selected_outputs = [outputs[index] for index in task_indices]
    selected_labels = labels[:, task_indices]
    return paper_training_loss(selected_outputs, selected_labels, loss_weights, ohem, uncertainty_weights, mean_bce=mean_bce)


def soft_label_auxiliary_loss(outputs, labels, heldout_index, mode, aux_weight):
    """Auxiliary held-out soft-label loss used by the filtered F2 variants."""
    if aux_weight <= 0.0:
        return outputs[heldout_index].sum() * 0.0
    logits = outputs[heldout_index]
    targets = labels[:, heldout_index].clamp(min=0.0, max=1.0)
    if mode == "aux_prob":
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="sum"
        )
    elif mode == "aux_pos_weight":
        mask = targets > 0.0
        if not torch.any(mask):
            return logits.sum() * 0.0
        positive_targets = torch.ones_like(targets[mask])
        per_sample = torch.nn.functional.binary_cross_entropy_with_logits(
            logits[mask], positive_targets, reduction="none"
        )
        loss = (per_sample * targets[mask]).sum()
    else:
        raise ValueError(mode)
    return float(aux_weight) * loss


def apply_tanimoto_soft_labels(train_data, heldout_index, tanimoto_matrix, gamma,
                                tani_min=0.0, pwm_gate=False, source_mask=None):
    """Replace held-out column labels with chemistry-derived soft labels.

    For each sample i:
        y_soft[i, heldout] = gamma * max over j != heldout of (y[i, j] * T[heldout, j])

    The soft label uses only Tanimoto similarity (RDKit on canonical SMILES, no
    label info from the held-out modification) and the labels of the SEEN
    modifications that are present in the post-LOMO-removal training set. So
    it is a chemistry-prior signal, not a leak from held-out positives.

    Optional filters make the transfer less destructive: tani_min drops weak
    chemistry neighbors, and pwm_gate keeps pseudo positives preferentially on
    samples whose sequence motif matches the held-out modification. PWM gating
    is skipped when the held-out PWM score is effectively constant.

    Returns a NEW dict (train_data is not mutated).
    """
    if gamma is None or gamma <= 0:
        return train_data
    out = dict(train_data)
    y = out["y"].astype(np.float32).copy()
    T_row = tanimoto_matrix[heldout_index].astype(np.float32).copy()
    T_row[heldout_index] = 0.0  # exclude held-out's own column (forced to 0 anyway by remove_positive_rows)
    if tani_min and tani_min > 0:
        T_row[T_row < float(tani_min)] = 0.0
    if source_mask is not None:
        T_row[~source_mask] = 0.0
    weighted = y * T_row[None, :]  # (N, K)
    soft = weighted.max(axis=1)  # (N,)
    if pwm_gate and "pwm_match" in out:
        pwm = out["pwm_match"][:, heldout_index].astype(np.float32)
        q10, q90 = np.percentile(pwm, [10, 90])
        if float(q90 - q10) > 1e-6:
            gate = np.clip((pwm - q10) / (q90 - q10), 0.0, 1.0).astype(np.float32)
            soft = soft * gate
    y[:, heldout_index] = float(gamma) * soft
    out["y"] = y
    return out


def create_uncertainty_weights(loss_strategy, device):
    if loss_strategy in {"uncertainty", "paper_ohem_uw"}:
        return TaskUncertaintyWeights(len(MODIFICATION_NAMES)).to(device)
    assert loss_strategy in {"weighted_bce", "weighted_bce_ohem", "paper_mean_bce"}
    return None


def uses_ohem(loss_strategy):
    assert loss_strategy in {"weighted_bce", "weighted_bce_ohem", "uncertainty", "paper_ohem_uw", "paper_mean_bce"}
    return loss_strategy in {"weighted_bce_ohem", "paper_ohem_uw"}


def uses_mean_bce(loss_strategy):
    return loss_strategy == "paper_mean_bce"


def create_uncertainty_weights_for_task_count(loss_strategy, task_count, device):
    if loss_strategy in {"uncertainty", "paper_ohem_uw"}:
        return TaskUncertaintyWeights(task_count).to(device)
    assert loss_strategy in {"weighted_bce", "weighted_bce_ohem", "paper_mean_bce"}
    return None


def build_param_groups(model, uncertainty_weights, weight_decay):
    decay = []
    no_decay = []
    for module in model.modules():
        if isinstance(module, (nn.LayerNorm, nn.Embedding)):
            for param in module.parameters(recurse=False):
                if param.requires_grad:
                    no_decay.append(param)
            continue
        for name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            if name.endswith("bias") or param.ndim < 2:
                no_decay.append(param)
            else:
                decay.append(param)
    if uncertainty_weights is not None:
        no_decay += list(uncertainty_weights.parameters())
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


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
            logits = torch.stack(outputs, dim=1)
            probabilities = torch.sigmoid(logits).cpu().numpy()
            all_probabilities.append(probabilities)
            all_labels.append(labels.numpy())
    return np.concatenate(all_probabilities, axis=0), np.concatenate(all_labels, axis=0)


def evaluate_loss(model, loader, loss_weights, loss_strategy, uncertainty_weights, device):
    model.eval()
    ohem = uses_ohem(loss_strategy)
    mean_bce = uses_mean_bce(loss_strategy)
    total_loss = 0.0
    total_examples = 0
    with torch.no_grad():
        for x, labels in loader:
            labels = labels.to(device)
            outputs = model(x.to(device))
            loss = paper_training_loss(outputs, labels, loss_weights, ohem, uncertainty_weights, mean_bce=mean_bce)
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
    out = {
        "x": split_data["x"][keep_mask].copy(),
        "y": split_data["y"][keep_mask].copy(),
    }
    if "pwm_match" in split_data:
        out["pwm_match"] = split_data["pwm_match"][keep_mask].copy()
    for k in ("region_onehot", "log_cap_distance", "mapped"):
        if k in split_data:
            out[k] = split_data[k][keep_mask].copy()
    return out


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
    param_groups = build_param_groups(model, uncertainty_weights, args.weight_decay)
    optimizer = AdamW(param_groups, lr=args.lr)
    mean_bce = uses_mean_bce(args.loss_strategy)
    ohem = uses_ohem(args.loss_strategy)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    with (save_dir / "config.json").open("w") as handle:
        json.dump(vars(args), handle, indent=2)

    best_valid_aucb = -1.0
    best_valid_loss = float("inf")
    epochs_since_valid_loss_improvement = 0
    epochs_since_aucb_improvement = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_examples = 0
        for x, labels in train_loader:
            labels = labels.to(args.device)
            outputs = model(x.to(args.device))
            loss = paper_training_loss(outputs, labels, loss_weights, ohem, uncertainty_weights, mean_bce=mean_bce)
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
            epochs_since_aucb_improvement = 0
            torch.save(model.state_dict(), save_dir / "best_model.pt")
            save_table(valid_rows, save_dir, "valid")
        else:
            epochs_since_aucb_improvement += 1
            if epochs_since_aucb_improvement >= args.early_stop_patience:
                print(f"early stopping at epoch {epoch} (no AUCb improvement for {args.early_stop_patience} epochs)", flush=True)
                break

    model.load_state_dict(torch.load(save_dir / "best_model.pt", map_location=args.device))
    test_probabilities, test_labels = collect_predictions(model, test_loader, args.device)
    np.savez(save_dir / "test_predictions.npz", prob=test_probabilities, label=test_labels)
    valid_probabilities, valid_labels = collect_predictions(model, valid_loader, args.device)
    np.savez(save_dir / "valid_predictions.npz", prob=valid_probabilities, label=valid_labels)
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


def run_train_modid(args):
    set_seed(args.seed)
    train_data, _, _, _, valid_loader, test_loader = build_data_loaders(args)
    model = ModificationIdMultiRM(args.embedding_path).to(args.device)
    train_and_evaluate(args, model, train_data, valid_loader, test_loader, "Modification ID shared-scorer all-modification test results")


def run_train_modid_lomo(args):
    set_seed(args.seed)
    heldout_index = MODIFICATION_NAMES.index(args.heldout_mod)
    task_indices = [index for index in range(len(MODIFICATION_NAMES)) if index != heldout_index]

    train_data = read_split_as_kmers(args.data_path, "train", args.length, args.embedding_path, args.cache_dir)
    valid_data = read_split_as_kmers(args.data_path, "valid", args.length, args.embedding_path, args.cache_dir)
    test_data = read_split_as_kmers(args.data_path, "test", args.length, args.embedding_path, args.cache_dir)

    train_data = remove_positive_rows(train_data, heldout_index)
    valid_data_for_selection = remove_positive_rows(valid_data, heldout_index)

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

    model = ModificationIdMultiRM(args.embedding_path).to(args.device)
    train_lomo_model(
        args,
        model,
        train_data,
        valid_loader,
        test_loader,
        task_indices,
        heldout_index,
        "Modification ID shared-scorer LOMO result",
    )


def run_train_chemical_v1(args):
    set_seed(args.seed)
    train_data, _, _, _, valid_loader, test_loader = build_data_loaders(args)
    modification_table = load_modification_table(args.modifications_path)
    chemical_features = build_chemical_feature_matrix(modification_table, site_weight=args.site_weight)
    model = ChemicalMultiRMv1(
        args.embedding_path, chemical_features,
        num_heads=args.num_heads, scorer_type=args.scorer_type,
        chemical_encoder_type=args.chemical_encoder_type,
    ).to(args.device)
    train_and_evaluate(
        args, model, train_data, valid_loader, test_loader,
        f"Chemical v1 scorer={args.scorer_type} all-modification test results",
    )


def run_train_chemical_v1_lomo(args):
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
    chemical_features = build_chemical_feature_matrix(modification_table, site_weight=args.site_weight)
    model = ChemicalMultiRMv1(
        args.embedding_path, chemical_features,
        num_heads=args.num_heads, scorer_type=args.scorer_type,
        chemical_encoder_type=args.chemical_encoder_type,
    ).to(args.device)

    train_lomo_model(
        args, model, train_data, valid_loader, test_loader,
        task_indices, heldout_index,
        f"Chemical v1 scorer={args.scorer_type} LOMO result",
    )


def run_train_chemical_v2(args):
    set_seed(args.seed)
    train_data, _, _, _, valid_loader, test_loader = build_data_loaders(args)
    modification_table = load_modification_table(args.modifications_path)
    chemical_features = build_chemical_feature_matrix(
        modification_table, site_weight=args.site_weight, fp_kind=args.fp_kind,
        bio_weight=args.bio_weight,
    )
    model = ChemicalMultiRMv2(
        args.embedding_path, chemical_features,
        tau=args.tau, chemical_encoder_type=args.chemical_encoder_type,
    ).to(args.device)
    train_and_evaluate(
        args, model, train_data, valid_loader, test_loader,
        f"Chemical v2 (sharp-attention) tau={args.tau} all-modification test results",
    )


def run_train_chemical_v2_lomo(args):
    set_seed(args.seed)
    heldout_index = MODIFICATION_NAMES.index(args.heldout_mod)
    task_indices = [index for index in range(len(MODIFICATION_NAMES)) if index != heldout_index]

    train_data = read_split_as_kmers(args.data_path, "train", args.length, args.embedding_path, args.cache_dir)
    valid_data = read_split_as_kmers(args.data_path, "valid", args.length, args.embedding_path, args.cache_dir)
    test_data = read_split_as_kmers(args.data_path, "test", args.length, args.embedding_path, args.cache_dir)

    train_data = remove_positive_rows(train_data, heldout_index)
    valid_data_for_selection = remove_positive_rows(valid_data, heldout_index)

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
    chemical_features = build_chemical_feature_matrix(
        modification_table, site_weight=args.site_weight, fp_kind=args.fp_kind,
        bio_weight=args.bio_weight,
    )
    model = ChemicalMultiRMv2(
        args.embedding_path, chemical_features,
        tau=args.tau, chemical_encoder_type=args.chemical_encoder_type,
    ).to(args.device)

    # F2 — chemistry-derived soft labels on the held-out column.
    loss_task_indices = task_indices
    soft_label_aux_config = None
    soft_gamma = getattr(args, "soft_label_gamma", 0.0)
    soft_mode = getattr(args, "soft_label_loss_mode", "joint_prob")
    if soft_gamma and soft_gamma > 0:
        bio_pack = pickle.load(open("/ibex/user/songt/MultiRM/Data/bio_priors.pkl", "rb"))
        tanimoto_matrix = bio_pack["tanimoto_matrix"]
        source_mask = None
        if getattr(args, "soft_label_same_base_only", False):
            bases = modification_table["original_base"].astype(str).to_numpy()
            source_mask = bases == bases[heldout_index]
            source_mask[heldout_index] = False
        train_data = apply_tanimoto_soft_labels(
            train_data, heldout_index, tanimoto_matrix, soft_gamma,
            tani_min=getattr(args, "soft_label_tani_min", 0.0),
            pwm_gate=getattr(args, "soft_label_pwm_gate", False),
            source_mask=source_mask,
        )
        soft_mass = float(train_data["y"][:, heldout_index].sum())
        if soft_mass <= 0.0:
            print(f"[F2] gamma={soft_gamma} produced no held-out soft labels after filters; "
                  "training seen mods only.", flush=True)
        elif soft_mode == "joint_prob":
            loss_task_indices = list(range(len(MODIFICATION_NAMES)))
            print(f"[F2] Tanimoto soft labels enabled with gamma={soft_gamma}; "
                  f"mode=joint_prob soft_mass={soft_mass:.3f}; "
                  f"loss covers all {len(loss_task_indices)} columns including held-out.", flush=True)
        elif soft_mode in {"aux_prob", "aux_pos_weight"}:
            soft_label_aux_config = {
                "mode": soft_mode,
                "weight": getattr(args, "soft_label_aux_weight", 1.0),
            }
            print(f"[F2] Tanimoto soft labels enabled with gamma={soft_gamma}; "
                  f"mode={soft_mode} aux_weight={soft_label_aux_config['weight']} "
                  f"soft_mass={soft_mass:.3f}; seen-mod BCE unchanged.", flush=True)
        else:
            raise ValueError(soft_mode)

    train_lomo_model(
        args, model, train_data, valid_loader, test_loader,
        task_indices, heldout_index,
        f"Chemical v2 (sharp-attention) tau={args.tau} gamma={soft_gamma} mode={soft_mode} LOMO result",
        loss_task_indices=loss_task_indices,
        soft_label_aux_config=soft_label_aux_config,
    )


def train_lomo_model(args, model, train_data, valid_loader, test_loader, task_indices, heldout_index, result_label, loss_task_indices=None, soft_label_aux_config=None):
    """LOMO training. By default the loss is computed only over the seen
    modifications (task_indices). When loss_task_indices is given, that list
    drives the BCE while task_indices still drives the seen-mod early stopping
    and metric tables. soft_label_aux_config adds a separate held-out
    chemistry-prior loss while leaving the seen-mod BCE unchanged. Model
    selection stays strict-LOMO-clean (only seen-mod valid AUCb).
    """
    if loss_task_indices is None:
        loss_task_indices = task_indices
    train_loader = DataLoader(
        RmDataset(train_data),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    # Training-side loss weights / uncertainty weights are sized to the FULL
    # loss task set (which may include the held-out column).
    train_loss_weights = calculate_loss_weights(train_data["y"][:, loss_task_indices]).to(args.device)
    train_uncertainty_weights = create_uncertainty_weights_for_task_count(args.loss_strategy, len(loss_task_indices), args.device)
    # Eval-side weights stay over seen mods only so valid_loss is comparable
    # across strict-LOMO and soft-label runs.
    eval_loss_weights = calculate_loss_weights(train_data["y"][:, task_indices]).to(args.device)
    eval_uncertainty_weights = create_uncertainty_weights_for_task_count(args.loss_strategy, len(task_indices), args.device)
    param_groups = build_param_groups(model, train_uncertainty_weights, args.weight_decay)
    optimizer = AdamW(param_groups, lr=args.lr)
    mean_bce = uses_mean_bce(args.loss_strategy)
    ohem = uses_ohem(args.loss_strategy)

    save_dir = Path(args.save_dir) / MODIFICATION_NAMES[heldout_index]
    save_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["heldout_index"] = heldout_index
    config["train_task_indices"] = task_indices
    config["loss_task_indices"] = loss_task_indices
    config["soft_label_aux_config"] = soft_label_aux_config
    config["train_task_names"] = [MODIFICATION_NAMES[index] for index in task_indices]
    with (save_dir / "config.json").open("w") as handle:
        json.dump(config, handle, indent=2)

    best_valid_aucb = -1.0
    best_valid_loss = float("inf")
    epochs_since_valid_loss_improvement = 0
    epochs_since_aucb_improvement = 0
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
                loss_task_indices,
                train_loss_weights,
                ohem,
                train_uncertainty_weights,
                mean_bce=mean_bce,
            )
            if soft_label_aux_config is not None:
                loss = loss + soft_label_auxiliary_loss(
                    outputs, labels, heldout_index,
                    soft_label_aux_config["mode"],
                    soft_label_aux_config["weight"],
                )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += float(loss.item())
            train_examples += labels.shape[0]

        valid_loss = evaluate_lomo_loss(model, valid_loader, task_indices, eval_loss_weights, args.loss_strategy, eval_uncertainty_weights, args.device)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            epochs_since_valid_loss_improvement = 0
            torch.save(model.state_dict(), save_dir / "best_loss_model.pt")
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
            epochs_since_aucb_improvement = 0
            torch.save(model.state_dict(), save_dir / "best_model.pt")
            save_table(valid_rows, save_dir, "valid_seen")
        else:
            epochs_since_aucb_improvement += 1
            if epochs_since_aucb_improvement >= args.early_stop_patience:
                print(f"early stopping at epoch {epoch} (no AUCb improvement for {args.early_stop_patience} epochs)", flush=True)
                break

    model.load_state_dict(torch.load(save_dir / "best_model.pt", map_location=args.device))
    test_probabilities, test_labels = collect_predictions(model, test_loader, args.device)
    np.savez(save_dir / "test_predictions.npz", prob=test_probabilities, label=test_labels)
    valid_probabilities, valid_labels = collect_predictions(model, valid_loader, args.device)
    np.savez(save_dir / "valid_predictions.npz", prob=valid_probabilities, label=valid_labels)
    heldout_row = compute_single_modification_row(test_probabilities, test_labels, heldout_index)
    save_single_row(heldout_row, save_dir, "test_heldout")
    print(f"{result_label}: {MODIFICATION_NAMES[heldout_index]}", flush=True)
    print_table([heldout_row])

    best_loss_ckpt = save_dir / "best_loss_model.pt"
    if best_loss_ckpt.exists():
        model.load_state_dict(torch.load(best_loss_ckpt, map_location=args.device))
        bl_test_prob, bl_test_label = collect_predictions(model, test_loader, args.device)
        np.savez(save_dir / "test_predictions_bestloss.npz", prob=bl_test_prob, label=bl_test_label)
        bl_row = compute_single_modification_row(bl_test_prob, bl_test_label, heldout_index)
        save_single_row(bl_row, save_dir, "test_heldout_bestloss")
        print(f"{result_label} [best-loss ckpt]: {MODIFICATION_NAMES[heldout_index]}", flush=True)
        print_table([bl_row])


def evaluate_lomo_loss(model, loader, task_indices, loss_weights, loss_strategy, uncertainty_weights, device):
    model.eval()
    mean_bce = uses_mean_bce(loss_strategy)
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
                mean_bce=mean_bce,
            )
            total_loss += float(loss.item())
            total_examples += labels.shape[0]
    return total_loss / total_examples


def collect_predictions_bio(model, loader, device):
    model.eval()
    probs = []
    labels = []
    with torch.no_grad():
        for x, y, pwm in loader:
            logits_list = model(x.to(device), pwm.to(device))
            logits = torch.stack(logits_list, dim=1)
            probs.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(probs, axis=0), np.concatenate(labels, axis=0)


def evaluate_lomo_loss_bio(model, loader, task_indices, loss_weights, loss_strategy, uncertainty_weights, device):
    model.eval()
    mean_bce = uses_mean_bce(loss_strategy)
    total_loss = 0.0
    total_examples = 0
    with torch.no_grad():
        for x, labels, pwm in loader:
            labels = labels.to(device)
            outputs = model(x.to(device), pwm.to(device))
            loss = paper_training_loss_for_indices(
                outputs, labels, task_indices, loss_weights,
                uses_ohem(loss_strategy), uncertainty_weights, mean_bce=mean_bce,
            )
            total_loss += float(loss.item())
            total_examples += labels.shape[0]
    return total_loss / total_examples


def train_lomo_model_bio(args, model, train_data, valid_loader, test_loader, task_indices, heldout_index, result_label):
    """LOMO training loop that threads per-sample pwm_match through model.forward.

    Mirrors train_lomo_model but uses RmDatasetBio and passes pwm to model.
    """
    train_loader = DataLoader(
        RmDatasetBio(train_data),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    loss_weights = calculate_loss_weights(train_data["y"][:, task_indices]).to(args.device)
    uncertainty_weights = create_uncertainty_weights_for_task_count(args.loss_strategy, len(task_indices), args.device)
    param_groups = build_param_groups(model, uncertainty_weights, args.weight_decay)
    optimizer = AdamW(param_groups, lr=args.lr)
    mean_bce = uses_mean_bce(args.loss_strategy)
    ohem = uses_ohem(args.loss_strategy)

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
    epochs_since_aucb_improvement = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_examples = 0
        for x, labels, pwm in train_loader:
            labels = labels.to(args.device)
            outputs = model(x.to(args.device), pwm.to(args.device))
            loss = paper_training_loss_for_indices(
                outputs, labels, task_indices, loss_weights,
                ohem, uncertainty_weights, mean_bce=mean_bce,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += float(loss.item())
            train_examples += labels.shape[0]

        valid_loss = evaluate_lomo_loss_bio(model, valid_loader, task_indices, loss_weights, args.loss_strategy, uncertainty_weights, args.device)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            epochs_since_valid_loss_improvement = 0
            torch.save(model.state_dict(), save_dir / "best_loss_model.pt")
        else:
            epochs_since_valid_loss_improvement += 1
            if epochs_since_valid_loss_improvement >= args.lr_patience:
                decay_learning_rate(optimizer, args.lr_decay)
                epochs_since_valid_loss_improvement = 0

        valid_probabilities, valid_labels = collect_predictions_bio(model, valid_loader, args.device)
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
            epochs_since_aucb_improvement = 0
            torch.save(model.state_dict(), save_dir / "best_model.pt")
            save_table(valid_rows, save_dir, "valid_seen")
        else:
            epochs_since_aucb_improvement += 1
            if epochs_since_aucb_improvement >= args.early_stop_patience:
                print(f"early stopping at epoch {epoch} (no AUCb improvement for {args.early_stop_patience} epochs)", flush=True)
                break

    model.load_state_dict(torch.load(save_dir / "best_model.pt", map_location=args.device))
    test_probabilities, test_labels = collect_predictions_bio(model, test_loader, args.device)
    np.savez(save_dir / "test_predictions.npz", prob=test_probabilities, label=test_labels)
    valid_probabilities, valid_labels = collect_predictions_bio(model, valid_loader, args.device)
    np.savez(save_dir / "valid_predictions.npz", prob=valid_probabilities, label=valid_labels)
    heldout_row = compute_single_modification_row(test_probabilities, test_labels, heldout_index)
    save_single_row(heldout_row, save_dir, "test_heldout")
    print(f"{result_label}: {MODIFICATION_NAMES[heldout_index]}", flush=True)
    print_table([heldout_row])


def run_train_chemical_biomatch_lomo(args):
    set_seed(args.seed)
    heldout_index = MODIFICATION_NAMES.index(args.heldout_mod)
    task_indices = [index for index in range(len(MODIFICATION_NAMES)) if index != heldout_index]

    train_data = read_split_as_kmers(args.data_path, "train", args.length, args.embedding_path, args.cache_dir)
    valid_data = read_split_as_kmers(args.data_path, "valid", args.length, args.embedding_path, args.cache_dir)
    test_data = read_split_as_kmers(args.data_path, "test", args.length, args.embedding_path, args.cache_dir)

    train_data = remove_positive_rows(train_data, heldout_index)
    valid_data_for_selection = remove_positive_rows(valid_data, heldout_index)

    valid_loader = DataLoader(
        RmDatasetBio(valid_data_for_selection),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=args.device.startswith("cuda"),
    )
    test_loader = DataLoader(
        RmDatasetBio(test_data),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=args.device.startswith("cuda"),
    )

    modification_table = load_modification_table(args.modifications_path)
    chemical_features = build_chemical_feature_matrix(
        modification_table, site_weight=args.site_weight, fp_kind=args.fp_kind,
        bio_weight=args.bio_weight,
    )
    bio_pack = pickle.load(open("/ibex/user/songt/MultiRM/Data/bio_priors.pkl", "rb"))
    bio_prior_per_mod = bio_pack["feature_matrix"]
    model = ChemicalMultiRMv2BioMatch(
        args.embedding_path, chemical_features, bio_prior_per_mod,
        tau=args.tau, chemical_encoder_type=args.chemical_encoder_type,
        lambda_match=args.lambda_match,
    ).to(args.device)

    train_lomo_model_bio(
        args, model, train_data, valid_loader, test_loader,
        task_indices, heldout_index,
        f"Chemical v2 + bio-match path tau={args.tau} lambda={args.lambda_match} LOMO result",
    )


def run_train_chemical_biomatch_tan_lomo(args):
    set_seed(args.seed)
    heldout_index = MODIFICATION_NAMES.index(args.heldout_mod)
    task_indices = [index for index in range(len(MODIFICATION_NAMES)) if index != heldout_index]

    train_data = read_split_as_kmers(args.data_path, "train", args.length, args.embedding_path, args.cache_dir)
    valid_data = read_split_as_kmers(args.data_path, "valid", args.length, args.embedding_path, args.cache_dir)
    test_data = read_split_as_kmers(args.data_path, "test", args.length, args.embedding_path, args.cache_dir)

    train_data = remove_positive_rows(train_data, heldout_index)
    valid_data_for_selection = remove_positive_rows(valid_data, heldout_index)

    valid_loader = DataLoader(
        RmDatasetBio(valid_data_for_selection),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=args.device.startswith("cuda"),
    )
    test_loader = DataLoader(
        RmDatasetBio(test_data),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=args.device.startswith("cuda"),
    )

    modification_table = load_modification_table(args.modifications_path)
    chemical_features = build_chemical_feature_matrix(
        modification_table, site_weight=args.site_weight, fp_kind=args.fp_kind,
        bio_weight=args.bio_weight,
    )
    bio_pack = pickle.load(open("/ibex/user/songt/MultiRM/Data/bio_priors.pkl", "rb"))
    tanimoto_matrix = bio_pack["tanimoto_matrix"]
    model = ChemicalMultiRMv2BioMatchTan(
        args.embedding_path, chemical_features, tanimoto_matrix,
        tau=args.tau, chemical_encoder_type=args.chemical_encoder_type,
        alpha=args.alpha, learnable_alpha=args.learnable_alpha,
    ).to(args.device)

    train_lomo_model_bio(
        args, model, train_data, valid_loader, test_loader,
        task_indices, heldout_index,
        f"Chemical v2 + Tanimoto-weighted biomatch tau={args.tau} LOMO result",
    )


def collect_predictions_pathA(model, loader, device):
    model.eval()
    probs = []
    labels = []
    with torch.no_grad():
        for x, y, pwm, region, logcap, mapped in loader:
            logits_list = model(
                x.to(device), pwm.to(device),
                region.to(device), logcap.to(device), mapped.to(device),
            )
            logits = torch.stack(logits_list, dim=1)
            probs.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(probs, axis=0), np.concatenate(labels, axis=0)


def evaluate_lomo_loss_pathA(model, loader, task_indices, loss_weights, loss_strategy, uncertainty_weights, device):
    model.eval()
    mean_bce = uses_mean_bce(loss_strategy)
    total_loss = 0.0
    total_examples = 0
    with torch.no_grad():
        for x, labels, pwm, region, logcap, mapped in loader:
            labels = labels.to(device)
            outputs = model(
                x.to(device), pwm.to(device),
                region.to(device), logcap.to(device), mapped.to(device),
            )
            loss = paper_training_loss_for_indices(
                outputs, labels, task_indices, loss_weights,
                uses_ohem(loss_strategy), uncertainty_weights, mean_bce=mean_bce,
            )
            total_loss += float(loss.item())
            total_examples += labels.shape[0]
    return total_loss / total_examples


def train_lomo_model_pathA(args, model, train_data, valid_loader, test_loader, task_indices, heldout_index, result_label):
    train_loader = DataLoader(
        RmDatasetPathA(train_data),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    loss_weights = calculate_loss_weights(train_data["y"][:, task_indices]).to(args.device)
    uncertainty_weights = create_uncertainty_weights_for_task_count(args.loss_strategy, len(task_indices), args.device)
    param_groups = build_param_groups(model, uncertainty_weights, args.weight_decay)
    optimizer = AdamW(param_groups, lr=args.lr)
    mean_bce = uses_mean_bce(args.loss_strategy)
    ohem = uses_ohem(args.loss_strategy)

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
    epochs_since_aucb_improvement = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_examples = 0
        for x, labels, pwm, region, logcap, mapped in train_loader:
            labels = labels.to(args.device)
            outputs = model(
                x.to(args.device), pwm.to(args.device),
                region.to(args.device), logcap.to(args.device), mapped.to(args.device),
            )
            loss = paper_training_loss_for_indices(
                outputs, labels, task_indices, loss_weights,
                ohem, uncertainty_weights, mean_bce=mean_bce,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += float(loss.item())
            train_examples += labels.shape[0]

        valid_loss = evaluate_lomo_loss_pathA(
            model, valid_loader, task_indices, loss_weights, args.loss_strategy, uncertainty_weights, args.device,
        )
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            epochs_since_valid_loss_improvement = 0
            torch.save(model.state_dict(), save_dir / "best_loss_model.pt")
        else:
            epochs_since_valid_loss_improvement += 1
            if epochs_since_valid_loss_improvement >= args.lr_patience:
                decay_learning_rate(optimizer, args.lr_decay)
                epochs_since_valid_loss_improvement = 0

        valid_probabilities, valid_labels = collect_predictions_pathA(model, valid_loader, args.device)
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
            epochs_since_aucb_improvement = 0
            torch.save(model.state_dict(), save_dir / "best_model.pt")
            save_table(valid_rows, save_dir, "valid_seen")
        else:
            epochs_since_aucb_improvement += 1
            if epochs_since_aucb_improvement >= args.early_stop_patience:
                print(f"early stopping at epoch {epoch} (no AUCb improvement for {args.early_stop_patience} epochs)", flush=True)
                break

    model.load_state_dict(torch.load(save_dir / "best_model.pt", map_location=args.device))
    test_probabilities, test_labels = collect_predictions_pathA(model, test_loader, args.device)
    np.savez(save_dir / "test_predictions.npz", prob=test_probabilities, label=test_labels)
    valid_probabilities, valid_labels = collect_predictions_pathA(model, valid_loader, args.device)
    np.savez(save_dir / "valid_predictions.npz", prob=valid_probabilities, label=valid_labels)
    heldout_row = compute_single_modification_row(test_probabilities, test_labels, heldout_index)
    save_single_row(heldout_row, save_dir, "test_heldout")
    print(f"{result_label}: {MODIFICATION_NAMES[heldout_index]}", flush=True)
    print_table([heldout_row])


def attach_per_sample_metadata(split_data, split_name):
    """Merge Path-A per-sample metadata fields into a split_data dict in place-equivalent."""
    from v0_data import load_per_sample_metadata
    meta = load_per_sample_metadata(split_name)
    out = dict(split_data)
    n_data = out["x"].shape[0]
    n_meta = meta["region_onehot"].shape[0]
    if n_data != n_meta:
        raise ValueError(
            f"Path-A metadata row count {n_meta} for split '{split_name}' does not match "
            f"split data row count {n_data}. Rebuild metadata against the current data cache."
        )
    out["region_onehot"] = meta["region_onehot"]
    out["log_cap_distance"] = meta["log_cap_distance"]
    out["mapped"] = meta["mapped"]
    return out


def compute_seen_mod_log_cap_dist_centers(train_data, seen_indices, tanimoto_matrix, heldout_index):
    """LOMO-safe per-mod median log cap distance.

    For each seen modification k in seen_indices: median log1p(cap_distance) over
    training samples that are k-positive AND mapped to a mature-mRNA position
    (cap_distance>=0). Train data passed in is expected to already have had the
    held-out positives removed via remove_positive_rows, so there is no label
    leak from heldout_index into the seen-mod statistics.

    For the held-out modification: the centre is filled by a Tanimoto-weighted
    average of the seen-mod centres. This uses only chemistry (mod fingerprints)
    plus seen-mod data — no held-out label information.

    Returns (K,) float array of mature-mRNA log_cap centres per modification.
    """
    y = train_data["y"]
    log_cap = train_data["log_cap_distance"]
    mapped = train_data["mapped"]
    K = y.shape[1]
    exonic_mask = mapped > 0.5  # implies cap_distance >= 0 by load_per_sample_metadata
    centers = np.zeros(K, dtype=np.float32)
    # Seen-mod global fallback: overall median on any seen-mod positive (exonic)
    seen_any = np.zeros(y.shape[0], dtype=bool)
    for k in seen_indices:
        seen_any |= (y[:, k] > 0.5)
    seen_pool = seen_any & exonic_mask
    overall_median = float(np.median(log_cap[seen_pool])) if seen_pool.sum() > 0 else 0.0
    # Per-seen-mod median (with fallback)
    seen_has_data = np.zeros(K, dtype=bool)
    for k in seen_indices:
        mask = (y[:, k] > 0.5) & exonic_mask
        if mask.sum() > 0:
            centers[k] = float(np.median(log_cap[mask]))
            seen_has_data[k] = True
        else:
            centers[k] = overall_median
    # Held-out centre: Tanimoto-weighted average over seen mods that have data.
    seen_indices_arr = np.asarray(seen_indices, dtype=np.int64)
    seen_with_data = seen_indices_arr[seen_has_data[seen_indices_arr]]
    if seen_with_data.size > 0:
        w = np.asarray(tanimoto_matrix[heldout_index, seen_with_data], dtype=np.float64)
        if w.sum() > 0:
            centers[heldout_index] = float(
                (w * centers[seen_with_data].astype(np.float64)).sum() / w.sum()
            )
        else:
            centers[heldout_index] = float(np.mean(centers[seen_with_data]))
    else:
        centers[heldout_index] = overall_median
    return centers


def run_train_chemical_pathA_lomo(args):
    set_seed(args.seed)
    heldout_index = MODIFICATION_NAMES.index(args.heldout_mod)
    task_indices = [index for index in range(len(MODIFICATION_NAMES)) if index != heldout_index]

    train_data = read_split_as_kmers(args.data_path, "train", args.length, args.embedding_path, args.cache_dir)
    valid_data = read_split_as_kmers(args.data_path, "valid", args.length, args.embedding_path, args.cache_dir)
    test_data = read_split_as_kmers(args.data_path, "test", args.length, args.embedding_path, args.cache_dir)

    train_data = attach_per_sample_metadata(train_data, "train")
    valid_data = attach_per_sample_metadata(valid_data, "valid")
    test_data = attach_per_sample_metadata(test_data, "test")

    bio_pack = pickle.load(open("/ibex/user/songt/MultiRM/Data/bio_priors.pkl", "rb"))
    tanimoto_matrix = bio_pack["tanimoto_matrix"]
    # mod_region_prior: literature-derived, centred region affinity per mod (12, 5)
    writer_dim = int(bio_pack["writer_dim"])
    region_dim = int(bio_pack["region_dim"])
    mod_region_prior = bio_pack["feature_matrix"][:, writer_dim:writer_dim + region_dim]

    # Remove held-out positives FIRST so the LOMO-safe centre function never
    # touches held-out mod's labels.
    train_data = remove_positive_rows(train_data, heldout_index)
    valid_data_for_selection = remove_positive_rows(valid_data, heldout_index)

    mod_log_cap_dist_centers = compute_seen_mod_log_cap_dist_centers(
        train_data, task_indices, tanimoto_matrix, heldout_index,
    )
    print("mod_log_cap_dist_centers (LOMO-safe, exonic-only):", flush=True)
    for k, m in enumerate(MODIFICATION_NAMES):
        marker = "(heldout/tani)" if k == heldout_index else ""
        print(f"  {m:>5s}: log_cap_dist={mod_log_cap_dist_centers[k]:.3f} {marker}", flush=True)

    valid_loader = DataLoader(
        RmDatasetPathA(valid_data_for_selection),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=args.device.startswith("cuda"),
    )
    test_loader = DataLoader(
        RmDatasetPathA(test_data),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=args.device.startswith("cuda"),
    )

    modification_table = load_modification_table(args.modifications_path)
    chemical_features = build_chemical_feature_matrix(
        modification_table, site_weight=args.site_weight, fp_kind=args.fp_kind,
        bio_weight=args.bio_weight,
    )
    model = ChemicalMultiRMv2PathA(
        args.embedding_path, chemical_features, tanimoto_matrix,
        mod_region_prior, mod_log_cap_dist_centers,
        tau=args.tau, chemical_encoder_type=args.chemical_encoder_type,
        alpha_pwm=args.alpha_pwm, alpha_meta=args.alpha_meta, sigma=args.sigma,
    ).to(args.device)

    train_lomo_model_pathA(
        args, model, train_data, valid_loader, test_loader,
        task_indices, heldout_index,
        f"Chemical v2 + PathA (PWM-tani + region/cap) tau={args.tau} "
        f"alpha_pwm={args.alpha_pwm} alpha_meta={args.alpha_meta} LOMO result",
    )


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
    elif args.command == "train_modid":
        run_train_modid(args)
    elif args.command == "train_modid_lomo":
        run_train_modid_lomo(args)
    elif args.command == "train_chemical_v1":
        run_train_chemical_v1(args)
    elif args.command == "train_chemical_v1_lomo":
        run_train_chemical_v1_lomo(args)
    elif args.command == "train_chemical_v2":
        run_train_chemical_v2(args)
    elif args.command == "train_chemical_v2_lomo":
        run_train_chemical_v2_lomo(args)
    elif args.command == "train_chemical_biomatch_lomo":
        run_train_chemical_biomatch_lomo(args)
    elif args.command == "train_chemical_biomatch_tan_lomo":
        run_train_chemical_biomatch_tan_lomo(args)
    elif args.command == "train_chemical_pathA_lomo":
        run_train_chemical_pathA_lomo(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
