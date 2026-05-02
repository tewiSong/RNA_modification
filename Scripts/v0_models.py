import math

import torch
from torch import nn


class SequenceEncoder(nn.Module):
    def __init__(self, input_dim=4, hidden_size=256, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_dim = hidden_size * 2
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x):
        hidden_states, _ = self.lstm(x)
        center_state = hidden_states[:, hidden_states.shape[1] // 2, :]
        return hidden_states, center_state


class InteractionScorer(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, center_states, context_states, modification_states):
        batch_size, num_modifications, hidden_dim = context_states.shape
        expanded_center = center_states[:, None, :].expand(batch_size, num_modifications, hidden_dim)
        expanded_modifications = modification_states[None, :, :].expand(batch_size, num_modifications, hidden_dim)
        features = torch.cat(
            [
                expanded_center,
                context_states,
                expanded_modifications,
                expanded_center * expanded_modifications,
            ],
            dim=-1,
        )
        logits = self.net(features.reshape(batch_size * num_modifications, hidden_dim * 4))
        return logits.reshape(batch_size, num_modifications)


class QueryFusion(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.query_projection = nn.Linear(hidden_dim, hidden_dim)
        self.key_projection = nn.Linear(hidden_dim, hidden_dim)
        self.value_projection = nn.Linear(hidden_dim, hidden_dim)
        self.scale = math.sqrt(hidden_dim)

    def forward(self, hidden_states, query_states):
        queries = self.query_projection(query_states)
        keys = self.key_projection(hidden_states)
        values = self.value_projection(hidden_states)
        attention_scores = torch.einsum("md,bld->bml", queries, keys) / self.scale
        attention_weights = torch.softmax(attention_scores, dim=-1)
        context_states = torch.einsum("bml,bld->bmd", attention_weights, values)
        return context_states


class ChemicalConditionedRM(nn.Module):
    def __init__(self, chemical_features, hidden_size=256, chem_hidden_size=512, dropout=0.3):
        super().__init__()
        hidden_dim = hidden_size * 2
        self.sequence_encoder = SequenceEncoder(hidden_size=hidden_size)
        self.register_buffer("chemical_features", torch.as_tensor(chemical_features, dtype=torch.float32))
        self.chemical_encoder = nn.Sequential(
            nn.Linear(chemical_features.shape[1], chem_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(chem_hidden_size, hidden_dim),
        )
        self.fusion = QueryFusion(hidden_dim)
        self.scorer = InteractionScorer(hidden_dim, dropout)

    def forward(self, x):
        hidden_states, center_state = self.sequence_encoder(x)
        modification_states = self.chemical_encoder(self.chemical_features)
        context_states = self.fusion(hidden_states, modification_states)
        return self.scorer(center_state, context_states, modification_states)


class ModificationIdConditionedRM(nn.Module):
    def __init__(self, num_modifications=12, hidden_size=256, dropout=0.3):
        super().__init__()
        hidden_dim = hidden_size * 2
        self.sequence_encoder = SequenceEncoder(hidden_size=hidden_size)
        self.modification_embedding = nn.Embedding(num_modifications, hidden_dim)
        self.register_buffer("modification_indices", torch.arange(num_modifications, dtype=torch.long))
        self.fusion = QueryFusion(hidden_dim)
        self.scorer = InteractionScorer(hidden_dim, dropout)

    def forward(self, x):
        hidden_states, center_state = self.sequence_encoder(x)
        modification_states = self.modification_embedding(self.modification_indices)
        context_states = self.fusion(hidden_states, modification_states)
        return self.scorer(center_state, context_states, modification_states)


class SequenceOnlyRM(nn.Module):
    def __init__(self, num_modifications=12, hidden_size=256, attention_hidden_size=100, dropout=0.3):
        super().__init__()
        hidden_dim = hidden_size * 2
        self.num_modifications = num_modifications
        self.sequence_encoder = SequenceEncoder(hidden_size=hidden_size)
        self.value_projection = nn.Linear(hidden_dim, attention_hidden_size)
        self.center_projection = nn.Linear(hidden_dim, attention_hidden_size)
        self.attention_vector = nn.Linear(attention_hidden_size, num_modifications)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 1),
            )
            for _ in range(num_modifications)
        ])

    def forward(self, x):
        hidden_states, center_state = self.sequence_encoder(x)
        value_term = self.value_projection(hidden_states)
        center_term = self.center_projection(center_state)[:, None, :]
        attention_scores = self.attention_vector(torch.tanh(value_term + center_term))
        attention_weights = torch.softmax(attention_scores, dim=1)
        context_states = torch.einsum("bld,blm->bmd", hidden_states, attention_weights)
        logits = []
        for index, head in enumerate(self.heads):
            logits.append(head(context_states[:, index, :]).squeeze(-1))
        return torch.stack(logits, dim=1)
