# -*- coding: utf-8 -*-
"""
BiLSTM + CRF cho VLSP NER.
Không phụ thuộc torchcrf.
"""

from typing import List, Optional

import torch
import torch.nn as nn


class LinearChainCRF(nn.Module):
    def __init__(self, num_tags: int):
        super().__init__()
        self.num_tags = num_tags
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def forward(self,
                emissions: torch.Tensor,
                tags: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        numerator = self._score_sentence(emissions, tags, mask)
        denominator = self._compute_log_partition(emissions, mask)
        return (denominator - numerator).mean()

    def decode(self,
               emissions: torch.Tensor,
               mask: torch.Tensor) -> List[List[int]]:
        return self._viterbi_decode(emissions, mask)

    def _score_sentence(self,
                        emissions: torch.Tensor,
                        tags: torch.Tensor,
                        mask: torch.Tensor) -> torch.Tensor:
        mask = mask.bool()
        first_tags = tags[:, 0]
        score = self.start_transitions[first_tags]
        score = score + emissions[:, 0].gather(1, first_tags.unsqueeze(1)).squeeze(1)

        for t in range(1, emissions.size(1)):
            current_mask = mask[:, t]
            prev_tags = tags[:, t - 1]
            current_tags = tags[:, t]
            transition_score = self.transitions[prev_tags, current_tags]
            emission_score = emissions[:, t].gather(1, current_tags.unsqueeze(1)).squeeze(1)
            score = score + (transition_score + emission_score) * current_mask

        lengths = mask.long().sum(dim=1) - 1
        last_tags = tags.gather(1, lengths.unsqueeze(1)).squeeze(1)
        score = score + self.end_transitions[last_tags]
        return score

    def _compute_log_partition(self,
                               emissions: torch.Tensor,
                               mask: torch.Tensor) -> torch.Tensor:
        mask = mask.bool()
        score = self.start_transitions + emissions[:, 0]

        for t in range(1, emissions.size(1)):
            broadcast_score = score.unsqueeze(2)
            broadcast_emission = emissions[:, t].unsqueeze(1)
            next_score = broadcast_score + self.transitions.unsqueeze(0) + broadcast_emission
            next_score = torch.logsumexp(next_score, dim=1)
            score = torch.where(mask[:, t].unsqueeze(1), next_score, score)

        score = score + self.end_transitions
        return torch.logsumexp(score, dim=1)

    def _viterbi_decode(self,
                        emissions: torch.Tensor,
                        mask: torch.Tensor) -> List[List[int]]:
        batch_size, seq_len, _ = emissions.shape
        mask = mask.bool()
        score = self.start_transitions + emissions[:, 0]
        history = []

        for t in range(1, seq_len):
            next_score = score.unsqueeze(2) + self.transitions.unsqueeze(0)
            best_score, best_path = next_score.max(dim=1)
            best_score = best_score + emissions[:, t]
            score = torch.where(mask[:, t].unsqueeze(1), best_score, score)
            history.append(best_path)

        score = score + self.end_transitions
        _, best_last_tag = score.max(dim=1)
        lengths = mask.long().sum(dim=1)
        predictions: List[List[int]] = []

        for i in range(batch_size):
            seq_end = lengths[i].item()
            last_tag = best_last_tag[i].item()
            best_tags = [last_tag]

            for hist_t in range(seq_end - 2, -1, -1):
                last_tag = history[hist_t][i][last_tag].item()
                best_tags.append(last_tag)

            best_tags.reverse()
            predictions.append(best_tags)

        return predictions


class BiLSTMCRF(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 num_labels: int,
                 embedding_dim: int = 128,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 dropout: float = 0.3,
                 pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx,
        )
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.emission = nn.Linear(hidden_dim, num_labels)
        self.crf = LinearChainCRF(num_labels)

    def forward(self,
                word_ids: torch.Tensor,
                mask: torch.Tensor,
                labels: Optional[torch.Tensor] = None):
        embedded = self.embedding(word_ids)
        encoded, _ = self.encoder(embedded)
        emissions = self.emission(self.dropout(encoded))

        loss = None
        if labels is not None:
            loss = self.crf(emissions=emissions, tags=labels, mask=mask)

        predictions = self.crf.decode(emissions=emissions, mask=mask)
        return loss, predictions
