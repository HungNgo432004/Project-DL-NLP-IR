# -*- coding: utf-8 -*-
"""
BiLSTM + CharCNN + CRF cho VLSP NER.
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

from src.bilstm_crf import LinearChainCRF


class CharCNNEncoder(nn.Module):
    def __init__(self,
                 char_vocab_size: int,
                 char_embedding_dim: int = 32,
                 char_channels: int = 64,
                 kernel_size: int = 3,
                 char_pad_idx: int = 0):
        super().__init__()
        self.char_embedding = nn.Embedding(
            num_embeddings=char_vocab_size,
            embedding_dim=char_embedding_dim,
            padding_idx=char_pad_idx,
        )
        self.conv = nn.Conv1d(
            in_channels=char_embedding_dim,
            out_channels=char_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.activation = nn.GELU()

    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, max_word = char_ids.shape
        chars = char_ids.reshape(batch_size * seq_len, max_word)
        embedded = self.char_embedding(chars).transpose(1, 2)
        conv_out = self.activation(self.conv(embedded))
        pooled = conv_out.max(dim=2).values
        return pooled.reshape(batch_size, seq_len, -1)


class BiLSTMCharCRF(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 char_vocab_size: int,
                 num_labels: int,
                 embedding_dim: int = 128,
                 char_embedding_dim: int = 32,
                 char_channels: int = 64,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 dropout: float = 0.3,
                 pad_idx: int = 0,
                 char_pad_idx: int = 0):
        super().__init__()
        self.word_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx,
        )
        self.char_encoder = CharCNNEncoder(
            char_vocab_size=char_vocab_size,
            char_embedding_dim=char_embedding_dim,
            char_channels=char_channels,
            char_pad_idx=char_pad_idx,
        )

        lstm_input_dim = embedding_dim + char_channels
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            input_size=lstm_input_dim,
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
                char_ids: torch.Tensor,
                mask: torch.Tensor,
                labels: Optional[torch.Tensor] = None):
        word_emb = self.word_embedding(word_ids)
        char_emb = self.char_encoder(char_ids)
        features = torch.cat([word_emb, char_emb], dim=-1)

        lengths = mask.long().sum(dim=1).cpu()
        packed = rnn_utils.pack_padded_sequence(
            features,
            lengths=lengths,
            batch_first=True,
            enforce_sorted=True,
        )
        packed_out, _ = self.encoder(packed)
        encoded, _ = rnn_utils.pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=word_ids.size(1),
        )

        emissions = self.emission(self.dropout(encoded))

        loss = None
        if labels is not None:
            loss = self.crf(emissions=emissions, tags=labels, mask=mask)

        predictions = self.crf.decode(emissions=emissions, mask=mask)
        return loss, predictions
