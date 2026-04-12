# -*- coding: utf-8 -*-
"""
Train baseline BiLSTM + CRF cho VLSP NER.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.bilstm_char_crf import BiLSTMCharCRF
from src.data_loader import (
    build_char_vocab,
    build_label_vocab,
    build_word_vocab,
    load_folder,
    normalize_bio,
    save_vocabs,
    train_dev_split,
)
from src.dataset import VLSPDatasetBiLSTM, bilstm_collate_fn
from src.trainer import BaselineNERTrainer, build_adam


if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train baseline BiLSTM + CRF cho VLSP NER')
    parser.add_argument('--data-dir', default='Data', help='Thu muc chua train/ va test/')
    parser.add_argument('--output-dir', default='output/bilstm_crf_baseline',
                        help='Noi luu model, vocab va history')
    parser.add_argument('--dev-ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--max-seq', type=int, default=128)
    parser.add_argument('--max-word', type=int, default=30)
    parser.add_argument('--embedding-dim', type=int, default=128)
    parser.add_argument('--char-embedding-dim', type=int, default=32)
    parser.add_argument('--char-channels', type=int, default=64)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--min-freq', type=int, default=1)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    data_dir = project_root / args.data_dir
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 72)
    print('VLSP NER BiLSTM + CRF Training')
    print('=' * 72)
    print(f'data_dir   : {data_dir}')
    print(f'output_dir : {output_dir}')
    print(f'device     : {args.device}')

    train_all = normalize_bio(
        load_folder(str(data_dir / 'train'), has_labels=True, verbose=True)
    )
    train_sents, dev_sents = train_dev_split(
        train_all,
        dev_ratio=args.dev_ratio,
        seed=args.seed,
    )

    label2id, id2label = build_label_vocab(train_sents)
    word2id, _ = build_word_vocab(train_sents, min_freq=args.min_freq)
    char2id, _ = build_char_vocab(train_sents)

    save_vocabs(label2id, word2id, char2id, str(output_dir / 'vocabs.pkl'))

    train_ds = VLSPDatasetBiLSTM(
        train_sents, word2id, char2id, label2id,
        max_seq=args.max_seq, max_word=args.max_word,
    )
    dev_ds = VLSPDatasetBiLSTM(
        dev_sents, word2id, char2id, label2id,
        max_seq=args.max_seq, max_word=args.max_word,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=bilstm_collate_fn,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=bilstm_collate_fn,
    )

    model = BiLSTMCharCRF(
        vocab_size=len(word2id),
        char_vocab_size=len(char2id),
        num_labels=len(label2id),
        embedding_dim=args.embedding_dim,
        char_embedding_dim=args.char_embedding_dim,
        char_channels=args.char_channels,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    optimizer = build_adam(model, lr=args.lr)

    trainer = BaselineNERTrainer(
        model=model,
        optimizer=optimizer,
        id2label=id2label,
        device=args.device,
        output_dir=str(output_dir),
    )
    history = trainer.fit(train_loader, dev_loader, epochs=args.epochs)

    with open(output_dir / 'history.json', 'w', encoding='utf-8') as fh:
        json.dump(history, fh, ensure_ascii=False, indent=2)

    print('\nPer-type F1 o epoch cuoi:')
    final_metrics = trainer.evaluate(dev_loader)
    for entity_type, stats in final_metrics['per_type'].items():
        print(
            f"  {entity_type:<5} "
            f"P={stats['precision']:.4f} "
            f"R={stats['recall']:.4f} "
            f"F1={stats['f1']:.4f} "
            f"support={stats['support']}"
        )

    print('\nHoan tat.')
    print(f"Best model : {output_dir / 'best_model.pt'}")
    print(f"History    : {output_dir / 'history.json'}")


if __name__ == '__main__':
    main()
