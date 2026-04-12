# -*- coding: utf-8 -*-
"""
Training loop cho baseline NER.
"""

from pathlib import Path
from typing import Dict, List

import torch
from torch.optim import Adam
from tqdm import tqdm

from src.metrics import ner_scores


class BaselineNERTrainer:
    def __init__(self,
                 model,
                 optimizer,
                 id2label: Dict[int, str],
                 device: str,
                 output_dir: str):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.id2label = id2label
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.best_f1 = -1.0

    def _move_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        moved = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(self.device)
            else:
                moved[key] = value
        return moved

    def train_epoch(self, loader) -> float:
        self.model.train()
        total_loss = 0.0

        for batch in tqdm(loader, desc='train', leave=False):
            batch = self._move_batch(batch)
            self.optimizer.zero_grad()
            loss, _ = self.model(
                word_ids=batch['word_ids'],
                char_ids=batch['char_ids'],
                mask=batch['mask'],
                labels=batch['label_ids'],
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / max(1, len(loader))

    def evaluate(self, loader) -> Dict:
        self.model.eval()
        all_true: List[List[str]] = []
        all_pred: List[List[str]] = []
        total_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(loader, desc='eval', leave=False):
                batch = self._move_batch(batch)
                loss, predictions = self.model(
                    word_ids=batch['word_ids'],
                    char_ids=batch['char_ids'],
                    mask=batch['mask'],
                    labels=batch['label_ids'],
                )
                total_loss += loss.item()

                masks = batch['mask'].detach().cpu()
                labels = batch['label_ids'].detach().cpu()

                for seq_labels, seq_preds, seq_mask in zip(labels, predictions, masks):
                    valid_positions = seq_mask.bool()
                    true_ids = seq_labels[valid_positions].tolist()
                    pred_ids = list(seq_preds)
                    all_true.append([self.id2label[idx] for idx in true_ids])
                    all_pred.append([self.id2label[idx] for idx in pred_ids])

        scores = ner_scores(all_true, all_pred)
        scores['loss'] = total_loss / max(1, len(loader))
        return scores

    def fit(self, train_loader, dev_loader, epochs: int) -> List[Dict]:
        history: List[Dict] = []

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            dev_metrics = self.evaluate(dev_loader)
            row = {
                'epoch': epoch,
                'train_loss': train_loss,
                'dev_loss': dev_metrics['loss'],
                'precision': dev_metrics['precision'],
                'recall': dev_metrics['recall'],
                'f1': dev_metrics['f1'],
            }
            history.append(row)

            print(
                f"Epoch {epoch:02d} | "
                f"train_loss={train_loss:.4f} | "
                f"dev_loss={dev_metrics['loss']:.4f} | "
                f"P={dev_metrics['precision']:.4f} | "
                f"R={dev_metrics['recall']:.4f} | "
                f"F1={dev_metrics['f1']:.4f}"
            )

            if dev_metrics['f1'] > self.best_f1:
                self.best_f1 = dev_metrics['f1']
                torch.save(
                    {
                        'model_state_dict': self.model.state_dict(),
                        'best_f1': self.best_f1,
                    },
                    self.output_dir / 'best_model.pt',
                )
                print(f"  saved -> {self.output_dir / 'best_model.pt'}")

        return history


def build_adam(model, lr: float = 1e-3) -> Adam:
    return Adam(model.parameters(), lr=lr)
