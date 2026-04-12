# -*- coding: utf-8 -*-
# src/dataset.py
"""
PyTorch Dataset classes cho VLSP NER.

  VLSPDatasetBiLSTM      — BiLSTM-CRF baseline (word + char features)
  VLSPDatasetPhoBERT     — PhoBERT + CRF main model
  VLSPTestDatasetPhoBERT — PhoBERT inference (test, không có label)
  bilstm_collate_fn      — collate function cho DataLoader (BiLSTM)

Lưu ý đặc thù VLSP:
  Token VLSP dùng '_' ghép âm tiết: nhà_hàng, Đồng_Xoài.
  PhoBERT cần chuỗi thường → PHẢI replace '_' → ' ' trước khi tokenize.
  Bỏ bước này mất ~2.5% F1.
"""

import torch
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple


# ════════════════════════════════════════════════════════════════════════════ #
#  1.  BiLSTM-CRF Dataset                                                     #
# ════════════════════════════════════════════════════════════════════════════ #

class VLSPDatasetBiLSTM(Dataset):
    """
    Dataset cho BiLSTM-CRF.

    Dùng kèm với bilstm_collate_fn để batch các câu có độ dài khác nhau.

    Output __getitem__:
        word_ids  : (seq_len,)            — word indices
        char_ids  : (seq_len, max_word)   — char indices per token
        label_ids : (seq_len,)            — NER label indices
        mask      : (seq_len,)  bool      — 1=real token, 0=padding
        seq_len   : int                   — độ dài thực (chưa pad)
    """

    def __init__(
        self,
        sentences : List[Dict],
        word2id   : Dict[str, int],
        char2id   : Dict[str, int],
        label2id  : Dict[str, int],
        max_seq   : int = 256,
        max_word  : int = 30,
    ):
        self.data     = sentences
        self.word2id  = word2id
        self.char2id  = char2id
        self.label2id = label2id
        self.max_seq  = max_seq
        self.max_word = max_word

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        sent   = self.data[idx]
        tokens = sent['tokens'][:self.max_seq]
        labels = sent['labels'][:self.max_seq]
        L      = len(tokens)

        # Word IDs — <UNK>=1, <PAD>=0
        wids = [self.word2id.get(t, 1) for t in tokens]
        wids += [0] * (self.max_seq - L)

        # Char IDs — shape (max_seq, max_word)
        cids = []
        for t in tokens:
            row = [self.char2id.get(c, 1) for c in t[:self.max_word]]
            row += [0] * (self.max_word - len(row))
            cids.append(row)
        cids += [[0] * self.max_word] * (self.max_seq - L)

        # Label IDs
        lids = [self.label2id.get(lbl, 0) for lbl in labels]
        lids += [0] * (self.max_seq - L)

        # Attention mask
        mask = [1] * L + [0] * (self.max_seq - L)

        return {
            'word_ids':  torch.tensor(wids, dtype=torch.long),
            'char_ids':  torch.tensor(cids, dtype=torch.long),
            'label_ids': torch.tensor(lids, dtype=torch.long),
            'mask':      torch.tensor(mask, dtype=torch.bool),
            'seq_len':   L,
        }


def bilstm_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function cho DataLoader khi dùng VLSPDatasetBiLSTM.

    Sắp xếp batch theo seq_len giảm dần (yêu cầu của pack_padded_sequence).
    Các tensor đã được padding đến max_seq trong Dataset nên không cần
    pad thêm ở đây — chỉ cần stack và trả về.

    Dùng:
        DataLoader(ds, batch_size=32, collate_fn=bilstm_collate_fn)
    """
    # Sắp xếp theo độ dài giảm dần
    batch = sorted(batch, key=lambda x: x['seq_len'], reverse=True)

    return {
        'word_ids':  torch.stack([b['word_ids']  for b in batch]),
        'char_ids':  torch.stack([b['char_ids']  for b in batch]),
        'label_ids': torch.stack([b['label_ids'] for b in batch]),
        'mask':      torch.stack([b['mask']      for b in batch]),
        'seq_lens':  torch.tensor([b['seq_len']  for b in batch],
                                   dtype=torch.long),
    }


# ════════════════════════════════════════════════════════════════════════════ #
#  2.  PhoBERT Dataset (train / dev)                                          #
# ════════════════════════════════════════════════════════════════════════════ #

class VLSPDatasetPhoBERT(Dataset):
    """
    Dataset cho PhoBERT + CRF.

    Quy trình tokenization:
      1. Thay '_' → ' ' trong mỗi token VLSP    (bắt buộc, đặc thù VLSP)
      2. Gọi tokenizer với is_split_into_words=True
      3. Align label:
           - subtoken đầu tiên của mỗi word → label thực
           - subtoken tiếp theo + special tokens → -100 (bỏ khi tính loss)

    Output __getitem__:
        input_ids      : (max_len,)  token ids PhoBERT
        attention_mask : (max_len,)  1=real, 0=pad
        labels         : (max_len,)  ids, -100 cho subtoken/special
    """

    def __init__(
        self,
        sentences  : List[Dict],
        label2id   : Dict[str, int],
        model_name : str = 'vinai/phobert-base',
        max_len    : int = 256,
        tokenizer  = None,
    ):
        self.data     = sentences
        self.label2id = label2id
        self.max_len  = max_len

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sent   = self.data[idx]
        tokens = sent['tokens']
        labels = sent['labels']

        # Bước đặc thù VLSP: normalize '_' → ' '
        norm_tokens = [t.replace('_', ' ') for t in tokens]

        enc = self.tokenizer(
            norm_tokens,
            is_split_into_words = True,
            max_length          = self.max_len,
            truncation          = True,
            padding             = 'max_length',
            return_tensors      = 'pt',
        )

        # Align label — chỉ label subtoken ĐẦU TIÊN của mỗi word
        word_ids  = enc.word_ids(batch_index=0)
        label_ids : List[int] = []
        prev_wid  : Optional[int] = None

        for wid in word_ids:
            if wid is None:              # [CLS], [SEP], [PAD]
                label_ids.append(-100)
            elif wid != prev_wid:        # first subtoken of a word
                label_ids.append(
                    self.label2id.get(labels[wid], 0)
                )
            else:                        # subsequent subtokens
                label_ids.append(-100)
            prev_wid = wid

        return {
            'input_ids':      enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'labels':         torch.tensor(label_ids, dtype=torch.long),
        }


# ════════════════════════════════════════════════════════════════════════════ #
#  3.  PhoBERT Dataset — Test / Inference (không có label)                    #
# ════════════════════════════════════════════════════════════════════════════ #

class VLSPTestDatasetPhoBERT(Dataset):
    """
    Dataset cho tập test (không có NER label).
    Dùng khi chạy inference để tạo file submission.

    Ngoài input_ids / attention_mask, trả về thêm:
        word_ids : List[Optional[int]]  — để map prediction về word gốc
        tokens   : List[str]            — token gốc (có '_')
        source   : str                  — tên file gốc
    """

    def __init__(
        self,
        sentences  : List[Dict],
        model_name : str = 'vinai/phobert-base',
        max_len    : int = 256,
        tokenizer  = None,
    ):
        self.data    = sentences
        self.max_len = max_len

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        sent        = self.data[idx]
        tokens      = sent['tokens']
        norm_tokens = [t.replace('_', ' ') for t in tokens]

        enc = self.tokenizer(
            norm_tokens,
            is_split_into_words = True,
            max_length          = self.max_len,
            truncation          = True,
            padding             = 'max_length',
            return_tensors      = 'pt',
        )

        return {
            'input_ids':      enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'word_ids':       enc.word_ids(batch_index=0),   # List[Optional[int]]
            'tokens':         tokens,
            'source':         sent.get('source', ''),
        }


# ════════════════════════════════════════════════════════════════════════════ #
#  4.  Smoke-test                                                              #
# ════════════════════════════════════════════════════════════════════════════ #

if __name__ == '__main__':
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from src.data_loader import (
        load_folder, normalize_bio,
        build_label_vocab, build_word_vocab, build_char_vocab,
        train_dev_split,
    )
    from torch.utils.data import DataLoader

    base = Path(__file__).parent.parent / 'Data'

    print("Doc du lieu ...")
    train_all = normalize_bio(load_folder(str(base / 'train'), verbose=False))
    train_sents, _ = train_dev_split(train_all, dev_ratio=0.1)

    label2id, id2label = build_label_vocab(train_sents)
    word2id,  _        = build_word_vocab(train_sents)
    char2id,  _        = build_char_vocab(train_sents)

    # ── BiLSTM Dataset + collate_fn ─────────────────────────────────────
    print("\n=== VLSPDatasetBiLSTM ===")
    ds_bi   = VLSPDatasetBiLSTM(train_sents[:20], word2id, char2id, label2id,
                                  max_seq=64, max_word=20)
    loader  = DataLoader(ds_bi, batch_size=4, collate_fn=bilstm_collate_fn)
    batch   = next(iter(loader))
    print(f"  word_ids  : {batch['word_ids'].shape}")
    print(f"  char_ids  : {batch['char_ids'].shape}")
    print(f"  label_ids : {batch['label_ids'].shape}")
    print(f"  mask      : {batch['mask'].shape}")
    print(f"  seq_lens  : {batch['seq_lens'].tolist()}")

    # ── PhoBERT Dataset ─────────────────────────────────────────────────
    print("\n=== VLSPDatasetPhoBERT ===")
    try:
        from transformers import AutoTokenizer
        tok    = AutoTokenizer.from_pretrained('vinai/phobert-base')
        ds_pb  = VLSPDatasetPhoBERT(train_sents[:4], label2id, tokenizer=tok)
        loader = DataLoader(ds_pb, batch_size=4)
        batch  = next(iter(loader))
        print(f"  input_ids      : {batch['input_ids'].shape}")
        print(f"  attention_mask : {batch['attention_mask'].shape}")
        print(f"  labels         : {batch['labels'].shape}")

        # Alignment check trên câu đầu
        sent    = train_sents[0]
        tokens  = sent['tokens']
        labels  = sent['labels']
        sample  = ds_pb[0]
        wids    = tok([t.replace('_', ' ') for t in tokens],
                      is_split_into_words=True,
                      max_length=64, truncation=True).word_ids()
        print("\n  Alignment check (10 token dau):")
        seen, count = set(), 0
        for wid, lid in zip(wids, sample['labels'].tolist()):
            if wid is None or wid in seen:
                continue
            seen.add(wid)
            tag = id2label.get(lid, '?') if lid != -100 else '-'
            print(f"    [{wid:2d}] {tokens[wid]:<20}  true={labels[wid]:<8}  enc={tag}")
            count += 1
            if count >= 10:
                break

    except Exception as e:
        print(f"  PhoBERT chua download: {e}")
