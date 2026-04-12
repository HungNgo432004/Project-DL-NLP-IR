# -*- coding: utf-8 -*-
# src/data_loader.py
"""
Parser cho dữ liệu VLSP NER — format CoNLL nhiều file.

Dataset thực tế:
  Data/train/  — 267 file .txt, mỗi file là 1 bài báo
                  Format: 5 cột  token | POS | chunk | NER | nested_NER
  Data/test/   — 45 file .txt, mỗi file là 1 bài báo
                  Format: 3 cột  token | POS | chunk  (KHÔNG có NER)

Mỗi file bắt đầu bằng <title>, <editor>, -DOCSTART- rồi đến câu <s>...</s>.
"""

import pickle
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ─── Hằng số ────────────────────────────────────────────────────────────────
VALID_ENTS = {'PER', 'LOC', 'ORG', 'MISC'}


# ════════════════════════════════════════════════════════════════════════════ #
#  1.  File-level parser                                                       #
# ════════════════════════════════════════════════════════════════════════════ #

def parse_vlsp_file(file_path: str, has_labels: bool = True) -> List[Dict]:
    """
    Đọc 1 file VLSP.

    Train (has_labels=True)  → 5 cột: token | POS | chunk | NER | nested
    Test  (has_labels=False) → 3 cột: token | POS | chunk

    Trả về list[dict] — mỗi dict là 1 câu:
        tokens, pos, chunks, labels, nested, source
    """
    sentences: List[Dict] = []
    current   = _new_sent(file_path)
    inside_s  = False

    with open(file_path, 'r', encoding='utf-8') as fh:
        for raw in fh:
            line     = raw.rstrip('\r\n')
            stripped = line.strip()

            if stripped.startswith('<s>'):
                inside_s = True
                current  = _new_sent(file_path)
                continue

            if stripped.startswith('</s>'):
                inside_s = False
                if current['tokens']:
                    sentences.append(current)
                continue

            if (not inside_s
                    or stripped == ''
                    or stripped.startswith('-DOCSTART-')
                    or stripped.startswith('<')):
                continue

            # Ưu tiên tab, fallback split()
            parts = line.split('\t') if '\t' in line else stripped.split()

            min_cols = 4 if has_labels else 3
            if len(parts) < min_cols:
                continue

            current['tokens'].append(parts[0])
            current['pos'].append(parts[1])
            current['chunks'].append(parts[2])

            if has_labels:
                current['labels'].append(parts[3])
                current['nested'].append(parts[4] if len(parts) > 4 else 'O')
            else:
                current['labels'].append('O')
                current['nested'].append('O')

    return sentences


def _new_sent(source: str) -> Dict:
    return {
        'tokens': [], 'pos': [], 'chunks': [],
        'labels': [], 'nested': [],
        'source': Path(source).name,
    }


# ════════════════════════════════════════════════════════════════════════════ #
#  2.  Folder-level loader                                                     #
# ════════════════════════════════════════════════════════════════════════════ #

def load_folder(folder_path: str,
                has_labels: bool = True,
                verbose: bool = True) -> List[Dict]:
    """
    Đọc toàn bộ *.txt trong folder, ghép thành 1 list câu.

    Args:
        folder_path : Data/train hoặc Data/test
        has_labels  : True  → có NER label (train)
                      False → không có NER (test)
        verbose     : in tiến trình mỗi 50 file
    """
    folder = Path(folder_path)
    files  = sorted(folder.glob('*.txt'))

    if not files:
        raise FileNotFoundError(f"Khong tim thay file .txt trong '{folder_path}'")

    all_sents: List[Dict] = []
    for i, fp in enumerate(files):
        sents = parse_vlsp_file(str(fp), has_labels=has_labels)
        all_sents.extend(sents)
        if verbose and (i + 1) % 50 == 0:
            print(f"  Doc {i+1}/{len(files)} file — {len(all_sents):,} cau")

    if verbose:
        label_str = "co NER" if has_labels else "khong NER"
        print(f"  OK {len(files)} file ({label_str}) -> {len(all_sents):,} cau")

    return all_sents


# ════════════════════════════════════════════════════════════════════════════ #
#  3.  BIO normalisation                                                       #
# ════════════════════════════════════════════════════════════════════════════ #

def normalize_bio(sentences: List[Dict],
                  fix_invalid: bool = True) -> List[Dict]:
    """
    Chuẩn hóa BIO sequence — sửa in-place, trả về list.

    Các lỗi được sửa:
      I-X ở đầu hoặc sau O       → B-X
      I-X sau B-Y / I-Y (Y != X) → B-X
      Nhãn entity không hợp lệ   → O   (nếu fix_invalid=True)

    Args:
        sentences    : list câu từ load_folder / parse_vlsp_file
        fix_invalid  : True  → đổi entity type lạ thành O
                       False → giữ nguyên entity type lạ (không sửa)
    """
    n_fixed = 0
    for sent in sentences:
        prev_type: Optional[str] = None
        for i, lbl in enumerate(sent['labels']):
            if lbl == 'O':
                prev_type = None

            elif lbl.startswith('B-'):
                etype = lbl[2:]
                if fix_invalid and etype not in VALID_ENTS:
                    sent['labels'][i] = 'O'
                    prev_type = None
                    n_fixed += 1
                else:
                    prev_type = etype

            elif lbl.startswith('I-'):
                etype = lbl[2:]
                if fix_invalid and etype not in VALID_ENTS:
                    sent['labels'][i] = 'O'
                    prev_type = None
                    n_fixed += 1
                elif etype != prev_type:          # I- sai vị trí → B-
                    sent['labels'][i] = f'B-{etype}'
                    prev_type = etype
                    n_fixed += 1
                # else: hợp lệ, giữ nguyên

            else:                                 # nhãn lạ hoàn toàn
                sent['labels'][i] = 'O'
                prev_type = None
                n_fixed += 1

    if n_fixed:
        print(f"  normalize_bio: da sua {n_fixed} nhan loi")
    return sentences


# ════════════════════════════════════════════════════════════════════════════ #
#  4.  Vocabulary builders                                                     #
# ════════════════════════════════════════════════════════════════════════════ #

def build_label_vocab(sentences: List[Dict]) -> Tuple[Dict[str, int],
                                                       Dict[int, str]]:
    """
    label → id.  O luôn = 0; các nhãn khác theo thứ tự gặp lần đầu.
    Thứ tự trả về: (label2id, id2label).
    """
    label2id: Dict[str, int] = {'O': 0}
    for sent in sentences:
        for lbl in sent['labels']:
            if lbl not in label2id:
                label2id[lbl] = len(label2id)
    id2label = {v: k for k, v in label2id.items()}
    return label2id, id2label


def build_word_vocab(sentences: List[Dict],
                     min_freq: int = 1) -> Tuple[Dict[str, int],
                                                  Dict[int, str]]:
    """
    word → id.  <PAD>=0, <UNK>=1; các từ còn lại theo tần suất giảm dần.
    """
    cnt = Counter(t for s in sentences for t in s['tokens'])
    w2i: Dict[str, int] = {'<PAD>': 0, '<UNK>': 1}
    for word, freq in cnt.most_common():
        if freq >= min_freq:
            w2i[word] = len(w2i)
    return w2i, {v: k for k, v in w2i.items()}


def build_char_vocab(sentences: List[Dict]) -> Tuple[Dict[str, int],
                                                      Dict[int, str]]:
    """
    char → id.  <PAD>=0, <UNK>=1; các ký tự còn lại sắp xếp theo bảng mã.
    """
    chars = set(c for s in sentences for t in s['tokens'] for c in t)
    c2i: Dict[str, int] = {'<PAD>': 0, '<UNK>': 1}
    for c in sorted(chars):
        c2i[c] = len(c2i)
    return c2i, {v: k for k, v in c2i.items()}


# ════════════════════════════════════════════════════════════════════════════ #
#  5.  Train / Dev split (split theo file, không phải câu)                    #
# ════════════════════════════════════════════════════════════════════════════ #

def train_dev_split(sentences: List[Dict],
                    dev_ratio: float = 0.1,
                    seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    """
    Tách train thành train + dev theo tỷ lệ dev_ratio.

    **Split theo file** — đảm bảo toàn bộ câu của 1 bài báo chỉ nằm
    trong 1 tập, tránh data leakage.

    Args:
        sentences : list câu đã qua normalize_bio
        dev_ratio : tỷ lệ file dành cho dev (mặc định 10%)
        seed      : random seed để kết quả có thể tái tạo

    Returns:
        (train_sents, dev_sents)
    """
    rng = random.Random(seed)

    by_file: Dict[str, List[Dict]] = defaultdict(list)
    for s in sentences:
        by_file[s['source']].append(s)

    files = sorted(by_file.keys())
    rng.shuffle(files)

    n_dev       = max(1, round(len(files) * dev_ratio))
    dev_files   = set(files[:n_dev])
    train_files = set(files[n_dev:])

    train_sents = [s for s in sentences if s['source'] in train_files]
    dev_sents   = [s for s in sentences if s['source'] in dev_files]

    print(f"  train/dev split: {len(train_files)} file train "
          f"({len(train_sents):,} cau) | "
          f"{len(dev_files)} file dev ({len(dev_sents):,} cau)")
    return train_sents, dev_sents


# ════════════════════════════════════════════════════════════════════════════ #
#  6.  Lưu / Tải vocab bằng pickle (để dùng lại khi inference)               #
# ════════════════════════════════════════════════════════════════════════════ #

def save_vocabs(label2id: Dict, word2id: Dict, char2id: Dict,
                save_path: str) -> None:
    """Lưu 3 vocab vào 1 file pickle."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump({'label2id': label2id,
                     'word2id':  word2id,
                     'char2id':  char2id}, f)
    print(f"  Vocab saved -> {save_path}")


def load_vocabs(save_path: str) -> Tuple[Dict, Dict, Dict, Dict, Dict, Dict]:
    """
    Tải vocab từ file pickle.
    Trả về: label2id, id2label, word2id, id2word, char2id, id2char
    """
    with open(save_path, 'rb') as f:
        d = pickle.load(f)
    label2id = d['label2id']
    word2id  = d['word2id']
    char2id  = d['char2id']
    return (label2id, {v: k for k, v in label2id.items()},
            word2id,  {v: k for k, v in word2id.items()},
            char2id,  {v: k for k, v in char2id.items()})


# ════════════════════════════════════════════════════════════════════════════ #
#  7.  Quick smoke-test                                                        #
# ════════════════════════════════════════════════════════════════════════════ #

if __name__ == '__main__':
    import sys
    base = Path(__file__).resolve().parent.parent / 'Data'

    print("=" * 60)
    print("  VLSP data_loader — smoke test")
    print("=" * 60)

    train_raw = load_folder(str(base / 'train'), has_labels=True)
    train     = normalize_bio(train_raw)
    test      = load_folder(str(base / 'test'),  has_labels=False)

    train_sents, dev_sents = train_dev_split(train, dev_ratio=0.1)

    label2id, id2label = build_label_vocab(train_sents)
    word2id,  id2word  = build_word_vocab(train_sents, min_freq=1)
    char2id,  id2char  = build_char_vocab(train_sents)

    print(f"\n  Labels ({len(label2id)}): {label2id}")
    print(f"  Word vocab : {len(word2id):,}")
    print(f"  Char vocab : {len(char2id):,}")
    print(f"  Test  sents: {len(test):,}")

    # Kiểm tra save/load vocab
    tmp = str(base.parent / 'output' / 'vocabs.pkl')
    save_vocabs(label2id, word2id, char2id, tmp)
    l2i, i2l, w2i, i2w, c2i, i2c = load_vocabs(tmp)
    assert l2i == label2id, "label2id mismatch!"
    print("  Vocab save/load: OK")

    # In câu ví dụ
    ex = train_sents[0]
    print(f"\n  Vi du cau (file: {ex['source']}):")
    for tok, lbl in zip(ex['tokens'][:8], ex['labels'][:8]):
        print(f"    {tok:<22}  {lbl}")
