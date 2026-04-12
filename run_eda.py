# -*- coding: utf-8 -*-
"""
run_eda.py
----------
Script chính chạy toàn bộ Phần 1-6 của NER_Vietnamese_VLSP.md.

Dùng:
    python run_eda.py                      # mặc định
    python run_eda.py --dev-ratio 0.15    # dev 15%
    python run_eda.py --no-plots          # chỉ in báo cáo, không vẽ
    python run_eda.py --out-dir results   # thư mục output tuỳ chỉnh

Output:
    output/
        label_vocab.txt        nhãn NER (text)
        vocabs.pkl             word/char/label vocab (pickle, dùng lại khi train)
        eda_train.png          6 biểu đồ tập train
        eda_dev.png            6 biểu đồ tập dev
        eda_test.png           6 biểu đồ tập test
        eda_comparison.png     so sánh 3 split
        report_train.txt       báo cáo văn bản tập train
        report_dev.txt         báo cáo văn bản tập dev
        report_test.txt        báo cáo văn bản tập test
"""

import argparse
import sys
import os
from pathlib import Path

# ── Thêm thư mục gốc vào PYTHONPATH ─────────────────────────────────────────
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# ── Import src modules ───────────────────────────────────────────────────────
from src.data_loader import (
    load_folder,
    normalize_bio,
    build_label_vocab,
    build_word_vocab,
    build_char_vocab,
    train_dev_split,
    save_vocabs,
)
from src.eda import (
    analyze_dataset,
    print_report,
    save_report,
    plot_eda,
    compare_splits,
    analyze_vlsp_specifics,
)


# ════════════════════════════════════════════════════════════════════════════ #
#  Argument parser                                                             #
# ════════════════════════════════════════════════════════════════════════════ #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='VLSP NER — Phan tich du lieu (Phan 1-6)'
    )
    p.add_argument('--data-dir',   default='Data',   help='Thu muc chua train/ va test/')
    p.add_argument('--out-dir',    default='output', help='Thu muc luu ket qua')
    p.add_argument('--dev-ratio',  type=float, default=0.1,
                   help='Ti le file danh cho dev (mac dinh 0.1 = 10%%)')
    p.add_argument('--seed',       type=int, default=42,
                   help='Random seed cho train/dev split')
    p.add_argument('--no-plots',   action='store_true',
                   help='Bo qua buoc ve bieu do (nhanh hon)')
    p.add_argument('--min-freq',   type=int, default=1,
                   help='Nguong tan suat toi thieu de dua vao word vocab')
    return p.parse_args()


# ════════════════════════════════════════════════════════════════════════════ #
#  Main                                                                        #
# ════════════════════════════════════════════════════════════════════════════ #

def main() -> None:
    args = parse_args()

    base_dir  = project_root / args.data_dir
    out_dir   = project_root / args.out_dir
    train_dir = base_dir / 'train'
    test_dir  = base_dir / 'test'

    out_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 65)
    print('  VLSP NER --- Phan Tich Du Lieu (Phan 1-6)')
    print('=' * 65)
    print(f'  data_dir  : {base_dir}')
    print(f'  out_dir   : {out_dir}')
    print(f'  dev_ratio : {args.dev_ratio}  seed={args.seed}')
    print(f'  min_freq  : {args.min_freq}')

    # ── PHAN 5: Doc & Tien Xu Ly ─────────────────────────────────────── #
    print('\n[PHAN 5] Doc & Tien Xu Ly Du Lieu VLSP')
    print('-' * 45)

    print(f'\n  Doc train ({train_dir.name}/) ...')
    train_raw  = load_folder(str(train_dir), has_labels=True,  verbose=True)

    print(f'\n  Chuan hoa BIO ...')
    train_norm = normalize_bio(train_raw)

    print(f'\n  Doc test ({test_dir.name}/) ...')
    test_sents = load_folder(str(test_dir),  has_labels=False, verbose=True)

    # Train / Dev split
    train_sents, dev_sents = train_dev_split(
        train_norm, dev_ratio=args.dev_ratio, seed=args.seed)

    print(f'\n  Split {int((1-args.dev_ratio)*100)}/{int(args.dev_ratio*100)}'
          f' (theo file):')
    print(f'    Train : {len(train_sents):,} cau')
    print(f'    Dev   : {len(dev_sents):,} cau')
    print(f'    Test  : {len(test_sents):,} cau')

    # ── Build vocab ──────────────────────────────────────────────────── #
    print('\n  Xay dung vocabulary ...')
    label2id, id2label = build_label_vocab(train_sents)
    word2id,  id2word  = build_word_vocab(train_sents, min_freq=args.min_freq)
    char2id,  id2char  = build_char_vocab(train_sents)

    print(f'    Labels ({len(label2id)}): {label2id}')
    print(f'    Word vocab : {len(word2id):,}')
    print(f'    Char vocab : {len(char2id):,}')

    # Lưu label vocab (text, dễ đọc)
    vocab_txt = out_dir / 'label_vocab.txt'
    with open(vocab_txt, 'w', encoding='utf-8') as f:
        f.write('# Label Vocabulary --- VLSP NER\n')
        f.write(f'# Tong so nhan: {len(label2id)}\n\n')
        for lbl, idx in sorted(label2id.items(), key=lambda x: x[1]):
            f.write(f'{idx}\t{lbl}\n')
    print(f'\n  OK label_vocab.txt -> {vocab_txt}')

    # Lưu vocab pickle (dùng lại khi train model)
    vocab_pkl = str(out_dir / 'vocabs.pkl')
    save_vocabs(label2id, word2id, char2id, vocab_pkl)

    # ── Vi du cau thuc te ────────────────────────────────────────────── #
    print('\n  [Vi du cau da parse - file 7818.txt]')
    ex7818 = [s for s in train_sents if s['source'] == '7818.txt']
    sent   = ex7818[0] if ex7818 else train_sents[0]
    print(f'  File: {sent["source"]}')
    header = f'  {"Token":<22} {"POS":<8} {"Chunk":<10} {"NER":<10} Nested'
    print(header)
    print('  ' + '-' * 60)
    for tok, pos, chk, lbl, nst in zip(
            sent['tokens'], sent['pos'], sent['chunks'],
            sent['labels'], sent['nested']):
        flag  = '  <- NER!'    if lbl != 'O' else ''
        nflag = f' +nested={nst}' if nst != 'O' else ''
        print(f'  {tok:<22} {pos:<8} {chk:<10} {lbl:<10} {nst}{flag}{nflag}')

    # ── PHAN 6: EDA ──────────────────────────────────────────────────── #
    print('\n\n[PHAN 6] Phan Tich Kham Pha Du Lieu EDA')
    print('-' * 45)

    stats_train = analyze_dataset(train_sents, name='Train')
    stats_dev   = analyze_dataset(dev_sents,   name='Dev')
    stats_test  = analyze_dataset(test_sents,  name='Test')

    print_report(stats_train)
    print_report(stats_dev)
    print_report(stats_test)

    analyze_vlsp_specifics(train_sents)

    # Luu bao cao text
    save_report(stats_train, str(out_dir / 'report_train.txt'))
    save_report(stats_dev,   str(out_dir / 'report_dev.txt'))
    save_report(stats_test,  str(out_dir / 'report_test.txt'))

    # Ve bieu do
    if not args.no_plots:
        print(f'\n  Ve bieu do -> {out_dir} ...')
        saved = []
        for st in [stats_train, stats_dev, stats_test]:
            p = plot_eda(st, save_dir=str(out_dir))
            saved.append(p)
        p_cmp = compare_splits(
            [stats_train, stats_dev, stats_test], save_dir=str(out_dir))
        if p_cmp:
            saved.append(p_cmp)
        for p in saved:
            print(f'  OK {p}')
    else:
        print('\n  (Bo qua ve bieu do theo --no-plots)')

    # ── Tom tat ──────────────────────────────────────────────────────── #
    print('\n' + '=' * 65)
    print('  TOM TAT')
    print('=' * 65)
    for st in [stats_train, stats_dev, stats_test]:
        print(f"  {st['name']:<6}: {st['n_sents']:>6,} cau | "
              f"{st['n_tokens']:>8,} token | avg={st['avg_len']:.1f}")

    print(f'\n  Labels       : {label2id}')
    print(f"  Entity (train): {dict(stats_train['ent_cnt'])}")
    print(f"\n  Token '_' (train): {stats_train['underscore_tokens']:,} "
          f"({stats_train['underscore_ratio']:.1f}%) "
          f"-> PHAI replace '_'->' ' truoc PhoBERT!")
    print(f'\n  OK Hoan tat! Output: {out_dir}')
    print('=' * 65)


if __name__ == '__main__':
    main()
