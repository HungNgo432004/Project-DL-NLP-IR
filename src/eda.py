# -*- coding: utf-8 -*-
# src/eda.py
"""
Phân tích khám phá dữ liệu (EDA) cho VLSP NER.

Public API:
  analyze_dataset(sentences, name)   → dict thống kê
  print_report(stats)                → in báo cáo văn bản
  save_report(stats, path)           → lưu báo cáo ra file .txt
  plot_eda(stats, save_dir)          → vẽ 6 subplot, lưu PNG
  compare_splits(stats_list, dir)    → vẽ so sánh đa tập
  analyze_vlsp_specifics(sents)      → in phân tích đặc trưng VLSP
"""

import collections
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')                   # non-interactive, lưu file PNG
import matplotlib.pyplot as plt
import numpy as np


# ─── Màu sắc entity ─────────────────────────────────────────────────────────
ENTITY_COLORS = {
    'PER' : '#3498db',
    'LOC' : '#2ecc71',
    'ORG' : '#e74c3c',
    'MISC': '#f39c12',
}


# ════════════════════════════════════════════════════════════════════════════ #
#  Utility                                                                     #
# ════════════════════════════════════════════════════════════════════════════ #

def extract_entities(
    tokens : List[str],
    labels : List[str],
) -> List[Tuple[int, int, str, str]]:
    """
    Trích xuất entity span từ BIO sequence.

    Trả về list (start_idx, end_idx_exclusive, entity_type, entity_text).
    """
    entities: List[Tuple] = []
    i = 0
    while i < len(labels):
        if labels[i].startswith('B-'):
            etype = labels[i][2:]
            span  = [tokens[i]]
            j     = i + 1
            while j < len(labels) and labels[j] == f'I-{etype}':
                span.append(tokens[j])
                j += 1
            entities.append((i, j, etype, ' '.join(span)))
            i = j
        else:
            i += 1
    return entities


# ════════════════════════════════════════════════════════════════════════════ #
#  Phân tích thống kê                                                          #
# ════════════════════════════════════════════════════════════════════════════ #

def analyze_dataset(sentences: List[Dict], name: str = 'Dataset') -> Dict:
    """
    Tính toàn bộ thống kê cho 1 tập dữ liệu.

    Returns dict với các key:
        name, n_sents, n_tokens, lengths, avg_len, max_len, min_len,
        underscore_tokens, underscore_ratio,
        label_cnt, ent_cnt, ent_by_type, nested_ent_cnt,
        pos_cnt, buckets
    """
    tokens_all = [t for s in sentences for t in s['tokens']]
    labels_all = [l for s in sentences for l in s['labels']]
    nested_all = [n for s in sentences for n in s['nested']]
    pos_all    = [p for s in sentences for p in s['pos']]
    lengths    = [len(s['tokens']) for s in sentences]

    if not tokens_all:
        raise ValueError(f"Dataset '{name}' rong (0 token).")

    # Token dùng gạch dưới (đặc trưng VLSP)
    underscore_cnt   = sum(1 for t in tokens_all if '_' in t)
    underscore_ratio = underscore_cnt / len(tokens_all) * 100

    # Label & entity
    label_cnt   = collections.Counter(labels_all)
    ent_cnt     = collections.Counter(
        lbl[2:] for lbl in labels_all if lbl.startswith('B-'))
    nested_cnt  = collections.Counter(
        n[2:]   for n   in nested_all  if n  != 'O')

    # Top entity text theo loại
    ent_by_type: Dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter)
    for sent in sentences:
        for _, _, etype, etext in extract_entities(sent['tokens'], sent['labels']):
            ent_by_type[etype][etext] += 1

    # POS
    pos_cnt = collections.Counter(pos_all)

    # Phân nhóm độ dài câu
    buckets = {'<=32': 0, '33-64': 0, '65-128': 0, '129-256': 0, '>256': 0}
    for ln in lengths:
        if ln <= 32:         buckets['<=32']    += 1
        elif ln <= 64:       buckets['33-64']   += 1
        elif ln <= 128:      buckets['65-128']  += 1
        elif ln <= 256:      buckets['129-256'] += 1
        else:                buckets['>256']    += 1

    return {
        'name':              name,
        'n_sents':           len(sentences),
        'n_tokens':          len(tokens_all),
        'lengths':           lengths,
        'avg_len':           sum(lengths) / len(lengths),
        'max_len':           max(lengths),
        'min_len':           min(lengths),
        'underscore_tokens': underscore_cnt,
        'underscore_ratio':  underscore_ratio,
        'label_cnt':         label_cnt,
        'ent_cnt':           ent_cnt,
        'ent_by_type':       ent_by_type,
        'nested_ent_cnt':    nested_cnt,
        'pos_cnt':           pos_cnt,
        'buckets':           buckets,
    }


# ════════════════════════════════════════════════════════════════════════════ #
#  In / Lưu báo cáo văn bản                                                   #
# ════════════════════════════════════════════════════════════════════════════ #

def _build_report_lines(stats: Dict) -> List[str]:
    """Tạo list dòng báo cáo — dùng chung cho print và save."""
    SEP  = '=' * 65
    lines: List[str] = []
    add   = lines.append

    add(SEP)
    add(f"  EDA --- {stats['name']}")
    add(SEP)

    add("")
    add("  [Tong quan]")
    add(f"    So cau          : {stats['n_sents']:,}")
    add(f"    Tong token      : {stats['n_tokens']:,}")
    add(f"    Do dai TB       : {stats['avg_len']:.1f} token/cau")
    add(f"    Do dai min/max  : {stats['min_len']} / {stats['max_len']}")
    add(f"    Token co '_'    : {stats['underscore_tokens']:,} "
        f"({stats['underscore_ratio']:.1f}%)")

    add("")
    add("  [Phan nhom do dai cau]")
    for bucket, cnt in stats['buckets'].items():
        bar = '#' * int(cnt / stats['n_sents'] * 40)
        pct = cnt / stats['n_sents'] * 100
        add(f"    {bucket:10s} : {cnt:5,}  ({pct:5.1f}%)  {bar}")

    add("")
    add("  [Phan phoi nhan NER]")
    max_cnt = max(stats['label_cnt'].values()) if stats['label_cnt'] else 1
    for lbl in sorted(stats['label_cnt']):
        cnt = stats['label_cnt'][lbl]
        bar = '#' * int(cnt / max_cnt * 30)
        add(f"    {lbl:12s} : {cnt:7,}  {bar}")

    add("")
    add("  [Entity count (B- tags)]")
    for etype, cnt in stats['ent_cnt'].most_common():
        add(f"    {etype:6s}: {cnt:,}")

    add("")
    if stats['nested_ent_cnt']:
        add("  [Nested entity (cot 5)]")
        for etype, cnt in stats['nested_ent_cnt'].most_common():
            add(f"    {etype:6s}: {cnt:,}")
    else:
        add("  [Nested entity] --- khong co (test set)")

    add("")
    add("  [Top 10 entity text theo loai]")
    for etype in ['PER', 'LOC', 'ORG', 'MISC']:
        if etype in stats['ent_by_type']:
            add(f"    [{etype}]")
            for text, cnt in stats['ent_by_type'][etype].most_common(10):
                add(f"      {text:<40s} ({cnt:,}x)")

    add("")
    add("  [Top 15 POS tag]")
    for pos, cnt in stats['pos_cnt'].most_common(15):
        bar = '#' * int(cnt / stats['n_tokens'] * 50)
        pct = cnt / stats['n_tokens'] * 100
        add(f"    {pos:8s} : {cnt:7,} ({pct:5.1f}%)  {bar}")

    add("")
    return lines


def print_report(stats: Dict) -> None:
    """In báo cáo EDA ra stdout."""
    for line in _build_report_lines(stats):
        print(line)


def save_report(stats: Dict, path: str) -> None:
    """Lưu báo cáo EDA ra file text (UTF-8)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(_build_report_lines(stats)))
        f.write('\n')
    print(f"  Report saved -> {path}")


# ════════════════════════════════════════════════════════════════════════════ #
#  Biểu đồ matplotlib                                                          #
# ════════════════════════════════════════════════════════════════════════════ #

def _get_font_prop():
    """Trả về FontProperties phù hợp (ưu tiên font có tiếng Việt)."""
    try:
        from matplotlib.font_manager import FontProperties, findfont
        for fname in ['DejaVu Sans', 'Arial Unicode MS', 'Noto Sans']:
            try:
                fp = FontProperties(family=fname)
                if findfont(fp, fallback_to_default=False):
                    return fp
            except Exception:
                pass
    except Exception:
        pass
    return None


def plot_eda(stats: Dict, save_dir: str = '.') -> str:
    """
    Vẽ bảng 6 biểu đồ EDA cho 1 tập (train / dev / test), lưu PNG.

    Ô 1: Histogram độ dài câu
    Ô 2: Entity count bar chart
    Ô 3: Pie chart nhãn NER (không tính O)
    Ô 4: POS tag top-15 (barh)
    Ô 5: Phân nhóm độ dài câu (bar)
    Ô 6: Bảng top entity text

    Trả về đường dẫn file PNG đã lưu.
    """
    name      = stats['name']
    safe_name = name.lower().replace(' ', '_').replace('/', '-')
    out_path  = os.path.join(save_dir, f'eda_{safe_name}.png')

    fig, axes = plt.subplots(2, 3, figsize=(19, 10))
    fig.suptitle(f'EDA  |  {name}', fontsize=15, fontweight='bold')

    # ── 1. Histogram độ dài câu ─────────────────────────────────────── #
    ax = axes[0, 0]
    ax.hist(stats['lengths'], bins=50,
            color='#3498db', edgecolor='white', alpha=0.85)
    ax.axvline(64,  color='#2ecc71', ls='--', lw=1.5, label='64')
    ax.axvline(128, color='#e74c3c', ls='--', lw=1.5, label='128')
    ax.axvline(256, color='#f39c12', ls='--', lw=1.5, label='256')
    ax.axvline(stats['avg_len'], color='#9b59b6', ls='-.',
               lw=1.5, label=f'Mean={stats["avg_len"]:.0f}')
    ax.set_title('Phan phoi do dai cau (token)', fontsize=11)
    ax.set_xlabel('So token')
    ax.set_ylabel('So cau')
    ax.legend(fontsize=8)

    # ── 2. Entity bar chart ─────────────────────────────────────────── #
    ax = axes[0, 1]
    ent_cnt = stats['ent_cnt']
    if ent_cnt:
        etypes = list(ent_cnt.keys())
        ecnts  = [ent_cnt[e] for e in etypes]
        ecols  = [ENTITY_COLORS.get(e, '#95a5a6') for e in etypes]
        bars   = ax.bar(etypes, ecnts, color=ecols, edgecolor='white', width=0.6)
        for bar, cnt in zip(bars, ecnts):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(ecnts) * 0.01,
                    f'{cnt:,}', ha='center', va='bottom', fontsize=9)
    else:
        ax.text(0.5, 0.5, 'Khong co entity\n(tap test)',
                ha='center', va='center', fontsize=11, transform=ax.transAxes)
    ax.set_title('So luong entity theo loai', fontsize=11)
    ax.set_xlabel('Loai entity')
    ax.set_ylabel('So luong')

    # ── 3. Label pie (bỏ O) ─────────────────────────────────────────── #
    ax = axes[0, 2]
    ent_labels = {k: v for k, v in stats['label_cnt'].items() if k != 'O'}
    if ent_labels:
        wedges, texts, autotexts = ax.pie(
            ent_labels.values(),
            labels    = ent_labels.keys(),
            colors    = [ENTITY_COLORS.get(k[2:], '#95a5a6')
                         for k in ent_labels],
            autopct   = '%1.1f%%',
            startangle= 90,
            wedgeprops= {'edgecolor': 'white', 'linewidth': 1.5},
        )
        for at in autotexts:
            at.set_fontsize(8)
    else:
        ax.text(0.5, 0.5, 'Khong co nhan NER',
                ha='center', va='center', fontsize=11, transform=ax.transAxes)
    ax.set_title('Ty le nhan NER (khong tinh O)', fontsize=11)

    # ── 4. POS barh top-15 ──────────────────────────────────────────── #
    ax = axes[1, 0]
    pos_items = stats['pos_cnt'].most_common(15)
    pos_tags  = [p[0] for p in pos_items]
    pos_cnts  = [p[1] for p in pos_items]
    ax.barh(pos_tags[::-1], pos_cnts[::-1],
            color='#9b59b6', edgecolor='white', alpha=0.85)
    ax.set_title('Top 15 POS tag', fontsize=11)
    ax.set_xlabel('So lan xuat hien')

    # ── 5. Length bucket bar ────────────────────────────────────────── #
    ax    = axes[1, 1]
    bkeys = list(stats['buckets'].keys())
    bvals = [stats['buckets'][k] for k in bkeys]
    bcols = ['#1abc9c', '#3498db', '#9b59b6', '#e67e22', '#c0392b']
    bars  = ax.bar(bkeys, bvals, color=bcols, edgecolor='white')
    for bar, val in zip(bars, bvals):
        pct = val / stats['n_sents'] * 100
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(bvals) * 0.01,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)
    ax.set_title('Phan nhom do dai cau', fontsize=11)
    ax.set_xlabel('Khoang token')
    ax.set_ylabel('So cau')

    # ── 6. Top entity table ─────────────────────────────────────────── #
    ax = axes[1, 2]
    ax.axis('off')

    rows: List[List[str]] = []
    for etype in ['PER', 'LOC', 'ORG', 'MISC']:
        if etype in stats['ent_by_type']:
            for text, cnt in stats['ent_by_type'][etype].most_common(3):
                short = text[:26] + '..' if len(text) > 26 else text
                rows.append([etype, short, str(cnt)])

    if rows:
        col_labels = ['Loai', 'Entity', 'N']
        tbl = ax.table(
            cellText  = rows,
            colLabels = col_labels,
            loc       = 'center',
            cellLoc   = 'left',
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1.1, 1.5)
        n_cols = len(col_labels)
        for j in range(n_cols):
            tbl[(0, j)].set_facecolor('#2c3e50')
            tbl[(0, j)].set_text_props(color='white', fontweight='bold')
        # Màu xen kẽ theo entity type
        etype_color = {'PER': '#d6eaf8', 'LOC': '#d5f5e3',
                       'ORG': '#fadbd8', 'MISC': '#fef9e7'}
        for r_idx, row in enumerate(rows, start=1):
            clr = etype_color.get(row[0], '#f2f3f4')
            for j in range(n_cols):
                tbl[(r_idx, j)].set_facecolor(clr)
    else:
        ax.text(0.5, 0.5, 'Khong co entity\n(tap test)',
                ha='center', va='center', fontsize=11, transform=ax.transAxes)

    ax.set_title('Top entity text theo loai', fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


# ════════════════════════════════════════════════════════════════════════════ #
#  So sánh nhiều split                                                         #
# ════════════════════════════════════════════════════════════════════════════ #

def compare_splits(stats_list: List[Dict], save_dir: str = '.') -> str:
    """
    Vẽ biểu đồ so sánh nhiều split dữ liệu (train / dev / test).

    Trả về đường dẫn PNG.  Trả về '' nếu chỉ có 1 split.
    """
    if len(stats_list) < 2:
        return ''

    n     = len(stats_list)
    names = [s['name'] for s in stats_list]
    split_colors = ['#3498db', '#2ecc71', '#e74c3c',
                    '#f39c12', '#9b59b6'][:n]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle('So sanh cac split du lieu', fontsize=14, fontweight='bold')

    # Số câu
    ax = axes[0]
    bars = ax.bar(names, [s['n_sents'] for s in stats_list],
                  color=split_colors, edgecolor='white')
    for bar, s in zip(bars, stats_list):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{s['n_sents']:,}", ha='center', va='bottom', fontsize=9)
    ax.set_title('So cau')
    ax.set_ylabel('Cau')

    # Số token
    ax = axes[1]
    bars = ax.bar(names, [s['n_tokens'] for s in stats_list],
                  color=split_colors, edgecolor='white')
    for bar, s in zip(bars, stats_list):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{s['n_tokens']:,}", ha='center', va='bottom', fontsize=9)
    ax.set_title('Tong so token')
    ax.set_ylabel('Token')

    # Entity count grouped bar
    ax    = axes[2]
    x     = np.arange(4)
    w     = 0.8 / n
    et    = ['PER', 'LOC', 'ORG', 'MISC']
    for i, st in enumerate(stats_list):
        vals = [st['ent_cnt'].get(e, 0) for e in et]
        rects = ax.bar(x + i * w, vals, w,
                       label=st['name'],
                       color=split_colors[i],
                       edgecolor='white', alpha=0.9)
    ax.set_xticks(x + w * (n - 1) / 2)
    ax.set_xticklabels(et)
    ax.set_title('Entity count theo loai')
    ax.legend(fontsize=8)

    plt.tight_layout()
    out = os.path.join(save_dir, 'eda_comparison.png')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out


# ════════════════════════════════════════════════════════════════════════════ #
#  Phân tích đặc trưng VLSP                                                    #
# ════════════════════════════════════════════════════════════════════════════ #

def analyze_vlsp_specifics(train_sents: List[Dict],
                            n_example: int = 20) -> None:
    """
    In phân tích các đặc điểm định dạng riêng của VLSP:
      1. Token ghép bằng '_' (từ ghép)
      2. Nested entity (cột 5) — điểm thách thức chính
      3. Token chứa dấu nháy trong entity span (tên biệt hiệu)
      4. Token viết tắt (POS = Ny)
      5. Câu 1-token (có thể gây lỗi batch)
    """
    SEP = '=' * 65
    print(f'\n{SEP}')
    print('  Phan Tich Dac Trung Format VLSP')
    print(SEP)

    tokens_all = [t for s in train_sents for t in s['tokens']]

    # 1. Token ghép '_'
    usc = [t for t in tokens_all if '_' in t]
    print(f"\n  1. Token ghep bang '_'  : {len(usc):,} / {len(tokens_all):,} "
          f"({len(usc)/len(tokens_all)*100:.1f}%)")
    shown: set = set()
    cnt_shown  = 0
    for t in tokens_all:
        if '_' in t and t not in shown:
            print(f"       {t}")
            shown.add(t)
            cnt_shown += 1
            if cnt_shown >= n_example:
                break

    # 2. Nested entity
    nested_ents = [
        (s['tokens'][i], s['labels'][i], s['nested'][i])
        for s in train_sents
        for i, n in enumerate(s['nested'])
        if n != 'O'
    ]
    print(f"\n  2. Nested entity (cot 5): {len(nested_ents):,} token")
    print(f"     Vi du (10 dau):")
    for tok, lbl4, lbl5 in nested_ents[:10]:
        print(f"       {tok:<22}  col4={lbl4:<10}  col5={lbl5}")

    # 3. Dấu nháy trong span
    quote_in_ent = sum(
        1
        for s in train_sents
        for tok, lbl in zip(s['tokens'], s['labels'])
        if (lbl.startswith('B-') or lbl.startswith('I-'))
           and ('"' in tok or "'" in tok)
    )
    print(f"\n  3. Token dau nhay trong entity span: {quote_in_ent:,}")

    # 4. Viết tắt POS=Ny
    ny_tokens = [
        (s['tokens'][i], s['labels'][i])
        for s in train_sents
        for i, p in enumerate(s['pos'])
        if p == 'Ny'
    ]
    print(f"\n  4. Token viet tat (POS=Ny): {len(ny_tokens):,}")
    for tok, lbl in ny_tokens[:10]:
        print(f"       {tok:<12}  {lbl}")

    # 5. Câu cực ngắn (1 token)
    short = [s for s in train_sents if len(s['tokens']) == 1]
    print(f"\n  5. Cau 1 token: {len(short):,}")
    for s in short[:5]:
        print(f"       '{s['tokens'][0]}'  {s['labels'][0]}  (file: {s['source']})")


# ════════════════════════════════════════════════════════════════════════════ #
#  Standalone run                                                               #
# ════════════════════════════════════════════════════════════════════════════ #

if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from src.data_loader import (
        load_folder, normalize_bio, train_dev_split,
    )

    base    = Path(__file__).parent.parent / 'Data'
    out_dir = str(Path(__file__).parent.parent / 'output')

    print("Doc train ...")
    train_all   = normalize_bio(load_folder(str(base / 'train'), has_labels=True))
    train_sents, dev_sents = train_dev_split(train_all)

    print("Doc test ...")
    test_sents  = load_folder(str(base / 'test'), has_labels=False)

    stats_tr = analyze_dataset(train_sents, 'Train')
    stats_dv = analyze_dataset(dev_sents,   'Dev')
    stats_te = analyze_dataset(test_sents,  'Test')

    print_report(stats_tr)
    print_report(stats_dv)
    print_report(stats_te)

    analyze_vlsp_specifics(train_sents)

    for st in [stats_tr, stats_dv, stats_te]:
        p = plot_eda(st, save_dir=out_dir)
        print(f"  OK {p}")

    p = compare_splits([stats_tr, stats_dv, stats_te], save_dir=out_dir)
    if p:
        print(f"  OK {p}")

    # Lưu báo cáo text
    save_report(stats_tr, os.path.join(out_dir, 'report_train.txt'))
    save_report(stats_dv, os.path.join(out_dir, 'report_dev.txt'))
