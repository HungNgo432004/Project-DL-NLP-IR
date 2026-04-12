# -*- coding: utf-8 -*-
"""
Utilities đánh giá NER theo entity-level cho nhãn BIO.
"""

from collections import Counter
from typing import Dict, List, Sequence, Tuple


Entity = Tuple[int, int, str]


def extract_entities(labels: Sequence[str]) -> List[Entity]:
    entities: List[Entity] = []
    i = 0
    while i < len(labels):
        label = labels[i]
        if label.startswith('B-'):
            entity_type = label[2:]
            j = i + 1
            while j < len(labels) and labels[j] == f'I-{entity_type}':
                j += 1
            entities.append((i, j, entity_type))
            i = j
        else:
            i += 1
    return entities


def ner_scores(true_seqs: Sequence[Sequence[str]],
               pred_seqs: Sequence[Sequence[str]]) -> Dict:
    tp = 0
    fp = 0
    fn = 0

    per_type_tp: Counter = Counter()
    per_type_fp: Counter = Counter()
    per_type_fn: Counter = Counter()

    for true_labels, pred_labels in zip(true_seqs, pred_seqs):
        true_entities = set(extract_entities(true_labels))
        pred_entities = set(extract_entities(pred_labels))

        common = true_entities & pred_entities
        misses = true_entities - pred_entities
        extras = pred_entities - true_entities

        tp += len(common)
        fn += len(misses)
        fp += len(extras)

        for _, _, entity_type in common:
            per_type_tp[entity_type] += 1
        for _, _, entity_type in misses:
            per_type_fn[entity_type] += 1
        for _, _, entity_type in extras:
            per_type_fp[entity_type] += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else 0.0)

    per_type = {}
    entity_types = sorted(set(per_type_tp) | set(per_type_fp) | set(per_type_fn))
    for entity_type in entity_types:
        etp = per_type_tp[entity_type]
        efp = per_type_fp[entity_type]
        efn = per_type_fn[entity_type]
        eprecision = etp / (etp + efp) if etp + efp else 0.0
        erecall = etp / (etp + efn) if etp + efn else 0.0
        ef1 = (2 * eprecision * erecall / (eprecision + erecall)
               if eprecision + erecall else 0.0)
        per_type[entity_type] = {
            'precision': eprecision,
            'recall': erecall,
            'f1': ef1,
            'support': etp + efn,
        }

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'per_type': per_type,
    }
