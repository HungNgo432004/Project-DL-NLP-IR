import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import streamlit as st
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

try:
    from torchcrf import CRF
except ImportError as exc:
    raise RuntimeError(
        "Missing dependency 'pytorch-crf'. Install with: pip install pytorch-crf"
    ) from exc

try:
    from underthesea import word_tokenize
except ImportError:
    word_tokenize = None


# -----------------------------
# App configuration
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "best_model.pt"
MODEL_NAME = "vinai/phobert-base"
MAX_LEN = 256
MAX_CHARS = None

# Fallback mapping when label_mapping.json is unavailable.
DEFAULT_ID2LABEL = {
    0: "O",
    1: "B-PER",
    2: "I-PER",
    3: "B-LOC",
    4: "I-LOC",
    5: "B-ORG",
    6: "I-ORG",
    7: "B-MISC",
    8: "I-MISC",
}

ENTITY_NAMES = {
    "PER": "Người (PER)",
    "LOC": "Địa điểm (LOC)",
    "ORG": "Tổ chức (ORG)",
    "MISC": "Khác (MISC)",
}

ENTITY_COLORS = {
    "PER": "#0E7490",
    "LOC": "#2563EB",
    "ORG": "#D97706",
    "MISC": "#7C3AED",
    "OTHER": "#475569",
}


class PhoBERTCRF(nn.Module):
    def __init__(
        self,
        num_labels: int,
        model_name: str = "vinai/phobert-base",
        dropout: float = 0.1,
        use_crf: bool = True,
    ):
        super().__init__()
        self.use_crf = use_crf
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden)
        self.classifier = nn.Linear(hidden, num_labels)

        if use_crf:
            self.crf = CRF(num_labels, batch_first=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None,
    ):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq = self.dropout(self.layer_norm(out.last_hidden_state))
        logits = self.classifier(seq)
        mask = attention_mask.bool()

        if self.use_crf:
            if labels is not None:
                safe_labels = labels.clone()
                safe_labels[safe_labels == -100] = 0
                loss = -self.crf(logits, safe_labels, mask=mask, reduction="mean")
                preds = self.crf.decode(logits, mask=mask)
                return loss, preds
            preds = self.crf.decode(logits, mask=mask)
            return None, preds

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            return loss, logits.argmax(-1)
        return None, logits.argmax(-1)


def _safe_torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _find_label_mapping(num_labels: int) -> Dict[int, str]:
    candidates = [
        PROJECT_ROOT / "label_mapping.json",
        PROJECT_ROOT / "output" / "label_mapping.json",
        PROJECT_ROOT / "output" / "phobert_crf" / "label_mapping.json",
    ]

    candidates.extend(PROJECT_ROOT.glob("**/label_mapping.json"))

    for fp in candidates:
        if not fp.exists() or not fp.is_file():
            continue
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            id2label = data.get("id2label", {})
            mapped = {int(k): str(v) for k, v in id2label.items()}
            if len(mapped) == num_labels:
                return mapped
        except Exception:
            continue

    if num_labels == len(DEFAULT_ID2LABEL):
        return DEFAULT_ID2LABEL

    dynamic = {0: "O"}
    for i in range(1, num_labels):
        dynamic[i] = f"LABEL_{i}"
    return dynamic


def _segment_words(text: str) -> List[str]:
    clean_text = re.sub(r"\s+", " ", text.strip())
    if not clean_text:
        return []

    if word_tokenize is not None:
        segmented = word_tokenize(clean_text, format="text")
        tokens = [tok for tok in segmented.split(" ") if tok]
        if tokens:
            return tokens

    return [tok for tok in clean_text.split(" ") if tok]


def _decode_entities(words: List[str], labels: List[str]) -> List[Dict[str, str]]:
    entities = []
    current_type = None
    current_tokens: List[str] = []

    def flush_entity():
        nonlocal current_type, current_tokens
        if current_type and current_tokens:
            entities.append(
                {
                    "entity": current_type,
                    "text": " ".join(current_tokens).replace("_", " "),
                }
            )
        current_type = None
        current_tokens = []

    for word, label in zip(words, labels):
        if label == "O" or "-" not in label:
            flush_entity()
            continue

        prefix, tag = label.split("-", 1)
        tag = tag.strip()

        if prefix == "B":
            flush_entity()
            current_type = tag
            current_tokens = [word]
        elif prefix == "I" and current_type == tag:
            current_tokens.append(word)
        else:
            flush_entity()
            if prefix == "I":
                current_type = tag
                current_tokens = [word]

    flush_entity()

    for item in entities:
        item["entity_name"] = ENTITY_NAMES.get(item["entity"], item["entity"])

    return entities


@st.cache_resource(show_spinner=False)
def load_inference_components():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    state_dict = _safe_torch_load(MODEL_PATH, device)
    num_labels = int(state_dict["classifier.weight"].shape[0])
    id2label = _find_label_mapping(num_labels)

    model = PhoBERTCRF(
        num_labels=num_labels,
        model_name=MODEL_NAME,
        use_crf=True,
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    return model, tokenizer, device, id2label


def predict_ner(text: str, model, tokenizer, device, id2label: Dict[int, str]):
    words = _segment_words(text)
    if not words:
        return [], []

    norm_words = [w.replace("_", " ") for w in words]

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id

    subtoken_ids: List[int] = []
    first_positions: List[int] = []

    for word in norm_words:
        wtoks = tokenizer.tokenize(word)
        if not wtoks:
            wtoks = [tokenizer.unk_token]
        wids = tokenizer.convert_tokens_to_ids(wtoks)

        # Keep index of first sub-token for each original word.
        first_positions.append(len(subtoken_ids) + 1)
        subtoken_ids.extend(wids)

    max_body = MAX_LEN - 2
    if len(subtoken_ids) > max_body:
        valid_first_positions = []
        kept_words = []

        for idx, pos in enumerate(first_positions):
            if pos - 1 < max_body:
                valid_first_positions.append(pos)
                kept_words.append(words[idx])
            else:
                break

        words = kept_words
        first_positions = valid_first_positions
        subtoken_ids = subtoken_ids[:max_body]

    input_ids = [cls_id] + subtoken_ids + [sep_id]
    attn_mask = [1] * len(input_ids)

    pad_len = MAX_LEN - len(input_ids)
    input_ids += [pad_id] * pad_len
    attn_mask += [0] * pad_len

    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    mask_tensor = torch.tensor([attn_mask], dtype=torch.long, device=device)

    with torch.no_grad():
        _, preds = model(input_ids=input_tensor, attention_mask=mask_tensor, labels=None)

    pred_ids = preds[0]
    word_labels = []
    for pos in first_positions:
        if pos < len(pred_ids):
            word_labels.append(id2label.get(int(pred_ids[pos]), "O"))
        else:
            word_labels.append("O")

    entities = _decode_entities(words, word_labels)
    return entities, list(zip(words, word_labels))


def _render_entity_badges(entities: List[Dict[str, str]]):
    if not entities:
        st.info("Không tìm thấy thực thể nào trong đoạn văn bản.", icon="ℹ️")
        return

    chips = []
    for ent in entities:
        color = ENTITY_COLORS.get(ent["entity"], ENTITY_COLORS["OTHER"])
        chips.append(
            f"""
            <span style=\"display:inline-block;margin:4px 6px 4px 0;padding:6px 14px;\
            border-radius:20px;background:{color};color:white;font-size:0.9rem;\
            font-weight:500;box-shadow: 0 2px 4px rgba(0,0,0,0.1);\">\
            <b>{ent['entity_name']}</b>: {ent['text']}</span>
            """
        )

    st.markdown("".join(chips), unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title="Nhận Dạng Thực Thể Tiếng Việt",
        page_icon="✨",
        layout="centered",
    )

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        .block-container {padding-top: 1.5rem; padding-bottom: 3rem; max-width: 950px;}
        .title-box {
            background: linear-gradient(135deg, #4f46e5 0%, #0ea5e9 100%);
            padding: 1.8rem 2rem;
            border-radius: 16px;
            color: white;
            margin-bottom: 1.5rem;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        }
        .title-box h2 {
            margin: 0;
            font-weight: 700;
            letter-spacing: -0.025em;
            color: white;
            font-size: 2rem;
        }
        .subtle {
            color: #475569;
            font-size: 0.95rem;
            display: flex;
            gap: 1.5rem;
            align-items: center;
            margin-bottom: 1.5rem;
            padding: 0.8rem 1.2rem;
            background: #f8fafc;
            border-radius: 10px;
            border: 1px solid #e2e8f0;
        }
        .metric-card {
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 1.5rem;
            background: white;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
            transition: all 0.2s ease-in-out;
            text-align: center;
        }
        .metric-card:hover {
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
            transform: translateY(-2px);
        }
        .metric-value {
            font-size: 2.2rem;
            font-weight: 700;
            color: #0f172a;
            margin-top: 0.5rem;
        }
        .metric-label {
            font-size: 0.95rem;
            color: #64748b;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        /* Style text area */
        .stTextArea textarea {
            border-radius: 12px !important;
            border: 1px solid #cbd5e1 !important;
            padding: 1rem !important;
            font-size: 1.05rem !important;
            line-height: 1.6 !important;
            box-shadow: inset 0 2px 4px 0 rgba(0, 0, 0, 0.03) !important;
        }
        .stTextArea textarea:focus {
            border-color: #3b82f6 !important;
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2) !important;
        }
        /* Style button */
        .stButton>button {
            border-radius: 12px !important;
            font-weight: 600 !important;
            font-size: 1.05rem !important;
            padding: 0.5rem 1rem !important;
            height: 3.2rem !important;
            transition: all 0.2s !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="title-box">
            <h2>✨ Hệ Thống Nhận Dạng Thực Thể</h2>
            <div style="margin-top:10px;opacity:0.9;font-size:1.05rem;line-height:1.5;">Phát hiện và phân loại các thực thể PER (Người), LOC (Địa điểm), ORG (Tổ chức), MISC (Khác) từ văn bản tiếng Việt sử dụng mô hình PhoBERT + CRF.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.spinner("Đang tải mô hình vào bộ nhớ..."):
        model, tokenizer, device, id2label = load_inference_components()

    max_chars_display = str(MAX_CHARS) if MAX_CHARS is not None else "Không giới hạn"
    st.markdown(
        f"""
        <div class='subtle'>
            <span>🚀 <b>Mô hình:</b> {MODEL_PATH.name}</span>
            <span>⚡ <b>Thiết bị:</b> {str(device).upper()}</span>
            <span>📝 <b>Giới hạn:</b> {max_chars_display}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    default_text = (
        "Ngày 20/10/2023, ông Nguyễn Văn A - CEO Vingroup - đã họp với UBND thành phố Hà Nội "
        "tại khách sạn Sofitel Legend Metropole để thảo luận về dự án xe điện thông minh."
    )

    if MAX_CHARS is None:
        text = st.text_area(
            "Nhập đoạn văn bản cần nhận dạng thực thể",
            value=default_text,
            height=150,
            help="(Không giới hạn ký tự)",
            placeholder="Ví dụ: Tôi gặp ông Trần Văn B tại Đà Nẵng...",
        )
    else:
        text = st.text_area(
            "Nhập đoạn văn bản cần nhận dạng thực thể",
            value=default_text,
            max_chars=MAX_CHARS,
            height=150,
            help="Giới hạn ký tự để đảm bảo tốc độ dự đoán (demo).",
            placeholder="Ví dụ: Tôi gặp ông Trần Văn B tại Đà Nẵng...",
        )

    col1, col2 = st.columns([1, 2])
    with col1:
        run_btn = st.button("🔍 Phân tích văn bản", type="primary", use_container_width=True)
    with col2:
        st.markdown("<div style='padding-top:0.8rem;'></div>", unsafe_allow_html=True)
        show_tokens = st.checkbox("Hiển thị nhãn chi tiết từng token (Debug mode)", value=False)

    if run_btn:
        cleaned = re.sub(r"\s+", " ", text).strip()
        if not cleaned:
            st.warning("⚠️ Vui lòng nhập văn bản trước khi phân tích.")
            return

        with st.spinner("Đang xử lý phân tích..."):
            entities, token_labels = predict_ner(cleaned, model, tokenizer, device, id2label)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🎯 Kết Quả Nhận Dạng")
        
        st.markdown("<div style='background: #f8fafc; padding: 1.5rem; border-radius: 12px; border: 1px solid #e2e8f0; margin-bottom: 1.5rem;'>", unsafe_allow_html=True)
        _render_entity_badges(entities)
        st.markdown("</div>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f"""
                <div class='metric-card'>
                    <div class='metric-label'>Số Thực Thể Tìm Thấy</div>
                    <div class='metric-value'>{len(entities)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            unique_types = sorted({e["entity"] for e in entities})
            types_str = ', '.join(unique_types) if unique_types else 'Không có'
            st.markdown(
                f"""
                <div class='metric-card'>
                    <div class='metric-label'>Các Loại Thực Thể</div>
                    <div class='metric-value' style='font-size: 1.4rem;'>{types_str}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        if entities:
            st.markdown("### 📊 Bảng Tổng Hợp Chi Tiết")
            rows = [
                {
                    "Loại Thực Thể": e["entity_name"],
                    "Mã Nhãn": e["entity"],
                    "Nội Dung": e["text"],
                }
                for e in entities
            ]
            st.dataframe(rows, use_container_width=True)

        if show_tokens:
            st.markdown("### 🔬 Nhãn Cấp Độ Token")
            st.dataframe(
                [
                    {
                        "Token": tok.replace("_", " "),
                        "Nhãn": lbl,
                    }
                    for tok, lbl in token_labels
                ],
                use_container_width=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📌 Ghi chú hệ thống"):
        st.info(
            """
            - Dấu `_` trong token (được tạo ra bởi bộ tách từ) đã được chuyển thành khoảng trắng khi hiển thị kết quả để dễ đọc hơn.
            - Nếu không tìm thấy tệp `label_mapping.json`, ứng dụng sẽ sử dụng ánh xạ mặc định gồm 9 nhãn tiêu chuẩn.
            - Để kết quả chính xác và đồng nhất với quá trình huấn luyện nhất, hãy đảm bảo đặt `label_mapping.json` cùng thư mục với tệp mô hình (`best_model.pt`).
            """
        )


if __name__ == "__main__":
    main()
