# Hướng Dẫn Chạy Notebook NER PhoBERT-CRF

Tài liệu này hướng dẫn cách chạy notebook [src/phobert/phobert_ner.ipynb](src/phobert/phobert_ner.ipynb), tập trung vào cách **tự thay đường dẫn** theo môi trường của bạn (local, Colab, Kaggle).

## 1. Nguyên tắc thay đường dẫn

Notebook có một số biến cấu hình đường dẫn trong các cell train/evaluate/inference. Bạn chỉ cần nhớ 3 loại đường dẫn:

- Đường dẫn dữ liệu huấn luyện.
- Đường dẫn thư mục lưu checkpoint.
- Đường dẫn file model để nạp lại khi đánh giá hoặc suy luận.

Thay vì dùng cứng một đường dẫn mẫu, hãy áp dụng quy tắc sau:

1. Nếu chạy local, ưu tiên dùng đường dẫn tương đối từ thư mục gốc project.
2. Nếu chạy trên nền tảng cloud, thay sang đường dẫn mount tương ứng của nền tảng đó.
3. Dùng cùng một logic đặt đường dẫn cho train và evaluate để tránh lệch dữ liệu/weights.

## 2. Cần sửa những biến nào

Trong notebook [src/phobert/phobert_ner.ipynb](src/phobert/phobert_ner.ipynb), bạn cần kiểm tra và thay các biến sau:

1. Cell train chính:
      - `TRAIN_PATH` (đường dẫn dữ liệu train).
      - `CFG["out_dir"]` (thư mục lưu checkpoint).
2. Cell evaluate độc lập:
      - `TRAIN_DIR` (đường dẫn dữ liệu để dựng lại tập test nếu cần).
      - `MODEL_PATH` (đường dẫn file `.pt` cần nạp).
3. Cell suy luận văn bản:
      - `MODEL_PATH` (đường dẫn file `.pt` cần nạp).

Mẫu thay đường dẫn (tham khảo):

```python
# Dữ liệu
TRAIN_PATH = Path("<duong_dan_den_thu_muc_du_lieu>")
TRAIN_DIR = "<duong_dan_den_thu_muc_du_lieu>"

# Nơi lưu model sau train
CFG["out_dir"] = "<thu_muc_luu_checkpoint>"

# Nơi load model để evaluate/infer
MODEL_PATH = "<duong_dan_day_du_den_file_best_model.pt>"
```

## 3. Cách kiểm tra đã thay đúng chưa

Trước khi chạy train/evaluate, kiểm tra nhanh:

1. Đường dẫn dữ liệu có tồn tại và đọc được file `.txt`.
2. Sau khi train, thư mục checkpoint có tạo ra file model.
3. `MODEL_PATH` trỏ đúng file model vừa train (hoặc model bạn muốn dùng).
4. Nếu có file ánh xạ nhãn đi kèm model, đặt cùng thư mục để evaluate ổn định.

## 4. Thứ tự chạy cell khuyến nghị

Dưới đây là hướng dẫn chi tiết theo số cell của notebook. Mở notebook [src/phobert/phobert_ner.ipynb](src/phobert/phobert_ner.ipynb) và chạy các cell theo thứ tự phù hợp với mục tiêu của bạn.

Chi tiết các cell (theo đánh số trong notebook):

- Cell 0 — Cài đặt thư viện: chạy để cài `seqeval`, `pytorch-crf` và tắt log không cần thiết. Chạy Cell 0 trước khi làm việc nếu notebook chưa cài phụ thuộc.

- Cell 1 — Thiết lập reproducibility & hàm đọc dữ liệu:
      - Thiết lập `SEED`, hàm `parse_vlsp_file()`, `normalize_bio()`, `build_label_vocab()` và `split_train_dev()`.
      - Chạy Cell 1 để có sẵn các hàm tiền xử lý và log.

- Cell 2 — `VLSPDatasetPhoBERT`:
      - Định nghĩa `Dataset` dùng tokenizer (phải chạy trước khi tạo `DataLoader`).

- Cell 3 — `PhoBERTCRF` model:
      - Định nghĩa mô hình PhoBERT + CRF. Chạy để có class model.

- Cell 4 — `NERTrainer`:
      - Định nghĩa trainer gồm hàm `train()`, `evaluate()` và `_save_checkpoint()` (lưu `best_model.pt`, `label_mapping.json`, `best_config.json`).
      - Chạy Cell 4 trước khi train hoặc evaluate.

- Cell 5 — Main (Load data -> Build vocab -> Train -> Evaluate):
      - Đây là cell huấn luyện chính. Trước khi chạy, sửa các biến:
            - `TRAIN_PATH` — đường dẫn tới dữ liệu huấn luyện của bạn.
            - `CFG["out_dir"]` — thư mục lưu checkpoint.
            - `CFG["device"]` — (tùy chọn) `"cuda"` hoặc `"cpu"`.
      - Chạy Cell 5 sẽ thực hiện chia dataset (80/10/10 theo notebook), xây vocab, tạo `DataLoader`, khởi tạo model và gọi `trainer.train()`.

- Cell 6 — Standalone Evaluate (chạy độc lập, không cần train lại):
      - Dùng để tải model có sẵn và đánh giá trên tập test. Trước khi chạy, sửa:
            - `MODEL_PATH` — đường dẫn đầy đủ tới file `.pt` bạn muốn nạp.
            - `TRAIN_DIR` — (nếu notebook cần tái dựng test set từ train folder).
      - Cell này sẽ cố gắng load `label_mapping.json` từ cùng thư mục với `MODEL_PATH` nếu có.

- Cell 7 — Cài `underthesea` (nếu cần cho tiền xử lý tiếng Việt để inference):
      - Chỉ cần chạy nếu bạn sử dụng phần dự đoán văn bản tương tác.

- Cell 8 — Khởi tạo hàm `predict_text()`:
      - Tải model (sử dụng `MODEL_PATH`) và tokenizer, xây input từ câu người dùng, và trả về nhãn theo từ.
      - Trước khi chạy, đảm bảo `MODEL_PATH` trỏ tới model đã có.

- Cell 9 — Vòng lặp tương tác (nhập câu từ người dùng):
      - Chạy cell này để nhập câu và nhận nhãn NER theo thời gian thực. Gõ `exit` để thoát.

## 5. Ví dụ lệnh terminal (mở notebook, kích hoạt virtualenv)

Mở PowerShell (Windows) từ thư mục gốc project và kích hoạt virtualenv, sau đó chạy Jupyter Lab/Notebook:

```powershell
& .\venv\Scripts\Activate.ps1
python -m pip install -r requirements_streamlit.txt
python -m jupyter lab
```

Hoặc mở Jupyter Notebook:

```powershell
& .\venv\Scripts\Activate.ps1
python -m jupyter notebook
```

Gợi ý: nếu bạn chỉ muốn chạy toàn bộ notebook không tương tác, có thể dùng `nbconvert` để chạy và lưu output:

```powershell
& .\venv\Scripts\Activate.ps1
python -m pip install nbconvert
jupyter nbconvert --to notebook --inplace --execute src/phobert/phobert_ner.ipynb
```

## 5. Lỗi thường gặp khi thay đường dẫn

- Không tìm thấy dữ liệu:
     - Sai thư mục dữ liệu hoặc sai cấp thư mục.
- Không tìm thấy model:
     - Sai `MODEL_PATH` hoặc chưa train xong.
- Báo lỗi do chạy sai thứ tự cell:
     - Chưa chạy các cell định nghĩa class/hàm nhưng đã evaluate/infer.

## 6. Chạy giao diện Streamlit (`app_streamlit.py`) và cách tải model từ Kaggle

Nếu bạn muốn chạy app web demo thay vì notebook, sử dụng `app_streamlit.py` trong thư mục gốc dự án. App cần file mô hình (`best_model.pt`) và (khuyến nghị) `label_mapping.json` để ánh xạ nhãn chính xác.

Các bước chung (bằng tiếng Việt):

1) Tải mô hình từ phần Output của notebook Kaggle

- Mở trang kernel: https://www.kaggle.com/code/khuynhongvn/dl-final2
- Chọn tab **Output** (hoặc **Files** / **Output**). Tìm thư mục `checkpoints/phobert_crf` trong cấu trúc output.
- Tải xuống các file cần thiết (bằng trình duyệt):
      - `best_model.pt`
      - `label_mapping.json` (nếu có)
      - `best_config.json` (tuỳ chọn)

Hoặc dùng Kaggle CLI để tải toàn bộ output (yêu cầu đã cài `kaggle` và đăng nhập):

```powershell
# Tải và giải nén output kernel vào thư mục local ./kaggle_output
kaggle kernels output khuynhongvn/dl-final2 -p ./kaggle_output --unzip
```

Sau khi tải, bạn sẽ thấy cấu trúc tương tự `./kaggle_output/checkpoints/phobert_crf/best_model.pt`.

2) Đặt file mô hình vào dự án

- Cách A (khuyên dùng): tạo thư mục checkpoint trong project và đặt file vào đó:

```powershell
mkdir checkpoints\phobert_crf
Move-Item .\kaggle_output\checkpoints\phobert_crf\best_model.pt .\checkpoints\phobert_crf\
Move-Item .\kaggle_output\checkpoints\phobert_crf\label_mapping.json .\checkpoints\phobert_crf\
```

- Cách B: nếu bạn muốn giữ file ở thư mục gốc project, đặt `best_model.pt` và `label_mapping.json` vào thư mục gốc.

3) Cập nhật `app_streamlit.py` nếu cần

- Mặc định hiện tại `app_streamlit.py` dùng `MODEL_PATH = PROJECT_ROOT / "best_model.pt"`. Nếu bạn đã đặt model trong `checkpoints/phobert_crf`, chỉnh sửa `MODEL_PATH` ở đầu file thành:

```python
MODEL_PATH = PROJECT_ROOT / "checkpoints" / "phobert_crf" / "best_model.pt"
```

Hoặc để nguyên (nếu bạn đã di chuyển `best_model.pt` vào thư mục gốc).

4) Cài môi trường và chạy app Streamlit

```powershell
& .\venv\Scripts\Activate.ps1
python -m pip install -r requirements_streamlit.txt
streamlit run app_streamlit.py
```

Sau khi app chạy, truy cập `http://localhost:8501` (hoặc địa chỉ Streamlit hiển thị) để dùng giao diện.

Ghi chú:
- `app_streamlit.py` có cơ chế tìm `label_mapping.json` ở một số vị trí, nhưng đặt `label_mapping.json` cùng thư mục với `best_model.pt` là cách an toàn nhất.
- Nếu model tải về kèm thư mục con (ví dụ `checkpoints/phobert_crf/best_model.pt`), bạn có thể để nguyên cấu trúc và chỉ sửa `MODEL_PATH` trong `app_streamlit.py` cho phù hợp.
