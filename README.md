# Hướng Dẫn Chạy Dự Án NER

## 1. Hướng dẫn Huấn luyện và Thực thi Mô hình BiLSTM-CRF (Baseline)

Mô hình BiLSTM-CRF (tích hợp trích xuất đặc trưng ký tự) được triển khai thông qua tệp mã nguồn `train.py`. Quá trình huấn luyện có thể được thực thi trực tiếp qua giao diện dòng lệnh (Command Line Interface - CLI) với các thiết lập tùy chỉnh.

### 1.1. Yêu cầu Tiền quyết
* Đảm bảo môi trường Python đã được kích hoạt (Virtual Environment).
* Dữ liệu huấn luyện phải được đặt trong cấu trúc thư mục quy định. Mặc định, hệ thống sẽ đọc dữ liệu từ thư mục `Data/train/`. Dữ liệu cần được định dạng theo tiêu chuẩn CoNLL (Word - Label).

### 1.2. Khởi chạy Huấn luyện Cơ bản
Để tiến hành huấn luyện mô hình với các siêu tham số (hyperparameters) mặc định, thực thi câu lệnh sau tại thư mục gốc của dự án:
```powershell
python train.py
```
Khi thực thi lệnh trên, hệ thống sẽ áp dụng các thông số thiết lập sẵn bao gồm:
* Nguồn dữ liệu: `Data`
* Thư mục đầu ra (lưu trọng số và từ điển): `output/bilstm_crf_baseline`
* Số vòng lặp huấn luyện (Epochs): `5`
* Kích thước lô (Batch size): `32`
* Tốc độ học (Learning rate): `1e-3`
* Tỉ lệ tập xác thực (Validation/Dev ratio): `0.1` (10%)
* Thiết bị (Device): Tự động ưu tiên `cuda` nếu có phần cứng hỗ trợ, ngược lại sẽ sử dụng `cpu`.

### 1.3. Tùy chỉnh Siêu tham số (Hyperparameters)
Tệp `train.py` hỗ trợ điều chỉnh các siêu tham số trong quá trình huấn luyện thông qua các đối số dòng lệnh (arguments). 

Một số tùy chỉnh phổ biến:
* **Điều chỉnh số vòng lặp và thư mục lưu trữ:**
  ```powershell
  python train.py --epochs 20 --output-dir "output/bilstm_v2"
  ```
* **Điều chỉnh cấu trúc kiến trúc mạng Nơ-ron:**
  ```powershell
  python train.py --hidden-dim 512 --num-layers 3 --embedding-dim 256
  ```

**Danh sách các đối số có thể cấu hình:**
* `--data-dir`: Đường dẫn thư mục chứa dữ liệu đầu vào.
* `--output-dir`: Đường dẫn thư mục lưu trữ mô hình đầu ra, bộ từ vựng và lịch sử huấn luyện.
* `--epochs`: Số vòng lặp huấn luyện tối đa.
* `--batch-size`: Số lượng mẫu xử lý trong mỗi lô (batch).
* `--lr`: Tốc độ học của thuật toán tối ưu hóa (Optimizer).
* `--max-seq`: Độ dài chuỗi tối đa của một câu.
* `--hidden-dim`: Kích thước của tầng ẩn (hidden state) trong mạng LSTM.
* `--num-layers`: Số lượng tầng (layers) của mạng LSTM.

### 1.4. Kết quả Đầu ra
Sau khi tiến trình huấn luyện hoàn tất, các tệp tin kết quả sẽ được tự động lưu tại thư mục đầu ra đã chỉ định (mặc định: `output/bilstm_crf_baseline`), bao gồm:
1. **`best_model.pt`**: Tệp tin lưu trữ trọng số của mô hình đạt hiệu năng (điểm F1) tốt nhất trên tập dữ liệu xác thực (Validation set).
2. **`vocabs.pkl`**: Tệp tin nhị phân chứa các bộ từ điển ánh xạ (mapping) từ vựng, ký tự và nhãn sang ID tương ứng. Đây là thành phần bắt buộc để tải lại mô hình phục vụ cho quá trình suy luận (Inference).
3. **`history.json`**: Tệp tin định dạng JSON ghi chú lại chi tiết lịch sử suy hao (loss) và các chỉ số đo lường (metrics) qua từng vòng lặp huấn luyện.

### 1.5. Suy luận (Inference)
Để chạy mô hình đã huấn luyện trên một văn bản bất kỳ, sử dụng script `predict.py`:
```powershell
python predict.py --text "Ông Nguyễn Văn A đến Hà Nội." --model-path "output/bilstm_crf_baseline/best_model.pt" --vocab-path "output/bilstm_crf_baseline/vocabs.pkl"
```

> [!TIP]
> Xem chi tiết tại [BILSTM_GUIDE.md](file:///d:/Project-DL-NLP-IR-main/BILSTM_GUIDE.md) để biết thêm về cách cấu hình và các tham số khác.

---

## 2. Hướng dẫn Huấn luyện Notebook NER PhoBERT-CRF (Main Model)

Tài liệu này hướng dẫn cách chạy notebook `src/phobert/phobert_ner.ipynb`, tập trung vào cách thiết lập đường dẫn tùy theo môi trường (local, Colab, Kaggle).

### 2.1. Nguyên tắc thiết lập đường dẫn
Notebook có một số biến cấu hình đường dẫn trong các cell huấn luyện (train), đánh giá (evaluate) và suy luận (inference). Cần chú ý 3 loại đường dẫn chính:
- Đường dẫn dữ liệu huấn luyện.
- Đường dẫn thư mục lưu checkpoint.
- Đường dẫn file mô hình để nạp lại khi đánh giá hoặc suy luận.

**Quy tắc:**
1. Nếu chạy trên máy cá nhân (local), ưu tiên dùng đường dẫn tương đối từ thư mục gốc dự án.
2. Nếu chạy trên nền tảng cloud (Kaggle, Colab), thay sang đường dẫn mount tương ứng của nền tảng đó.
3. Đảm bảo tính thống nhất trong cách đặt đường dẫn giữa quá trình train và evaluate để tránh sự cố lệch dữ liệu/trọng số.

### 2.2. Các biến cấu hình cần kiểm tra
Trong notebook `src/phobert/phobert_ner.ipynb`, cần kiểm tra và thay thế các biến sau:

**1. Cell huấn luyện chính:**
- `TRAIN_PATH`: Đường dẫn dữ liệu huấn luyện.
- `CFG["out_dir"]`: Thư mục lưu checkpoint mô hình.

**2. Cell đánh giá độc lập:**
- `TRAIN_DIR`: Đường dẫn dữ liệu để dựng lại tập test (nếu cần thiết).
- `MODEL_PATH`: Đường dẫn tệp `.pt` cần nạp.

**3. Cell suy luận văn bản:**
- `MODEL_PATH`: Đường dẫn tệp `.pt` cần nạp.

**Ví dụ thiết lập đường dẫn (Tham khảo):**
```python
# Dữ liệu
TRAIN_PATH = Path("<duong_dan_den_thu_muc_du_lieu>")
TRAIN_DIR = "<duong_dan_den_thu_muc_du_lieu>"

# Nơi lưu model sau train
CFG["out_dir"] = "<thu_muc_luu_checkpoint>"

# Nơi load model để evaluate/infer
MODEL_PATH = "<duong_dan_day_du_den_file_best_model.pt>"
```

### 2.3. Quy trình thực thi các cell (Khuyến nghị)
Thực thi tuần tự các ô mã (cell) trong notebook theo hướng dẫn dưới đây để đảm bảo mô hình hoạt động ổn định:

- **Cell 0 — Cài đặt thư viện:** Cài đặt `seqeval`, `pytorch-crf` và điều chỉnh logging. Thực thi đầu tiên nếu môi trường chưa có đủ phụ thuộc.
- **Cell 1 — Thiết lập Reproducibility & Tiền xử lý:** Khởi tạo `SEED` và các hàm đọc dữ liệu, phân chia tập (train/dev).
- **Cell 2 — `VLSPDatasetPhoBERT`:** Định nghĩa `Dataset` tương thích với Tokenizer của PhoBERT.
- **Cell 3 — `PhoBERTCRF` model:** Khởi tạo kiến trúc mô hình PhoBERT + CRF.
- **Cell 4 — `NERTrainer`:** Định nghĩa class huấn luyện bao gồm các hàm `train()`, `evaluate()` và lưu trữ checkpoint.
- **Cell 5 — Main:** Ô mã thực thi luồng huấn luyện chính. Lưu ý cập nhật `TRAIN_PATH`, `CFG["out_dir"]`, và `CFG["device"]` trước khi chạy.
- **Cell 6 — Standalone Evaluate:** Đánh giá mô hình đã huấn luyện. Cần đảm bảo `MODEL_PATH` trỏ chính xác đến tệp `.pt` đã có.
- **Cell 7 & 8 & 9 — Suy luận Tương tác:** Cài đặt thư viện `underthesea` (nếu cần), khởi tạo hàm `predict_text()` và chạy vòng lặp suy luận trực tiếp từ đầu vào văn bản.

### 2.4. Lệnh chạy Notebook trên Local
Khởi chạy PowerShell từ thư mục gốc, kích hoạt môi trường ảo và mở Jupyter Lab:
```powershell
& .\venv\Scripts\Activate.ps1
python -m pip install -r requirements_streamlit.txt
python -m jupyter lab
```
(Tùy chọn) Chạy notebook tự động bằng `nbconvert`:
```powershell
jupyter nbconvert --to notebook --inplace --execute src/phobert/phobert_ner.ipynb
```

### 2.5. Xử lý lỗi thường gặp
- **Không tìm thấy dữ liệu:** Kiểm tra lại độ chính xác của đường dẫn thư mục hoặc cấu trúc dữ liệu.
- **Không tìm thấy mô hình:** Đảm bảo quá trình huấn luyện đã lưu thành công tệp tin hoặc `MODEL_PATH` đang trỏ đúng vị trí.
- **Lỗi thiếu định nghĩa biến/hàm:** Nguyên nhân do thực thi các ô mã không theo thứ tự quy định.

---

## 3. Triển khai Giao diện Streamlit (`app_streamlit.py`)

Giao diện Web Demo được xây dựng bằng Streamlit nhằm cung cấp công cụ nhận dạng thực thể trực quan. Ứng dụng yêu cầu tệp trọng số (`best_model.pt`) và tập tin ánh xạ nhãn (`label_mapping.json`).

### 3.1. Chuẩn bị Mô hình (Từ Kaggle)
1. Truy cập Kernel Kaggle: `https://www.kaggle.com/code/khuynhongvn/dl-final2`
2. Tại mục **Output**, tìm và tải thư mục `checkpoints/phobert_crf`.
3. Giải nén và thu thập các tệp: `best_model.pt`, `label_mapping.json`.

*(Tùy chọn) Sử dụng Kaggle CLI:*
```powershell
kaggle kernels output khuynhongvn/dl-final2 -p ./kaggle_output --unzip
```

### 3.2. Cấu hình Dự án
Tạo thư mục checkpoint trong dự án và di chuyển các tệp đã tải vào:
```powershell
mkdir checkpoints\phobert_crf
Move-Item .\kaggle_output\checkpoints\phobert_crf\best_model.pt .\checkpoints\phobert_crf\
Move-Item .\kaggle_output\checkpoints\phobert_crf\label_mapping.json .\checkpoints\phobert_crf\
```
*Lưu ý: Có thể đặt `best_model.pt` trực tiếp vào thư mục gốc của dự án, lúc này có thể bỏ qua bước cấu hình đường dẫn phía dưới.*

### 3.3. Cập nhật mã nguồn `app_streamlit.py`
Mở `app_streamlit.py` và cập nhật biến `MODEL_PATH` để trỏ đúng vị trí lưu tệp `.pt`.
```python
MODEL_PATH = PROJECT_ROOT / "checkpoints" / "phobert_crf" / "best_model.pt"
```

### 3.4. Khởi chạy Ứng dụng
```powershell
& .\venv\Scripts\Activate.ps1
python -m pip install -r requirements_streamlit.txt
streamlit run app_streamlit.py
```
Sau khi khởi chạy thành công, truy cập địa chỉ `http://localhost:8501` trên trình duyệt để sử dụng giao diện hệ thống.
