    # Starter · PostgreSQL + pgvector

Bộ khởi động **tùy chọn** cho lab cá nhân. Docker Compose tạo PostgreSQL + pgvector local. `src/` đã có đọc/chuẩn hóa PDF, chunking, embedder local, lưu/tìm vector và chạy benchmark retrieval tự động. Phần audit thủ công về coherence, metadata và grounding vẫn cần hoàn thiện trong báo cáo. Vẫn có thể chọn Qdrant, Chroma hoặc vector store tương đương theo [GUIDE.md](../GUIDE.md).

## Cấu trúc

```text
starter/
├── docker-compose.yml     # PostgreSQL 16 + pgvector, chỉ mở cổng local
├── init.sql               # bật extension khi tạo database mới
├── requirements.txt       # psycopg, pypdf, sentence-transformers, pgvector
├── .env.example           # cấu hình mẫu; copy thành .env
├── scripts/check_setup.py # kiểm tra kết nối và extension (đã chạy được)
├── src/
│   ├── models.py          # cấu trúc dữ liệu mẫu
│   ├── ingestion.py       # đọc PDF, chuẩn hóa và giữ truy vết trang/section
│   ├── chunking.py        # fixed-size và paragraph-aware
│   ├── embeddings.py      # adapter Sentence Transformers local
│   ├── store.py           # PostgreSQL + pgvector, cosine search, metadata filter
│   ├── evaluation.py      # Hit@1/Evidence Hit@3 và trace tự động
│   └── run.py             # chạy ingestion đến retrieval, ghi trace/kết quả
├── data/                  # bạn tự chuẩn bị corpus được phép sử dụng
└── ../results/            # trace, summary và metadata mỗi lần chạy
```

## 1. Cài và kiểm tra database

Cần Docker Desktop hoặc Docker Engine + Compose, Python **3.11–3.14**, và mạng để tải image/dependencies lần đầu. Các lệnh sau chạy từ thư mục `starter/`:

```bash
cd starter
python3 -m venv .venv
```

Kích hoạt `.venv`: macOS/Linux dùng `source .venv/bin/activate`; Windows PowerShell dùng `.venv\Scripts\Activate.ps1`. Sau đó:

```bash
python -m pip install -r requirements.txt
# macOS/Linux: cp .env.example .env
# Windows PowerShell: Copy-Item .env.example .env
docker compose -p day6_student_starter up -d --wait
python scripts/check_setup.py
```

Kỳ vọng: hiện tên database, phiên bản PostgreSQL và pgvector, sau đó `Setup OK`. Đây **chỉ là kiểm tra hạ tầng**, không phải kết quả benchmark. Docker dùng cổng `127.0.0.1:55433` và mật khẩu cố định **chỉ dùng local cho lab**; nếu cổng bận, sửa đồng thời port trong `docker-compose.yml` và `DATABASE_URL` trong `.env`. `init.sql` chỉ chạy khi volume mới được tạo; nếu database sẵn có chưa bật extension, xem log Docker và kiểm tra cấu hình thay vì xóa volume chứa dữ liệu.

Dừng container nhưng giữ dữ liệu:

```bash
docker compose -p day6_student_starter stop
```

## 2. Embedding model

Mặc định `.env.example` dùng `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` chạy local; đổi `EMBEDDING_MODEL` nếu chọn model khác. Model local không cần API key. Ghi tên model, số chiều và version thư viện vào báo cáo; dùng cùng model cho toàn bộ chiến lược và vector query tương thích vector document.

## 3. Hoàn thiện pipeline

1. Đặt corpus được phép sử dụng trong repo và hoàn thiện manifest. Đọc [GUIDE.md](../GUIDE.md) về **30 trang nội dung**, quyền dữ liệu và metadata.
2. Dùng cùng corpus, model và bộ câu hỏi cho A/B/C. Chạy `python -m src.run` từ thư mục `starter/`; pipeline tạo embedding, lưu/tìm vector và ghi trace top-3 cùng Hit@1/Evidence Hit@3.
3. Mỗi lần chạy tạo thư mục riêng trong `results/` với `retrieval_traces.jsonl`, `retrieval_summary.json` và `run_metadata.json`.
4. Hoàn thiện audit thủ công theo [BENCHMARK.md](../BENCHMARK.md): coherence trên 10 vị trí cố định, metadata trên 5 câu test và grounding trên đủ 5 câu test. Exact quote match tự động chỉ là proxy retrieval, không thay nhãn thủ công.

Lưu code, cấu hình, kết quả và lệnh chạy để người chấm tái lập; không commit key, dữ liệu mật hoặc dữ liệu không có quyền chia sẻ.
