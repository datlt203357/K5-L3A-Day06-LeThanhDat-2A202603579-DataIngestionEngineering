# Báo cáo cá nhân — Day 6

## Dữ liệu và pipeline
- Chủ đề: Sổ tay sinh viên 2026 của Đại học Bách khoa Hà Nội, nguồn `https://ctsv.hust.edu.vn/tan-sinh-vien/nhung-dieu-can-biet`, truy cập 2026-11-13. PDF có 55 trang vật lý; loại bìa 1, mục lục 4, thông tin xuất bản 55 còn 52 trang nội dung hữu ích. Theo manifest, quyền local cho nghiên cứu/thử nghiệm phi thương mại chỉ dựa trên ngoại lệ luật định nếu đáp ứng điều kiện; chưa có giấy phép mở hoặc văn bản chấp thuận tái phân phối từ HUST. Không xem đây là quyền công khai PDF/toàn văn hoặc trace chứa nhiều nội dung nguồn.
- Đọc PDF bằng pypdf; chuẩn hóa Unicode NFC, gộp whitespace trong từng dòng và giữ ranh giới đoạn, cắt khoảng trắng đầu/cuối. Trang rỗng/không đọc được: không có. Trang 8 có hiệu chỉnh `�ến độ` → `tiến độ` đã đối chiếu trực quan; lần chạy cuối khớp toàn bộ số ký tự manifest (114.469 ký tự chuẩn hóa trên 52 trang nội dung). Chunk sau token split có tối đa 120 token theo tokenizer; giới hạn model là 128 token và không chunk nào vượt giới hạn.
- Embedding local: Sentence Transformers 6.3.0, `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, 384 chiều; document dùng `encode_document`, query dùng `encode_query`, embeddings được chuẩn hóa và cùng model dùng cho A/B/C. Vector store: PostgreSQL 16.16 + extension pgvector 0.8.7, adapter Python `pgvector` 0.5.1, psycopg 3.3.8; cosine distance. Lệnh chạy từ `starter/`: `docker compose -p day6_student_starter up -d --wait`, `python scripts/check_setup.py`, `python -m src.run`. Vector lưu trong Docker volume `pg_data`; truy vấn cosine không có chỉ mục ANN nên là exact search trong cách triển khai này. API cost = 0; tài nguyên máy local chưa quy đổi.
- Metadata mỗi chunk: `document_id` và `source` là chuỗi lấy từ source trong manifest; `page` là số trang PDF vật lý; `section` là chuỗi `section_id` gán theo khoảng trang trong `manifest.sections`. `section` giúp giới hạn retrieval vào phạm vi chủ đề do câu hỏi cung cấp; hạn chế là section khá rộng và filter không đảm bảo đoạn evidence cụ thể nằm trong top-3.
- Nạp lại cùng corpus không cộng dồn bản ghi: `PgVectorStore.upsert` thay thế các dòng của strategy rồi upsert theo khóa `(strategy, chunk_id)`. Bảng hiện có đúng 511/444/444 dòng A/B/C, khớp số chunk của lượt chạy; khi ranh giới chunk thay đổi, thao tác thay thế cũng loại chunk cũ.

### Môi trường và thống kê chunk sau token split

Lượt chạy được lưu lúc `2026-11-14 03:01:38 UTC` trên Windows 10, Python 3.13.11. Các phiên bản thư viện chính: `sentence-transformers` 6.3.0, `transformers` 5.22.2, `torch` 2.15.0, `tokenizers` 0.23.3, `numpy` 2.5.4, `pypdf` 6.21.4, `psycopg` 3.3.8, Python adapter `pgvector` 0.5.1 và `python-dotenv` 1.2.4. PostgreSQL server 16.16, extension pgvector 0.8.7. Run metadata ghi model ID nhưng không pin revision/hash của model; đây là giới hạn tái lập môi trường.

Số token đếm bằng tokenizer của model, có tính special tokens ở mỗi chunk. Tổng ký tự/tokens là tổng trên các chunk sau token split; vì vậy A tính cả phần overlap lặp lại, còn B/C tính cả dấu phân cách đoạn được ghép vào chunk.

| Strategy | Chunks | Ký tự tổng | Ký tự trung bình / trung vị (min–max) | Tokens tổng | Tokens trung bình / trung vị (min–max) |
|---|---:|---:|---:|---:|---:|
| A | 511 | 125.341 | 245,29 / 249 (53–500) | 37.459 | 73,31 / 70 (20–120) |
| B | 444 | 115.012 | 259,04 / 243 (14–499) | 33.922 | 76,40 / 71 (8–120) |
| C | 444 | 115.012 | 259,04 / 243 (14–499) | 33.922 | 76,40 / 71 (8–120) |

## Cấu hình và benchmark
- A: fixed-size 500 ký tự, overlap 50; B: chunk theo đoạn, tối đa 500 ký tự; C: cùng chunk/embedding như B và áp dụng filter `section` chỉ khi câu hỏi có filter. C khác B ở bước truy xuất filter.
- Có 20 câu (15 dev/5 test), 6 câu có filter (4 dev/2 test), với ba giá trị section khác nhau. Quote đã được đối chiếu với trang nguồn chuẩn hóa; q06/q07 được mở rộng để bao gồm đủ các ý trong đáp án. Nguy cơ rò rỉ: dev/test cùng một tài liệu và cùng các section, nên chủ đề/cách diễn đạt gần nhau có thể xuất hiện ở cả hai split; kết quả không đại diện cho tài liệu khác.
- **Không có tiêu chí chọn chiến lược được ghi trước khi xem test.** Sau khi xem kết quả, không chọn một chiến lược thắng tổng quát và không dùng test để chỉnh tham số. Nếu cần chọn cấu hình triển khai hoặc điều chỉnh theo kết quả test, cần một test mới chưa xem.

## Kết quả
| Strategy | Split | Hit@1 (n/N) | Evidence Hit@3 (n/N) | Chunk count | Thời gian | Chi phí |
|---|---:|---:|---:|---:|---|---:|
| A | dev | 7/15 | 8/15 | 511 | Dùng chung lượt chạy | 0 API |
| A | test | 0/5 | 0/5 | 511 | Dùng chung lượt chạy | 0 API |
| B | dev | 10/15 | 12/15 | 444 | Dùng chung lượt chạy | 0 API |
| B | test | 1/5 | 1/5 | 444 | Dùng chung lượt chạy | 0 API |
| C | dev | 10/15 | 12/15 | 444 | Dùng chung lượt chạy | 0 API |
| C | test | 1/5 | 1/5 | 444 | Dùng chung lượt chạy | 0 API |

| Strategy | Coherence trung bình /2 (n) | Metadata đủ và đúng (n/N) | Filter tuân thủ (n/N hoặc N/A) | Grounding 0/1/2 (số câu) và trung bình /2 |
|---|---:|---:|---:|---|
| A | 1.1/2 (10) | 15/15 | 1/2 câu test có filter | 1/3/1; TB 1.0/2 (5) |
| B | 1.4/2 (10) | 15/15 | 1/2 câu test có filter | 1/3/1; TB 1.0/2 (5) |
| C | 1.4/2 (10) | 15/15 | 2/2 câu test có filter | 1/3/1; TB 1.0/2 (5) |

Hit@1 và Evidence Hit@3 là proxy tự động: chunk phải cùng trang gold và chứa nguyên văn evidence quote; chúng không thay cho đánh giá thủ công. Filter compliance tính ở mức câu hỏi: cả ba kết quả top-3 phải khớp filter được cung cấp. A/B không áp filter nhưng có thể tình cờ trả về đúng phạm vi; câu không có filter không tính vào mẫu. Trong cột grounding, bộ ba số lần lượt là điểm 0/1/2.

Thời gian chung của lượt chạy `run_20261114T030138Z`: ingestion 33.812 s; chunking 0.019 s; embedding + store 37.692 s; retrieval 2.497 s. Đây là thời gian gộp A/B/C, không phân bổ theo chiến lược. Model chạy local nên chi phí API là 0; chưa quy đổi chi phí tài nguyên máy.

Đường dẫn trace: `results/run_20261114T030138Z/retrieval_traces.jsonl`. Nhãn và lý do cho 30 mẫu coherence, 45 bản ghi metadata và 15 câu grounding được lưu tại `results/run_20261114T030138Z/manual_audit.csv`. Mẫu coherence gồm 10 trang nội dung đầu theo số trang vật lý: 2, 3, 5, 6, 7, 8, 9, 10, 11, 12; dùng cùng vị trí nội dung cho A/B/C. **Các vị trí được chọn sau khi đã xem kết quả retrieval, không phải mẫu preregistered.** Metadata được kiểm trên 5 câu test × top-3; grounding được chấm trên 5 câu test mỗi cấu hình theo thang 0–2. Người rà soát: Lê Thành Đạt (tác giả), đã xác nhận toàn bộ nhãn thủ công. Tôi đối chiếu từng dòng với trace, câu hỏi, manifest, dữ liệu chunk trong PostgreSQL và trang PDF; không có người chấm thứ hai độc lập. Theo RULES.md, báo cáo vẫn khai báo đã dùng Codex trong quá trình xây dựng pipeline và phân tích; kết luận audit cuối cùng do tác giả xác nhận. Quote q07 được sửa sau lượt chạy để giữ số thứ tự 2 và 3 đúng như trang 10; top-3 vẫn không chứa đủ cả hai ý nên nhãn retrieval q07 vẫn là miss.

## Phân tích và kết luận
- A–B: B cao hơn A trên dev (Evidence Hit@3 12/15 so với 8/15), test (1/5 so với 0/5) và coherence (1.4 so với 1.1/2). A tạo nhiều hơn B 67 chunk và tổng đầu vào cao hơn 3.537 token, chủ yếu do overlap lặp lại và special tokens được tính theo từng chunk; đây không phải lượng văn bản nguồn duy nhất tăng thêm. Các thời gian chỉ đo chung, không đủ để kết luận chênh lệch thời gian từng chiến lược. Ví dụ q16 (trang PDF 46): top-1 A/B/C dừng trước phần xử lý sự cố “24/7”; q19 (trang 51): top-3 cả ba chiến lược không lấy được evidence về mục đích làm thêm.
- B–C: nội dung chunk, số lượng và coherence giống nhau. Trên dev, cả A/B/C đều có top-3 tuân thủ filter ở 4/4 câu có filter; trên test, A/B là 1/2 và C là 2/2. Đây là quan sát hậu nghiệm trên hai câu test, không phải tiêu chí chọn đã đăng ký trước. q20 (trang 52) cho thấy C lọc đúng section chia sẻ kinh nghiệm nhưng top-3 vẫn thiếu câu nói cụ thể về khoản “lệ phí”; filter thu hẹp phạm vi mà không bảo đảm tìm được evidence.
- Metadata `document_id`/`source`/`page`/`section` đúng ở 45/45 chunk top-3 đã kiểm (5 câu test × top-3 × 3 chiến lược). Grounding trung bình đều 1.0/2 trên tập test nhỏ; q18 của A được chấm đủ khi ghép hai mảnh top-3, nhưng evidence bị phân mảnh qua nhiều chunk nên chỉ số exact-evidence tự động không ghi nhận hit. q16 và q17 chỉ có bằng chứng một phần; q19 không có bằng chứng.
- Không công bố chiến lược thắng tổng quát vì không ghi tiêu chí lựa chọn trước khi xem test. Trên dev, B/C có retrieval và coherence tốt hơn A; B/C có cùng nội dung và cùng 444 chunk mỗi strategy (database lưu mỗi strategy thành các dòng riêng). C có lợi thế vận hành khi filter section được cung cấp trước, nhưng hiệu quả riêng trên test mới chỉ dựa trên 2 câu filter. Tập test nhỏ nên các kết quả này chỉ mô tả corpus/bộ câu hỏi hiện tại.
- Về metadata usefulness: 45/45 chunk top-3 có `document_id`, `source`, `page`, `section` đúng; `section` cho phép truy vết và thu hẹp theo 7 phạm vi trong manifest. Hạn chế: một tài liệu duy nhất và section khá rộng, nên metadata đầy đủ không đồng nghĩa evidence đúng lọt top-3.
- Quyền tái phân phối sổ tay vẫn cần xác minh theo manifest. Câu q16/q17/q20 cho thấy ranh giới token có thể làm mất phần đầu/cuối evidence. Grounding đã được tác giả xác nhận; chỉ có một người chấm nên vẫn có nguy cơ thiên lệch chủ quan. Chi phí máy local chưa được quy đổi.
- Chọn mẫu coherence sau khi đã xem điểm retrieval tạo nguy cơ chọn mẫu hậu nghiệm; điểm coherence hiện là đánh giá thăm dò. Tác giả là người chấm duy nhất; dù đã xác nhận nhãn, vẫn có nguy cơ thiên lệch kỳ vọng và không có chấm chéo độc lập. Tập test chỉ có 5 câu, cùng một tài liệu; không khái quát hóa ngoài corpus.
- Nếu mở rộng, giữ câu hoàn chỉnh hơn khi chia theo token (hoặc bổ sung overlap token-aware), chọn tham số trên dev và dùng test mới nếu đã điều chỉnh theo kết quả test.
