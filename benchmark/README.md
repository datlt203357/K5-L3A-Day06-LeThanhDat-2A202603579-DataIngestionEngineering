# Benchmark cho Sổ tay sinh viên 2026

## Bộ câu hỏi

`questions.jsonl` có 20 câu: `q01`–`q15` là dev, `q16`–`q20` là test. Mỗi câu có đáp án kỳ vọng, trích dẫn evidence, ID nguồn và số trang PDF vật lý. Trang nguồn dùng số trang trong file PDF, không dùng số in trên trang.

Có 6 câu dùng filter metadata `section`: bốn câu dev (`q03`, `q10`, `q12`, `q15`) và hai câu test (`q18`, `q20`). Các giá trị filter phải được gán cho chunk theo `sections` trong `data/manifest.json`. Câu không có filter dùng `null`; cấu hình C phải chạy các câu đó không filter.

## Yêu cầu trước khi chạy

- Đối chiếu từng `evidence_quote` với PDF và xác nhận nó đủ hỗ trợ `expected_answer`; sửa quote nếu pipeline trích xuất có khác biệt về dấu câu/Unicode.
- Giữ nguyên câu hỏi, gold và split trong suốt quá trình thử. Chọn cấu hình bằng dev; không dùng test để tinh chỉnh. Nếu đã dùng test để sửa, ghi rõ trong báo cáo và tạo test mới nếu có thể.
- Không đưa đáp án hoặc gold source vào query hay filter. Filter chỉ dùng section mà ngữ cảnh câu hỏi đã nêu.
- Dùng cùng corpus, embedding model, vector store và metric cho A/B/C. So sánh A–B về chunking và B–C về metadata filter.
- Lưu top-3 đầy đủ cho mỗi câu: chunk ID/text, score, source/page/section, gold và nhãn đánh giá.

## Chỉ số và audit cần báo cáo

1. **Retrieval:** Hit@1 và Evidence Hit@3, báo số câu đúng/tổng câu riêng cho dev và test.
2. **Chunk coherence:** chọn trước cùng 10 vị trí nguồn; chấm chunk bao phủ mỗi vị trí từ 0–2 cho từng cấu hình, kèm lý do và điểm trung bình.
3. **Metadata usefulness:** kiểm tra `source`, `page` và metadata cần thiết trong top-3 của 5 câu test; báo số bản ghi metadata đủ/đúng trên tổng kiểm tra. Với câu có filter, báo số câu mà top-3 tuân thủ filter trên tổng câu filter.
4. **Grounding quality:** chấm top-3 của cả 5 câu test theo thang 0–2: 0 không có evidence, 1 evidence thiếu điều kiện/ngữ cảnh, 2 đủ evidence và truy được nguồn. Báo số câu ở từng mức, trung bình và lý do.
5. **Data strategy impact:** so sánh A–B và B–C, gồm số chunk, phân bố độ dài, thời gian ingest, chi phí, lỗi và đánh đổi.

Grounding ở đây đánh giá độ đủ của evidence retrieval, không phải độ chính xác của câu trả lời do LLM tạo. Ghi nhãn thủ công riêng với kết quả tự động; nêu rõ người chấm và giới hạn của corpus/test nhỏ.
