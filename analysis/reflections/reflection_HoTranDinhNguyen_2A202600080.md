# Báo cáo Cá nhân - Hồ Trần Đình Nguyên - 2A202600080

## 1. Vai trò và phạm vi công việc

Trong Lab Day 14, em phụ trách chính các phần liên quan đến Data Pipeline và Retrieval:

- Thiết kế và xây dựng `data/synthetic_gen.py` để tạo Golden Dataset 90 cases (80 factual + 10 hard).
- Xây dựng `engine/vector_store.py`: chunking, indexing vào ChromaDB, và hàm search.
- Xây dựng `engine/retrieval_eval.py`: tính toán Hit Rate và MRR trên toàn bộ dataset.
- Điền Section 0 (Retrieval Quality) trong `analysis/failure_analysis.md` với số liệu thật từ benchmark.

---

## 2. Đóng góp kỹ thuật cụ thể

### 2.1. Golden Dataset (data/synthetic_gen.py)

Em xây dựng pipeline tạo dataset từ 3 file TSV thô (S08, S09, S10) với các bước:

1. **Load và chuẩn hóa**: Đọc với `encoding="latin-1"` để xử lý ký tự đặc biệt. Loại bỏ BOM ký tự (`ï»¿`) khỏi tên cột bằng `str.lstrip()`.
2. **Lọc dữ liệu bẩn**: Câu hỏi bị corrupted (ví dụ `"MassachuS08_setts"`) được lọc bằng regex `S0[89]_|S10_`. Loại bỏ các case có answer là yes/no vì không thể xác minh.
3. **Quality filter nghiêm ngặt**: Chỉ giữ lại case có `answer` xuất hiện chính xác trong file text → đảm bảo 100% context có thật.
4. **Cân bằng difficulty**: Sample đều 3 mức easy/medium/hard với `random_state=42` để tái lập được.
5. **10 Hard Cases**: Bao phủ 9 loại: out_of_context (×2), prompt_injection, goal_hijacking, conflicting, ambiguous, multi_turn, negation_trap, multi_turn_correction, latency_stress.

Kết quả: 90 cases với `expected_retrieval_ids` mapping chính xác theo article file.

### 2.2. Vector Store với Paragraph-based Chunking (engine/vector_store.py)

Trước khi quyết định chunking strategy, em phân tích phân phối độ dài thực tế của 8,948 paragraphs:
- 65% nằm trong khoảng 200–1000 chars
- Median: 409 chars

Từ đó thiết kế **paragraph-based chunking** gồm 3 bước:
1. Split theo `\n\n` để lấy đơn vị semantic tự nhiên
2. Merge đoạn < 150 chars vào đoạn kế tiếp
3. Split đoạn > 1000 chars tại sentence boundary

Embedding model chọn `multi-qa-MiniLM-L6-cos-v1` vì được thiết kế cho QA semantic search (512 token limit phù hợp với chunk size thực tế).

Kết quả: 9,570 chunks từ 150 files, lưu vào ChromaDB PersistentClient.

Thiết kế kỹ thuật đáng chú ý:
- **Singleton pattern** cho `_collection` để tránh tạo lại client mỗi lần gọi `search()`
- `article_file` dùng `file.name.replace(".txt.clean", "")` thay vì `file.stem` để tránh giữ lại `.txt` trong ID

### 2.3. Retrieval Evaluation (engine/retrieval_eval.py)

Em xây dựng `RetrievalEvaluator` với 2 metrics chuẩn:
- **Hit Rate**: Kiểm tra article đúng có nằm trong top-K kết quả không
- **MRR**: Đo thứ hạng của article đúng, phạt nặng nếu đứng sau

Thiết kế deduplication: search top_k×3 chunks rồi dedup theo `article_file`, đảm bảo sau khi loại trùng vẫn đủ K unique articles. Thứ tự giữ nguyên theo similarity → article nào có chunk tốt nhất đứng trước.

Vì `search()` là synchronous nhưng `evaluate_batch()` là async, em dùng `asyncio.to_thread()` để tránh block event loop.

Kết quả thực đo: **Hit Rate 86.67%** (78/90 cases), **MRR 0.7707** — theo số liệu benchmark tích hợp trong `failure_analysis.md`.

---

## 3. Những vấn đề kỹ thuật em đã gặp

### 3.1. UnicodeDecodeError trên dữ liệu thô

File TSV và file text đều encode theo `latin-1` nhưng code mẫu đọc bằng `utf-8-sig`. Lỗi này không xuất hiện ngay mà chỉ xuất hiện khi gặp ký tự đặc biệt trong nội dung. Em phát hiện qua traceback và đổi encoding toàn bộ pipeline.

### 3.2. BOM khiến ArticleTitle toàn NaN

Cột đầu tiên trong TSV có BOM `ï»¿` dính vào tên cột. Pandas đọc được nhưng tên cột bị sai → lookup bằng `"ArticleTitle"` trả về NaN hết. Em fix bằng:

```python
combined.columns = combined.columns.str.strip().str.lstrip('﻿').str.lstrip('ï»¿')
```

### 3.3. article_file ID không khớp với expected_retrieval_ids

Bug quan trọng nhất: dùng `file.stem` trên file `S08_set1_a1.txt.clean` trả về `S08_set1_a1.txt` (giữ lại `.txt`), trong khi `expected_retrieval_ids` lưu `S08_set1_a1`. Nếu không phát hiện, Hit Rate sẽ là 0% dù retrieval hoạt động đúng. Fix bằng `file.name.replace(".txt.clean", "")`.

### 3.4. Chunks cũ lẫn chunk mới trong ChromaDB

Sau khi sửa chunking logic, ChromaDB vẫn còn chunks từ lần build trước. Vì `get_or_create_collection()` không tự xóa, kết quả search bị lẫn dữ liệu cũ. Em phải delete collection và rebuild với `force=True`.

### 3.5. Blocking call trong async context

`search()` gọi ChromaDB là synchronous. Gọi trực tiếp trong `async evaluate_batch()` block toàn bộ event loop. Fix bằng `await asyncio.to_thread(search, question, top_k)`.

---

## 4. Kiến thức kỹ thuật em hiểu rõ hơn sau bài lab

### 4.1. Hit Rate vs MRR

Hit Rate chỉ trả lời "có tìm được không" (binary). MRR trả lời thêm "tìm được ở thứ hạng bao nhiêu". Kết quả Hit Rate 86.67% và MRR 0.7707 cho thấy: hệ thống tìm được đúng article trong phần lớn cases, nhưng không phải lúc nào cũng đưa nó lên top-1. Đây là tín hiệu cho thấy reranking sẽ cải thiện được MRR mà không cần thay đổi chunking hay embedding.

### 4.2. Tại sao Paragraph-based Chunking tốt hơn Fixed-size

Fixed-size chunking cắt theo ký tự không quan tâm ý nghĩa, dễ cắt đứt giữa câu hoặc gộp 2 đoạn không liên quan. Paragraph-based giữ nguyên semantic unit tự nhiên của văn bản. Khi embedding được tính trên một đoạn coherent, cosine similarity với câu hỏi chính xác hơn.

### 4.3. Singleton pattern trong long-running service

Nếu tạo `chromadb.PersistentClient` mỗi lần gọi `search()`, overhead là đáng kể và có thể gây lỗi file lock. Singleton đảm bảo chỉ khởi tạo 1 lần, tái sử dụng connection trong suốt lifetime của process.

### 4.4. Tầm quan trọng của data quality filter

Context quality ban đầu chỉ đạt ~52% do giữ lại các case yes/no và case không tìm được answer trong text. Sau khi áp dụng strict filter (exact answer must appear in text), chất lượng đạt 100% nhưng số lượng giảm. Đây là trade-off điển hình: ít case hơn nhưng đáng tin hơn.

---

## 5. Điều em rút ra

Phần việc em làm là nền tảng của toàn bộ pipeline: nếu dataset sai, hoặc retrieval ID không khớp, toàn bộ benchmark sẽ ra số liệu vô nghĩa dù các phần sau chạy đúng. Điều này cho thấy trong một hệ thống evaluation factory, data quality và ID consistency là điều kiện tiên quyết, không phải afterthought.

Em cũng nhận ra rằng số liệu retrieval (Hit Rate, MRR) cần được đo và báo cáo trước khi đánh giá generation. Nếu retrieval đã sai từ đầu, LLM không có context đúng và sẽ hallucinate — lỗi đó không phải lỗi của LLM.

---

## 6. Tự đánh giá đóng góp

Phần đóng góp của em tập trung ở ba mặt:

- Xây dựng Golden Dataset chất lượng cao với đầy đủ metadata và hard cases đa dạng
- Thiết kế vector store với chunking strategy có cơ sở từ phân tích dữ liệu thực tế
- Đạt Hit Rate 86.67% / MRR 0.7707 — đủ để xác nhận retrieval stage hoạt động tốt và không phải bottleneck chính của pipeline

Nếu có thêm thời gian, em muốn thực hiện ablation study thật sự: chạy fixed-size chunking, đo Hit Rate/MRR giảm xuống bao nhiêu, và dùng số liệu đó để chứng minh quyết định thiết kế có cơ sở định lượng thay vì chỉ lý luận lý thuyết.
