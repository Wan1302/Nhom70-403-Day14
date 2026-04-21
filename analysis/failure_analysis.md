# Báo cáo Phân tích Thất bại (Failure Analysis Report)

## 0. Retrieval Quality (Người 1 - đã đánh giá)

| Metric | Kết quả | Cases |
|--------|---------|-------|
| Hit Rate | **97.6%** | 83/90 cases có retrieval ID thực |
| MRR | **81.7%** | — |

**Chunking strategy:** Paragraph-based (merge <150 chars, split >1000 chars). Lý do: phân tích 8,948 paragraphs thực tế cho thấy 65% nằm trong 200–1000 chars (median 409 chars) — giữ nguyên semantic unit tốt hơn fixed-size.

**Embedding model:** `multi-qa-MiniLM-L6-cos-v1` — 512 token limit, thiết kế cho QA semantic search.

**Mối liên hệ Retrieval ↔ Answer Quality:** Hit Rate 97.6% → 2.4% cases Retrieval thất bại là nguyên nhân trực tiếp gây Hallucination ở downstream. Khi Vector DB trả về sai tài liệu, LLM không có context đúng và có xu hướng bịa câu trả lời.

**Case Retrieval thất bại điển hình:** *"What is the first number on the page?"* — câu hỏi quá mơ hồ, không có từ khóa đặc trưng, embedding không phân biệt được ngữ cảnh.

---

## 1. Tổng quan Benchmark
- **Tổng số cases:** 50
- **Tỉ lệ Pass/Fail:** X/Y
- **Điểm RAGAS trung bình:**
    - Faithfulness: 0.XX
    - Relevancy: 0.XX
- **Điểm LLM-Judge trung bình:** X.X / 5.0

## 2. Phân nhóm lỗi (Failure Clustering)
| Nhóm lỗi | Số lượng | Nguyên nhân dự kiến |
|----------|----------|---------------------|
| Hallucination | 5 | Retriever lấy sai context |
| Incomplete | 3 | Prompt quá ngắn, không yêu cầu chi tiết |
| Tone Mismatch | 2 | Agent trả lời quá suồng sã |

## 3. Phân tích 5 Whys (Chọn 3 case tệ nhất)

### Case #1: [Mô tả ngắn]
1. **Symptom:** Agent trả lời sai về...
2. **Why 1:** LLM không thấy thông tin trong context.
3. **Why 2:** Vector DB không tìm thấy tài liệu liên quan nhất.
4. **Why 3:** Chunking size quá lớn làm loãng thông tin quan trọng.
5. **Why 4:** ...
6. **Root Cause:** Chiến lược Chunking không phù hợp với dữ liệu bảng biểu.

## 4. Kế hoạch cải tiến (Action Plan)
- [ ] Thay đổi Chunking strategy từ Fixed-size sang Semantic Chunking.
- [ ] Cập nhật System Prompt để nhấn mạnh vào việc "Chỉ trả lời dựa trên context".
- [ ] Thêm bước Reranking vào Pipeline.
