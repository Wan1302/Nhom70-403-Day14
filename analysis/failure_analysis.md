# Báo cáo Phân tích Thất bại (Failure Analysis Report)

> Báo cáo này được cập nhật dựa trên `reports/summary.json` và `reports/benchmark_results.json` sinh ra sau lần chạy benchmark gần nhất cho `Agent_V2_Optimized`.

## 0. Chất lượng Retrieval

| Metric | Kết quả | Ghi chú |
|--------|---------|--------|
| Hit Rate | **86.67%** | 78/90 case có ít nhất một `expected_retrieval_id` xuất hiện trong top-k kết quả truy hồi |
| MRR | **0.7707** | Tài liệu đúng thường xuất hiện khá cao trong danh sách truy hồi, nhưng vẫn còn một số case bị nhiễu hoặc truy hồi sai article |

**Chiến lược chunking hiện tại:** V2 đang dùng paragraph-based chunking, còn V1 dùng fixed-size chunking. Kết quả benchmark cho thấy V2 cải thiện được điểm tổng hợp và relevancy, nhưng một số case factoid ngắn vẫn bị ảnh hưởng bởi prompt quá thận trọng hoặc retrieval ở article gần nghĩa.

**Embedding model hiện tại:** `multi-qa-MiniLM-L6-cos-v1` trong ChromaDB. Nếu nhóm muốn tối ưu chi phí ở phần evaluator riêng, có thể cân nhắc `text-embedding-3-small`, nhưng retrieval benchmark phải bám đúng embedding đã dùng khi lập chỉ mục.

**Mối liên hệ giữa Retrieval và Answer Quality:**  
Khi `Hit Rate` thấp, LLM thường không có đủ thông tin để trả lời đúng nên hoặc rơi vào fallback, hoặc dựa vào context sai article. Tuy nhiên benchmark cũng cho thấy retrieval đúng chưa đủ: có ít nhất 7 case `hit_rate = 1.0` nhưng agent vẫn trả `"I don't have enough information to answer this."`, tức là lỗi nằm ở bước generation/prompting chứ không phải retrieval.

**Quan sát quan trọng:**  
Những case mơ hồ, câu hỏi chỉ cần một fact rất ngắn, hoặc có nhiều article gần nghĩa nhau là nhóm dễ lỗi nhất. Đây là nhóm nên ưu tiên trong red teaming và failure analysis.

**Giải thích về `retrieved_ids` bị lặp trong `benchmark_results.json`:**  
Trong pipeline hiện tại, retrieval thực tế chạy ở **cấp chunk**, tức là ChromaDB trả về nhiều đoạn văn khác nhau. Tuy nhiên khi ghi ra `benchmark_results.json`, hệ thống đang map mỗi chunk về `article_file`. Vì vậy nếu top-k có nhiều chunk đều thuộc cùng một article, danh sách `retrieved_ids` sẽ xuất hiện nhiều ID giống nhau, ví dụ cùng là `S08_set2_a2`. Điều này không nhất thiết làm sai `Hit Rate` hoặc `MRR` ở cấp article, nhưng nó làm report khó đọc hơn và che khuất độ đa dạng thực sự của top-k retrieval. Nói cách khác, đây là hệ quả của việc **retrieval theo chunk nhưng đánh giá theo article**, không phải do dữ liệu benchmark bị trùng dòng.

---

## 1. Tổng quan Benchmark

| Chỉ số | Giá trị |
|--------|---------|
| Tổng số cases | **90** |
| Số case pass | **89** |
| Số case fail | **1** |
| Tỉ lệ pass | **98.89%** |
| Điểm RAGAS trung bình - Faithfulness | **0.8778** |
| Điểm RAGAS trung bình - Relevancy | **0.6956** |
| Điểm Retrieval trung bình - Hit Rate | **0.8667** |
| Điểm Retrieval trung bình - MRR | **0.7707** |
| Điểm LLM Judge trung bình | **4.6316** / 5.0 |
| Agreement Rate trung bình | **0.9889** |
| Latency trung bình | **1.1524** giây/case |
| Tổng token sử dụng | **62,894** |
| Tổng chi phí generation | **0.010269 USD** |
| Tổng chi phí judge | **0.209048 USD** |
| Tổng chi phí ước tính | **0.219317 USD** |

### Nhận xét nhanh
- Retrieval đang đủ tốt để làm nền cho benchmark, nhưng chưa thật sự ổn định ở các câu hỏi mơ hồ hoặc các article gần nghĩa.
- Multi-judge hoạt động ổn định, thể hiện qua `agreement_rate = 98.89%`, nghĩa là hai judge khá nhất quán.
- V2 được `APPROVE` trong regression vì `avg_score` tăng **+0.2074** và `relevancy` tăng **+0.0868**, dù `agreement_rate` và `faithfulness` giảm nhẹ.

---

## 2. Phân nhóm lỗi (Failure Clustering)

| Nhóm lỗi | Số lượng | Dấu hiệu điển hình | Nguyên nhân dự kiến |
|----------|----------|--------------------|---------------------|
| Hallucination / High-confidence mismatch | **3** | Judge vẫn chấm tương đối cao nhưng faithfulness/relevancy rất thấp | Agent bám prompt hoặc answer format chưa tốt, làm câu trả lời lệch khỏi context thật |
| Incomplete / Over-conservative fallback | **11** | Có context đúng hoặc hit rate cao nhưng agent vẫn trả `"I don't have enough information to answer this."` | Prompt anti-hallucination quá chặt, model ngại trả lời các fact ngắn |
| Tone Mismatch | **9** | Điểm tone từ judge thấp hơn phần accuracy/safety | Một số câu trả lời quá cụt hoặc quá máy móc vì prompt tối giản |
| Retrieval Miss | **12** | Không tìm thấy article đúng trong top-k | Chunking/indexing chưa ổn với câu hỏi mơ hồ, nhiều article gần nghĩa nhau |
| Ambiguous Handling Error | **3** | Case multi-turn hoặc thiếu ngữ cảnh không được xử lý theo cách clarify tối ưu | Agent chưa có chiến lược làm rõ yêu cầu, thường fallback chung chung |

### Cách đọc bảng này
- Lỗi lớn nhất hiện tại không phải judge, mà là sự kết hợp giữa retrieval chưa đủ sạch và generation quá thận trọng.
- Nhóm lỗi đáng ưu tiên nhất là `Retrieval Miss` và `Incomplete`, vì hai nhóm này kéo tụt trực tiếp hit rate, faithfulness và score đầu ra.
- Một số case xấu nhất có `hit_rate = 1.0` nhưng vẫn fail hoặc gần fail, cho thấy prompt/generation hiện là nút thắt quan trọng không kém retrieval.

---

## 3. Phân tích 5 Whys

> Chọn 3 case tệ nhất theo tổng hợp của `judge.final_score`, `faithfulness`, và `retrieval.hit_rate`.

### Case #1: "Who was the victor of the 'War of Currents'?"
1. **Triệu chứng:** Agent trả lời `"I don't have enough information to answer this."`, trong khi đáp án đúng là `Nikola Tesla`.
2. **Why 1:** Retriever không lấy được đúng article mong đợi (`hit_rate = 0.0`).
3. **Why 2:** Query có tính khái niệm lịch sử, dễ bị kéo sang các article cùng thời kỳ hoặc cùng domain nhưng không chứa fact đúng.
4. **Why 3:** Vector similarity hiện chưa có reranking hoặc lexical rescue nên dễ ưu tiên article “na ná” thay vì article đúng.
5. **Why 4:** Bộ index hiện vẫn dựa chủ yếu vào embedding ở cấp chunk, chưa có cơ chế ưu tiên entity exact match.
6. **Root Cause:** Retrieval pipeline chưa đủ mạnh cho các câu hỏi cần khớp thực thể/lịch sử chính xác.

### Case #2: "In Turkish, which syllable usually has the stress?"
1. **Triệu chứng:** Đây là case fail duy nhất trong benchmark; agent fallback dù context đầu tiên đã chứa đúng câu trả lời.
2. **Why 1:** Model trả `"I don't have enough information to answer this."` dù `hit_rate = 1.0`.
3. **Why 2:** Prompt anti-hallucination yêu cầu chỉ trả lời khi model “chắc chắn”, khiến mô hình quá dè dặt với các fact ngắn.
4. **Why 3:** Câu trả lời mong đợi rất ngắn và nằm trong đoạn có thêm nhiều ngoại lệ, làm model không tự tin trích xuất câu trả lời trực tiếp.
5. **Why 4:** Không có bước post-processing hoặc extraction-specific prompting cho loại câu hỏi factoid ngắn.
6. **Root Cause:** Generation prompt đang quá bảo thủ; hệ thống thiếu chế độ “extractive answer” cho các câu hỏi fact-based đơn giản.

### Case #3: "What did he do after that?"
1. **Triệu chứng:** Agent trả về fallback chung, trong khi ground truth mong đợi một câu trả lời kiểu clarify rõ hơn.
2. **Why 1:** Câu hỏi phụ thuộc ngữ cảnh hội thoại trước nhưng benchmark hiện chạy theo single-turn.
3. **Why 2:** Retriever vẫn cố kéo về một số article gần nghĩa thay vì nhận diện đây là case thiếu ngữ cảnh.
4. **Why 3:** System prompt chỉ có một fallback chung `"I don't have enough information to answer this."`, không phân biệt giữa `out-of-context` và `need clarification`.
5. **Why 4:** Judge chấp nhận tương đối tốt câu fallback, nhưng relevancy/faithfulness vẫn bằng 0 nên case này kéo tụt metric phân tích.
6. **Root Cause:** Thiết kế agent hiện chưa hỗ trợ tốt ambiguous / multi-turn handling; prompt chưa tách rõ giữa “refuse” và “clarify”.

---

## 4. Kế hoạch cải tiến

### Ưu tiên cao
- [ ] Thêm reranking hoặc exact-match rescue cho các câu hỏi factoid / entity-based để giảm `Retrieval Miss`.
- [ ] Tách prompt thành hai mode: `extract answer directly` cho factoid và `clarify / insufficient context` cho ambiguous case.
- [ ] Giữ anti-hallucination, nhưng nới lỏng điều kiện fallback để mô hình không bỏ qua các đáp án đã hiện rõ trong context.
- [ ] Ghi log rõ `retrieved_ids`, `top chunk`, và câu context được dùng để trace các case fail về đúng tài liệu gốc.

### Ưu tiên trung bình
- [ ] Chuẩn hóa output schema giữa retrieval, ragas và judge để báo cáo dễ phân tích hơn.
- [ ] Tách riêng chiến lược xử lý `out_of_context`, `ambiguous`, và `multi-turn` thay vì dùng một fallback duy nhất.
- [ ] Theo dõi thêm thời gian toàn pipeline để xác nhận mục tiêu performance `< 2 phút cho 50 cases`.

### Ưu tiên bổ sung
- [ ] Thêm nhiều hard cases hơn cho negation trap, historical entity collision, và follow-up question.
- [ ] Tối ưu batch size hoặc chia phase retrieval/judge để giảm tổng chi phí benchmark.
- [ ] Dùng `benchmark_results.json` để tạo dashboard mini theo nhóm lỗi thay vì chỉ đọc raw JSON thủ công.

---

## 5. Kết luận

Benchmark hiện tại cho thấy hệ thống đã ở mức khá tốt về retrieval (`Hit Rate 86.67%`), judge consistency (`Agreement Rate 98.89%`) và score tổng (`4.63/5.0`). Tuy vậy, lỗi cốt lõi chưa nằm ở judge mà nằm ở hai điểm: retrieval chưa đủ bền ở các case mơ hồ / entity-heavy, và generation prompt còn quá thận trọng nên hay fallback ngay cả khi đã có context đúng.

V2 vẫn được `APPROVE` vì cải thiện `avg_score` và `relevancy` so với V1. Để tăng điểm lab và làm báo cáo thuyết phục hơn, nhóm nên ưu tiên sửa retrieval cho các case khó và tối ưu prompt để biến các case “retrieved đúng nhưng không dám trả lời” thành câu trả lời trực tiếp, ngắn gọn và bám context.
