# Báo cáo Cá nhân - Hồ Đắc Toàn - 2A202600057

## 1. Phạm vi phụ trách

Trong Lab Day 14, em phụ trách chính các hạng mục thuộc phần việc của Người 2:

- Hoàn thiện `agent/main_agent.py` để thay agent placeholder bằng RAG agent thật.
- Tạo hai phiên bản agent `v1` và `v2` để phục vụ benchmark so sánh.
- Hoàn thiện `engine/llm_judge.py` để chấm output bằng LLM Judge theo rubric rõ ràng.
- Xây dựng `multi-judge consensus`, tính `agreement_rate`, theo dõi `conflicts` và `judge_cost_usd`.
- Bổ sung kiểm tra `position bias` để đánh giá mức độ lệch do thứ tự trình bày đáp án.

Phần em làm tập trung vào hai lớp lõi của evaluation pipeline:

- lớp **Agent**: sinh câu trả lời và trả về retrieval evidence
- lớp **Judge**: chấm chất lượng câu trả lời một cách có cấu trúc

Các còn lại là thành viên khác phụ trách; phần của em là làm cho agent và judge tương thích, chạy được ổn định trong pipeline chung.

## 2. Engineering Contribution (15 điểm)

### 2.1. Đóng góp vào các module

Em trực tiếp triển khai hai module có độ khó kỹ thuật cao:

- `agent/main_agent.py`
- `engine/llm_judge.py`

Các commit chính chứng minh phần đóng góp:

- `72aece9` - `feat: implement real RAG agent V1/V2`
- `2763adc` - `feat: multi-judge evaluation engine`
- `4c940aa` - `update judge b llm model, update batchsize to 5`

Ngoài ra, file `agent/main_agent.py` còn được cập nhật để đồng bộ lại mô tả V1 theo đúng implementation thật: **V1 dùng fixed-size chunking**.

### 2.2. Hoàn thiện RAG Agent thật

Trong `agent/main_agent.py`, em thay stub bằng pipeline RAG thật:

`question -> retrieval -> build context -> OpenAI generation -> answer + contexts + retrieved_ids + metadata`

Agent hiện trả về đầy đủ các trường mà benchmark cần dùng:

- `answer`
- `contexts`
- `retrieved_ids`
- `metadata.model`
- `metadata.tokens_used`
- `metadata.prompt_tokens`
- `metadata.completion_tokens`
- `metadata.cost_usd`
- `metadata.latency_ms`

Điểm quan trọng là output của agent không chỉ để “trả lời”, mà còn để benchmark tính retrieval metrics, token usage, cost và latency.

### 2.3. Tối ưu prompt theo cấu trúc XML và R-T-C-F

Ngoài việc chỉ gọi model để sinh câu trả lời, em còn tối ưu prompt để agent và judge ổn định hơn trong benchmark.

Với agent, em tổ chức prompt theo hướng **XML-structured prompting**:

- `<system_role>`
- `<instructions>`
- `<constraints>`
- `<context>`
- `<user_question>`

Em xem đây là một cách triển khai thực tế của khung **R-T-C-F**:

- **R - Role**: xác định rõ model là QA Agent chỉ được trả lời từ context
- **T - Task**: nhiệm vụ là trả lời câu hỏi dựa trên tài liệu đã retrieve
- **C - Constraints**: cấm dùng internal knowledge, cấm suy diễn, cấm hallucinate
- **F - Format/Fallback**: quy định rõ kiểu output và câu fallback cố định khi thiếu thông tin

Điểm lợi của cách viết này là prompt rõ vai trò, rõ nhiệm vụ, rõ ràng buộc và giảm mơ hồ khi benchmark trên nhiều test case liên tiếp.

### 2.4. Tạo Version 1 và Version 2

Theo code hiện tại, em xác định rõ khác biệt giữa hai version như sau:

- `v1`: dùng `search_v1`, retrieval trên **fixed-size chunks**
- `v2`: dùng `search`, retrieval trên **paragraph-based chunks**

Phần chunking này được implement ở `engine/vector_store.py`:

- `FIXED_CHUNK_SIZE = 200`
- `chunk_text_fixed()` cho V1
- `chunk_text()` cho V2

Ở đây em chỉ tổ chức lại `vector_store.py`; phần em phụ trách là:

- tổ chức `MainAgent(version="v1" | "v2")`
- nối đúng `search_v1` và `search` vào agent
- giữ cho hai version có cùng interface để benchmark so sánh công bằng

Hiện tại cả 2 version sẽ là như này:

- **V1 là baseline fixed-size chunking**
- **V2 là bản cải tiến chuyển sang paragraph chunking**

Cả hai version cùng:

- prompt generation
- `top_k = 5`
- output schema giống nhau

Như vậy benchmark đo chủ yếu tác động của retrieval strategy lên answer quality, thay vì thay quá nhiều biến cùng lúc.

### 2.5. Hoàn thiện Multi-Judge

Trong `engine/llm_judge.py`, em triển khai:

- rubric 3 chiều: `accuracy`, `tone`, `safety`
- 2 judge model chạy song song bằng `asyncio.gather`
- tính `agreement_rate`
- phát hiện `conflicts` khi lệch quá 1 điểm
- trả về `judge_cost_usd`
- trả về reasoning theo từng dimension

Hiện cấu hình model là:

- RAG generation: `gpt-4o-mini`
- Judge A: `gpt-4o`
- Judge B: `gpt-4.1-mini`

Như vậy hệ thống hiện dùng 3 model khác nhau, đúng yêu cầu benchmark phân tách rõ agent và judge.

### 2.6. Chống Context Bleed trong prompt judge

Khi thiết kế `engine/llm_judge.py`, em không đưa dữ liệu vào prompt theo kiểu text phẳng, mà bọc bằng các tag có ngữ nghĩa rõ:

- `<question>`
- `<expected_answer>`
- `<agent_answer>`
- `<input_data>`
- `<output_format>`

Cách này giúp giảm hiện tượng **Context Bleed**, tức model trộn lẫn vai trò của các trường đầu vào hoặc suy luận sai do biên giữa các phần prompt không rõ.

Đặc biệt trong judge, việc tách rời `expected_answer` và `agent_answer` bằng tag rõ ràng rất quan trọng vì nếu prompt lẫn lộn hai phần này, điểm chấm sẽ thiếu ổn định và khó tin cậy.

### 2.7. Ép định dạng JSON để benchmark parse ổn định

Một phần tối ưu prompt khác em làm là **ép định dạng đầu ra** của judge về JSON cố định.

Trong rubric, em ghi rõ:

- chỉ trả về JSON hợp lệ
- không bọc trong markdown code block
- không thêm lời dẫn

Sau đó em định nghĩa schema output cụ thể gồm:

- `accuracy_reasoning`
- `tone_reasoning`
- `safety_reasoning`
- `accuracy`
- `tone`
- `safety`

Việc ép format kiểu này rất quan trọng với benchmark vì pipeline phía sau cần parse tự động. Nếu output tự do thì hệ thống rất dễ lỗi hoặc phải viết parser quá mong manh.

### 2.8. Giảm Position Bias bằng cách đảo thứ tự prompt

Trong `engine/llm_judge.py`, em còn giảm `Position Bias` bằng cách cho hai judge nhìn hai thứ tự prompt khác nhau:

- Judge A: `expected_answer -> agent_answer`
- Judge B: `agent_answer -> expected_answer`

Đây cũng là một phần của tối ưu prompt, vì cùng một nội dung nhưng cách sắp xếp khác nhau có thể làm judge chấm lệch. Em chủ động kiểm soát điểm này ngay từ prompt design thay vì chỉ xử lý sau khi benchmark xong.

### 2.9. Làm parser phòng thủ cho output của judge

Một vấn đề thực tế của LLM-as-Judge là model không phải lúc nào cũng trả đúng JSON như prompt yêu cầu. Nếu parse cứng, benchmark rất dễ vỡ ở runtime.

Em xử lý bằng `_parse_score()` theo hướng phòng thủ:

- strip markdown fence nếu có
- thử parse JSON chuẩn
- nếu fail thì fallback bằng regex
- nếu vẫn không đủ dữ liệu thì dùng giá trị trung tính

Điểm em ưu tiên ở đây là **pipeline không bị crash vì output malformed**.

## 3. Technical Depth

### 3.1. MRR

`MRR` là `Mean Reciprocal Rank`, dùng để đo tài liệu đúng xuất hiện ở vị trí nào trong danh sách retrieval.

Ví dụ:

- đúng ở hạng 1 -> `1.0`
- đúng ở hạng 2 -> `0.5`
- đúng ở hạng 3 -> `0.333...`

MRR quan trọng hơn việc chỉ biết “có retrieve đúng hay không”, vì trong RAG, tài liệu đúng xuất hiện càng sớm thì model càng dễ dùng đúng context hơn.

### 3.2. Cohen's Kappa và Agreement Rate

Trong pipeline hiện tại, em đang dùng `agreement_rate` như một chỉ số thực dụng:

- nếu chênh lệch ở từng dimension <= 1 thì xem là đồng thuận

Về mặt lý thuyết, `Cohen's Kappa` là thước đo mạnh hơn vì nó loại bỏ phần đồng ý ngẫu nhiên giữa hai judge. Em chưa implement Kappa trong code hiện tại, nhưng em hiểu đây là hướng mở rộng đúng nếu muốn đánh giá độ tin cậy của judge sâu hơn thay vì chỉ dừng ở agreement rate.

### 3.3. Position Bias

`Position Bias` là hiện tượng LLM Judge cho điểm bị ảnh hưởng bởi thứ tự trình bày hai đáp án, thay vì chỉ nhìn nội dung.

Để giảm rủi ro này, em dùng hai cách:

- Judge A nhìn theo thứ tự `expected_answer -> agent_answer`
- Judge B nhìn theo thứ tự `agent_answer -> expected_answer`

Ngoài ra em có thêm `check_position_bias()` để chạy cùng một judge với hai thứ tự prompt khác nhau, từ đó đo `bias_delta`.

### 3.4. XML, R-T-C-F, Context Bleed và JSON Enforcement

Qua phần prompt engineering cho agent và judge, em hiểu rõ hơn một số nguyên tắc giúp hệ thống ổn định:

- **XML structure** giúp phân tách rõ vai trò, nhiệm vụ, context và output format
- **R-T-C-F** giúp prompt có cấu trúc rõ ràng hơn: Role, Task, Constraints, Format
- **Context Bleed prevention** giúp model không nhầm lẫn giữa `expected_answer`, `agent_answer` và `question`
- **JSON enforcement** giúp output chấm điểm parse được ổn định trong pipeline tự động

Những kỹ thuật này không chỉ làm prompt “đẹp hơn”, mà trực tiếp làm benchmark đáng tin cậy hơn vì giảm sai lệch do format và giảm lỗi tích hợp giữa các module.

### 3.5. Trade-off giữa chi phí và chất lượng

Qua phần agent và judge, em hiểu rõ hơn trade-off của hệ thống evaluation:

- model mạnh hơn cho judge thường đáng tin hơn nhưng đắt hơn
- multi-judge tăng độ tin cậy nhưng cũng tăng chi phí benchmark
- prompt strict giúp giảm hallucination nhưng có thể làm tăng false refusal

Vì vậy trong code em giữ lại đầy đủ:

- token usage
- generation cost
- judge cost
- latency

để các quyết định tối ưu sau này có dữ liệu định lượng thay vì cảm tính.

## 4. Problem Solving (10 điểm)

### 4.1. Giải quyết biên async/sync

Một vấn đề quan trọng em gặp là retrieval trong vector store là synchronous, nhưng agent lại chạy trong async pipeline. Nếu gọi trực tiếp, retrieval sẽ block event loop.

Em xử lý bằng:

```python
retrieved = await asyncio.to_thread(self._search_fn, question, self._top_k)
```

Giải pháp này giúp giữ nguyên vector layer hiện có nhưng vẫn làm agent chạy ổn trong benchmark async.

### 4.2. Giải quyết key mismatch giữa các module

Khi tích hợp agent với retrieval layer, em phải kiểm tra kỹ schema trả về của `search()` và `search_v1()` để tránh lỗi runtime do dùng nhầm key.

Thay vì giả định mơ hồ, em bám đúng output retrieval gồm:

- `text`
- `article_file`
- `article_title`
- `chunk_index`

Từ đó mới chuẩn hóa output agent sao cho runner, evaluator và judge đều đọc được.

### 4.3. Giữ benchmark dễ giải thích

Trong quá trình làm, em chủ động giữ khác biệt giữa V1 và V2 ở mức có kiểm soát:

- cùng prompt
- cùng `top_k`
- cùng schema output
- chỉ thay retrieval strategy

Việc này giúp benchmark ra kết quả dễ giải thích hơn:

- nếu V2 tốt hơn thì có thể quy phần lớn công lao cho chunking strategy
- nếu V2 chưa tốt hơn, nhóm cũng biết phải phân tích lại retrieval design thay vì nghi ngờ hàng loạt biến khác

## 5. Tự đánh giá:

Em tự đáng giá theo đóng góp của mình bám sát 3 tiêu chí cá nhân:

- **Engineering Contribution**: có code thật ở các module khó, có commit chứng minh, có vai trò rõ trong agent và multi-judge.
- **Technical Depth**: hiểu và giải thích được MRR, Cohen's Kappa, Position Bias, cùng trade-off chi phí và chất lượng.
- **Problem Solving**: xử lý được các vấn đề tích hợp thực tế như async/sync boundary, schema mismatch, malformed judge output.

## 6. Nếu có thêm thời gian

Ba việc em muốn làm tiếp để tăng chất lượng bài lab:

- thêm manual spot-check cho judge để hoàn thành đúng Phase 3 - Step 9
- viết test riêng cho `_parse_score()` với nhiều kiểu output lỗi
- thêm một lớp cải tiến rõ hơn cho V2 như reranking để khoảng cách V2 so với V1 thuyết phục hơn, thay vì chỉ dừng ở chunking strategy
