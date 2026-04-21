# Báo cáo Cá nhân - Hồ Trọng Duy Quang

## 1. Vai trò và phạm vi công việc

Trong Lab Day 14, em phụ trách chính các phần liên quan đến:

- Thiết kế và hoàn thiện `engine/runner.py` để chạy benchmark bất đồng bộ, thu thập latency, token usage và cost.
- Xây dựng `ExpertEvaluator` trong `main.py`, tích hợp chấm điểm bằng RAGAS và kết hợp thêm retrieval metrics như `Hit Rate` và `MRR`.
- Viết logic `Regression Testing` trong `main.py` để so sánh `V1` và `V2`, đồng thời tạo phần tổng hợp kết quả trong `reports/summary.json`.
- Hoàn thiện `analysis/failure_analysis.md` dựa trên dữ liệu thật từ benchmark thay vì để template trống.
- Hỗ trợ tích hợp sau khi pull code của thành viên khác về, rà soát lại interface giữa `MainAgent`, `LLMJudge`, `BenchmarkRunner` và luồng sinh report.

## 2. Đóng góp kỹ thuật cụ thể

### 2.1. Async Runner và thống kê hiệu năng

Em hoàn thiện `engine/runner.py` để mỗi test case không chỉ trả về đáp án mà còn có thêm các thông tin phục vụ đánh giá hệ thống:

- `latency` và `latency_ms`
- `tokens_used`
- `generation_cost_usd`
- `judge_cost_usd`
- `total_cost_usd`
- `status`

Phần này quan trọng vì rubric yêu cầu không chỉ benchmark chất lượng mà còn phải theo dõi hiệu năng và chi phí. Em cũng bổ sung hàm `summarize_results()` để dễ tổng hợp:

- thời gian trung bình
- tổng token
- tổng cost
- pass rate

### 2.2. Tích hợp RAGAS và retrieval metrics

Trong `main.py`, em xây dựng `ExpertEvaluator` để chấm:

- `faithfulness`
- `relevancy`
- `hit_rate`
- `mrr`

Ban đầu phần RAGAS bị lỗi do API của thư viện trong môi trường không khớp với cách gọi cũ. Em đã kiểm tra lại signature thực tế của các metric và sửa logic gọi sao cho tương thích với version đang cài. Sau khi sửa, benchmark mới dùng được RAGAS thật thay vì rơi xuống fallback heuristic.

Đây là một phần em thấy rất quan trọng vì nếu evaluator sai thì toàn bộ kết quả benchmark sẽ không đáng tin, dù pipeline vẫn chạy.

### 2.3. So sánh V1 và V2

Em hoàn thiện logic regression trong `main.py` để so sánh trực tiếp giữa:

- `Agent_V1_Base`
- `Agent_V2_Optimized`

Sau khi rà lại yêu cầu trong `GRADING_RUBRIC.md`, em đã chỉnh release gate theo hướng bám sát bài lab hơn: không dùng các ngưỡng tuyệt đối tự đặt, mà so sánh dựa trên delta thực giữa `V1` và `V2`. Kết quả được lưu vào `summary.json` với các trường như:

- `decision`
- `reason`
- `delta`
- `improved_metrics`
- `regressed_metrics`
- `unchanged_metrics`

### 2.4. Failure Analysis

Em điền đầy đủ `analysis/failure_analysis.md` bằng số liệu thật từ:

- `reports/summary.json`
- `reports/benchmark_results.json`

Em không chỉ điền số liệu tổng mà còn:

- phân nhóm lỗi
- chọn các case xấu nhất
- viết 5 Whys
- đề xuất action plan theo mức ưu tiên

Phần này giúp kết nối benchmark kỹ thuật với góc nhìn phân tích nguyên nhân gốc rễ, đúng tinh thần của một hệ thống evaluation factory.

## 3. Những vấn đề kỹ thuật em đã gặp

Trong quá trình làm, em gặp ba vấn đề chính:

### 3.1. Lệch interface sau khi pull code nhóm

Sau khi pull code của thành viên khác về, một số phần bị lệch nhau:

- `MainAgent` yêu cầu chạy `version="v1"` và `version="v2"`
- judge thật nằm ở `engine/llm_judge.py`
- một số logic cũ trong `main.py` vẫn còn placeholder

Em đã phải rà lại toàn bộ luồng để đảm bảo benchmark không còn dùng judge giả hoặc so sánh sai version.

### 3.2. RAGAS bị fail do lệch API

Đây là lỗi kỹ thuật em thấy quan trọng nhất. Bề ngoài hệ thống vẫn báo `ragas_available = True`, nhưng khi chấm từng case lại fail vì metric object không có method như code cũ mong đợi. Em đã kiểm tra trực tiếp trong môi trường để xác nhận nguyên nhân và sửa lại helper gọi metric theo đúng signature thật.

### 3.3. Dữ liệu retrieval hiển thị theo article bị lặp

Trong `benchmark_results.json`, có nhiều trường hợp `retrieved_ids` lặp cùng một `article_file` vì retrieval chạy theo chunk nhưng kết quả lại được map lên cấp article. Em đã phân tích ảnh hưởng của vấn đề này và xác định rằng:

- nó không làm hỏng hoàn toàn `Hit Rate` và `MRR`
- nhưng làm report khó đọc và khó phân tích hơn

Sau khi cân nhắc phạm vi công việc cá nhân, em quyết định không ép sửa sâu phần này khi không thật sự cần cho việc nộp bài.

## 4. Kiến thức kỹ thuật em hiểu rõ hơn sau bài lab

### 4.1. MRR là gì

`MRR` là viết tắt của `Mean Reciprocal Rank`. Với mỗi query, ta tìm vị trí đầu tiên mà tài liệu đúng xuất hiện trong danh sách retrieval. Nếu tài liệu đúng đứng ở vị trí:

- 1 thì điểm là `1.0`
- 2 thì điểm là `0.5`
- 3 thì điểm là `0.333...`

Sau đó lấy trung bình trên toàn bộ dataset. MRR hữu ích vì nó không chỉ kiểm tra “có tìm thấy hay không”, mà còn đo xem hệ thống có đưa tài liệu đúng lên đầu danh sách hay không.

### 4.2. Agreement Rate và Multi-Judge

Khi dùng nhiều judge model, `agreement_rate` cho biết mức độ đồng thuận giữa các judge. Nếu chỉ dùng một judge đơn lẻ, kết quả có thể bị lệch theo prompt, model bias hoặc thứ tự trình bày. Multi-judge giúp tăng độ tin cậy của benchmark.

### 4.3. Position Bias

`Position Bias` là hiện tượng judge có thể chấm khác nhau chỉ vì thứ tự hiển thị câu trả lời thay đổi. Ví dụ nếu đáp án chuẩn luôn được đặt trước, model judge có thể vô thức ưu tiên phần xuất hiện đầu. Đây là lý do tại sao trong lab có ý tưởng dùng thứ tự prompt khác nhau giữa các judge để giảm thiên lệch này.

### 4.4. Trade-off giữa chi phí và chất lượng

Một điểm em học được rõ hơn là benchmark AI không chỉ là “điểm cao”, mà còn là:

- chi phí bao nhiêu
- bao lâu chạy xong
- chất lượng tăng có đáng với chi phí tăng không

Trong kết quả của nhóm, chi phí judge lớn hơn chi phí generation khá nhiều. Điều này cho thấy nếu muốn tối ưu hệ thống ở bước tiếp theo, không thể chỉ tập trung vào agent mà còn phải tối ưu cả tầng evaluation.

## 5. Điều em rút ra về mặt kỹ thuật và làm việc nhóm

Bài lab này giúp em hiểu rõ hơn rằng việc xây một evaluation pipeline đáng tin khó ở chỗ tích hợp:

- dataset
- retrieval
- generation
- judge
- metrics
- report

Nếu từng module chạy riêng lẻ nhưng schema không khớp nhau, cả pipeline vẫn fail. Vì vậy, phần việc em làm không chỉ là viết code cho từng file mà còn là làm cho toàn bộ dòng chảy dữ liệu chạy trơn tru từ benchmark đến báo cáo.

Về làm việc nhóm, em nhận ra phần “sau khi pull code của nhau về” rất quan trọng. Nhiều lỗi không đến từ thuật toán mà đến từ interface, naming hoặc assumption khác nhau giữa các thành viên. Việc rà lại và đồng bộ các phần đó là một đóng góp kỹ thuật thật sự, không chỉ là “sửa lỗi vặt”.

## 6. Tự đánh giá đóng góp

Em đánh giá phần đóng góp của mình có chiều sâu ở ba mặt:

- xây và hoàn thiện benchmark runner cùng phần thống kê chi phí/hiệu năng
- làm evaluator và sửa tích hợp RAGAS để kết quả benchmark có ý nghĩa
- biến số liệu benchmark thành báo cáo failure analysis có thể dùng cho nộp bài

Nếu có thêm thời gian, em muốn cải thiện tiếp:

- đo thời gian toàn pipeline thay vì chỉ riêng generation
- làm sạch hơn phần retrieval report ở cấp article/chunk
- thêm dashboard nhỏ để nhìn các nhóm lỗi nhanh hơn thay vì đọc raw JSON
