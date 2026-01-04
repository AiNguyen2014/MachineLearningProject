# MachineLearningProject
## PHÂN CỤM DỮ LIỆU BIỂU HIỆN GEN VỚI ENSEMBLE LEARNING

#Link https://machinelearningproject-30id.onrender.com

📌 Giới thiệu đề tài

Đồ án này tập trung vào bài toán phân cụm dữ liệu biểu hiện gen (Gene Expression Data) nhằm khám phá các nhóm mẫu sinh học có đặc điểm tương đồng mà không cần nhãn trước.
Do dữ liệu gen thường có số chiều rất lớn, nhiều nhiễu và cấu trúc phức tạp, nhóm lựa chọn cách tiếp cận Unsupervised Learning kết hợp Ensemble Learning để nâng cao độ ổn định và chất lượng phân cụm.

🎯 Mục tiêu
- Áp dụng các thuật toán phân cụm đơn lẻ để khám phá cấu trúc dữ liệu gen.
- Kết hợp nhiều mô hình phân cụm bằng Ensemble Learning nhằm:
    - Giảm sự phụ thuộc vào một thuật toán duy nhất.
    - Tăng độ ổn định và độ tin cậy của kết quả phân cụm.
- Đánh giá và so sánh hiệu quả giữa mô hình đơn lẻ và mô hình ensemble.

## Các mô hình phân cụm được sử dụng
🔹 1. K-Means++ là phiên bản cải tiến của K-Means, giúp lựa chọn tâm cụm ban đầu thông minh hơn. Thuật toán khởi tạo các centroid sao cho chúng cách xa nhau nhất có thể, từ đó:
  - Giảm nguy cơ rơi vào cực trị cục bộ.
  - Cải thiện tốc độ hội tụ và chất lượng phân cụm.

🔹 2. Hierarchical Clustering là phương pháp phân cụm theo cấu trúc phân cấp, không cần xác định trước số cụm. Thuật toán xây dựng cây phân cấp (dendrogram) bằng cách:
  - Gộp dần các điểm hoặc cụm gần nhau (Agglomerative)
  - Hoặc tách dần từ một cụm lớn (Divisive)

🔹 3. Gaussian Mixture Model (GMM)
GMM là mô hình phân cụm dựa trên xác suất, giả định dữ liệu được sinh ra từ nhiều phân phối Gaussian khác nhau. Mỗi điểm dữ liệu được gán vào cụm dựa trên xác suất thuộc về từng Gaussian, thay vì gán cứng như K-Means.
🔗 Ensemble Learning trong phân cụm: Sau khi thực hiện phân cụm bằng các mô hình đơn lẻ, nhóm áp dụng Ensemble Clustering để tổng hợp kết quả.
Ý tưởng chính:
  - Kết hợp nhiều kết quả phân cụm khác nhau
  - Tạo ra một phân cụm cuối cùng ổn định và đáng tin cậy hơn
Cách tiếp cận này giúp:
  - Giảm ảnh hưởng của nhiễu
  - Tận dụng điểm mạnh của từng thuật toán
  - Cải thiện chất lượng phân cụm tổng thể
