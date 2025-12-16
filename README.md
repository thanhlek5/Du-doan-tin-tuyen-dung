# Dự đoán tin tuyển dụng giả mạo

## Dataset:
Nguồn dữ liệu lấy từ kaggle: [DATASET](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction/discussion/data?sort=hotness)

Dữ liệu được lấy từ Phòng thí nghiệm An ninh hệ thống thông tin (laboratory of information & Communication Systems Security) thuộc đại học Aegean.

Các tập tin này được thu thập trong khoảng từ năm 2012 tới 2014.

Dữ liệu bao gồm 17.880 tin tuyển dụng thực tế, trong đó có khoảng 866 tin giả mạo. Dữ liệu được dán nhãn thủ công để phục vụ nghiên cứu về vấn đề lừa đảo việc làm.

## Việc nhóm:
    + mỗi người đều làm tất cả các phần và sẽ lấy những phần được cho là tốt nhất.
    + mỗi người dùng nhánh (branch) riêng với các phần riêng vd: thành ở phần tiền xử lý dữ liệu -> branch name : preprocessor/thanh. 

### Câu hỏi nghiên cứu
+ Thuật toán máy học nào cho hiệu suất cao nhất trong việc phân loại tin tuyển dụng.
+ Làm thế nào để cân bằng giữa chỉ số precision và recall -> dùng f1 và PR-AUC
+ Việc áp dụng các kỹ thuật tái lấy mẫu như smote thì liệu có ảnh hưởng như thế nào đến khả năng phát hiện tin giả của mô hình.
+ Những đặc trưng chủ yếu về văn bản hoặc từ khóa (keywords) nào mang tính phân biệt cao nhất giúp nhận diện một tin tuyển dụng giả mạo.
+ Phương pháp trích xuất đặc trung nào (TF-IDF, Count Vectorized, Word Embeddings như Word2Vec/BERT) mang lại đầu vào tốt nhất cho mô hình phân loại trong ngữ cảnh này.
+ Làm thế nào để giải thích lý do mô hình đánh dấu một tin giả mạo để người dùng cuối tin tưởng -> dùng biểu đồ của SHAP (waterfall, Beeswarm)

### Các phần cần được xử lý:
+ tiền xử lý dữ liệu.
+ phân tích dữ liệu tìm insign và các cột quan trọng 
+ Huấn luyện mô hình.
+ Đánh giá mô hình và chọn mô hình tốt nhất.
+ Làm báo cáo.
+ Làm slide ppt.

### Cấu trúc thư mục

```
├── configs  ## Thư mục chứa các siêu tham số của mô hình dùng để tinh chỉnh
|   |──file.json
├── models ## Thư mục chứa cac model đã huẫn luyện
│   |──file.pkl
├── notebooks ## các nghiên cứu và phân tích, tiền xử lý dữ liệu, huấn luyện
│   |──file.ipynb
├── reports ## các báo cáo và ppt
|   |──file.docs,ppt
├── src ## chứa các hàm đề hỗ trợ cho việc làm 
│   |── preprocessor_utils.py ## chứa các hàm cho tiền xử lý dữ liệu, chia tập
|   |── model_utils.py ## chứa các hàm cho việc huấn luyện 
|   |──eval_metric.py ## Chứa các hàm cho việc đánh giá 
├── .gitignore
├── README.md
└── requirements.txt ## các thư việc sử dụng
```


















