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
+ Những đặc trưng chủ yếu về văn bản hoặc từ khóa (keywords) nào mang tính phân biệt cao nhất giúp nhận diện một tin tuyển dụng giả mạo. -> EDA
+ Phương pháp trích xuất đặc trung nào (TF-IDF, Count Vectorized, Word Embeddings như Word2Vec/BERT) mang lại đầu vào tốt nhất cho mô hình phân loại trong ngữ cảnh này. -> CountVector


### Các phần cần được xử lý:
+ tiền xử lý dữ liệu.
+ phân tích dữ liệu tìm insign và các cột quan trọng 
+ Huấn luyện mô hình.
+ Đánh giá mô hình và chọn mô hình tốt nhất.
+ Làm báo cáo.
+ Làm slide ppt.

### Cấu trúc thư mục
** Nhớ tạo thư mục tên data để lưu dữ liệu không để file dữ liệu ở bên ngoài **
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

## Tạo pipeline cho việc Tiền xử lý dữ liệu. 
### xử lý mất cân bằng dữ liệu (imbalanced)

Bộ dữ liệu bị mất cân bằng khá cao (17.200/800) nên ta cần đưa ra phương án để xử lý

Ta có 2 hướng để xử lý:

**Hướng 1** 
Ta sẽ smote dữ liệu giả mạo lên cho gần bằng với với dữ liệu không giả mạo -> ta sẽ thêm tớ 16.400 dữ liệu mới tỷ lệ 1:20 

rủi ro: 
+ nếu chọn cách này nghĩa là ta đang ép thuật toán bịa ra rất nhiều dữ liệu ảo từ 1 lượng thông tin rất ít -> mô hình có khả năng cao sẽ học thuộc những dữ liệu ảo đó -> khả năng học vẹt rất cao. 
+ tốc độ và tài nguyên vì 17.880 (dữ liệu thật) và 17.200 (dứ liệu ảo) -> 34.400 dòng dữ liệu.

**Hướng 2**
Ta sẽ kết hợp giũa smote và undersampling: 
ta sẽ smote dữ liệu giả mạo lên 5 lần là tỷ lệ 1:5 và sẽ undersampling dữ liệu không giả mạo xuống gần bằng với dữ liệu giả mạo.
-> 1 pipeline cho hướng 1 và 1 pipeline cho hướng 2
### Xử lý giá trị trống:
ta sẽ chia làm 3 nhóm dữ liệu: 
+ số -> thay bằng median (trung vị) 
+ chữ -> thay bằng "unknow"
+ văn bản -> thay bằng "missing"
### chỉnh sửa, thêm và bớt cột (features)
+ Bỏ cột job_id
+ Thêm cột keynote và chain -> giá trị là 0 và 1 
+ ta sẽ gộp tất cả các features chữ và văn bbản vào cột mới là combined_text
### chuẩn hóa dữ liệu:
+ Với dữ liệu dạng số ta không cần thiết phải chuẩn hóa nữa vì chúng chỉ có 2 giá trị là `0 và 1`.
+ với  cột combined_text ta sẽ làm sạch chúng (bỏ các kí tự đặc biệt) -> chuẩn hóa (bỏ các từ stopwords,...) -> vector hóa chúng.
+ ở phần vector hóa ta có rât nhiều thuật toán và mô hình để xử lý  countVector, TF-IDF, Word2Vec.
-> tạo pipeline cho 3 cái trên 1 cái là dùng CountVector, 1 cái dùng TF-IDF, 1 cái dùng Word2Vec 

tổng là có 5 pipeline tất cả


















