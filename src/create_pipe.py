import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer 
from sklearn.pipeline import Pipeline as SklearnPipeline
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import os 
import sys 
from feature_engineering import FeatureEngineer
from text_cleaner import TextCleaner 
from vectorizers import Word2VecTransformer
project_root = os.path.dirname(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)



def create_preprocessing_pipeline(vectorizer_type='tfidf'):
    """
    Hàm này chỉ trả về Pipeline xử lý dữ liệu: 
    Feature Eng -> Clean -> Vectorize.
    
    KHÔNG CÓ SMOTE (Cân bằng dữ liệu).
    KHÔNG CÓ MODEL (XGBoost).
    """
    
    # 1. Định nghĩa cột
    num_cols = ['telecommuting', 'has_company_logo', 'has_questions']
    text_cols = ['title', 'location', 'department', 'company_profile', 'description', 
                'requirements', 'benefits', 'employment_type', 'required_experience', 
                'required_education', 'industry', 'function']
    
    # 2. Chọn Vectorizer
    if vectorizer_type == 'count':
        vec_step = CountVectorizer(max_features=5000)
    elif vectorizer_type == 'tfidf':
        vec_step = TfidfVectorizer(max_features=5000)
    elif vectorizer_type == 'word2vec':
        vec_step = Word2VecTransformer(vector_size=100, window=5, min_count=2)
    
    # 3. Preprocessor (ColumnTransformer)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), num_cols),
            ('text', SklearnPipeline([
                ('cleaner', TextCleaner()), 
                ('vec', vec_step)
            ]), text_cols),
            ('flags', 'passthrough', ['key_note'])
        ],
        remainder='drop'
    )
    
    # 4. Các bước Pipeline
    steps = [
        ('engineer', FeatureEngineer()), 
        ('preprocessor', preprocessor)
    ]
    
    # === ĐÃ XÓA PHẦN 5 (SMOTE) ===
    # Pipeline này giờ chỉ biến đổi dữ liệu thô thành số
    
    return ImbPipeline(steps=steps)

path_data = os.path.join(project_root,"data","data_train.csv")
df = pd.read_csv(path_data)
X_train= df.drop("fraudulent",axis= 1)
y_train = df['fraudulent']


# --- 1. CẤU HÌNH CÁC PHƯƠNG ÁN TIỀN XỬ LÝ ---
# (Lưu ý: Tên file mình đổi tiền tố thành 'Preprocess_' cho dễ phân biệt)
vectorizer_types = ["tfidf", "count", "word2vec"]


print("\n🚀 BẮT ĐẦU CHẠY TIỀN XỬ LÝ & LƯU DỮ LIỆU SẠCH...\n")

# --- 3. VÒNG LẶP XỬ LÝ ---
for vec_type in vectorizer_types:
    # Đặt tên file tự động
    name = f"Preprocess_{vec_type}" 
    print(f"⏳ Đang xử lý: {name}...")
    
    # A. Gọi hàm tạo Pipeline (Không truyền imbalance_strategy nữa)
    preprocessor = create_preprocessing_pipeline(vectorizer_type=vec_type)
    
    # B. Fit & Transform (Thay vì fit_resample)
    # Hàm này chỉ học từ vựng và biến đổi thành số. KHÔNG sinh thêm dữ liệu.
    X_processed = preprocessor.fit_transform(X_train, y_train)
    
    # C. Lưu kết quả
    
    # 1. Lưu Pipeline (Chứa logic xử lý)
    pipe_path = os.path.join(project_root,"models", f"{name}_pipeline.pkl")
    joblib.dump(preprocessor, pipe_path)
    
    # 2. Lưu Dữ liệu sạch
    # Lưu ý: y_train giữ nguyên vì ta không SMOTE
    data_path = os.path.join(project_root,"models", f"{name}_data.pkl")
    joblib.dump((X_processed, y_train), data_path)
    
    print(f"   ✅ Pipeline lưu tại: {pipe_path}")
    print(f"   ✅ Dữ liệu sạch ({X_processed.shape}) lưu tại: {data_path}\n")

print("🎉 HOÀN TẤT! Bạn đã có bộ dữ liệu sạch (chưa cân bằng).")


project_root,"fraud-detection-post","models"