import pandas as pd
import numpy as np
import re
import nltk
import spacy
from gensim.models import Word2Vec
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer 
from sklearn.pipeline import Pipeline as SklearnPipeline
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import os 
import sys 
project_root = os.path.abspath(os.path.join(os.getcwd(),".."))
if project_root not in sys.path:
    sys.path.append(project_root)
from preprocessor_utils import split_data    


class TextCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        # --- 1. MUA GIA VỊ VÀ CHẢO (Khởi tạo tài nguyên ngay trong Class) ---
        print("Dataset initialization: Loading NLP resources...")
        
        # Tải NLTK Stopwords (Tự động tải nếu chưa có)
        try:
            self.stopwords = set(nltk.corpus.stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords')
            nltk.download('punkt')
            self.stopwords = set(nltk.corpus.stopwords.words('english'))
            
        # Tải SpaCy Model (Tự động tải nếu chưa có)
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading language model for the spaCy POS tagger\n"
                "(don't worry, this will only happen once)")
            from spacy.cli import download
            download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
            
    def fit(self, X, y=None):
        return self
    # --- 2. CÔNG THỨC NẤU ĂN (Chuyển hàm vào trong class) ---
    def _internal_clear_text(self, text: str) -> str:
        """Hàm làm sạch dùng tài nguyên nội bộ (self.stopwords)"""
        if not isinstance(text, str):
            return ""
        whitelist = {'show', 'unless', 'me', 'anywhere', 'he', 'again', 'from', 'my', 'may', 'before', 'full', 'name', 'done', 'nothing', 'others', 'per', 'above', 'below', 'six', 'your', 'down', 'own', 'hence', 'thereby', 'within', 'call', 'ours', 'third', 'must', 'off', 'say', 'ten', 'eight', 'his', 'should', 'serious', 'any', 'otherwise', 'mostly', 'much', 'several', 'under', 'no', 'amount', 'toward', 'amongst', 'via', 'mine', 'hundred', 'whose'}
        final_stopwords = self.stopwords - whitelist 
        # Logic làm sạch cũ của bạn
        text = text.lower()
        text = re.sub(r'<.*?>', "", text)
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
        
        tokens = nltk.word_tokenize(text)
        # QUAN TRỌNG: Dùng self.stopwords thay vì biến toàn cục
        tokens = [word for word in tokens if word not in final_stopwords]
        return " ".join(tokens)

    def _internal_normalize_text(self, text: str) -> str:
        """Hàm chuẩn hóa dùng tài nguyên nội bộ (self.nlp)"""
        # QUAN TRỌNG: Dùng self.nlp thay vì biến toàn cục
        doc = self.nlp(text)
        normalized_words = [token.lemma_ for token in doc]
        return ' '.join(normalized_words)

    def transform(self, X):
        # X là DataFrame
        X_filled = X.fillna("missing")
        
        # Gộp cột
        combined = X_filled.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
        
        # Gọi hàm nội bộ (dùng self.)
        # Bước 1: Clear text
        cleaned = combined.apply(self._internal_clear_text)
        
        # Bước 2: Normalize
        normalized = cleaned.apply(self._internal_normalize_text)
        
        return normalized

# --- ĐỊNH NGHĨA CLASS WORD2VEC TRANSFORMER ---
class Word2VecTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, vector_size=100, window=5, min_count=1):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.model = None

    def fit(self, X, y=None):
        """
        Huấn luyện mô hình Word2Vec trên tập dữ liệu.
        X: List hoặc Series các chuỗi văn bản (đã được clean).
        """
        # Tách từ (Tokenize) đơn giản bằng split()
        # Lưu ý: X nên là văn bản đã qua bước TextCleaner (sạch sẽ)
        sentences = [str(text).split() for text in X]
        
        # Train mô hình Gensim Word2Vec
        self.model = Word2Vec(sentences, 
                            vector_size=self.vector_size, 
                            window=self.window, 
                            min_count=self.min_count, 
                            workers=4)
        return self

    def transform(self, X):
        """
        Chuyển đổi văn bản thành Vector trung bình (Average Word Vector).
        """
        # Hàm con để tính vector trung bình cho 1 câu
        def get_avg_vector(text):
            if self.model is None:
                return np.zeros(self.vector_size)
                
            words = str(text).split()
            # Lấy vector của các từ có trong từ điển của model
            word_vectors = [self.model.wv[w] for w in words if w in self.model.wv]
            
            # Nếu câu không có từ nào trong từ điển (hoặc rỗng) -> Trả về vector 0
            if len(word_vectors) == 0:
                return np.zeros(self.vector_size)
            
            # Tính trung bình cộng các vector
            return np.mean(word_vectors, axis=0)
            
        # Áp dụng cho toàn bộ dữ liệu X
        # np.vstack giúp xếp chồng các vector thành ma trận
        return np.vstack([get_avg_vector(text) for text in X])
    
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy() # Copy ra để không ảnh hưởng dữ liệu gốc
        
        # Tạo biến tạm gộp text để tìm kiếm cho nhanh
        # (Chỉ dùng nội bộ để tìm features, không ghi đè cột description gốc)
        temp_text = X['description'].fillna('') + " " + X['requirements'].fillna('')
        
        # 1. TẠO CỘT 'chain' (Phát hiện mã rác bot spam)
        garbage_char = '0fa3f7c5e23a16de16a841e368006cae916884407d90b154dfef3976483a71ae'
        X['chain'] = temp_text.apply(lambda x: 1 if garbage_char in str(x) else 0)
        
        # 2. TẠO CỘT 'key_note' (Từ khóa mạo danh Dầu khí)
        keys = ['aker', 'subsea', 'action', 'novation']
        pattern = '|'.join(keys) # Tạo regex: "aker|subsea|action..."
        X['key_note'] = temp_text.str.contains(pattern, case=False, na=False).astype(int)
        
        return X

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
            ('flags', 'passthrough', ['key_note', 'chain'])
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

path_data = os.path.join(project_root,"fraud-detection-post","data","data_train.csv")
df = pd.read_csv(path_data)
X_train,y_train = split_data(df)

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
    pipe_path = os.path.join(project_root,"fraud-detection-post","models", f"{name}_pipeline.pkl")
    joblib.dump(preprocessor, pipe_path)
    
    # 2. Lưu Dữ liệu sạch
    # Lưu ý: y_train giữ nguyên vì ta không SMOTE
    data_path = os.path.join(project_root,"fraud-detection-post","models", f"{name}_data.pkl")
    joblib.dump((X_processed, y_train), data_path)
    
    print(f"   ✅ Pipeline lưu tại: {pipe_path}")
    print(f"   ✅ Dữ liệu sạch ({X_processed.shape}) lưu tại: {data_path}\n")

print("🎉 HOÀN TẤT! Bạn đã có bộ dữ liệu sạch (chưa cân bằng).")


project_root,"fraud-detection-post","models"