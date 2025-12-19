from sklearn.base import BaseEstimator, TransformerMixin
from gensim.models import Word2Vec
import numpy as np 


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