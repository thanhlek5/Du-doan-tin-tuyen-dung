from sklearn.base import BaseEstimator, TransformerMixin
import re 

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy() # Copy ra để không ảnh hưởng dữ liệu gốc
        
        # Tạo biến tạm gộp text để tìm kiếm cho nhanh
        # (Chỉ dùng nội bộ để tìm features, không ghi đè cột description gốc)
        temp_text = X['description'].fillna('') + " " + X['requirements'].fillna('')
        
        
        # 2. TẠO CỘT 'key_note' (Từ khóa mạo danh Dầu khí)
        keys = ["petroleum", "oil gas","oil energy","data entry","typing","clerical","work home","training provided"," encouraged", "administrative assistant" ,"clerk" ]
        pattern = r'('+'|'.join(map(re.escape,keys)) + r')' # Tạo regex: "aker|subsea|action..."
        X['key_note'] = temp_text.str.contains(pattern, case=False, na=False).astype(int)
        
        return X