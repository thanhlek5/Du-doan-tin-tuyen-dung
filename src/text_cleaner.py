from sklearn.base import BaseEstimator, TransformerMixin
import nltk
import spacy
import re


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