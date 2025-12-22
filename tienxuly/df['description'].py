df['description'].fillna('', inplace=True)
df['requirements'].fillna('', inplace=True)
df['benefits'].fillna('', inplace=True)
Categorical fields:
python# Tạo category 'Unknown' cho missing
df['employment_type'].fillna('Unknown', inplace=True)
df['required_experience'].fillna('Unknown', inplace=True)
df['industry'].fillna('Unknown', inplace=True)
Salary_range:
python# Drop do quá nhiều missing (85%) và khó impute
df.drop('salary_range', axis=1, inplace=True)
Bước 2: Text Preprocessing
Định nghĩa hàm clean_text:
pythonimport re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def clean_text(text):
    # Lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove numbers (tùy chọn)
    # text = re.sub(r'\d+', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Tokenization
    tokens = text.split()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words]
    
    # Stemming (hoặc Lemmatization)
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(w) for w in tokens]
    
    return ' '.join(tokens)
Áp dụng:
python# Clean các text columns
text_cols = ['title', 'location', 'company_profile', 
             'description', 'requirements', 'benefits']

for col in text_cols:
    df[col + '_clean'] = df[col].apply(clean_text)
Bước 3: Feature Engineering
Tạo features mới từ text:
python# Độ dài description
df['description_length'] = df['description'].apply(len)
df['description_word_count'] = df['description'].apply(lambda x: len(x.split()))

# Độ dài requirements
df['requirements_length'] = df['requirements'].apply(len)

# Có salary_range hay không
df['has_salary_range'] = df['salary_range'].notna().astype(int)

# Số lượng từ trong title
df['title_word_count'] = df['title'].apply(lambda x: len(str(x).split()))

# Location có phải remote không
df['is_remote'] = df['location'].apply(lambda x: 1 if 'remote' in str(x).lower() else 0)

# Có chứa các từ khóa lừa đảo không
fraud_keywords = ['urgent', 'immediate', 'apply now', 'high income', 'earn money']
df['has_fraud_keywords'] = df['description_clean'].apply(
    lambda x: 1 if any(kw in x for kw in fraud_keywords) else 0
)
Bước 4: Encoding Categorical Variables
pythonfrom sklearn.preprocessing import LabelEncoder

# Label encoding cho categorical features
cat_cols = ['employment_type', 'required_experience', 
            'required_education', 'industry', 'function']

label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
Bước 5: Text Vectorization
Option 1: TF-IDF
pythonfrom sklearn.feature_extraction.text import TfidfVectorizer

# Combine tất cả text fields
df['combined_text'] = (df['title_clean'] + ' ' + 
                       df['company_profile_clean'] + ' ' +
                       df['description_clean'] + ' ' +
                       df['requirements_clean'] + ' ' +
                       df['benefits_clean'])

# TF-IDF vectorization
tfidf = TfidfVectorizer(max_features=1000,  # Giới hạn số features
                        min_df=5,            # Bỏ từ xuất hiện < 5 docs
                        max_df=0.8,          # Bỏ từ quá phổ biến
                        ngram_range=(1, 2))  # Unigrams và bigrams

X_text = tfidf.fit_transform(df['combined_text'])
Option 2: Count Vectorizer (Bag of Words)
pythonfrom sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=1000)
X_text = cv.fit_transform(df['combined_text'])
Bước 6: Feature Scaling
pythonfrom sklearn.preprocessing import StandardScaler

# Chuẩn hóa numerical features
num_features = ['description_length', 'description_word_count', 
                'requirements_length', 'title_word_count']

scaler = StandardScaler()
df[num_features] = scaler.fit_transform(df[num_features])
Bước 7: Combine All Features
pythonimport numpy as np
from scipy.sparse import hstack

# Binary và engineered features
binary_features = ['telecommuting', 'has_company_logo', 'has_questions',
                   'has_salary_range', 'is_remote', 'has_fraud_keywords']

# Categorical encoded features  
cat_encoded_features = [col + '_encoded' for col in cat_cols]

# Numerical features
numerical_features = ['description_length', 'description_word_count',
                      'requirements_length', 'title_word_count']

# Combine
X_structured = df[binary_features + cat_encoded_features + numerical_features].values

# Kết hợp với text features (TF-IDF)
X_final = hstack([X_structured, X_text])

# Target
y = df['fraudulent'].values
3.1.6. Handling Imbalanced Data
Phương pháp 1: SMOTE (Synthetic Minority Over-sampling Technique)
pythonfrom imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

print(f"Before SMOTE: {Counter(y_train)}")
print(f"After SMOTE: {Counter(y_resampled)}")
Phương pháp 2: Class Weights
pythonfrom sklearn.utils.class_weight import compute_class_weight

# Tính class weights
class_weights = compute_class_weight('balanced', 
                                     classes=np.unique(y_train),
                                     y=y_train)

# Tạo dictionary
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

# Sử dụng trong model
model = RandomForestClassifier(class_weight=class_weight_dict)
Phương pháp 3: Combination (SMOTE + Tomek Links)
pythonfrom imblearn.combine import SMOTETomek

smt = SMOTETomek(random_state=42)
X_resampled, y_resampled = smt.fit_resample(X_train, y_train)
Lựa chọn phương pháp:

SMOTE: Thử nghiệm đầu tiên
Class weights: Nếu không muốn thay đổi data distribution
Combination: Nếu SMOTE đơn thuần không đủ

3.1.7. Train-Test Split
pythonfrom sklearn.model_selection import train_test_split

# Stratified split để giữ tỷ lệ class
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, 
    test_size=0.2,        # 80-20 split
    stratify=y,           # Giữ tỷ lệ fraudulent/legitimate
    random_state=42       # Reproducibility
)

print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
print(f"Train fraud ratio: {y_train.sum() / len(y_train):.2%}")
print(f"Test fraud ratio: {y_test.sum() / len(y_test):.2%}")
```

### **3.2. Phương pháp đề xuất**


3.2.2. Pipeline for Each Model
Pipeline Example với Logistic Regression:
pythonfrom sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Create pipeline
lr_pipeline = Pipeline([
    ('scaler', StandardScaler(with_mean=False)),  # Cho sparse matrix
    ('classifier', LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    ))
])

# Train
lr_pipeline.fit(X_train, y_train)

# Predict
y_pred = lr_pipeline.predict(X_test)
y_pred_proba = lr_pipeline.predict_proba(X_test)[:, 1]
3.2.3. Cross-Validation Strategy
pythonfrom sklearn.model_selection import StratifiedKFold, cross_val_score

# 5-fold stratified CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# CV scores
cv_scores = cross_val_score(
    lr_pipeline, 
    X_train, 
    y_train,
    cv=skf,
    scoring='f1'  # Hoặc 'roc_auc', 'precision', 'recall'
)

print(f"CV F1 scores: {cv_scores}")
print(f"Mean F1: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
3.2.4. Hyperparameter Tuning
Grid Search cho Random Forest:
pythonfrom sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'class_weight': ['balanced', 'balanced_subsample']
}

# Grid search với CV
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=2
)

# Fit
grid_search.fit(X_train, y_train)

# Best parameters
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best F1 score: {grid_search.best_score_:.3f}")

# Use best model
best_rf = grid_search.best_estimator_
Random Search cho XGBoost:
pythonfrom sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb

# Parameter distributions
param_dist = {
    'max_depth': [3, 4, 5, 6, 7, 8],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [100, 200, 300, 500],
    'min_child_weight': [1, 3, 5, 7],
    'gamma': [0, 0.1, 0.2, 0.3],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'reg_alpha': [0, 0.1, 0.5, 1],
    'reg_lambda': [0, 0.1, 0.5, 1],
    'scale_pos_weight': [1, 5, 10, 19.6]  # Ratio of neg/pos
}

# Random search
random_search = RandomizedSearchCV(
    xgb.XGBClassifier(random_state=42, use_label_encoder=False),
    param_distributions=param_dist,
    n_iter=50,  # Số combinations thử
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=2,
    random_state=42
)

random_search.fit(X_train, y_train)

print(f"Best parameters: {random_search.best_params_}")
print(f"Best F1 score: {random_search.best_score_:.3f}")

best_xgb = random_search.best_estimator_
3.2.5. Model Evaluation Metrics
pythonfrom sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    roc_a