from sklearn.metrics import accuracy_score, f1_score, average_precision_score, precision_recall_curve, confusion_matrix, classification_report, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt

class Metric:
    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test

    def _get_probs(self):
        """Hàm phụ trợ: Lấy xác suất dự đoán (tránh viết lặp code)"""
        if hasattr(self.model, "predict_proba"):
            # Lấy cột 1 (xác suất lớp Positive/Fraud)
            return self.model.predict_proba(self.X_test)[:, 1]
        elif hasattr(self.model, "decision_function"):
            return self.model.decision_function(self.X_test)
        else:
            raise AttributeError("Model không hỗ trợ predict_proba hoặc decision_function")

    def f1_benchmark(self, X_train, y_train):
        """Kiểm tra nhanh trên tập Train để check Overfitting"""
        y_pred = self.model.predict(X_train)
        return {
            "accuracy": accuracy_score(y_train, y_pred),
            "f1": f1_score(y_train, y_pred)
        }

    def AUPRC(self, plot=False):
        """Tính và vẽ đường AUPRC"""
        y_probs = self._get_probs() # Gọi hàm phụ trợ
        score = average_precision_score(self.y_test, y_probs)

        if plot:
            precision, recall, _ = precision_recall_curve(self.y_test, y_probs)
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, marker='.', label=f'{self.model.__class__.__name__} (AUPRC = {score:.4f})')
            plt.xlabel('Recall (Độ phủ)')
            plt.ylabel('Precision (Độ chính xác)')
            plt.title('Precision-Recall Curve')
            plt.legend()
            plt.grid(True)
            plt.show()

        return score

    def evaluate_model(self, model_name=None, threshold=0.5):
        """Đánh giá toàn diện với ngưỡng tùy chỉnh"""
        # Nếu không truyền tên, tự lấy tên class của model
        if model_name is None:
            model_name = self.model.__class__.__name__
            
        print(f"--- ĐÁNH GIÁ: {model_name.upper()} (Threshold={threshold}) ---")

        # 1. Lấy xác suất và áp dụng ngưỡng
        y_probs = self._get_probs()
        y_pred_new = (y_probs >= threshold).astype(int)

        # 2. Báo cáo chi tiết
        print(classification_report(self.y_test, y_pred_new))

        # 3. Các chỉ số quan trọng
        roc = roc_auc_score(self.y_test, y_probs)
        pr_auc = average_precision_score(self.y_test, y_probs)

        print(f"ROC-AUC: {roc:.4f}")
        print(f"PR-AUC (AUPRC): {pr_auc:.4f} (Quan trọng cho Fraud)")

        # 4. Vẽ Confusion Matrix đẹp hơn
        cm = confusion_matrix(self.y_test, y_pred_new)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Normal', 'Fraud'],
                    yticklabels=['Normal', 'Fraud'])
        plt.xlabel('Dự đoán (Predicted)')
        plt.ylabel('Thực tế (True)')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.show()