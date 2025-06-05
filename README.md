# 📊 Telecom Churn Prediction

Ứng dụng web dự đoán khả năng khách hàng rời mạng (churn) cho doanh nghiệp viễn thông bằng mô hình học máy và giao diện Streamlit.

---

## 🚀 Giới thiệu

**Churn** là hiện tượng khách hàng ngừng sử dụng dịch vụ. Việc dự đoán churn giúp doanh nghiệp chủ động giữ chân khách hàng. Dự án này:

- Dự đoán churn dựa trên dữ liệu khách hàng
- Áp dụng các mô hình học máy mạnh mẽ và kỹ thuật nâng cao
- Triển khai ứng dụng trực quan với **Streamlit**

---

## 🧠 Mô hình & Kỹ thuật

### 🎯 Xử lý dữ liệu:
- Mã hóa đặc trưng: `OneHotEncoder`, `OrdinalEncoder`, `TargetEncoder`
- Chuẩn hóa: `StandardScaler`
- Xử lý thiếu: `SimpleImputer`

### 🤖 Thuật toán học máy:
- **Thuật toán cơ bản**:
  - `Logistic Regression`
  - `Decision Tree`
  - `SVM`

- **Thuật toán ensemble**:
  - `Random Forest`
  - `Gradient Boosting`
  - `AdaBoost`
  - `XGBoost`
  - **Stacking Classifier** ⬅️ *(thuật toán tổ hợp nhiều mô hình để tối ưu hóa hiệu suất)*

### 📊 Đánh giá mô hình:
- Confusion Matrix, Accuracy, F1-score
- ROC AUC, ROC Curve
- Cross-validation, GridSearchCV, RandomizedSearchCV

### ⚖️ Cân bằng dữ liệu:
- Sử dụng `SMOTE` từ `imblearn` để xử lý mất cân bằng dữ liệu lớp

---

## 🖥️ Giao diện người dùng (Streamlit)

Người dùng có thể:
- Nhập thông tin khách hàng
- Nhấn nút "Dự đoán" để xem kết quả dự đoán churn
- Xem trực quan các biểu đồ phân tích dữ liệu và hiệu suất mô hình

---

## 📦 Yêu cầu hệ thống

- Python >= 3.7

### Cài đặt thư viện:

```bash
pip install -r requirements.txt
