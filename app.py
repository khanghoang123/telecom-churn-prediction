import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from category_encoders import TargetEncoder

# Tải mô hình
model = joblib.load('stacking_model.pkl')
ce_target = joblib.load('target_encoder.pkl')      
preprocessor = joblib.load('preprocessor.pkl') 
# Nếu bạn có encoder riêng cho 'City', thì load encoder:
# ce_target = joblib.load('encoder.pkl')

st.set_page_config(page_title="Telecom Churn Prediction", layout="wide")
st.title("📞 Telecom Customer Churn Prediction App")
st.markdown("Dự đoán khách hàng có rời bỏ dịch vụ hay không dựa vào các đặc điểm cá nhân và hành vi sử dụng dịch vụ.")

# ==== Nhập thông tin khách hàng ====
with st.form("customer_form"):
    gender = st.selectbox("Giới tính", ["Male", "Female"])
    age = st.slider("Tuổi", 18, 100, 30)
    married = st.selectbox("Tình trạng hôn nhân", ["Yes", "No"])
    dependents = st.number_input("Số người phụ thuộc", min_value=0, max_value=10, value=0)
    city = st.text_input("Thành phố", "Los Angeles")
    referrals = st.slider("Số lượt giới thiệu", 0, 20, 0)
    tenure = st.slider("Số tháng sử dụng", 1, 100, 12)
    streaming_music = st.selectbox("Nghe nhạc trực tuyến?", ['Yes', 'No'])
    online_security = st.selectbox("Bảo mật trực tuyến?", ['Yes', 'No'])
    unlimited_data = st.selectbox("Dữ liệu không giới hạn?", ['Yes', 'No'])
    payment_method = st.selectbox("Phương thức thanh toán", ['Bank Withdrawal', 'Credit Card', 'Mailed Check', 'Electronic Check'])
    paperless_billing = st.selectbox("Hóa đơn điện tử?", ['Yes', 'No'])
    online_backup = st.selectbox("Sao lưu trực tuyến?", ['Yes', 'No'])
    streaming_movies = st.selectbox("Xem phim trực tuyến?", ['Yes', 'No'])
    streaming_tv = st.selectbox("Xem TV trực tuyến?", ['Yes', 'No'])
    premium_tech_support = st.selectbox("Hỗ trợ kỹ thuật cao cấp?", ['Yes', 'No'])
    device_protection_plan = st.selectbox("Gói bảo vệ thiết bị?", ['Yes', 'No'])
    offer = st.selectbox("Gói khuyến mãi", ['None', 'Offer A', 'Offer B', 'Offer C', 'Offer D', 'Offer E'])
    phone_service = st.selectbox("Có sử dụng điện thoại?", ['Yes', 'No'])
    avg_long_distance = st.number_input("Phí gọi xa trung bình hàng tháng", min_value=0.0, value=10.0)
    multiple_lines = st.selectbox("Nhiều đường dây?", ['Yes', 'No'])
    internet_service = st.selectbox("Có Internet?", ['Yes', 'No'])
    internet_type = st.selectbox("Loại Internet", ['Cable', 'Fiber Optic', 'DSL'])
    avg_gb = st.number_input("Sử dụng dữ liệu trung bình (GB/tháng)", min_value=0.0, value=10.0)
    contract = st.selectbox("Loại hợp đồng", ['Month-to-Month', 'One Year', 'Two Year'])
    monthly_charge = st.number_input("Chi phí hàng tháng", min_value=0.0, value=50.0)
    total_charges = st.number_input("Tổng chi phí", min_value=0.0, value=600.0)
    total_refunds = st.number_input("Tổng hoàn tiền", min_value=0.0, value=0.0)
    total_extra = st.number_input("Phí dữ liệu phát sinh", min_value=0.0, value=0.0)
    total_long = st.number_input("Tổng phí gọi xa", min_value=0.0, value=50.0)
    total_revenue = st.number_input("Tổng doanh thu", min_value=0.0, value=650.0)
    submit = st.form_submit_button("Dự đoán")

if submit:
    # ==== Tiền xử lý dữ liệu ====
    input_df = pd.DataFrame([{
        'Gender': gender,
        'Age': age,
        'Married': married,
        'Number of Dependents': dependents,
        'City': city,
        'Number of Referrals': referrals,
        'Tenure in Months': tenure,
        'Offer': offer,
        'Phone Service': phone_service,
        'Avg Monthly Long Distance Charges': avg_long_distance,
        'Multiple Lines': multiple_lines,
        'Internet Service': internet_service,
        'Internet Type': internet_type,
        'Avg Monthly GB Download': avg_gb,
        'Contract': contract,
        'Monthly Charge': monthly_charge,
        'Total Charges': total_charges,
        'Total Refunds': total_refunds,
        'Total Extra Data Charges': total_extra,
        'Total Long Distance Charges': total_long,
        'Total Revenue': total_revenue,
        'Streaming Music': streaming_music,
        'Online Security': online_security,
        'Unlimited Data': unlimited_data,
        'Payment Method': payment_method,
        'Paperless Billing': paperless_billing,
        'Online Backup': online_backup,
        'Streaming Movies': streaming_movies,
        'Streaming TV': streaming_tv,
        'Premium Tech Support': premium_tech_support,
        'Device Protection Plan': device_protection_plan,
    }])

    # Tạo cột đặc trưng mới
    input_df['Refund_Rate'] = input_df['Total Refunds'] / (input_df['Total Revenue'] + 1e-6)
    input_df['Charge_per_Month'] = input_df['Total Charges'] / (input_df['Tenure in Months'] + 1e-6)
    input_df['Data_per_Month'] = input_df['Avg Monthly GB Download'] / (input_df['Tenure in Months'] + 1e-6)
    input_df['New_Customer'] = input_df['Tenure in Months'].apply(lambda x: 1 if x <= 6 else 0)
    input_df['Low_Revenue_Flag'] = input_df['Total Revenue'].apply(lambda x: 1 if x < 400 else 0)    

    input_df['City'] = ce_target.transform(input_df['City'])
    
    input_processed = preprocessor.transform(input_df)


    # ==== Dự đoán ====
    prediction = model.predict(input_processed)[0]
    proba = model.predict_proba(input_processed)[0][1]

    st.subheader("🔍 Kết quả dự đoán:")
    if prediction == 1:
        st.error(f"Khách hàng **CÓ KHẢ NĂNG RỜI BỎ** với xác suất {proba:.2%}")
    else:
        st.success(f"Khách hàng **CÓ XU HƯỚNG Ở LẠI** với xác suất {(1 - proba):.2%}")
