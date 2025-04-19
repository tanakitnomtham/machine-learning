# โค้ดนี้เป็นการทำนาย เวลาที่ใช้จริงในการ ขับรถยนต์ไฟฟ้าโดยใช้ข้อมูลชุดเดียวกันเพื่อเป็นตัวช่วยในการวางแผนการเดินทางได้อย่างมีประสิทธิภาพมากยิ่งขึ้น
# นำเข้า library ต่าง ๆ ที่ใช้สำหรับการวิเคราะห์ข้อมูล, การทำ visualization และ machine learning เช่น pandas สำหรับจัดการข้อมูล, sklearn สำหรับการเทรนโมเดล และ seaborn, matplotlib สำหรับการสร้างกราฟ
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.model_selection import LeaveOneOut
import numpy as np

#โหลดข้อมูล โดยอัพโหลดจากคอมพิวเตอร์ของเรา
file_path = "/content/drive/MyDrive/ev_journeys 2.csv"
df = pd.read_csv(file_path)

#ตรวจสอบข้อมูลในรูปแบบ Data Frame & ตวรจสอบแถวและคอลลัมน์ของข้อมูล
df.head()
df.shape

#แสดงแถวของข้อมูลก่อนการกรอง
rows_before = len(df)
print(f"จำนวนแถวเริ่มต้น: {rows_before}")

#กรองข้อมูลตามเงื่อนไขให้สอดคล้องกับความเป็นจริง
#drivable_km_before ควรมากกว่า drivable_km_after (แปลว่าใช้งานแบตเตอรี่ไปแล้ว)
#total_km_before ควรน้อยกว่า total_km_after (แปลว่ารถได้เดินทางจริง)
condition_drivable = df['drivable_km_before'] > df['drivable_km_after']
condition_total_km = df['total_km_before'] < df['total_km_after']
combined_condition = (condition_drivable) & (condition_total_km)
df_cleaned_strict = df[combined_condition].copy()

# แสดงผลลัพธ์
rows_after = len(df_cleaned_strict)
print(f"จำนวนแถวหลังกรองตามเงื่อนไขทั้งสอง: {rows_after}")
print(f"จำนวนแถวที่ถูกลบออก: {rows_before - rows_after}")

#แสดงตัวอย่างข้อมูลที่กรองแล้ว
print("\nข้อมูล 5 แถวแรกหลังกรองตามเงื่อนไข:")
print(df_cleaned_strict.head())

#บันทึกข้อมูลหลังการกรองให้อยู่ในตัวแปรเดิม
df = df_cleaned_strict

#ตรวจสอบตัวอย่างข้อมูลหลังจากการกรอง
df.head()

#จัดการ Missing Value ใน Features ที่จะนำมาสร้างเป็น Target
df = df.dropna(subset=["date_before", "timestamp_before", "date_after", "timestamp_after"])

# แปลงเป็น datetime และรองรับ timezone
# ให้คอมพิวเตอร์สามารถเข้าในว่าเป็นวันที่และเวลา
df["start_time"] = pd.to_datetime(df["date_before"] + " " + df["timestamp_before"], errors='coerce', utc=True)
df["end_time"] = pd.to_datetime(df["date_after"] + " " + df["timestamp_after"], errors='coerce', utc=True)

# สร้าง Feature actual_drive_time (หน่วย: นาที) และตัดค่าติดลบออก
df["actual_drive_time"] = (df["end_time"] - df["start_time"]).dt.total_seconds() / 60
df = df[df["actual_drive_time"] >= 0]

#ลบข้อมูล ที่ไม่ได้ใช้ในการ Prediction
df.drop(columns=[
    'battery_percent_after', 'drivable_km_after',
    'total_km_after', 'temperature_after',
    'date_before', 'date_after', 'timestamp_after', 'timestamp_before','start_time','end_time'
], inplace=True)

#ตรวจสอบข้อมูลอีกครั้งหลังจากการจัดการข้อมูล
df.head()

# แสดงความสัมพัธ์ของ Features กับ Target ด้วย Correlation Matrix
numeric_df = df.select_dtypes(include=['number'])
corr_matrix = numeric_df.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation")
plt.show()

# ฟังก์ชันแสดงกราฟ Bar Chart ของ Top Features ที่สัมพันธ์กับ Target
# เรียงอันดับจากความสัมพันธ์ มากที่สุดไปน้อยที่สุด
def plot_top_features_correlation(corr_target, top_features, target):
    """สร้าง Bar Chart แสดงค่า Correlation ของ Top Features กับ Target Variable"""
    plt.figure(figsize=(10, 6))
    corr_values = corr_target[top_features]
    plt.bar(top_features, corr_values)
    plt.xlabel("Features")
    plt.ylabel("Absolute Correlation with Target")
    plt.title(f"Top Features Correlation with {target}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

# เรียกใช้งานในโปรแกรมหลัก (Main Program)
if __name__ == "__main__":
    # เลือกฟีเจอร์ที่มี Correlation สูงสุดกับ Target
    target ='actual_drive_time'
    corr_target = corr_matrix[target].drop(target).abs()
    top_features = corr_target.nlargest(n).index.tolist() #สามารถเลือก Features ที่จะใช้จากการจัดอันดับได้โดยแก้ไข้ค่า n 
    print("🔹 Features ที่เลือก:", top_features)

    # วาดกราฟ Bar Chart
    plot_top_features_correlation(corr_target, top_features, target)

    # แสดง Heatmap ของเฉพาะฟีเจอร์ที่เลือก + target
    plt.figure(figsize=(8, 6))
    sns.heatmap(numeric_df[top_features + [target]].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Top n Feature Correlation with Target")
    plt.show()

# นำข้อมูลจาก Top Features และ Target ไป Train
# แยก feature และ target
X_train = df[top_features]
y_train = df['actual_drive_time']


# กำหนดโมเดล
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42),
}

# ประเมิน Model โดยใช้ Leave-One-Out Cross Validation
# สร้าง Leave-One-Out Cross Validation
loo = LeaveOneOut()
cv_results = {name: {"y_true": [], "y_pred": [], "errors": []} for name in models}

# วนลูป LOO-CV
for train_index, test_index in loo.split(X_train):
    X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]

    for name, model in models.items():
        model.fit(X_train_fold, y_train_fold)  # เทรนโมเดล
        y_pred = model.predict(X_test_fold)[0]  # ทำนายค่า

        # บันทึกค่าจริง, ค่าพยากรณ์ และ Error
        y_true_value = y_test_fold.values[0]
        error = y_true_value - y_pred

        cv_results[name]["y_true"].append(y_true_value)
        cv_results[name]["y_pred"].append(y_pred)
        cv_results[name]["errors"].append(error)

# คำนวณ RMSE, MAE, R² และ Std ของแต่ละ Error
for name, scores in cv_results.items():
    y_true = np.array(scores["y_true"])
    y_pred = np.array(scores["y_pred"])
    errors = np.array(scores["errors"])  # Error ของแต่ละ fold

    rmse = np.sqrt(np.mean(errors**2))  # ค่าเฉลี่ย RMSE
    mae = np.mean(np.abs(errors))  # ค่าเฉลี่ย MAE
    r2 = r2_score(y_true, y_pred)  # คำนวณค่า R²

    # คำนวณ Standard Deviation (SD)

    error_std = np.std(errors)  # SD ของ Error

    print(f"{name}:")
    print(f"  RMSE: {rmse:.4f} ")
    print(f"  MAE: {mae:.4f}")
    print(f"  R² Score: {r2:.4f}")
    print(f"  Error SD: {error_std:.4f}\n")




