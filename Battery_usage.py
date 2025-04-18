# สวัสดีครับ วันนี้ผมจะมาแชร์ตัวอย่างโค้ดสำหรับการทำ Machine Learning เพื่อพยากรณ์การใช้งานแบตเตอรี่ (Battery Usage Prediction)
# โค้ดนี้ออกแบบมาเพื่อเป็นตัวช่วยให้ผู้ขับขี่รถยนต์ไฟฟ้า (EV) สามารถวางแผนการเดินทางได้อย่างมีประสิทธิภาพมากขึ้น
# ข้อมูลที่ใช้ในการเทรนโมเดลมาจากการเก็บจริงของผู้ใช้งานรถยนต์ไฟฟ้าในชีวิตประจำวัน
# โดยสะท้อนถึงพฤติกรรมการขับขี่จริง ซึ่งเหมาะสำหรับนำมาประยุกต์ใช้ในงานด้าน Machine Learning

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

#ตรวจสอบจำนวนแถวก่อนกรอง
rows_before = len(df)
print(f"จำนวนแถวเริ่มต้น: {rows_before}")

#กรองข้อมูลตามเงื่อนไข โดยกรองข้อมูลที่ทำให้เกิดข้อผิดพลาด
#drivable_km_before ควรมากกว่า drivable_km_after (แปลว่าใช้งานแบตเตอรี่ไปแล้ว)
#total_km_before ควรน้อยกว่า total_km_after (แปลว่ารถได้เดินทางจริง)
condition_drivable = df['drivable_km_before'] > df['drivable_km_after']
condition_total_km = df['total_km_before'] < df['total_km_after']
combined_condition = (condition_drivable) & (condition_total_km)
df_cleaned_strict = df[combined_condition].copy()



#แสดงจำนวนข้อมูลหลังการกรอง
print("\nข้อมูล 5 แถวแรกหลังกรองตามเงื่อนไข:")
print(df_cleaned_strict.head())

#สร้าง Feature สำหรับการทำนายแบตเตอรี่ที่ใช้ และลบ Feature ที่ไม่จำเป็นออก เช่น ข้อมูลในอนาคต วันที่และเวลา
df = df_cleaned_strict #บันทึกข้อมูลมาที่ตัวแปรเดิม
df["battery_usage"] = df["battery_percent_before"] - df["battery_percent_after"] #สร้าง Target feature เพื่อให้ Model ทำนาย
df = df.drop(columns=['battery_percent_after'])
df = df.drop(columns=['actual_drive_time', 'start_time', 'end_time', 
                      'total_km_after', 'temperature_after', 'drivable_km_after'])

#ตรวจสอบ Feature ที่ใช้หลังจากการคัดเลือก Feature
df.head()

#คำนวณ Correlation Matrix และแสดง Heatmap รวม
#เลือกเฉพาะคอลัมน์ที่เป็นตัวเลข (numeric_df) เพราะ correlation คำนวณได้เฉพาะตัวเลข 
#สร้าง correlation matrix สำหรับดูความสัมพันธ์ระหว่างแต่ละ feature 
#ใช้ heatmap จาก seaborn แสดงความสัมพันธ์แบบภาพ สีแดง-น้ำเงินสื่อถึงความสัมพันธ์เชิงบวก/ลบ
numeric_df = df.select_dtypes(include=['number'])
corr_matrix = numeric_df.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation")
plt.show()

#แสดงกราฟ Bar Chart ของ Top Features ที่สัมพันธ์กับ Target
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

#เรียกใช้งาน Feature ในโปรแกรมหลัก (Main Program) :
  #เลือก target ที่เราสนใจจะพยากรณ์หรือวิเคราะห์: battery_usage
  #คำนวณค่า correlation ของทุกฟีเจอร์กับ target แล้วแปลงเป็นค่า absolute (ค่าสัมบูรณ์)
  #เลือก top n features ที่มีความสัมพันธ์กับ target มากที่สุด(เลือกจำนวณ feature ที่จะใช้ได้ โดยเปลี่ยนค่า n)
  #แสดงชื่อ features ที่เลือกออกทางหน้าจอ
  #เรียกฟังก์ชัน plot_top_features_correlation() เพื่อแสดง bar chart

#สร้าง heatmap ใหม่เฉพาะฟีเจอร์ที่เลือก พร้อมกับ target เพื่อดูความสัมพันธ์ในกลุ่มย่อยที่น่าสนใจ
if __name__ == "__main__":
    # เลือกฟีเจอร์ที่มี Correlation สูงสุดกับ Target
    target ='battery_usage'
    corr_target = corr_matrix[target].drop(target).abs()
    top_features = corr_target.nlargest(n).index.tolist()
    print("🔹 Features ที่เลือก:", top_features)

    # วาดกราฟ Bar Chart
    plot_top_features_correlation(corr_target, top_features, target)

    # แสดง Heatmap ของเฉพาะฟีเจอร์ที่เลือก + target
    plt.figure(figsize=(8, 6))
    sns.heatmap(numeric_df[top_features + [target]].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Top 5 Feature Correlation with Target")
    plt.show()

#เตรียม Model สำหรับ Train ชุดข้อมูล

# แยก feature และ target
X_train = df[top_features]
y_train = df['battery_usage']


# กำหนดโมเดล
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42),
}

# ประเมินข้อมูลโดยใช้ Leave one out cross validation
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








