import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Bài 1
du_lieu = pd.DataFrame({
    "Tuổi": [18, 19, 20, 21, 20, 22, 20, 19, 18, 21, 20, 200],
    "Điểm": [7.5, 8.0, np.nan, 6.5, 9.0, 7.0, 8.5, 7.5, 8.0, 6.0, 9.5, 7.0]
}, dtype=float)

print("Dữ liệu ban đầu:")
print(du_lieu)

du_lieu = du_lieu.fillna(du_lieu.mean())

for cot in du_lieu.columns:
    trung_binh = du_lieu[cot].mean()
    do_lech_chuan = du_lieu[cot].std()
    ngoai_le = (du_lieu[cot] > trung_binh + 3 * do_lech_chuan) | (du_lieu[cot] < trung_binh - 3 * do_lech_chuan)
    trung_binh_moi = du_lieu.loc[~ngoai_le, cot].mean()
    du_lieu.loc[ngoai_le, cot] = trung_binh_moi

print("\nDữ liệu sau khi làm sạch:")
print(du_lieu)

# Bài 2
du_lieu_2 = np.array([
    [20, 10],
    [30, 30],
    [40, 50],
    [50, 70],
    [60, 100]
])

standard_scaler = StandardScaler()
minmax_scaler = MinMaxScaler()

du_lieu_standard = standard_scaler.fit_transform(du_lieu_2)
du_lieu_minmax = minmax_scaler.fit_transform(du_lieu_2)

print("\nDữ liệu ban đầu:")
print(du_lieu_2)
print("\nKết quả StandardScaler:")
print(du_lieu_standard)
print("\nKết quả MinMaxScaler:")
print(du_lieu_minmax)

# Bài 3
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nKích thước dữ liệu gốc:", X.shape)
print("Kích thước X_train:", X_train.shape)
print("Kích thước X_test:", X_test.shape)
print("Kích thước y_train:", y_train.shape)
print("Kích thước y_test:", y_test.shape)

# Bài 4
digits = load_digits()
X_digits = digits.data
y_digits = digits.target

X_train, X_test, y_train, y_test = train_test_split(
    X_digits, y_digits, test_size=0.2, random_state=42
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)

print("\nKích thước tập Train:", X_train.shape)
print("Kích thước tập Test:", X_test.shape)
print("Độ chính xác của mô hình:", pipeline.score(X_test, y_test))
