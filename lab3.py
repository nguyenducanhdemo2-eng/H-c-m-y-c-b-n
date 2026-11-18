import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Bài 1
np.random.seed(42)
X = np.arange(1, 101).reshape(-1, 1)
y = 2 * X.ravel() + 1 + np.random.normal(0, 10, 100)

model = LinearRegression()
model.fit(X, y)

print("Bài 1")
print("Hệ số hồi quy:", model.coef_[0])
print("Hệ số chặn:", model.intercept_)

# Bài 2
diabetes = load_diabetes()
X = diabetes.data[:, 2:3]
y = diabetes.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\nBài 2")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R-squared:", r2_score(y_test, y_pred))
print("MAE là sai số tuyệt đối trung bình, càng nhỏ càng tốt.")
print("MSE là trung bình bình phương sai số, càng nhỏ càng tốt.")
print("R-squared cho biết mức độ giải thích dữ liệu của mô hình, càng gần 1 càng tốt.")

# Bài 3
thu_tu = np.argsort(X_test[:, 0])

plt.scatter(X_test[:, 0], y_test, color="blue", label="Dữ liệu thực tế")
plt.plot(X_test[thu_tu, 0], y_pred[thu_tu], color="red", label="Đường hồi quy")
plt.xlabel("Đặc trưng BMI")
plt.ylabel("Chỉ số bệnh tiến triển")
plt.title("Bài 3 - Hồi quy tuyến tính")
plt.legend()
plt.show()

# Bài 4
X = diabetes.data
y = diabetes.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\nBài 4")
print("MSE:", mean_squared_error(y_test, y_pred))

plt.scatter(y_test, y_pred, color="blue")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red")
plt.xlabel("Giá trị thực tế")
plt.ylabel("Giá trị dự đoán")
plt.title("Bài 4 - So sánh kết quả dự đoán")
plt.show()
