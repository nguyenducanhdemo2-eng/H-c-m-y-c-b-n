import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_digits
from sklearn.linear_model import LinearRegression

print("Bài 1")
mang = np.arange(1, 11)
print("Mảng từ 1 đến 10:")
print(mang)

sinh_vien = pd.DataFrame({
    "Họ tên": [
        "Nguyễn Văn An",
        "Trần Thị Bình",
        "Lê Văn Cường",
        "Phạm Thị Dung",
        "Hoàng Văn Em"
    ],
    "Tuổi": [18, 19, 18, 20, 19],
    "Điểm toán": [8.0, 7.5, 9.0, 6.5, 8.5]
})

print("\nDanh sách sinh viên:")
print(sinh_vien)
print("\nThông tin tổng quan:")
sinh_vien.info()
print("\nĐiểm trung bình môn toán:", sinh_vien["Điểm toán"].mean())

print("\nBài 2")
# 1. Dự đoán giá nhà dựa trên diện tích: Supervised
# 2. Gom nhóm khách hàng dựa trên hành vi mua sắm: Unsupervised
# 3. Phân loại email là Spam hay không: Supervised
# 4. Nhận diện chữ viết tay từ hình ảnh: Supervised
print("Đã phân loại 4 bài toán Machine Learning trong code.")

print("\nBài 3")
iris = load_iris()
X_iris = iris.data
y_iris = iris.target

model = LinearRegression()
model.fit(X_iris, y_iris)
du_doan = model.predict(X_iris[:5])

print("Kết quả dự đoán 5 mẫu đầu tiên:")
print(du_doan)

print("\nBài 4")
digits = load_digits()
X_digits = digits.data
y_digits = digits.target

print("Hình dạng dữ liệu:", X_digits.shape)
print("Hình dạng nhãn:", y_digits.shape)
print("Nhãn của mẫu đầu tiên:", y_digits[0])

plt.imshow(digits.images[0], cmap="gray")
plt.title("Mẫu dữ liệu đầu tiên")
plt.axis("off")
plt.show()
