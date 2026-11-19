from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

thu_muc = Path(__file__).resolve().parent
file_goc = thu_muc / "data" / "titanic_raw.csv"
file_ket_qua = thu_muc / "data" / "titanic_processed.csv"
thu_muc_hinh = thu_muc / "hinh_anh"
thu_muc_hinh.mkdir(exist_ok=True)

# 1. Đọc dữ liệu
du_lieu = pd.read_csv(file_goc)
print("ASSIGNMENT 1 - TIEN XU LY DU LIEU TITANIC")
print("Du lieu ban dau:", du_lieu.shape[0], "dong,", du_lieu.shape[1], "cot")
print("So dong trung lap theo PassengerId:", du_lieu["PassengerId"].duplicated().sum())
print("\nGia tri thieu ban dau:")
print(du_lieu.isnull().sum()[du_lieu.isnull().sum() > 0].to_string())

# 2. Làm sạch dữ liệu
ket_qua = du_lieu.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"]).copy()
ket_qua["Age"] = ket_qua["Age"].fillna(ket_qua["Age"].median())
ket_qua["Fare"] = ket_qua["Fare"].fillna(ket_qua["Fare"].median())
ket_qua["Embarked"] = ket_qua["Embarked"].fillna(ket_qua["Embarked"].mode()[0])

# 3. Mã hóa và tạo đặc trưng
ket_qua["Sex"] = ket_qua["Sex"].map({"female": 0, "male": 1})
ket_qua["FamilySize"] = ket_qua["SibSp"] + ket_qua["Parch"] + 1
for ben in ["C", "Q", "S"]:
    ket_qua["Embarked_" + ben] = (ket_qua["Embarked"] == ben).astype(int)
ket_qua = ket_qua.drop(columns=["Embarked"])
ket_qua = ket_qua[[
    "Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare",
    "FamilySize", "Embarked_C", "Embarked_Q", "Embarked_S"
]]

# 4. Lưu kết quả
ket_qua.to_csv(file_ket_qua, index=False)
print("\nSau khi xu ly:", ket_qua.shape[0], "dong,", ket_qua.shape[1], "cot")
print("Tong gia tri thieu con lai:", ket_qua.isnull().sum().sum())
print("So cot chu va nhan dang:", len(ket_qua.select_dtypes(include="object").columns))
print("So nguoi khong song sot:", (ket_qua["Survived"] == 0).sum())
print("So nguoi song sot:", (ket_qua["Survived"] == 1).sum())
print("Da luu: data/titanic_processed.csv")

# 5. Vẽ biểu đồ
cot_thieu = ["Age", "Cabin", "Embarked"]
truoc = du_lieu[cot_thieu].isnull().sum()
sau = ket_qua.isnull().sum().sum()
plt.figure(figsize=(8, 4.5))
plt.bar(cot_thieu, truoc, label="Truoc xu ly")
plt.bar(cot_thieu, [0, 0, 0], label="Sau xu ly")
plt.title("Gia tri thieu trong du lieu Titanic")
plt.ylabel("So gia tri thieu")
plt.legend()
plt.tight_layout()
plt.savefig(thu_muc_hinh / "gia_tri_thieu.png", dpi=150)
plt.close()

plt.figure(figsize=(6, 4))
plt.bar(["Khong song sot", "Song sot"], [
    (ket_qua["Survived"] == 0).sum(), (ket_qua["Survived"] == 1).sum()
])
plt.title("So luong hanh khach theo ket qua")
plt.ylabel("So hanh khach")
plt.tight_layout()
plt.savefig(thu_muc_hinh / "ty_le_song_sot.png", dpi=150)
plt.close()
print("Da luu 2 bieu do trong thu muc hinh_anh/")
