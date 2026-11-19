# Assignment 1 - Học máy cơ bản

## Đề tài
Tiền xử lý dữ liệu Titanic để chuẩn bị dự đoán khả năng sống sót của hành khách (bài toán phân loại).

## Nguồn dữ liệu
Data Science Dojo: https://github.com/datasciencedojo/datasets/blob/master/titanic.csv

## Kết quả
- Dữ liệu ban đầu: 891 dòng, 12 cột.
- Giá trị thiếu: Age 177, Cabin 687, Embarked 2.
- Dữ liệu sau xử lý: 891 dòng, 11 cột, không còn giá trị thiếu.
- Xử lý: loại bỏ cột không cần thiết, thay dữ liệu khuyết, mã hóa Sex và Embarked, thêm FamilySize.
- Chưa huấn luyện mô hình: sẽ thực hiện trong giai đoạn 2.

## Tệp
- `assignment1.py`: mã nguồn.
- `data/titanic_raw.csv`: dữ liệu gốc.
- `data/titanic_processed.csv`: dữ liệu sau tiền xử lý.
- `anh_ket_qua.svg`: hình kết quả thực thi.
- `ket_qua_chay.txt`: kết quả thực thi dạng text.

## Chạy chương trình
```bash
pip install pandas matplotlib
python assignment1.py
```
Hai biểu đồ sẽ được tạo ở thư mục `hinh_anh` khi chạy.
