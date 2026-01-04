# 🧬 Gene Expression Ensemble Clustering

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

Ứng dụng Streamlit thực hiện **Ensemble Clustering** trên dữ liệu gene expression để phân loại ung thư ALL/AML.

## 📋 Mô tả

Dự án này triển khai 3 thuật toán clustering:
- **K-Means++**: Improved initialization for K-Means
- **Hierarchical Clustering**: Agglomerative với single linkage
- **GMM**: Gaussian Mixture Model với diagonal covariance
- **Ensemble**: Kết hợp 3 phương pháp trên bằng weighted co-association matrix

## 🚀 Deploy lên Streamlit Cloud

### Bước 1: Chuẩn bị GitHub Repository

1. **Tạo repository mới trên GitHub** (hoặc sử dụng repo hiện tại)
   - Đảm bảo repo là **PUBLIC** hoặc bạn có quyền kết nối với Streamlit Cloud

2. **Push code lên GitHub:**

```bash
# Khởi tạo git (nếu chưa có)
git init

# Add tất cả files
git add .

# Commit
git commit -m "Initial commit: Streamlit app for gene expression clustering"

# Add remote (thay YOUR_USERNAME và YOUR_REPO)
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git

# Push lên GitHub
git push -u origin main
```

### Bước 2: Deploy trên Streamlit Cloud

1. **Truy cập:** https://share.streamlit.io/

2. **Đăng nhập** bằng GitHub account

3. **Click "New app"**

4. **Điền thông tin:**
   - **Repository:** Chọn repo của bạn (ví dụ: `AiNguyen2014/MachineLearningProject`)
   - **Branch:** `main` (hoặc branch bạn muốn deploy)
   - **Main file path:** `app.py`

5. **Click "Deploy"** và đợi vài phút

6. **Done!** App của bạn sẽ có URL dạng: `https://your-app.streamlit.app`

### Bước 3: Cấu trúc thư mục cần thiết

```
MachineLearningProject/
├── app.py                          # ✅ Main Streamlit app
├── requirements.txt                # ✅ Dependencies
├── .streamlit/
│   └── config.toml                 # ✅ Streamlit config
├── utils/
│   ├── preprocessing.py            # ✅ Data preprocessing
│   ├── clustering.py               # ✅ Clustering algorithms
│   ├── ensemble.py                 # ✅ Ensemble logic
│   └── visualization.py            # ✅ Visualization functions
├── data_processed_72.csv           # ✅ Processed data
├── actual.csv                      # ✅ True labels
└── README.md                       # ✅ This file
```

## 📦 Dependencies

Tất cả dependencies được liệt kê trong `requirements.txt`:

```
streamlit==1.31.0
numpy==1.24.3
pandas==2.0.3
matplotlib==3.7.2
seaborn==0.12.2
scikit-learn==1.3.0
scipy==1.11.1
plotly==5.17.0
```

## 🏃 Chạy Local

Để test app trên máy local:

```bash
# Cài dependencies
pip install -r requirements.txt

# Chạy app
streamlit run app.py
```

App sẽ mở tại: http://localhost:8501

## 📊 Dataset

- **Source:** Golub et al. (1999) - "Molecular Classification of Cancer"
- **Samples:** 72 (38 ALL + 34 AML)
- **Features:** 100 genes được chọn lọc
- **Files:**
  - `data_processed_72.csv`: Dữ liệu gene expression đã tiền xử lý
  - `actual.csv`: Nhãn thực tế (ALL/AML)

## ✨ Features

- **Interactive Clustering:** Chạy 3 thuật toán clustering + ensemble
- **Real-time Parameters:** Điều chỉnh trọng số và threshold
- **Visualization:** 2D SVD projection, confusion matrix, radar chart
- **Metrics:** Silhouette, ARI, NMI, Purity
- **Comparison:** So sánh performance giữa các phương pháp

## 🐛 Troubleshooting

### Lỗi: "ModuleNotFoundError"
- Đảm bảo `requirements.txt` đầy đủ
- Kiểm tra version của các packages

### Lỗi: "File not found"
- Đảm bảo `data_processed_72.csv` và `actual.csv` có trong repo
- Kiểm tra đường dẫn trong code

### App chạy chậm
- Streamlit tự động cache functions với `@st.cache_data`
- Lần đầu sẽ chậm, lần sau sẽ nhanh hơn

### Lỗi deploy
- Kiểm tra logs trong Streamlit Cloud dashboard
- Đảm bảo repo là public hoặc có quyền truy cập
- File size không quá 1GB

## 📝 License

MIT License

## 👤 Author

**Trang Tran**
- GitHub: [@AiNguyen2014](https://github.com/AiNguyen2014)
- Project: Machine Learning - Gene Expression Analysis

## 📚 References

- Golub et al. (1999). Molecular classification of cancer: class discovery and class prediction by gene expression monitoring
- Fred & Jain (2005). Combining multiple clusterings using evidence accumulation

---

**Status:** ✅ Ready to deploy on Streamlit Cloud!
