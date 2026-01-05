"""
Streamlit App for Gene Expression Ensemble Clustering
Author: Trang Tran
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Import custom utilities
from utils.preprocessing import prepare_processed_data, get_svd_projection
from utils.clustering import run_all_clustering
from utils.ensemble import ensemble_clustering, get_cluster_distribution
from utils.visualization import (
    calculate_all_metrics,
    create_metrics_table,
    plot_metrics_comparison
)

# Page configuration
st.set_page_config(
    page_title="Phân cụm Ensemble - Nhóm 14",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Green theme
st.markdown("""
<style>
    /* Main content area - light green background */
    .main {
        background-color: #f0f8f5;
    }
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background-color: #f0f8f5;
    }
    
    /* Headers - keep black */
    h1 {
        color: #2c3e50;
        font-weight: 600;
        letter-spacing: -0.5px;
    }
    
    h2, h3 {
        color: #34495e;
        font-weight: 500;
    }
    
    /* Sidebar styling - darker green background */
    [data-testid="stSidebar"] {
        background-color: #d5f4e6;
    }
    
    /* Metrics cards */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        color: #2c3e50;
    }
    
    [data-testid="stMetricLabel"] {
        color: #34495e;
        font-weight: 500;
    }
    
    /* Info boxes */
    .stAlert {
        background-color: #d5f4e6;
        border-left: 4px solid #27ae60;
    }
    
    /* Success boxes */
    .stSuccess {
        background-color: #d5f4e6;
        color: #155724;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #27ae60 0%, #229954 100%);
        color: white;
        border-radius: 6px;
        border: none;
        padding: 0.6rem 2.5rem;
        font-weight: 600;
        transition: all 0.3s;
        box-shadow: 0 2px 8px rgba(39, 174, 96, 0.3);
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #229954 0%, #1e8449 100%);
        box-shadow: 0 4px 12px rgba(39, 174, 96, 0.4);
        transform: translateY(-1px);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #d5f4e6;
        padding: 10px;
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 12px 24px;
        background-color: white;
        color: #2c3e50;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #27ae60 0%, #229954 100%);
        color: white;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #d5f4e6;
        color: #2c3e50;
        border-radius: 6px;
        font-weight: 500;
    }
    
    /* Dataframe */
    [data-testid="stDataFrame"] {
        background-color: white;
    }
</style>
""", unsafe_allow_html=True)


# Helper functions
def purity_score(y_true, y_pred):
    """Calculate purity score"""
    contingency_matrix = pd.crosstab(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix.values, axis=0)) / np.sum(contingency_matrix.values)


@st.cache_data
def load_sample_data():
    """Load sample data for demo mode"""
    data_dict = {}
    
    # Load processed data
    if Path("data_processed_72.csv").exists():
        df = pd.read_csv("data_processed_72.csv", index_col=0)
        X = df.values
        data_dict['X'] = X
        data_dict['df'] = df
        
        # Load ground truth labels
        if Path("actual.csv").exists():
            df_actual = pd.read_csv("actual.csv")
            if 'cancer' in df_actual.columns:
                y_true = df_actual['cancer'].map({'ALL': 0, 'AML': 1}).values
                data_dict['y_true'] = y_true
                data_dict['df_actual'] = df_actual
    
    return data_dict


def run_clustering_pipeline(X, weights, threshold):
    """Run the full clustering pipeline"""
    # Run individual algorithms
    results = run_all_clustering(X, n_clusters=2)
    
    # Extract labels from results
    labels_dict = {
        'kmeans': results['kmeans']['labels'],
        'hierarchical': results['hierarchical']['labels'],
        'gmm': results['gmm']['labels']
    }
    
    # Run ensemble
    labels_ensemble, C_matrix = ensemble_clustering(
        labels_dict,
        weights=weights,
        threshold=threshold
    )
    labels_dict['ensemble'] = labels_ensemble
    
    return labels_dict, results, C_matrix


def display_results(X, labels_dict, y_true, weights, C_matrix):
    """Display clustering results with 4 tabs"""
    
    # Project to 2D
    X_2d = get_svd_projection(X, n_components=2)
    
    # Calculate metrics for all methods
    metrics_dict = {}
    for name, labels in labels_dict.items():
        metrics = calculate_all_metrics(X, labels, y_true)
        metrics_dict[name] = metrics
    
    # Tabs - thêm About tab
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Clustering Results",
        "📈 Metrics Comparison",
        "� Co-association Matrix",
        "ℹ️ About"
    ])
    
    # TAB 1: Clustering Results
    with tab1:
        st.header("Clustering Results")
        
        # Plot giống như trong Colab - 6 scatter plots riêng lẻ
        
        # Row 1: K-Means++ và Hierarchical
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("K-Means++")
            fig1, ax1 = plt.subplots(figsize=(8, 6))
            colors_km = ['#2ecc71' if l == 0 else '#9b59b6' for l in labels_dict['kmeans']]
            ax1.scatter(X_2d[:, 0], X_2d[:, 1], c=colors_km, s=80, alpha=0.8, edgecolors='black', linewidth=0.5)
            sil_km = metrics_dict['kmeans']['silhouette']
            ari_km = metrics_dict['kmeans']['ari']
            nmi_km = metrics_dict['kmeans']['nmi']
            pur_km = metrics_dict['kmeans']['purity']
            ax1.set_title(f"K-Means++\nSil={sil_km:.3f}, ARI={ari_km:.3f}, NMI={nmi_km:.3f}, Purity={pur_km:.3f}", 
                         fontsize=14, fontweight='bold')
            ax1.set_xlabel("PC1")
            ax1.set_ylabel("PC2")
            ax1.grid(True, alpha=0.3)
            legend_cluster = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ecc71', markersize=10, label='Cluster 0'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#9b59b6', markersize=10, label='Cluster 1')
            ]
            ax1.legend(handles=legend_cluster, loc='lower right', framealpha=0.9)
            plt.tight_layout()
            st.pyplot(fig1)
        
        with col2:
            st.subheader("Hierarchical (Single)")
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            colors_hier = ['#2ecc71' if l == 0 else '#9b59b6' for l in labels_dict['hierarchical']]
            ax2.scatter(X_2d[:, 0], X_2d[:, 1], c=colors_hier, s=80, alpha=0.8, edgecolors='black', linewidth=0.5)
            sil_h = metrics_dict['hierarchical']['silhouette']
            ari_h = metrics_dict['hierarchical']['ari']
            nmi_h = metrics_dict['hierarchical']['nmi']
            pur_h = metrics_dict['hierarchical']['purity']
            ax2.set_title(f"Hierarchical (Single)\nSil={sil_h:.3f}, ARI={ari_h:.3f}, NMI={nmi_h:.3f}, Purity={pur_h:.3f}", 
                         fontsize=14, fontweight='bold')
            ax2.set_xlabel("PC1")
            ax2.set_ylabel("PC2")
            ax2.grid(True, alpha=0.3)
            ax2.legend(handles=legend_cluster, loc='lower right', framealpha=0.9)
            plt.tight_layout()
            st.pyplot(fig2)
        
        # Row 2: GMM và Ground Truth
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("GMM")
            fig3, ax3 = plt.subplots(figsize=(8, 6))
            colors_gmm = ['#2ecc71' if l == 0 else '#9b59b6' for l in labels_dict['gmm']]
            ax3.scatter(X_2d[:, 0], X_2d[:, 1], c=colors_gmm, s=80, alpha=0.8, edgecolors='black', linewidth=0.5)
            sil_g = metrics_dict['gmm']['silhouette']
            ari_g = metrics_dict['gmm']['ari']
            nmi_g = metrics_dict['gmm']['nmi']
            pur_g = metrics_dict['gmm']['purity']
            ax3.set_title(f"GMM\nSil={sil_g:.3f}, ARI={ari_g:.3f}, NMI={nmi_g:.3f}, Purity={pur_g:.3f}", 
                         fontsize=14, fontweight='bold')
            ax3.set_xlabel("PC1")
            ax3.set_ylabel("PC2")
            ax3.grid(True, alpha=0.3)
            ax3.legend(handles=legend_cluster, loc='lower right', framealpha=0.9)
            plt.tight_layout()
            st.pyplot(fig3)
        
        with col4:
            st.subheader("Ground Truth (ALL vs AML)")
            fig4, ax4 = plt.subplots(figsize=(8, 6))
            if y_true is not None:
                colors_gt = ['#3498db' if y == 0 else '#e74c3c' for y in y_true]
                ax4.scatter(X_2d[:, 0], X_2d[:, 1], c=colors_gt, s=80, alpha=0.8, edgecolors='black', linewidth=0.5)
                ax4.set_title("Ground Truth\n(ALL vs AML)", fontsize=14, fontweight='bold')
                ax4.set_xlabel("PC1")
                ax4.set_ylabel("PC2")
                ax4.grid(True, alpha=0.3)
                legend_gt = [
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db', markersize=10, label='ALL'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', markersize=10, label='AML')
                ]
                ax4.legend(handles=legend_gt, loc='lower right', framealpha=0.9)
            else:
                ax4.text(0.5, 0.5, "No Ground Truth Available", ha='center', va='center', transform=ax4.transAxes, fontsize=14)
                ax4.set_xlabel("PC1")
                ax4.set_ylabel("PC2")
            plt.tight_layout()
            st.pyplot(fig4)
        
        # Row 3: ENSEMBLE và Ensemble vs Ground Truth
        col5, col6 = st.columns(2)
        
        with col5:
            st.subheader("🏆 ENSEMBLE")
            fig5, ax5 = plt.subplots(figsize=(8, 6))
            colors_ens = ['#2ecc71' if l == 0 else '#9b59b6' for l in labels_dict['ensemble']]
            ax5.scatter(X_2d[:, 0], X_2d[:, 1], c=colors_ens, s=100, alpha=0.9, edgecolors='gold', linewidth=2)
            sil_e = metrics_dict['ensemble']['silhouette']
            ari_e = metrics_dict['ensemble']['ari']
            nmi_e = metrics_dict['ensemble']['nmi']
            pur_e = metrics_dict['ensemble']['purity']
            ax5.set_title(f"ENSEMBLE\nSil={sil_e:.3f}, ARI={ari_e:.3f}, NMI={nmi_e:.3f}, Purity={pur_e:.3f}", 
                         fontsize=14, fontweight='bold', color='darkgreen')
            ax5.set_xlabel("PC1")
            ax5.set_ylabel("PC2")
            ax5.grid(True, alpha=0.3)
            for spine in ax5.spines.values():
                spine.set_edgecolor('gold')
                spine.set_linewidth(3)
            ax5.legend(handles=legend_cluster, loc='lower right', framealpha=0.9)
            plt.tight_layout()
            st.pyplot(fig5)
        
        with col6:
            st.subheader("Ensemble vs Ground Truth")
            fig6, ax6 = plt.subplots(figsize=(8, 6))
            if y_true is not None:
                correct = np.sum(labels_dict['ensemble'] == y_true)
                incorrect = len(y_true) - correct
                colors_correct = ['green' if labels_dict['ensemble'][i] == y_true[i] else 'red' for i in range(len(y_true))]
                ax6.scatter(X_2d[:, 0], X_2d[:, 1], c=colors_correct, s=80, alpha=0.8, edgecolors='black', linewidth=0.5)
                ax6.set_title(f"Ensemble vs Ground Truth\nĐúng: {correct}/72 ({100*correct/72:.1f}%)", 
                             fontsize=14, fontweight='bold')
                ax6.set_xlabel("PC1")
                ax6.set_ylabel("PC2")
                ax6.grid(True, alpha=0.3)
                legend_correct = [
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label=f'Đúng ({correct})'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label=f'Sai ({incorrect})')
                ]
                ax6.legend(handles=legend_correct, loc='lower right', framealpha=0.9)
            else:
                ax6.text(0.5, 0.5, "No Ground Truth Available", ha='center', va='center', transform=ax6.transAxes, fontsize=14)
                ax6.set_xlabel("PC1")
                ax6.set_ylabel("PC2")
            plt.tight_layout()
            st.pyplot(fig6)
        
        # Cluster distributions
        st.markdown("---")
        st.subheader("Phân bố cụm")
        
        dist_cols = st.columns(4)
        for idx, (name, labels) in enumerate(labels_dict.items()):
            dist = get_cluster_distribution(labels)
            with dist_cols[idx]:
                st.write(f"**{name.upper()}**")
                for cluster_id, count in sorted(dist.items()):
                    st.write(f"Cluster {cluster_id}: {count} samples")
    
    # TAB 2: Metrics Comparison
    with tab2:
        st.header("So sánh Metrics")
        
        # Metrics table
        st.subheader("Bảng Metrics")
        df_metrics = create_metrics_table(metrics_dict)
        
        # Highlight best values
        st.dataframe(
            df_metrics.style.highlight_max(axis=0, color='lightgreen'),
            use_container_width=True
        )
        
        # Bar charts (giống Colab)
        st.subheader("Biểu đồ so sánh")
        fig_bars = plot_metrics_comparison(metrics_dict)
        st.pyplot(fig_bars)
    
    # TAB 3: Co-association Matrix
    with tab3:
        st.header("Ma trận Đồng liên kết (Co-association Matrix)")
        
        # Display co-association matrix
        fig_coassoc, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(C_matrix, cmap='Greens', cbar=True, square=True, ax=ax,
                   cbar_kws={'label': 'Điểm đồng liên kết'})
        ax.set_title("Ma trận Đồng liên kết có Trọng số", fontsize=14, fontweight='600', color='#2c3e50')
        ax.set_xlabel("Chỉ số mẫu", fontsize=11)
        ax.set_ylabel("Chỉ số mẫu", fontsize=11)
        plt.tight_layout()
        st.pyplot(fig_coassoc)
        
        st.markdown("""
        ### Giải thích về Ma trận Đồng liên kết
        
        Ma trận đồng liên kết thể hiện sự **đồng thuận** giữa các thuật toán phân cụm:
        - Mỗi ô (i,j) cho biết tần suất mẫu i và j được gán vào cùng một cụm
        - Giá trị cao hơn (màu xanh đậm hơn) = sự đồng thuận mạnh hơn giữa các thuật toán
        - Ma trận này được tính bằng phiếu bầu có trọng số từ K-Means++, Hierarchical và GMM
        - Phân cụm ensemble cuối cùng sử dụng đồng thuận dựa trên đồ thị từ ma trận này
        """)
    
    # TAB 4: About
    with tab4:
        st.header("Về Dự án")
        
        col_a, col_b = st.columns([2, 1])
        
        with col_a:
            st.markdown("""
            ## 🧬 Phân cụm Ensemble cho Dữ liệu Gene Expression
            
            ### Tổng quan
            Ứng dụng này thực hiện **phân cụm ensemble** trên dữ liệu biểu hiện gen (gene expression) 
            để phân loại hai loại ung thư máu: Bạch cầu lympho cấp (ALL) và Bạch cầu tủy cấp (AML).
            
            ### Các thuật toán
            Ba thuật toán phân cụm bổ trợ cho nhau được kết hợp:
            
            **1. K-Means++ (Dựa trên tâm cụm)**
            - Cải tiến khởi tạo so với K-Means thông thường
            - Nhanh và hiệu quả cho dữ liệu lớn
            - Phù hợp cho cụm hình cầu
            
            **2. Hierarchical Clustering (Dựa trên kết nối)**
            - Phương pháp tích tụ với liên kết đơn (single linkage)
            - Nắm bắt cấu trúc phân cấp trong dữ liệu
            - Không giả định về hình dạng cụm
            
            **3. Gaussian Mixture Model (Dựa trên phân phối)**
            - Phân cụm xác suất sử dụng thuật toán EM
            - Gán cụm mềm (soft assignment)
            - Xử lý tốt các cụm chồng lấp
            
            ### Phương pháp Ensemble
            Ba thuật toán được kết hợp bằng **ma trận đồng liên kết có trọng số** (weighted co-association matrix):
            - Mỗi thuật toán bỏ phiếu cho các cặp mẫu (cùng cụm hoặc khác cụm)
            - Phiếu bầu được gán trọng số dựa trên độ quan trọng của thuật toán
            - Phân cụm đồng thuận dựa trên đồ thị để trích xuất cụm cuối cùng
            - Phương pháp này vững hơn bất kỳ thuật toán đơn lẻ nào
            
            ### Các chỉ số đánh giá
            
            **Chỉ số nội bộ** (không cần nhãn thực):
            - **Silhouette Score**: Đo độ gắn kết và tách biệt của cụm (-1 đến 1, càng cao càng tốt)
            
            **Chỉ số bên ngoài** (so sánh với nhãn ALL/AML thực):
            - **ARI (Adjusted Rand Index)**: Độ tương đồng với nhãn thực (0 đến 1, điều chỉnh cho ngẫu nhiên)
            - **NMI (Normalized Mutual Information)**: Thông tin chung với nhãn thực (0 đến 1)
            - **Purity**: Tỷ lệ mẫu được gán đúng (0 đến 1)
            """)
        
        with col_b:
            st.markdown("""
            ### Tập dữ liệu
            
            **Nguồn**  
            Golub et al. (1999)  
            *"Molecular Classification of Cancer"*
            
            **Mẫu**  
            - Tổng: 72 bệnh nhân
            - 38 ca ALL
            - 34 ca AML
            
            **Đặc trưng**  
            - 7,129 gen (ban đầu)
            - 100 gen (được chọn theo variance)
            
            **Tiền xử lý**
            1. Gộp tập train & test
            2. Chuẩn hóa Z-score
            3. Chọn đặc trưng (top 100 gen)
            4. SVD để trực quan hóa 2D
            
            ---
            
            ### Tài liệu tham khảo
            
            📄 **Golub et al. (1999)**  
            "Molecular classification of cancer: class discovery and class prediction by gene expression monitoring"  
            *Science*, 286(5439), 531-537
            
            📄 **Fred & Jain (2005)**  
            "Combining multiple clusterings using evidence accumulation"  
            *IEEE TPAMI*, 27(6), 835-850
            
            ---
            
            ### Tác giả
            **Nhóm 14**  
            Đồ án Machine Learning  
            Năm 2026
            """)


def run_demo_mode(weights, threshold):
    """Run demo mode with sample data"""
    st.header("Gene Expression Data - Demo")
    
    # Load sample data
    sample_data = load_sample_data()
    
    if 'X' not in sample_data:
        st.error("❌ Data file not found: data_processed_72.csv")
        return
    
    X = sample_data['X']
    df = sample_data['df']
    y_true = sample_data.get('y_true', None)
    
    # Display data info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Samples", X.shape[0])
    with col2:
        st.metric("Genes", X.shape[1])
    with col3:
        if y_true is not None:
            st.metric("Labels", "Available")
        else:
            st.metric("Labels", "N/A")
    
    st.markdown("")
    
    # HIỂN THỊ DỮ LIỆU ĐẦU VÀO
    with st.expander("� View Raw Data", expanded=False):
        st.subheader("Original Data (first 5 rows)")
        
        # Load raw data
        try:
            df_train_raw = pd.read_csv("data_set_ALL_AML_train.csv", index_col=0)
            df_test_raw = pd.read_csv("data_set_ALL_AML_independent.csv", index_col=0)
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.write("**Training set (38 samples)**")
                st.dataframe(df_train_raw.head(5), use_container_width=True)
                st.caption(f"Shape: {df_train_raw.shape}")
            
            with col_b:
                st.write("**Test set (34 samples)**")
                st.dataframe(df_test_raw.head(5), use_container_width=True)
                st.caption(f"Shape: {df_test_raw.shape}")
            
            # Hiển thị Ground Truth Labels
            if 'df_actual' in sample_data:
                st.write("**Ground Truth Labels**")
                st.dataframe(sample_data['df_actual'].head(10), use_container_width=True)
        except Exception as e:
            st.warning(f"Cannot load raw data: {e}")
    
    # HIỂN THỊ DỮ LIỆU ĐÃ XỬ LÝ
    with st.expander("🔧 View Processed Data", expanded=False):
        st.subheader("After Preprocessing")
        st.markdown("""
        **Processing steps:**
        1. Merge train + test → 72 samples
        2. Z-score normalization (StandardScaler)
        3. Feature selection: Top 100 genes by variance
        """)
        
        st.write("**Processed data (first 5 rows)**")
        st.dataframe(df.head(5), use_container_width=True)
        st.caption(f"Shape: {df.shape}")
        
        # Trực quan hóa dữ liệu processed
        st.subheader("📈 2D Visualization (SVD)")
        
        X_2d_preview = get_svd_projection(X, n_components=2)
        
        fig_preview, ax = plt.subplots(figsize=(10, 6))
        
        if y_true is not None:
            for label_val, label_name, color in [(0, 'ALL', '#3498db'), (1, 'AML', '#e74c3c')]:
                mask = y_true == label_val
                ax.scatter(X_2d_preview[mask, 0], X_2d_preview[mask, 1], 
                          c=color, s=80, alpha=0.7, label=label_name, edgecolors='white', linewidth=0.5)
        else:
            ax.scatter(X_2d_preview[:, 0], X_2d_preview[:, 1], 
                      c='#7f8c8d', s=80, alpha=0.7, edgecolors='white', linewidth=0.5)
        
        ax.set_xlabel('SVD Component 1', fontsize=11)
        ax.set_ylabel('SVD Component 2', fontsize=11)
        ax.set_title('Processed Data - 2D Projection', fontsize=13, fontweight='500')
        ax.grid(True, alpha=0.2)
        if y_true is not None:
            ax.legend(loc='best', frameon=False)
        
        plt.tight_layout()
        st.pyplot(fig_preview)
    
    st.markdown("")
    
    # Run clustering
    if st.button("Run Clustering", type="primary"):
        with st.spinner("Running clustering algorithms..."):
            labels_dict, results, C_matrix = run_clustering_pipeline(X, weights, threshold)
        
        # Display results
        st.success("Completed!")
        display_results(X, labels_dict, y_true, weights, C_matrix)


def main():
    # Header - Green theme
    st.title("🧬 Gene Expression Clustering")
    st.caption("Ensemble clustering for ALL/AML cancer classification")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    st.sidebar.markdown("")
    
    # Ensemble weights
    st.sidebar.subheader("Ensemble Weights")
    w_kmeans = st.sidebar.slider("K-Means++", 0.0, 1.0, 0.30, 0.05)
    w_hier = st.sidebar.slider("Hierarchical", 0.0, 1.0, 0.35, 0.05)
    w_gmm = st.sidebar.slider("GMM", 0.0, 1.0, 0.35, 0.05)
    
    # Normalize weights
    total = w_kmeans + w_hier + w_gmm
    if total > 0:
        weights = {
            'kmeans': w_kmeans / total,
            'hierarchical': w_hier / total,
            'gmm': w_gmm / total
        }
    else:
        weights = {'kmeans': 0.33, 'hierarchical': 0.33, 'gmm': 0.34}
    
    # Display normalized weights
    st.sidebar.caption(f"""
    **Normalized:**  
    KM: {weights['kmeans']:.2f} | HC: {weights['hierarchical']:.2f} | GMM: {weights['gmm']:.2f}
    """)
    
    # Threshold
    st.sidebar.markdown("")
    threshold = st.sidebar.slider("Consensus Threshold", 0.5, 1.0, 0.70, 0.05)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Dataset Info")
    st.sidebar.markdown("""
    **Samples:** 72  
    (38 ALL, 34 AML)
    
    **Features:** 100 genes  
    (selected by variance)
    
    **Source:**  
    Golub et al. (1999)
    """)
    
    # Run demo mode (only mode available)
    run_demo_mode(weights, threshold)


if __name__ == "__main__":
    main()
