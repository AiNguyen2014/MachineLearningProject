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

# Import custom utilities
from utils.preprocessing import prepare_processed_data, get_svd_projection
from utils.clustering import run_all_clustering
from utils.ensemble import ensemble_clustering, get_cluster_distribution
from utils.visualization import (
    calculate_all_metrics,
    plot_clusters_interactive,
    plot_comparison_grid,
    plot_metrics_comparison,
    plot_confusion_matrix,
    create_metrics_table,
    plot_radar_chart
)

# Page configuration
st.set_page_config(
    page_title="🧬 Gene Expression Clustering",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2C3E50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #34495E;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F8F9FA;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #3498DB;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load processed data"""
    data_path = Path("data_processed_72.csv")
    if not data_path.exists():
        st.error(f"Không tìm thấy file: {data_path}")
        st.stop()
    
    X, df = prepare_processed_data(str(data_path))
    return X, df


@st.cache_data
def load_actual_labels():
    """Load actual cancer labels"""
    actual_path = Path("actual.csv")
    if not actual_path.exists():
        return None
    
    df_actual = pd.read_csv(actual_path)
    y_true = df_actual["cancer"].map({"ALL": 0, "AML": 1}).values
    return y_true


@st.cache_data
def run_clustering_pipeline(X, weights, threshold):
    """Run full clustering pipeline"""
    # Run all clustering algorithms
    results = run_all_clustering(X, n_clusters=2)
    
    # Extract labels
    labels_dict = {
        'kmeans': results['kmeans']['labels'],
        'hierarchical': results['hierarchical']['labels'],
        'gmm': results['gmm']['labels']
    }
    
    # Run ensemble
    ensemble_labels, C_matrix = ensemble_clustering(
        labels_dict,
        weights=weights,
        threshold=threshold
    )
    
    # Add ensemble to results
    labels_dict['ensemble'] = ensemble_labels
    
    return labels_dict, results, C_matrix


def main():
    # Header
    st.markdown('<p class="main-header">🧬 Gene Expression Ensemble Clustering</p>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">ALL/AML Cancer Classification using K-Means++, Hierarchical, and GMM</p>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Cấu hình")
    st.sidebar.markdown("---")
    
    # Ensemble weights
    st.sidebar.subheader("Trọng số Ensemble")
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
    st.sidebar.info(f"""
    **Trọng số chuẩn hóa:**
    - K-Means++: {weights['kmeans']:.2f}
    - Hierarchical: {weights['hierarchical']:.2f}
    - GMM: {weights['gmm']:.2f}
    """)
    
    # Threshold
    threshold = st.sidebar.slider("Consensus Threshold", 0.5, 1.0, 0.70, 0.05)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Về Dataset")
    st.sidebar.info("""
    - **Samples:** 72 (38 ALL, 34 AML)
    - **Features:** 100 genes (selected)
    - **Source:** Golub et al. (1999)
    """)
    
    # Load data
    with st.spinner("Đang tải dữ liệu..."):
        X, df = load_data()
        y_true = load_actual_labels()
    
    # Display data info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Số mẫu", X.shape[0])
    with col2:
        st.metric("🧬 Số genes", X.shape[1])
    with col3:
        if y_true is not None:
            st.metric("✅ Labels có sẵn", "Yes")
        else:
            st.metric("✅ Labels có sẵn", "No")
    
    st.markdown("---")
    
    # Run clustering
    with st.spinner("Đang chạy clustering algorithms..."):
        labels_dict, results, C_matrix = run_clustering_pipeline(X, weights, threshold)
        X_2d = get_svd_projection(X, n_components=2)
    
    # Calculate metrics for all methods
    metrics_dict = {}
    for name, labels in labels_dict.items():
        metrics = calculate_all_metrics(X, labels, y_true)
        metrics_dict[name] = metrics
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Clustering Results",
        "📈 Metrics Comparison", 
        "💬 Nhận xét",
        "� Co-association Matrix",
        "ℹ️ About"
    ])
    
    # TAB 1: Clustering Results
    with tab1:
        st.header("Kết quả Clustering")
        
        # Plot giống như trong Colab - 6 scatter plots riêng lẻ
        from matplotlib.lines import Line2D
        
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
    
    # TAB 3: Confusion Matrix (bỏ tab này vì không có trong Colab)
    with tab3:
        st.header("Nhận xét kết quả")
        
        if y_true is not None:
            st.markdown(f"""
            ### NHẬN XÉT KẾT QUẢ
            
            **1. ENSEMBLE kết hợp 3 thuật toán với trọng số:**
            - K-Means++: {weights['kmeans']:.2f} ({weights['kmeans']*100:.0f}%)
            - Hierarchical: {weights['hierarchical']:.2f} ({weights['hierarchical']*100:.0f}%)
            - GMM: {weights['gmm']:.2f} ({weights['gmm']*100:.0f}%)
            
            **2. Kết quả cho thấy:**
            - Ensemble đạt Silhouette = {metrics_dict['ensemble']['silhouette']:.4f}
            - Ensemble đạt ARI = {metrics_dict['ensemble']['ari']:.4f} (so với nhãn thực ALL/AML)
            - Ensemble đạt Purity = {metrics_dict['ensemble']['purity']:.4f}
            
            **3. So sánh với từng model đơn lẻ:**
            - K-Means++: ARI = {metrics_dict['kmeans']['ari']:.4f}
            - Hierarchical: ARI = {metrics_dict['hierarchical']['ari']:.4f}
            - GMM: ARI = {metrics_dict['gmm']['ari']:.4f}
            """)
            
            # Show detailed metrics table
            st.subheader("Chi tiết đầy đủ")
            st.dataframe(df_metrics, use_container_width=True)
        else:
            st.warning("Không có ground truth labels.")
    
    # TAB 4: Detailed Analysis (giữ Co-association matrix từ ensemble)
    with tab4:
        st.header("Phân tích chi tiết")
        
        # Co-association matrix (có trong ensemble logic)
        st.subheader("Co-association Matrix")
        fig_coassoc, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(C_matrix, cmap='YlOrRd', cbar=True, square=True, ax=ax)
        ax.set_title("Weighted Co-association Matrix", fontsize=14, fontweight='bold')
        st.pyplot(fig_coassoc)
        
        st.markdown("""
        ### Giải thích Co-association Matrix
        - Mỗi ô (i,j) thể hiện mức độ đồng thuận các mẫu i và j nằm cùng cụm
        - Giá trị cao (đỏ) = các thuật toán đồng ý rằng 2 mẫu cùng cụm
        - Ma trận này được tính bằng trọng số từ 3 thuật toán
        """)
    
    # TAB 5: About
    with tab5:
        st.header("Giới thiệu")
        
        st.markdown("""
        ## 🧬 Gene Expression Clustering App
        
        ### Mục đích
        Ứng dụng này thực hiện **Ensemble Clustering** trên dữ liệu gene expression 
        để phân loại ung thư ALL/AML.
        
        ### Thuật toán sử dụng
        1. **K-Means++**: Improved initialization for K-Means
        2. **Hierarchical Clustering**: Agglomerative với single linkage
        3. **GMM**: Gaussian Mixture Model với diagonal covariance
        4. **Ensemble**: Weighted co-association matrix + consensus clustering
        
        ### Metrics đánh giá
        - **Silhouette Score**: Đánh giá internal quality (-1 to 1, càng cao càng tốt)
        - **ARI (Adjusted Rand Index)**: So sánh với ground truth (0 to 1)
        - **NMI (Normalized Mutual Information)**: Mutual information chuẩn hóa (0 to 1)
        - **Purity**: Tỷ lệ mẫu được gán đúng cluster (0 to 1)
        
        ### Dataset
        - **Source**: Golub et al. (1999) - "Molecular Classification of Cancer"
        - **Samples**: 72 (38 ALL + 34 AML)
        - **Features**: 100 genes được chọn lọc
        
        ### Tác giả
        - **Name**: Trang Tran
        - **Project**: Machine Learning - Gene Expression Analysis
        - **Year**: 2026
        
        ---
        
        ### 📚 References
        - Golub et al. (1999). Molecular classification of cancer: class discovery and class prediction by gene expression monitoring
        - Fred & Jain (2005). Combining multiple clusterings using evidence accumulation
        """)
        
        st.success("✅ App đã sẵn sàng để deploy lên Streamlit Cloud!")


if __name__ == "__main__":
    main()
