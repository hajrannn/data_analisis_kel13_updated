"""
====================================================================
DASHBOARD ANALISIS & PREDIKSI PENDIDIKAN INDONESIA
Streamlit Web Application
====================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime

# ====================================================================
# PAGE CONFIGURATION
# ====================================================================

st.set_page_config(
    page_title="Dashboard Pendidikan Indonesia",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .insight-box {
        background-color: #e8f4f8;
        border-left: 5px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.3rem;
    }
    </style>
""", unsafe_allow_html=True)

# ====================================================================
# LOAD DATA FUNCTION
# ====================================================================

@st.cache_data
def load_all_data():
    """Load semua data yang sudah diproses"""
    try:
        # Load processed education data
        df_main = pd.read_csv('processed_education_data.csv')
        df_main['Jenjang_Pendidikan'] = pd.Categorical(
            df_main['Jenjang_Pendidikan'],
            categories=['SD', 'SMP', 'SMA'],
            ordered=True
        )
        
        # Load trend analysis
        df_trends = pd.read_csv('yearly_trends_analysis.csv')
        
        # Load rankings
        df_rankings = pd.read_csv('provincial_rankings.csv')
        
        # Load predictions
        df_predictions = pd.read_csv('predictions_2030.csv')
        
        # Load summary report
        with open('analysis_summary_report.json', 'r') as f:
            summary_report = json.load(f)
        
        return df_main, df_trends, df_rankings, df_predictions, summary_report
    
    except FileNotFoundError as e:
        st.error(f"❌ File tidak ditemukan: {e}")
        st.info("💡 Pastikan Anda sudah menjalankan preprocessing dan analisis terlebih dahulu!")
        st.stop()

# Load data
df_main, df_trends, df_rankings, df_predictions, summary_report = load_all_data()

# ====================================================================
# SIDEBAR - FILTERS & NAVIGATION
# ====================================================================

st.sidebar.markdown("# 🎛️ Dashboard Controls")
st.sidebar.markdown("---")

# Navigation
page = st.sidebar.radio(
    "📍 Navigasi Halaman",
    ["🏠 Overview", "📈 Analisis Trend", "🏆 Ranking Provinsi", "🔮 Prediksi 2030", "📊 Perbandingan Detail"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔍 Filter Data")

# Filter Jenjang Pendidikan
selected_jenjang = st.sidebar.multiselect(
    "Jenjang Pendidikan",
    options=['SD', 'SMP', 'SMA'],
    default=['SD', 'SMP', 'SMA']
)

# Filter Provinsi
all_provinces = sorted(df_main['Provinsi'].unique())
selected_provinces = st.sidebar.multiselect(
    "Pilih Provinsi",
    options=all_provinces,
    default=all_provinces[:5]  # Default 5 provinsi pertama
)

# Filter Tahun
year_range = st.sidebar.slider(
    "Range Tahun",
    min_value=int(df_main['Tahun'].min()),
    max_value=int(df_main['Tahun'].max()),
    value=(int(df_main['Tahun'].min()), int(df_main['Tahun'].max()))
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ Informasi Dataset")
st.sidebar.info(f"""
**Periode Data**: 2015-2023  
**Jumlah Provinsi**: {df_main['Provinsi'].nunique()}  
**Jenjang**: SD, SMP, SMA  
**Total Records**: {len(df_main):,}
""")

# ====================================================================
# HELPER FUNCTIONS
# ====================================================================

def create_line_chart(data, x_col, y_col, color_col, title):
    """Create interactive line chart"""
    fig = px.line(
        data,
        x=x_col,
        y=y_col,
        color=color_col,
        markers=True,
        title=title,
        template="plotly_white"
    )
    fig.update_layout(
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def create_bar_chart(data, x_col, y_col, title, orientation='v'):
    """Create interactive bar chart"""
    fig = px.bar(
        data,
        x=x_col if orientation == 'v' else y_col,
        y=y_col if orientation == 'v' else x_col,
        title=title,
        template="plotly_white",
        orientation=orientation
    )
    return fig

def create_heatmap(data, x_col, y_col, value_col, title):
    """Create heatmap"""
    pivot_data = data.pivot_table(
        values=value_col,
        index=y_col,
        columns=x_col,
        aggfunc='mean'
    )
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=pivot_data.columns,
        y=pivot_data.index,
        colorscale='RdYlGn',
        text=np.round(pivot_data.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Nilai (%)")
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title=x_col,
        yaxis_title=y_col,
        template="plotly_white"
    )
    
    return fig

# ====================================================================
# PAGE 1: OVERVIEW
# ====================================================================

if page == "🏠 Overview":
    st.markdown('<p class="main-header">📚 Dashboard Analisis Pendidikan Indonesia</p>', unsafe_allow_html=True)
    st.markdown("### Analisis Tingkat Penyelesaian Pendidikan 2015-2023 & Prediksi 2030")
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📊 Total Provinsi",
            value=df_main['Provinsi'].nunique()
        )
    
    with col2:
        avg_completion_2023 = df_main[df_main['Tahun'] == 2023]['Nilai'].mean()
        st.metric(
            label="📈 Rata-rata Nasional 2023",
            value=f"{avg_completion_2023:.2f}%"
        )
    
    with col3:
        avg_growth = df_predictions['Avg_Annual_Growth'].mean()
        st.metric(
            label="🚀 Rata-rata Pertumbuhan/Tahun",
            value=f"{avg_growth:+.2f}%"
        )
    
    with col4:
        avg_prediction_2030 = df_predictions['Prediction_2030_Final'].mean()
        st.metric(
            label="🔮 Prediksi Rata-rata 2030",
            value=f"{avg_prediction_2030:.2f}%"
        )
    
    st.markdown("---")
    
    # Trend Overview
    st.markdown('<p class="sub-header">📊 Trend Nasional 2015-2023</p>', unsafe_allow_html=True)
    
    # Aggregate data per year and education level
    trend_overview = df_main.groupby(['Tahun', 'Jenjang_Pendidikan'])['Nilai'].mean().reset_index()
    
    fig_overview = create_line_chart(
        trend_overview,
        'Tahun',
        'Nilai',
        'Jenjang_Pendidikan',
        'Rata-rata Tingkat Penyelesaian Pendidikan per Jenjang (2015-2023)'
    )
    st.plotly_chart(fig_overview, use_container_width=True)
    
    # Key Insights
    st.markdown('<p class="sub-header">💡 Key Insights</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("#### 🎯 Trend Direction")
        
        total_naik = len(df_trends[df_trends['Status_Trend'] == 'Naik'])
        total_transisi = len(df_trends)
        pct_naik = (total_naik / total_transisi) * 100
        
        st.write(f"**{pct_naik:.1f}%** dari transisi menunjukkan kenaikan")
        st.write(f"Dari {total_transisi:,} transisi yang dianalisis:")
        st.write(f"- ✅ Naik: {total_naik:,}")
        st.write(f"- ❌ Turun: {len(df_trends[df_trends['Status_Trend'] == 'Turun']):,}")
        st.write(f"- ➖ Stabil: {len(df_trends[df_trends['Status_Trend'] == 'Stabil']):,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("#### 🏆 Top 5 Provinsi Terbaik 2023")
        
        top_provinces_2023 = df_rankings[df_rankings['Tahun'] == 2023].groupby('Provinsi').agg({
            'Ranking': 'mean',
            'Nilai': 'mean'
        }).sort_values('Ranking').head(5)
        
        for i, (prov, data) in enumerate(top_provinces_2023.iterrows(), 1):
            st.write(f"{i}. **{prov}** - Avg: {data['Nilai']:.2f}%")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Distribution by Education Level
    st.markdown('<p class="sub-header">📚 Distribusi per Jenjang Pendidikan (2023)</p>', unsafe_allow_html=True)
    
    data_2023 = df_main[df_main['Tahun'] == 2023]
    
    fig_dist = go.Figure()
    
    for jenjang in ['SD', 'SMP', 'SMA']:
        jenjang_data = data_2023[data_2023['Jenjang_Pendidikan'] == jenjang]['Nilai']
        fig_dist.add_trace(go.Box(
            y=jenjang_data,
            name=jenjang,
            boxmean='sd'
        ))
    
    fig_dist.update_layout(
        title="Distribusi Nilai per Jenjang Pendidikan (2023)",
        yaxis_title="Nilai (%)",
        template="plotly_white"
    )
    
    st.plotly_chart(fig_dist, use_container_width=True)

# ====================================================================
# PAGE 2: ANALISIS TREND
# ====================================================================

elif page == "📈 Analisis Trend":
    st.markdown('<p class="main-header">📈 Analisis Trend Tahunan</p>', unsafe_allow_html=True)
    
    # Filter data based on selection
    filtered_data = df_main[
        (df_main['Jenjang_Pendidikan'].isin(selected_jenjang)) &
        (df_main['Provinsi'].isin(selected_provinces)) &
        (df_main['Tahun'] >= year_range[0]) &
        (df_main['Tahun'] <= year_range[1])
    ]
    
    if len(filtered_data) == 0:
        st.warning("⚠️ Tidak ada data untuk filter yang dipilih. Silakan ubah filter.")
    else:
        # Trend per Provinsi
        st.markdown("### 📊 Trend per Provinsi & Jenjang")
        
        fig_trends = create_line_chart(
            filtered_data,
            'Tahun',
            'Nilai',
            'Provinsi',
            f'Trend Tingkat Penyelesaian Pendidikan'
        )
        
        # Add facet for education level
        fig_trends = px.line(
            filtered_data,
            x='Tahun',
            y='Nilai',
            color='Provinsi',
            facet_col='Jenjang_Pendidikan',
            markers=True,
            title='Trend per Jenjang Pendidikan',
            template="plotly_white"
        )
        
        st.plotly_chart(fig_trends, use_container_width=True)
        
        # Year-over-Year Growth Analysis
        st.markdown("### 📊 Analisis Pertumbuhan Year-over-Year")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Filter trends data
            filtered_trends = df_trends[
                (df_trends['Jenjang_Pendidikan'].isin(selected_jenjang)) &
                (df_trends['Provinsi'].isin(selected_provinces))
            ]
            
            # Average growth by province
            avg_growth_by_prov = filtered_trends.groupby('Provinsi')['Perubahan_Absolut'].mean().sort_values(ascending=False)
            
            fig_growth = go.Figure(go.Bar(
                x=avg_growth_by_prov.values,
                y=avg_growth_by_prov.index,
                orientation='h',
                marker=dict(
                    color=avg_growth_by_prov.values,
                    colorscale='RdYlGn',
                    showscale=True
                )
            ))
            
            fig_growth.update_layout(
                title="Rata-rata Pertumbuhan per Provinsi",
                xaxis_title="Perubahan Rata-rata (poin %)",
                yaxis_title="Provinsi",
                template="plotly_white",
                height=400
            )
            
            st.plotly_chart(fig_growth, use_container_width=True)
        
        with col2:
            # Trend status distribution
            trend_status_count = filtered_trends['Status_Trend'].value_counts()
            
            fig_status = go.Figure(data=[go.Pie(
                labels=trend_status_count.index,
                values=trend_status_count.values,
                hole=0.4,
                marker=dict(colors=['#2ecc71', '#e74c3c', '#95a5a6'])
            )])
            
            fig_status.update_layout(
                title="Distribusi Status Trend",
                template="plotly_white",
                height=400
            )
            
            st.plotly_chart(fig_status, use_container_width=True)
        
        # Detailed Trend Table
        st.markdown("### 📋 Detail Perubahan Tahunan")
        
        # Show recent trends
        recent_trends = filtered_trends[
            filtered_trends['Tahun_Ke'] == filtered_trends['Tahun_Ke'].max()
        ].sort_values('Perubahan_Absolut', ascending=False)
        
        st.dataframe(
            recent_trends[['Provinsi', 'Jenjang_Pendidikan', 'Tahun_Dari', 'Tahun_Ke', 
                          'Nilai_Awal', 'Nilai_Akhir', 'Perubahan_Absolut', 'Status_Trend']],
            use_container_width=True
        )

# ====================================================================
# PAGE 3: RANKING PROVINSI
# ====================================================================

elif page == "🏆 Ranking Provinsi":
    st.markdown('<p class="main-header">🏆 Ranking Provinsi</p>', unsafe_allow_html=True)
    
    # Year selection for ranking
    selected_year_ranking = st.selectbox(
        "Pilih Tahun untuk Ranking",
        options=sorted(df_rankings['Tahun'].unique(), reverse=True),
        index=0
    )
    
    # Filter rankings
    filtered_rankings = df_rankings[
        (df_rankings['Tahun'] == selected_year_ranking) &
        (df_rankings['Jenjang_Pendidikan'].isin(selected_jenjang))
    ]
    
    # Tabs for different education levels
    tabs = st.tabs(['📊 Semua Jenjang'] + [f"📚 {j}" for j in selected_jenjang])
    
    with tabs[0]:
        st.markdown(f"### Ranking Provinsi {selected_year_ranking} - Semua Jenjang")
        
        # Average ranking across all education levels
        avg_ranking = filtered_rankings.groupby('Provinsi').agg({
            'Ranking': 'mean',
            'Nilai': 'mean'
        }).sort_values('Ranking').reset_index()
        
        avg_ranking['Ranking_Overall'] = range(1, len(avg_ranking) + 1)
        
        # Create ranking visualization
        fig_overall = go.Figure()
        
        fig_overall.add_trace(go.Bar(
            x=avg_ranking['Provinsi'],
            y=avg_ranking['Nilai'],
            marker=dict(
                color=avg_ranking['Nilai'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Nilai (%)")
            ),
            text=avg_ranking['Nilai'].round(2),
            textposition='outside'
        ))
        
        fig_overall.update_layout(
            title=f"Ranking Provinsi Berdasarkan Rata-rata Nilai ({selected_year_ranking})",
            xaxis_title="Provinsi",
            yaxis_title="Nilai Rata-rata (%)",
            template="plotly_white",
            height=600,
            xaxis={'tickangle': -45}
        )
        
        st.plotly_chart(fig_overall, use_container_width=True)
        
        # Top 10 Table
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🥇 Top 10 Provinsi Terbaik")
            top_10 = avg_ranking.head(10)[['Ranking_Overall', 'Provinsi', 'Nilai']]
            st.dataframe(top_10, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("#### 🔻 Bottom 10 Provinsi")
            bottom_10 = avg_ranking.tail(10)[['Ranking_Overall', 'Provinsi', 'Nilai']]
            st.dataframe(bottom_10, use_container_width=True, hide_index=True)
    
    # Individual education level tabs
    for i, jenjang in enumerate(selected_jenjang, 1):
        with tabs[i]:
            st.markdown(f"### Ranking {jenjang} - {selected_year_ranking}")
            
            jenjang_ranking = filtered_rankings[
                filtered_rankings['Jenjang_Pendidikan'] == jenjang
            ].sort_values('Ranking')
            
            # Bar chart
            fig_jenjang = create_bar_chart(
                jenjang_ranking.head(20),
                'Provinsi',
                'Nilai',
                f'Top 20 Provinsi - {jenjang} ({selected_year_ranking})'
            )
            fig_jenjang.update_xaxes(tickangle=-45)
            
            st.plotly_chart(fig_jenjang, use_container_width=True)
            
            # Full ranking table
            st.markdown("#### 📋 Ranking Lengkap")
            st.dataframe(
                jenjang_ranking[['Ranking', 'Provinsi', 'Nilai', 'Persentil']],
                use_container_width=True,
                hide_index=True
            )
    
    # Ranking Evolution (Heatmap)
    st.markdown("### 📊 Evolusi Ranking dari Waktu ke Waktu")
    
    selected_jenjang_heatmap = st.selectbox(
        "Pilih Jenjang untuk Heatmap",
        options=selected_jenjang
    )
    
    heatmap_data = df_rankings[
        (df_rankings['Jenjang_Pendidikan'] == selected_jenjang_heatmap) &
        (df_rankings['Provinsi'].isin(selected_provinces if selected_provinces else all_provinces[:15]))
    ]
    
    fig_heatmap = create_heatmap(
        heatmap_data,
        'Tahun',
        'Provinsi',
        'Nilai',
        f'Heatmap Nilai {selected_jenjang_heatmap} per Provinsi & Tahun'
    )
    
    st.plotly_chart(fig_heatmap, use_container_width=True)

# ====================================================================
# PAGE 4: PREDIKSI 2030
# ====================================================================

elif page == "🔮 Prediksi 2030":
    st.markdown('<p class="main-header">🔮 Prediksi Tingkat Penyelesaian 2030</p>', unsafe_allow_html=True)
    
    st.info("💡 Prediksi menggunakan kombinasi rata-rata pertumbuhan historis dan linear regression")
    
    # Filter predictions
    filtered_predictions = df_predictions[
        (df_predictions['Jenjang_Pendidikan'].isin(selected_jenjang)) &
        (df_predictions['Provinsi'].isin(selected_provinces if selected_provinces else all_provinces))
    ]
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_current = filtered_predictions['Nilai_2023'].mean()
        st.metric(
            "📊 Rata-rata Nilai 2023",
            f"{avg_current:.2f}%"
        )
    
    with col2:
        avg_pred = filtered_predictions['Prediction_2030_Final'].mean()
        st.metric(
            "🔮 Prediksi Rata-rata 2030",
            f"{avg_pred:.2f}%"
        )
    
    with col3:
        avg_expected_growth = filtered_predictions['Expected_Growth_2023_2030'].mean()
        st.metric(
            "📈 Pertumbuhan yang Diharapkan",
            f"{avg_expected_growth:+.2f} poin",
            delta=f"{avg_expected_growth:+.2f}"
        )
    
    st.markdown("---")
    
    # Prediction Comparison
    st.markdown("### 📊 Perbandingan Nilai 2023 vs Prediksi 2030")
    
    # Select education level for detailed view
    selected_jenjang_pred = st.selectbox(
        "Pilih Jenjang Pendidikan",
        options=selected_jenjang
    )
    
    jenjang_pred_data = filtered_predictions[
        filtered_predictions['Jenjang_Pendidikan'] == selected_jenjang_pred
    ].sort_values('Prediction_2030_Final', ascending=False)
    
    # Create comparison chart
    fig_comparison = go.Figure()
    
    fig_comparison.add_trace(go.Bar(
        name='Nilai 2023',
        x=jenjang_pred_data['Provinsi'],
        y=jenjang_pred_data['Nilai_2023'],
        marker_color='lightblue'
    ))
    
    fig_comparison.add_trace(go.Bar(
        name='Prediksi 2030',
        x=jenjang_pred_data['Provinsi'],
        y=jenjang_pred_data['Prediction_2030_Final'],
        marker_color='darkblue'
    ))
    
    fig_comparison.update_layout(
        title=f'Perbandingan Nilai 2023 vs Prediksi 2030 - {selected_jenjang_pred}',
        xaxis_title='Provinsi',
        yaxis_title='Nilai (%)',
        barmode='group',
        template='plotly_white',
        height=600,
        xaxis={'tickangle': -45}
    )
    
    st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Confidence Level Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Distribusi Confidence Level")
        
        confidence_dist = filtered_predictions['Confidence_Level'].value_counts()
        
        fig_confidence = go.Figure(data=[go.Pie(
            labels=confidence_dist.index,
            values=confidence_dist.values,
            hole=0.4
        )])
        
        fig_confidence.update_layout(
            title="Confidence Level Prediksi",
            template="plotly_white"
        )
        
        st.plotly_chart(fig_confidence, use_container_width=True)
    
    with col2:
        st.markdown("### 📈 Rata-rata Pertumbuhan Tahunan")
        
        growth_by_jenjang = filtered_predictions.groupby('Jenjang_Pendidikan')['Avg_Annual_Growth'].mean().sort_values(ascending=False)
        
        fig_growth = go.Figure(go.Bar(
            x=growth_by_jenjang.index,
            y=growth_by_jenjang.values,
            marker=dict(
                color=growth_by_jenjang.values,
                colorscale='RdYlGn',
                showscale=False
            ),
            text=growth_by_jenjang.values.round(3),
            textposition='outside'
        ))
        
        fig_growth.update_layout(
            title="Rata-rata Pertumbuhan per Jenjang",
            xaxis_title="Jenjang Pendidikan",
            yaxis_title="Pertumbuhan Tahunan (%)",
            template="plotly_white"
        )
        
        st.plotly_chart(fig_growth, use_container_width=True)
    
    # Top Predictions
    st.markdown("### 🏆 Top 10 Prediksi Tertinggi 2030")
    
    top_predictions = filtered_predictions.nlargest(10, 'Prediction_2030_Final')[
        ['Provinsi', 'Jenjang_Pendidikan', 'Nilai_2023', 'Prediction_2030_Final', 
         'Expected_Growth_2023_2030', 'Confidence_Level']
    ]
    
    st.dataframe(top_predictions, use_container_width=True, hide_index=True)
    
    # Provinces needing attention
    st.markdown("### ⚠️ Provinsi yang Memerlukan Perhatian")
    
    attention_needed = filtered_predictions[
        filtered_predictions['Expected_Growth_2023_2030'] < 1
    ].sort_values('Expected_Growth_2023_2030')
    
    if len(attention_needed) > 0:
        st.dataframe(
            attention_needed[['Provinsi', 'Jenjang_Pendidikan', 'Nilai_2023', 
                             'Prediction_2030_Final', 'Expected_Growth_2023_2030']].head(10),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.success("✅ Semua provinsi menunjukkan pertumbuhan positif!")
    
    # Detailed Prediction Table
    st.markdown("### 📋 Tabel Prediksi Lengkap")
    
    st.dataframe(
        filtered_predictions[['Provinsi', 'Jenjang_Pendidikan', 'Nilai_2015', 'Nilai_2023',
                             'Prediction_2030_Final', 'Avg_Annual_Growth', 'Confidence_Level']],
        use_container_width=True,
        hide_index=True
    )

# ====================================================================
# PAGE 5: PERBANDINGAN DETAIL
# ====================================================================

elif page == "📊 Perbandingan Detail":
    st.markdown('<p class="main-header">📊 Perbandingan Detail Antar Provinsi</p>', unsafe_allow_html=True)
    
    # Province comparison
    st.markdown("### 🔍 Pilih Provinsi untuk Perbandingan Detail")
    
    col1, col2 = st.columns(2)
    
    with col1:
        compare_province_1 = st.selectbox(
            "Provinsi 1",
            options=all_provinces,
            index=0
        )
    
    with col2:
        compare_province_2 = st.selectbox(
            "Provinsi 2",
            options=all_provinces,
            index=1 if len(all_provinces) > 1 else 0
        )
    
    if compare_province_1 == compare_province_2:
        st.warning("⚠️ Pilih provinsi yang berbeda untuk perbandingan")
    else:
        # Filter data for comparison
        compare_data = df_main[
            (df_main['Provinsi'].isin([compare_province_1, compare_province_2])) &
            (df_main['Jenjang_Pendidikan'].isin(selected_jenjang))
        ]
        
        # Line chart comparison
        st.markdown(f"### 📈 Trend Perbandingan: {compare_province_1} vs {compare_province_2}")
        
        fig_compare = px.line(
            compare_data,
            x='Tahun',
            y='Nilai',
            color='Provinsi',
            facet_col='Jenjang_Pendidikan',
            markers=True,
            title=f'Perbandingan Trend {compare_province_1} vs {compare_province_2}',
            template='plotly_white'
        )
        
        fig_compare.update_yaxes(matches=None)
        st.plotly_chart(fig_compare, use_container_width=True)
        
        # Statistics comparison
        st.markdown("### 📊 Statistik Perbandingan")
        
        stats_compare = compare_data.groupby(['Provinsi', 'Jenjang_Pendidikan']).agg({
            'Nilai': ['mean', 'min', 'max', 'std'],
            'Growth_YoY_Pct': 'mean'
        }).round(2)
        
        stats_compare.columns = ['Rata-rata', 'Min', 'Max', 'Std Dev', 'Avg Growth (%)']
        st.dataframe(stats_compare, use_container_width=True)
        
        # Prediction comparison
        st.markdown("### 🔮 Perbandingan Prediksi 2030")
        
        pred_compare = df_predictions[
            df_predictions['Provinsi'].isin([compare_province_1, compare_province_2])
        ]
        
        fig_pred_compare = go.Figure()
        
        for prov in [compare_province_1, compare_province_2]:
            prov_data = pred_compare[pred_compare['Provinsi'] == prov]
            
            fig_pred_compare.add_trace(go.Bar(
                name=f'{prov} - 2023',
                x=prov_data['Jenjang_Pendidikan'],
                y=prov_data['Nilai_2023'],
                marker_pattern_shape=".",
            ))
            
            fig_pred_compare.add_trace(go.Bar(
                name=f'{prov} - 2030 (Prediksi)',
                x=prov_data['Jenjang_Pendidikan'],
                y=prov_data['Prediction_2030_Final'],
            ))
        
        fig_pred_compare.update_layout(
            title=f'Perbandingan Nilai 2023 & Prediksi 2030',
            xaxis_title='Jenjang Pendidikan',
            yaxis_title='Nilai (%)',
            barmode='group',
            template='plotly_white'
        )
        
        st.plotly_chart(fig_pred_compare, use_container_width=True)
        
        # Gap analysis
        st.markdown("### 📏 Analisis Gap")
        
        for jenjang in selected_jenjang:
            st.markdown(f"#### {jenjang}")
            
            data_j = compare_data[compare_data['Jenjang_Pendidikan'] == jenjang]
            
            if len(data_j) > 0:
                prov1_latest = data_j[data_j['Provinsi'] == compare_province_1]['Nilai'].iloc[-1]
                prov2_latest = data_j[data_j['Provinsi'] == compare_province_2]['Nilai'].iloc[-1]
                
                gap = prov1_latest - prov2_latest
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(compare_province_1, f"{prov1_latest:.2f}%")
                
                with col2:
                    st.metric(compare_province_2, f"{prov2_latest:.2f}%")
                
                with col3:
                    st.metric("Gap", f"{abs(gap):.2f} poin", 
                             delta=f"{gap:+.2f}" if gap > 0 else f"{gap:.2f}")
        
        # Multi-province comparison
        st.markdown("---")
        st.markdown("### 📊 Perbandingan Multi-Provinsi")
        
        multi_provinces = st.multiselect(
            "Pilih beberapa provinsi untuk perbandingan",
            options=all_provinces,
            default=all_provinces[:5]
        )
        
        if len(multi_provinces) > 0:
            selected_year_multi = st.slider(
                "Pilih Tahun",
                min_value=int(df_main['Tahun'].min()),
                max_value=int(df_main['Tahun'].max()),
                value=int(df_main['Tahun'].max())
            )
            
            multi_data = df_main[
                (df_main['Provinsi'].isin(multi_provinces)) &
                (df_main['Tahun'] == selected_year_multi) &
                (df_main['Jenjang_Pendidikan'].isin(selected_jenjang))
            ]
            
            # Radar chart
            fig_radar = go.Figure()
            
            for prov in multi_provinces:
                prov_data = multi_data[multi_data['Provinsi'] == prov]
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=prov_data['Nilai'].values,
                    theta=prov_data['Jenjang_Pendidikan'].values,
                    fill='toself',
                    name=prov
                ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )
                ),
                showlegend=True,
                title=f"Perbandingan Multi-Provinsi ({selected_year_multi})",
                template='plotly_white'
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)
            
            # Heatmap comparison
            st.markdown("### 🗺️ Heatmap Perbandingan")
            
            heatmap_multi = multi_data.pivot_table(
                values='Nilai',
                index='Provinsi',
                columns='Jenjang_Pendidikan',
                aggfunc='mean'
            )
            
            fig_heatmap_multi = go.Figure(data=go.Heatmap(
                z=heatmap_multi.values,
                x=heatmap_multi.columns,
                y=heatmap_multi.index,
                colorscale='RdYlGn',
                text=np.round(heatmap_multi.values, 2),
                texttemplate='%{text}',
                textfont={"size": 12},
                colorbar=dict(title="Nilai (%)")
            ))
            
            fig_heatmap_multi.update_layout(
                title=f"Heatmap Nilai per Provinsi & Jenjang ({selected_year_multi})",
                xaxis_title="Jenjang Pendidikan",
                yaxis_title="Provinsi",
                template="plotly_white",
                height=400
            )
            
            st.plotly_chart(fig_heatmap_multi, use_container_width=True)

# ====================================================================
# FOOTER
# ====================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; padding: 2rem 0;'>
    <p><strong>Dashboard Analisis Pendidikan Indonesia</strong></p>
    <p>Data: Tingkat Penyelesaian Pendidikan 2015-2023 | Prediksi: 2030</p>
    <p>Dibuat oleh Kelompok 13</p>
</div>
""", unsafe_allow_html=True)