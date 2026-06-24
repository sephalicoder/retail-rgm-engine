import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Retail Pricing Intelligence",
    page_icon="🏪",
    layout="wide"
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background:#F8F9FB; }
[data-testid="stSidebar"] {
    background:#0A1628 !important;
    border-right:none;
}
[data-testid="stSidebar"] * { color:#CBD5E1 !important; }
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,[data-testid="stSidebar"] h4 {
    color:#F1F5F9 !important;
}
section[data-testid="stSidebar"] label {
    color:#94A3B8 !important;
    font-size:11px !important;
    font-weight:600;
    text-transform:uppercase;
    letter-spacing:0.06em;
}
.stSelectbox > div > div {
    background:#132039 !important;
    border:1px solid #1E3A5F !important;
    border-radius:8px !important;
}
.main .block-container { background:#F8F9FB; padding-top:1.5rem; }
[data-testid="metric-container"] {
    background:#FFFFFF;
    border:1px solid #E2E8F0;
    border-radius:12px;
    padding:1rem 1.2rem;
    box-shadow:0 1px 3px rgba(0,0,0,0.06);
}
[data-testid="metric-container"] label { color:#64748B !important; font-size:12px !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color:#0F172A !important; font-size:22px !important;
}
[data-testid="stMetricDelta"] { color:#2563EB !important; }
.stButton > button {
    background:#2563EB !important; color:#fff !important;
    border:none !important; border-radius:8px !important;
    font-weight:600 !important; width:100%;
    font-size:13px !important;
}
.stButton > button:hover { background:#1D4ED8 !important; }
hr { border-color:#E2E8F0 !important; }

/* Tab Configuration Styles */
.stTabs [data-baseweb="tab"] {
    color: #64748B !important; 
    font-size:13px !important; font-weight:600 !important;
    padding:8px 20px !important;
}
.stTabs [aria-selected="true"] { color:#2563EB !important; }

/* Force High Contrast Black Text Inside Tab Content Blocks */
.stTabs [data-baseweb="tab-panel"] {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 0 12px 12px 12px;
    padding: 20px 24px;
    color: #000000 !important;
}
.stTabs [data-baseweb="tab-panel"] p, 
.stTabs [data-baseweb="tab-panel"] li,
.stTabs [data-baseweb="tab-panel"] span,
.stTabs [data-baseweb="tab-panel"] h4 { 
    color: #000000 !important; 
}
</style>
""", unsafe_allow_html=True)

# ── Load assets ───────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    model              = joblib.load('forecast_model.pkl')
    forecast_features  = joblib.load('forecast_features.pkl')
    cluster_elasticity = joblib.load('cluster_elasticity.pkl')
    store_cluster_map  = joblib.load('store_cluster_map.pkl')
    store_features     = joblib.load('store_segments.pkl')
    cannibal_matrix    = joblib.load('cannibalization_matrix.pkl')
    opt_results        = joblib.load('optimization_results.pkl')
    return (model, forecast_features, cluster_elasticity,
            store_cluster_map, store_features, cannibal_matrix, opt_results)

@st.cache_data
def load_data():
    train    = pd.read_csv('train.csv',    parse_dates=['Date'])
    features = pd.read_csv('features.csv', parse_dates=['Date'])
    stores   = pd.read_csv('stores.csv')
    df = train.merge(features, on=['Store','Date','IsHoliday'], how='left')
    df = df.merge(stores, on='Store', how='left')
    md_cols = ['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5']
    for col in md_cols:
        df[col] = df[col].fillna(0).clip(lower=0)
    df['total_markdown'] = df[md_cols].sum(axis=1)
    return df

(model, forecast_features, cluster_elasticity,
 store_cluster_map, store_features, cannibal_matrix, opt_results) = load_assets()
df = load_data()

COST_RATE  = 0.15
BREAKEVEN  = COST_RATE

cluster_labels = {
    v['label']: k for k, v in cluster_elasticity.items()
}
cluster_label_map = {
    k: v['label'] for k, v in cluster_elasticity.items()
}
SEGMENT_COLORS = {
    'Small Low-Traffic':    '#DC2626',
    'Mid-Size Stable':      '#D97706',
    'Large Promotional':    '#059669',
    'High-Volume Premium':  '#2563EB',
}

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;padding-bottom:16px;
         border-bottom:1px solid #1E3A5F;margin-bottom:16px">
        <div style="width:36px;height:36px;background:#2563EB;border-radius:8px;
             display:flex;align-items:center;justify-content:center;font-size:18px">🏪</div>
        <div>
            <div style="font-size:14px;font-weight:700;color:#F1F5F9">Pricing Intelligence</div>
            <div style="font-size:11px;color:#64748B">Walmart RGM Engine</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("Navigate", [
        "📊 Overview",
        "🎯 Segment Elasticity",
        "💰 Optimizer",
        "🔀 Cannibalization",
        "📈 Forecast Explorer",
        "📋 Methodology"
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("#### Filters")
    selected_segment = st.selectbox(
        "Store Segment",
        ['All'] + list(SEGMENT_COLORS.keys())
    )
    selected_store = st.selectbox(
        "Store",
        ['All'] + sorted(df['Store'].unique().tolist())
    )
    selected_dept = st.selectbox(
        "Department",
        ['All'] + sorted(df['Dept'].unique().tolist())
    )

# ── Header ────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:6px">
    <span style="font-size:11px;font-weight:700;color:#2563EB;
          text-transform:uppercase;letter-spacing:0.1em">
        Revenue Growth Management
    </span>
    <h1 style="color:#0F172A;font-size:24px;font-weight:700;margin:4px 0 2px">
        Retail Markdown Optimization Engine
    </h1>
    <p style="color:#64748B;font-size:13px;margin:0">
        Walmart store sales · LightGBM forecasting · Segment-level price elasticity ·
        Constrained optimization · 421K weekly records
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

k1,k2,k3,k4,k5 = st.columns(5)
k1.metric("Total Records",      "421,570",  "45 stores · 81 depts")
k2.metric("Forecast WMAE",      "0.077",    "vs 0.10 benchmark")
k3.metric("Store Segments",     "4",        "K-means clustering")
k4.metric("Global Elasticity",  "+0.0094",  "log-log OLS")
k5.metric("Breakeven Elast.",   f">{BREAKEVEN:.2f}", "for markdown ROI")

st.divider()

# ════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.markdown("### 📊 Dataset Overview")

    col1, col2 = st.columns(2)

    with col1:
        weekly = df.groupby('Date')['Weekly_Sales'].sum().reset_index()
        fig, ax = plt.subplots(figsize=(9, 4))
        fig.patch.set_facecolor('#FFFFFF')
        ax.set_facecolor('#F8F9FB')
        ax.plot(weekly['Date'], weekly['Weekly_Sales']/1e6,
                color='#2563EB', linewidth=1.8)
        ax.fill_between(weekly['Date'], weekly['Weekly_Sales']/1e6,
                        alpha=0.08, color='#2563EB')
        holidays = df[df['IsHoliday']==True]['Date'].unique()
        for hd in holidays[:20]:
            ax.axvline(hd, color='#DC2626', alpha=0.15, linewidth=1)
        ax.set_title('Total Weekly Sales ($M) — red lines = holidays',
                     fontweight='bold', fontsize=11)
        ax.set_ylabel('Sales ($M)')
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x,_: f'${x:.0f}M'))
        for sp in ax.spines.values(): sp.set_color('#E2E8F0')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        seg_sales = df.copy()
        seg_sales['cluster'] = seg_sales['Store'].map(store_cluster_map)
        seg_sales['segment'] = seg_sales['cluster'].map(cluster_label_map)
        seg_agg = (seg_sales.groupby('segment')['Weekly_Sales']
                            .mean().reset_index()
                            .sort_values('Weekly_Sales', ascending=True))

        fig, ax = plt.subplots(figsize=(9, 4))
        fig.patch.set_facecolor('#FFFFFF')
        ax.set_facecolor('#F8F9FB')
        colors = [SEGMENT_COLORS.get(s, '#94A3B8') for s in seg_agg['segment']]
        bars = ax.barh(seg_agg['segment'], seg_agg['Weekly_Sales']/1e3,
                       color=colors, height=0.5)
        for bar, val in zip(bars, seg_agg['Weekly_Sales']):
            ax.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2,
                    f'${val/1e3:.1f}K', va='center', fontsize=10, fontweight='bold')
        ax.set_xlabel('Avg Weekly Sales ($K)')
        ax.set_title('Average Weekly Sales by Store Segment',
                     fontweight='bold', fontsize=11)
        for sp in ax.spines.values(): sp.set_color('#E2E8F0')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("#### Top 10 Departments by Average Weekly Sales")
    dept_summary = (df.groupby('Dept')['Weekly_Sales']
                      .agg(['mean','sum','std'])
                      .round(0)
                      .sort_values('mean', ascending=False)
                      .head(10)
                      .reset_index())
    dept_summary.columns = ['Dept','Avg Weekly Sales ($)','Total Sales ($)','Std Dev ($)']
    dept_summary['Avg Weekly Sales ($)'] = dept_summary['Avg Weekly Sales ($)'].apply(lambda x: f'${x:,.0f}')
    dept_summary['Total Sales ($)']      = dept_summary['Total Sales ($)'].apply(lambda x: f'${x:,.0f}')
    dept_summary['Std Dev ($)']          = dept_summary['Std Dev ($)'].apply(lambda x: f'${x:,.0f}')
    st.dataframe(dept_summary, use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════
# PAGE 2 — SEGMENT ELASTICITY
# ════════════════════════════════════════════════════════════════
elif page == "🎯 Segment Elasticity":
    st.markdown("### 🎯 Price Elasticity by Store Segment")
    st.markdown("""
    <div style="background:#EFF6FF;border:1px solid #BFDBFE;border-radius:10px;
         padding:12px 16px;margin-bottom:16px;font-size:13px;color:#1E40AF">
        <b>What this shows:</b> The same markdown spend produces different sales lifts
        across store segments. This is the foundation of segment-specific pricing strategy.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        ids    = sorted(cluster_elasticity.keys())
        labels = [cluster_elasticity[i]['label'] for i in ids]
        coefs  = [cluster_elasticity[i]['elasticity'] for i in ids]
        colors = [SEGMENT_COLORS.get(l, '#94A3B8') for l in labels]

        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor('#FFFFFF')
        ax.set_facecolor('#F8F9FB')
        bars = ax.barh(labels, coefs, color=colors, height=0.5)
        ax.axvline(BREAKEVEN, color='#DC2626', linewidth=1.5,
                   linestyle='--', label=f'Breakeven threshold ({BREAKEVEN})')
        for bar, coef in zip(bars, coefs):
            color = '#059669' if coef > BREAKEVEN else '#DC2626'
            ax.text(coef + 0.0003,
                    bar.get_y() + bar.get_height()/2,
                    f'{coef:+.4f}', va='center',
                    fontsize=10, fontweight='bold', color=color)
        ax.set_xlabel('Markdown Elasticity (log-log OLS with dept FE + holiday controls)')
        ax.set_title('Markdown Elasticity by Store Segment\n'
                     'Dashed line = breakeven threshold for markdown ROI',
                     fontweight='bold', fontsize=11, pad=10)
        ax.legend(fontsize=9)
        for sp in ax.spines.values(): sp.set_color('#E2E8F0')
        ax.grid(axis='x', color='#F1F5F9', linewidth=0.8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        for i in sorted(cluster_elasticity.keys()):
            info  = cluster_elasticity[i]
            label = info['label']
            elast = info['elasticity']
            color = SEGMENT_COLORS.get(label, '#94A3B8')
            is_profitable = elast > BREAKEVEN
            badge_color = '#D1FAE5' if is_profitable else '#FEE2E2'
            badge_text  = '#065F46' if is_profitable else '#991B1B'
            badge_label = 'ROI positive' if is_profitable else 'Below breakeven'
            
            r2_val = info.get('r2', 0.0)
            obs_raw = info.get('baseline_weekly_sales', info.get('obs', info.get('n', 'N/A')))
            obs_str = f"{obs_raw:,.0f}" if isinstance(obs_raw, (int, float)) else str(obs_raw)

            st.markdown(f"""
            <div style="border:1px solid #E2E8F0;border-radius:10px;
                 padding:12px;margin-bottom:10px;border-left:4px solid {color}">
                <div style="display:flex;justify-content:space-between;align-items:center">
                    <div style="font-size:13px;font-weight:600;color:#0F172A">{label}</div>
                    <div style="font-size:10px;font-weight:700;padding:2px 8px;
                         border-radius:20px;background:{badge_color};color:{badge_text}">
                        {badge_label}
                    </div>
                </div>
                <div style="font-size:22px;font-weight:700;color:{color};margin:6px 0 2px">
                    {elast:+.4f}
                </div>
                <div style="font-size:12px;color:#475569;font-family:monospace;">
                    elasticity · R² {r2_val:.3f} · n={obs_str}
                </div>
            </div>
            """, unsafe_allow_html=True)

    with st.expander("Identification Strategy & Limitations"):
        st.markdown("""
        **What the fixed-effects model controls for:**
        Store fixed effects absorb time-invariant store characteristics.
        Department fixed effects absorb category-level baseline demand differences.

        **What this is NOT:**
        This is a correlational estimate, not causal. Markdowns are not randomly assigned.

        **How to use these estimates:**
        Treat as directional signals and relative comparisons across segments,
        not as precise causal parameters.
        """)

# ════════════════════════════════════════════════════════════════
# PAGE 3 — OPTIMIZER
# ════════════════════════════════════════════════════════════════
elif page == "💰 Optimizer":
    st.markdown("### 💰 Markdown Optimization by Segment")

    col1, col2, col3 = st.columns(3)
    with col1:
        seg_choice = st.selectbox("Select segment", list(SEGMENT_COLORS.keys()))
    with col2:
        budget_pct = st.slider("Max markdown budget (% of baseline sales)",
                               0.05, 0.50, 0.30, 0.05)
    with col3:
        cost_rate = st.slider("Markdown cost rate", 0.05, 0.30, 0.15, 0.01)

    seg_id = cluster_labels.get(seg_choice)
    if seg_id is not None:
        elast    = cluster_elasticity[seg_id]['elasticity']
        seg_stores = [s for s, c in store_cluster_map.items() if c == seg_id]
        recent = df[
            (df['Store'].isin(seg_stores)) &
            (df['Date'] >= df['Date'].max() - pd.Timedelta(weeks=12))
        ]
        baseline = recent['Weekly_Sales'].mean() if len(recent) > 0 else 50000

        md_range = np.linspace(0, budget_pct, 100)
        net_revs, sales_lifts = [], []

        for md in md_range:
            lift  = np.exp(elast * np.log1p(md)) - 1
            sales = baseline * (1 + lift)
            net   = sales * (1 - cost_rate * md)
            sales_lifts.append(lift * 100)
            net_revs.append(net)

        best_idx  = int(np.argmax(net_revs))
        best_md   = md_range[best_idx]
        best_net  = net_revs[best_idx]
        best_lift = sales_lifts[best_idx]

        r1,r2,r3,r4 = st.columns(4)
        r1.metric("Baseline weekly sales",       f"${baseline:,.0f}")
        r2.metric("Optimal markdown intensity",  f"{best_md*100:.1f}%")
        r3.metric("Predicted sales lift",        f"{best_lift:.2f}%")
        r4.metric("Net revenue vs baseline",     f"${best_net-baseline:+,.0f}")

        col_a, col_b = st.columns(2)
        with col_a:
            fig, ax = plt.subplots(figsize=(8, 4))
            fig.patch.set_facecolor('#FFFFFF')
            ax.set_facecolor('#F8F9FB')
            ax.plot(md_range*100, net_revs, color='#2563EB', linewidth=2, label='Net revenue')
            ax.axhline(baseline, color='#94A3B8', linewidth=1, linestyle='--',
                       label='Baseline (no markdown)')
            ax.axvline(best_md*100, color='#059669', linewidth=1.5, linestyle=':',
                       label=f'Optimal: {best_md*100:.1f}%')
            ax.fill_between(md_range*100, net_revs, baseline,
                            where=[n > baseline for n in net_revs],
                            alpha=0.1, color='#059669')
            ax.fill_between(md_range*100, net_revs, baseline,
                            where=[n <= baseline for n in net_revs],
                            alpha=0.1, color='#DC2626')
            ax.set_xlabel('Markdown Intensity (%)')
            ax.set_ylabel('Net Weekly Revenue ($)')
            ax.set_title('Revenue Tradeoff Frontier', fontweight='bold')
            ax.legend(fontsize=8)
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x,_: f'${x:,.0f}'))
            for sp in ax.spines.values(): sp.set_color('#E2E8F0')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_b:
            fig, ax = plt.subplots(figsize=(8, 4))
            fig.patch.set_facecolor('#FFFFFF')
            ax.set_facecolor('#F8F9FB')
            ax2 = ax.twinx()
            ax.plot(md_range*100, sales_lifts, color='#2563EB', linewidth=2,
                    label='Sales lift (%)')
            ax2.plot(md_range*100,
                     [(n-baseline)/baseline*100 for n in net_revs],
                     color='#059669', linewidth=2, linestyle='--',
                     label='Net revenue change (%)')
            ax.set_xlabel('Markdown Intensity (%)')
            ax.set_ylabel('Sales Lift (%)', color='#2563EB')
            ax2.set_ylabel('Net Revenue Change (%)', color='#059669')
            ax.set_title('Volume vs Margin Tradeoff', fontweight='bold')
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1+lines2, labels1+labels2, fontsize=8)
            for sp in ax.spines.values(): sp.set_color('#E2E8F0')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        if elast > cost_rate:
            st.success(f"Elasticity {elast:.4f} > breakeven {cost_rate:.2f} — markdowns are ROI-positive.")
        else:
            st.error(f"Elasticity {elast:.4f} < breakeven {cost_rate:.2f} — markdowns destroy value at this cost rate.")

    st.divider()
    st.markdown("#### Optimization Results Across All Segments")
    if isinstance(opt_results, pd.DataFrame) and len(opt_results) > 0:
        st.dataframe(opt_results, use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════
# PAGE 4 — CANNIBALIZATION
# ════════════════════════════════════════════════════════════════
elif page == "🔀 Cannibalization":
    st.markdown("### 🔀 Cross-Department Cannibalization Matrix")
    st.markdown("""
    <div style="background:#EFF6FF;border:1px solid #BFDBFE;border-radius:10px;
         padding:12px 16px;margin-bottom:16px;font-size:13px;color:#1E40AF">
        <b>How to read this:</b> Each cell shows the elasticity of the row department
        with respect to markdown activity in the column department.
        Negative = cannibalization. Positive = complementary. Diagonal = own-price elasticity.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        fig, ax = plt.subplots(figsize=(9, 7))
        fig.patch.set_facecolor('#FFFFFF')
        cm_vals = cannibal_matrix.astype(float)
        sns.heatmap(
            cm_vals, annot=True, fmt='.4f',
            cmap='RdYlGn', center=0, ax=ax,
            linewidths=0.8, linecolor='#E2E8F0',
            annot_kws={'size':11,'weight':'bold'},
            cbar_kws={'label':'Elasticity coefficient'}
        )
        for i in range(len(cm_vals)):
            ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False,
                         edgecolor='#2563EB', lw=2.5))
        ax.set_title('Cross-Department Cannibalization Matrix',
                     fontsize=12, fontweight='bold', pad=14)
        ax.set_xlabel('Source of markdown', fontsize=11)
        ax.set_ylabel('Affected department', fontsize=11)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        cm_vals = cannibal_matrix.astype(float)
        depts   = list(cm_vals.index)
        st.markdown("**Own-price elasticities (diagonal):**")
        for d in depts:
            val   = cm_vals.loc[d, d]
            color = '#059669' if val > 0 else '#DC2626'
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;
                 padding:6px 0;border-bottom:1px solid #F1F5F9">
                <span style="font-size:12px;color:#374151">{d}</span>
                <span style="font-size:12px;font-weight:600;color:{color}">{val:+.4f}</span>
            </div>
            """, unsafe_allow_html=True)

        cross_effects = []
        for i, row in enumerate(depts):
            for j, col in enumerate(depts):
                if i != j:
                    val = cm_vals.loc[row, col]
                    if abs(val) > 0.001:
                        cross_effects.append((row, col, val))

        if cross_effects:
            st.markdown("<br><b>Significant cross-effects:</b>", unsafe_allow_html=True)
            for target, source, val in sorted(cross_effects, key=lambda x: x[2]):
                etype = "Cannibalization" if val < 0 else "Complementary"
                color = '#DC2626' if val < 0 else '#059669'
                st.markdown(f"""
                <div style="font-size:11px;color:#374151;padding:5px 0;
                     border-bottom:1px solid #F8F9FB">
                    {source} to {target}:
                    <span style="color:{color};font-weight:600">{val:+.4f} ({etype})</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("No significant cannibalization detected — departments operate independently.")

# ════════════════════════════════════════════════════════════════
# PAGE 5 — FORECAST EXPLORER
# ════════════════════════════════════════════════════════════════
elif page == "📈 Forecast Explorer":
    st.markdown("### 📈 Demand Forecast Explorer")

    col1, col2 = st.columns(2)
    with col1:
        store_id = st.selectbox("Store", sorted(df['Store'].unique()))
    with col2:
        dept_id = st.selectbox("Department", sorted(df['Dept'].unique()))

    sub = df[(df['Store']==store_id) & (df['Dept']==dept_id)].sort_values('Date').copy()

    if len(sub) < 10:
        st.warning("Not enough data for this store-dept combo. Try another.")
    else:
        md_cols = ['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5']
        for col in md_cols:
            sub[col] = sub[col].fillna(0).clip(lower=0)
        sub['total_markdown'] = sub[md_cols].sum(axis=1)
        sub['week']           = sub['Date'].dt.isocalendar().week.astype(int)
        sub['month']          = sub['Date'].dt.month
        sub['quarter']        = sub['Date'].dt.quarter
        sub['year']           = sub['Date'].dt.year
        sub['week_trend']     = ((sub['Date'] - df['Date'].min()).dt.days/7).astype(int)
        sub = sub.sort_values(['Store','Dept','Date'])
        sub['sales_lag_52']   = sub['Weekly_Sales'].shift(52)
        sub['sales_lag_4']    = sub['Weekly_Sales'].shift(4)
        sub['sales_lag_1']    = sub['Weekly_Sales'].shift(1)
        sub['rolling_mean_4'] = sub['Weekly_Sales'].shift(1).rolling(4).mean()
        sub['cluster']        = sub['Store'].map(store_cluster_map)

        from sklearn.preprocessing import LabelEncoder
        sub['Type_enc'] = LabelEncoder().fit_transform(sub['Type'].astype(str))

        FEATURES = [
            'Store','Dept','week','month','quarter','year','week_trend',
            'IsHoliday','Size','Type_enc',
            'CPI','Unemployment','Fuel_Price',
            'total_markdown','MarkDown1','MarkDown2',
            'sales_lag_52','sales_lag_4','sales_lag_1','rolling_mean_4',
            'cluster'
        ]
        sub_clean = sub.dropna(subset=FEATURES).copy()

        if len(sub_clean) > 5:
            preds = model.predict(sub_clean[FEATURES])
            sub_clean['predicted'] = preds

            col_a, col_b = st.columns([3,1])
            with col_a:
                fig, ax = plt.subplots(figsize=(11, 4))
                fig.patch.set_facecolor('#FFFFFF')
                ax.set_facecolor('#F8F9FB')
                ax.plot(sub_clean['Date'], sub_clean['Weekly_Sales'],
                        label='Actual', color='#2563EB', linewidth=1.8)
                ax.plot(sub_clean['Date'], sub_clean['predicted'],
                        label='Predicted', color='#E74C3C',
                        linewidth=1.5, linestyle='--')
                for hd in sub_clean[sub_clean['IsHoliday']==True]['Date']:
                    ax.axvline(hd, color='#F59E0B', alpha=0.3, linewidth=1)
                ax.set_title(f'Store {store_id} | Dept {dept_id} — Forecast vs Actual',
                             fontweight='bold', fontsize=11)
                ax.set_ylabel('Weekly Sales ($)')
                ax.legend()
                ax.yaxis.set_major_formatter(
                    ticker.FuncFormatter(lambda x,_: f'${x:,.0f}'))
                for sp in ax.spines.values(): sp.set_color('#E2E8F0')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            with col_b:
                actuals = sub_clean['Weekly_Sales']
                preds_  = sub_clean['predicted']
                mae  = np.mean(np.abs(actuals - preds_))
                wmae = mae / actuals.mean()
                corr = np.corrcoef(actuals, preds_)[0,1]
                st.metric("MAE",         f"${mae:,.0f}")
                st.metric("WMAE",        f"{wmae:.3f}")
                st.metric("Correlation", f"{corr:.3f}")
                segment = cluster_label_map.get(
                    store_cluster_map.get(store_id), 'Unknown')
                st.info(f"Segment: {segment}")
        else:
            st.warning("Not enough complete rows after feature engineering.")

# ════════════════════════════════════════════════════════════════
# PAGE 6 — METHODOLOGY
# ════════════════════════════════════════════════════════════════
elif page == "📋 Methodology":
    st.markdown("### 📋 Methodology, Assumptions & Limitations")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📦 Data & Assumptions",
        "🔍 Identification Strategy",
        "🏗️ Model Architecture",
        "⚠️ Failure Modes"
    ])

    with tab1:
        st.markdown("""
        #### Data Sources
        - **Primary:** Walmart Recruiting Store Sales Forecasting (Kaggle)
          Weekly sales by store/department, markdown flags, CPI, fuel price, holiday flags.
          421,570 records across 45 stores and 81 departments (2010-2012).

        #### Stated Limitations on Price Proxy
        This dataset contains **no explicit unit prices**. Markdown activity
        (MarkDown1-5) is used as a price proxy: higher markdown = lower effective price.
        MarkDown columns are 64-74% missing — missing values are treated as zero
        (no active promotion), not imputed.

        #### Synthetic Data (Clearly Labeled)
        A synthetic competitor price signal was added as a random walk correlated
        with our price proxy. It is **not real market data** and is included solely
        to demonstrate the system architecture supports competitive pricing inputs.

        #### Cost & Margin Assumptions
        - Markdown cost rate: **15%** (discount passed to customer as fraction of sales)
        - Breakeven elasticity: **0.15** (minimum elasticity for markdown ROI)
        """)

    with tab2:
        st.markdown("""
        #### What the Model Estimates
        Log-log OLS regression of log(weekly_sales) on log(markdown intensity),
        with store fixed effects, department fixed effects, linear time trend,
        holiday controls, and macro covariates (CPI, fuel price).

        #### What This Is NOT
        This is a **correlational estimate, not causal**. Markdowns are not
        randomly assigned. This endogeneity biases estimates.

        #### How to Use These Estimates Responsibly
        Treat elasticities as directional signals and relative comparisons
        across segments, not as precise causal parameters.
        """)

    with tab3:
        st.markdown("""
        #### Pipeline Architecture

        | Phase | Method | Output |
        |---|---|---|
        | EDA | Pandas, Seaborn | Seasonality, holiday effects, markdown sparsity |
        | Segmentation | K-means (k=4) | 4 business segments |
        | Elasticity | Log-log OLS with store + dept FEs | Segment-level elasticity |
        | Forecasting | LightGBM with lag features | WMAE 0.077 |
        | Optimization | scipy.optimize | Optimal markdown per segment |
        | Deployment | Streamlit | This app |

        #### Train/Test Split
        Time-based — last 12 weeks = test. Random shuffle is never used on time series.
        """)

    with tab4:
        st.markdown("""
        #### Known Failure Modes

        | Scenario | Risk | Mitigation |
        |---|---|---|
        | New department (no history) | lag features = NaN | Use category-level average |
        | Demand shock | Out-of-distribution | Retrain with recent data |
        | Price extrapolation | Unrealistic lift | Clip to observed range |
        | New store | No cluster assignment | Assign to nearest centroid |

        #### What I Would Do With Real Data
        - Replace markdown proxy with actual unit prices from POS systems
        - Use cost shocks or A/B pricing tests for causal identification
        - Retrain quarterly as pricing environment shifts
        """)