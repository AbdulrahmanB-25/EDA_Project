import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# Disable LaTeX math parsing so $ signs in labels are treated as plain text
mpl.rcParams["text.usetex"] = False
mpl.rcParams["mathtext.default"] = "regular"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Riyadh Restaurant Explorer",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    #MainMenu, footer, header { visibility: hidden; }

    [data-testid="stSidebar"] { background: #0f0f14; border-right: 1px solid #222230; }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div { color: #c8c8d8; }
    [data-testid="stSidebar"] .stButton button {
        background: transparent;
        color: #a0a0c0 !important;
        border: 1px solid #2a2a3e;
        border-radius: 8px;
        font-family: 'DM Sans', sans-serif;
        font-size: 0.88rem;
        text-align: left;
        transition: all 0.15s;
    }
    [data-testid="stSidebar"] .stButton button:hover {
        background: #1e1e2e !important;
        color: #ffffff !important;
        border-color: #3a3a5e;
    }

    .page-header {
        font-family: 'Syne', sans-serif;
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: #0f0f14;
        line-height: 1.1;
    }
    .page-subtitle { font-size: 1rem; color: #666; margin-top: 4px; }

    .kpi-card {
        background: #fff;
        border: 1px solid #e8e8f0;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .kpi-value {
        font-family: 'Syne', sans-serif;
        font-size: 2rem;
        font-weight: 800;
        color: #0f0f14;
        line-height: 1;
    }
    .kpi-label {
        font-size: 0.75rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 6px;
        font-weight: 500;
    }

    .section-title {
        font-family: 'Syne', sans-serif;
        font-size: 1.15rem;
        font-weight: 700;
        color: #0f0f14;
        border-left: 4px solid #c0152a;
        padding-left: 12px;
        margin: 20px 0 14px 0;
    }

    .insight-box {
        background: #fff8f8;
        border-left: 3px solid #c0152a;
        border-radius: 0 8px 8px 0;
        padding: 10px 14px;
        font-size: 0.83rem;
        color: #2a2a2a;
        margin-top: 8px;
        line-height: 1.6;
    }

    .model-card {
        background: #fff;
        border: 1px solid #e8e8f0;
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .model-name { font-family: 'Syne', sans-serif; font-weight: 700; font-size: 1rem; color: #0f0f14; }
    .model-acc  { font-family: 'Syne', sans-serif; font-size: 1.8rem; font-weight: 800; color: #0f0f14; }

    .badge { display:inline-block; padding:3px 10px; border-radius:20px; font-size:0.72rem; font-weight:600; letter-spacing:0.04em; }
    .badge-green  { background: #d4f5e5; color: #0f5c30; }
    .badge-orange { background: #fde8c8; color: #7a3800; }
    .badge-red    { background: #fdd8dc; color: #7a0010; }

    .warn-box {
        background: #fffbea;
        border: 1px solid #e8c840;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 0.83rem;
        color: #5a4000;
        margin-top: 12px;
        line-height: 1.5;
    }

    [data-testid="metric-container"] {
        background: #fff;
        border: 1px solid #e8e8f0;
        border-radius: 12px;
        padding: 16px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }

    .stTabs [data-baseweb="tab-list"] { gap: 8px; background: transparent; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 18px;
        font-family: 'DM Sans', sans-serif;
        font-size: 0.88rem;
        font-weight: 500;
        border: 1px solid #ddddef;
        background: #fff;
        color: #444 !important;
    }
    .stTabs [aria-selected="true"] {
        background: #c0152a !important;
        color: #ffffff !important;
        border-color: #c0152a !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("clean_data.csv")
    df["category"] = (
        df["category"]
        .astype(str)
        .str.strip("[]")
        .str.replace("'", "", regex=False)
        .str.strip()
    )
    return df

df = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 8px 0 16px 0;'>
        <div style='font-family: Syne, sans-serif; font-size: 1.3rem; font-weight: 800;
                    color: #fff; letter-spacing: -0.02em;'>🍽️ Riyadh<br>Restaurants</div>
        <div style='font-size: 0.75rem; color: #555; margin-top: 4px;'>Unit 3 Final Project</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("<div style='font-size:0.7rem; color:#555; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px;'>Navigation</div>", unsafe_allow_html=True)

    pages = {
        "🏠  Overview":         "overview",
        "📊  EDA":              "eda",
        "📍  Geography":        "geography",
        "💰  Price & Quality":  "price",
        "🤖  ML Classification":"ml",
    }

    if "page" not in st.session_state:
        st.session_state.page = "overview"

    for label, key in pages.items():
        if st.button(label, key=f"nav_{key}", use_container_width=True):
            st.session_state.page = key

    st.divider()

    st.markdown("<div style='font-size:0.7rem; color:#555; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px;'>Filters</div>", unsafe_allow_html=True)

    price_options = sorted(df["price_level"].dropna().unique().tolist())
    selected_prices = st.multiselect(
        "Price Level", options=price_options, default=price_options,
        format_func=lambda x: "Unspecified" if x == 0 else ("$" * int(x)).replace("$$", "$ $")
    )

    min_rating, max_rating = float(df["rating"].min()), float(df["rating"].max())
    rating_range = st.slider("Rating Range", min_value=min_rating, max_value=max_rating,
                             value=(min_rating, max_rating), step=0.1)

    top_neighborhoods = sorted(df["neighborhoods"].value_counts().head(50).index.tolist())
    selected_neighborhoods = st.multiselect("Neighborhoods", options=top_neighborhoods, default=[])

    st.divider()
    st.caption("Data: Kaggle / Foursquare")

# ── Apply filters ─────────────────────────────────────────────────────────────
filtered = df[
    (df["price_level"].isin(selected_prices)) &
    (df["rating"] >= rating_range[0]) &
    (df["rating"] <= rating_range[1])
]
if selected_neighborhoods:
    filtered = filtered[filtered["neighborhoods"].isin(selected_neighborhoods)]

page = st.session_state.page

# helper to style plots consistently
def style_ax(ax, fig):
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#fafafa')
    for spine in ax.spines.values():
        spine.set_color('#e8e8f0')
    ax.tick_params(labelsize=8)

RED  = "#c0152a"
REDS = ["#c0152a", "#a01020", "#800818", "#600010", "#400008"]

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "overview":
    st.markdown('<div class="page-header">Riyadh Restaurant Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Exploratory analysis of 5,000+ food & dining venues across Riyadh, Saudi Arabia</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    k1, k2, k3, k4, k5 = st.columns(5)
    for col, val, label in [
        (k1, f"{len(filtered):,}",                    "Total Venues"),
        (k2, f"{filtered['rating'].mean():.2f}",       "Avg Rating"),
        (k3, f"{filtered['category'].nunique():,}",    "Categories"),
        (k4, f"{filtered['neighborhoods'].nunique():,}","Neighborhoods"),
        (k5, f"{int(filtered['price_level'].median())}","Median Price Lvl"),
    ]:
        col.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{val}</div>
            <div class="kpi-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()
    st.markdown('<div class="section-title">Quick Overview</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Rating Distribution**")
        fig, ax = plt.subplots(figsize=(5, 3))
        style_ax(ax, fig)
        sns.histplot(filtered["rating"], bins=20, kde=True, ax=ax, color=RED, alpha=0.7)
        ax.axvline(filtered["rating"].mean(), color="#0f0f14", linestyle="--", lw=1.5,
                   label=f"Mean: {filtered['rating'].mean():.2f}")
        ax.set_xlabel("Rating", fontsize=9); ax.set_ylabel("Count", fontsize=9)
        ax.legend(fontsize=8)
        st.pyplot(fig); plt.close()
        st.markdown('<div class="insight-box">Most venues rate 7–9. Mean ~7.82 and median ~7.90 are nearly identical — a symmetric, healthy distribution.</div>', unsafe_allow_html=True)

    with c2:
        st.markdown("**Top 5 Categories**")
        top5 = filtered["category"].value_counts().head(5)
        fig, ax = plt.subplots(figsize=(5, 3))
        style_ax(ax, fig)
        sns.barplot(y=top5.index, x=top5.values, ax=ax,
                    palette=["#c0152a","#e06070","#d08090","#e0a8b0","#f0d0d5"])
        ax.set_xlabel("Count", fontsize=9); ax.set_ylabel("")
        st.pyplot(fig); plt.close()
        st.markdown('<div class="insight-box">Coffee Shop leads with ~2,050 venues — 3× more than Burger Joint in 2nd place.</div>', unsafe_allow_html=True)

    with c3:
        st.markdown("**Price Level Split**")
        price_counts = filtered["price_level"].value_counts().sort_index()
        labels = ["Unspecified" if x == 0 else "$" * int(x) for x in price_counts.index]
        fig, ax = plt.subplots(figsize=(5, 3))
        style_ax(ax, fig)
        ax.bar(labels, price_counts.values,
               color=["#ccc","#c0152a","#c0152a","#8b0000","#4a0000"][:len(labels)],
               edgecolor='white', lw=0.5)
        ax.set_xlabel("Price Level", fontsize=9); ax.set_ylabel("Count", fontsize=9)
        st.pyplot(fig); plt.close()
        st.markdown('<div class="insight-box">Budget dining ($) dominates at ~2,400 venues. Premium dining ($$+) is under 4% of all venues.</div>', unsafe_allow_html=True)

    st.divider()
    with st.expander("📋 Data Preview", expanded=False):
        st.dataframe(
            filtered[["name","category","rating","price_level","neighborhoods","latitude","longitude"]]
            .sort_index().head(50), use_container_width=True
        )
        st.caption(f"Showing 50 of {len(filtered):,} filtered rows")

    with st.expander("📊 Summary Statistics", expanded=False):
        st.dataframe(
            filtered[["rating","price_level","total_photos","total_ratings","total_tips"]].describe(),
            use_container_width=True
        )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "eda":
    st.markdown('<div class="page-header">Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">6 research questions answered through visualizations</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📊 Distributions & Categories", "🏆 Ratings Deep Dive", "🔗 Correlations & Variety"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="section-title">Top 10 Most Common Categories</div>', unsafe_allow_html=True)
            top_cat = filtered["category"].value_counts().head(10)
            fig, ax = plt.subplots(figsize=(6, 4.5))
            style_ax(ax, fig)
            colors = [RED if i == 0 else "#c87880" for i in range(len(top_cat))]
            sns.barplot(y=top_cat.index, x=top_cat.values, ax=ax, palette=colors)
            ax.set_xlabel("Count", fontsize=9); ax.set_ylabel("")
            st.pyplot(fig); plt.close()
            st.markdown('<div class="insight-box">Coffee Shop dominates at ~2,050 venues — over 3× more than Burger Joint (~650). The remaining 8 range between 270–470 venues.</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="section-title">Top 10 Highest Rated Categories</div>', unsafe_allow_html=True)
            cat_rating = filtered.groupby("category")["rating"].mean().sort_values(ascending=False).head(10)
            fig, ax = plt.subplots(figsize=(6, 4.5))
            style_ax(ax, fig)
            colors2 = [RED if i == 0 else "#c87880" for i in range(len(cat_rating))]
            sns.barplot(y=cat_rating.index, x=cat_rating.values, ax=ax, palette=colors2)
            ax.set_xlabel("Average Rating", fontsize=9); ax.set_ylabel("")
            ax.set_xlim(0, 10)
            st.pyplot(fig); plt.close()
            st.markdown('<div class="insight-box">All top-rated venues are multi-concept combinations (e.g. Restaurant & Juice Bar ~9.4). None of the most common categories appear here — popularity ≠ quality.</div>', unsafe_allow_html=True)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="section-title">Rating by Price Level</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 4))
            style_ax(ax, fig)
            sns.boxplot(x=filtered["price_level"], y=filtered["rating"], ax=ax,
                       palette=["#ccc","#c0152a","#c0152a","#8b0000","#4a0000"])
            ax.set_xlabel("Price Level", fontsize=9); ax.set_ylabel("Rating", fontsize=9)
            st.pyplot(fig); plt.close()
            st.markdown('<div class="insight-box">Price levels 0–2 share nearly identical medians (~7.9–8.0). Paying more does NOT yield better ratings.</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="section-title">Ratings Across Top Neighborhoods</div>', unsafe_allow_html=True)
            top_neigh_idx = filtered["neighborhoods"].value_counts().head(10).index
            fig, ax = plt.subplots(figsize=(6, 4))
            style_ax(ax, fig)
            sns.boxplot(
                data=filtered[filtered["neighborhoods"].isin(top_neigh_idx)],
                x="neighborhoods", y="rating", ax=ax, color=RED
            )
            plt.xticks(rotation=45, ha="right", fontsize=7)
            ax.set_xlabel(""); ax.set_ylabel("Rating", fontsize=9)
            st.pyplot(fig); plt.close()
            st.markdown('<div class="insight-box">Al Qairawan and Al Malqa lead at ~8.2 median. Al Yasmeen has the widest spread and lowest outlier (~4.7). Quality is fairly uniform city-wide.</div>', unsafe_allow_html=True)

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="section-title">Correlation Heatmap</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor('#ffffff')
            corr = filtered[["rating","price_level","total_photos","total_ratings"]].corr()
            sns.heatmap(corr, annot=True, cmap="RdYlBu_r", ax=ax, fmt=".2f",
                       annot_kws={"size": 10}, linewidths=0.5)
            ax.tick_params(labelsize=9)
            st.pyplot(fig); plt.close()
            st.markdown('<div class="insight-box">total_photos ↔ total_ratings: 0.61 (strongest). price_level shows near-zero correlation with everything.</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="section-title">Food Variety by Price Level (Top 5)</div>', unsafe_allow_html=True)
            variety_table = (
                filtered.groupby(["neighborhoods","price_level"])["category"]
                .nunique().unstack()
            )
            if not variety_table.empty:
                top5 = variety_table.sum(axis=1).sort_values(ascending=False).head(5).index
                fig, ax = plt.subplots(figsize=(6, 4))
                fig.patch.set_facecolor('#ffffff')
                sns.heatmap(variety_table.loc[top5], cmap="YlOrRd", annot=True, fmt=".0f",
                           ax=ax, linewidths=0.5, annot_kws={"size": 9})
                ax.set_xlabel("Price Level", fontsize=9); ax.set_ylabel("")
                ax.tick_params(labelsize=8)
                st.pyplot(fig); plt.close()
            st.markdown('<div class="insight-box">Al Malqa leads with 42 unique categories at price level 1. Diversity drops sharply at higher price tiers.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: GEOGRAPHY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "geography":
    st.markdown('<div class="page-header">Geographic Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Restaurant distribution across Riyadh\'s neighborhoods</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title">Top 15 Neighborhoods by Count</div>', unsafe_allow_html=True)
        top_neigh = filtered["neighborhoods"].value_counts().head(15)
        fig, ax = plt.subplots(figsize=(6, 5.5))
        style_ax(ax, fig)
        colors = [RED if i < 3 else "#c87880" for i in range(len(top_neigh))]
        sns.barplot(y=top_neigh.index, x=top_neigh.values, ax=ax, palette=colors)
        ax.set_xlabel("Number of Restaurants", fontsize=9); ax.set_ylabel("")
        st.pyplot(fig); plt.close()
        st.markdown('<div class="insight-box">Hiteen leads with ~365 restaurants, followed by Dhahrat Laban (~328) and Al Malqa (~315). Strong northern/northwestern concentration.</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-title">Restaurant Locations Map</div>', unsafe_allow_html=True)
        try:
            import geopandas as gpd
            import os

            geo_path = "Saudi-Arabia-Regions-Cities-and-Districts/geojson"
            cities_path    = f"{geo_path}/cities.geojson"
            districts_path = f"{geo_path}/districts.geojson"

            if os.path.exists(cities_path) and os.path.exists(districts_path):
                cities_gdf    = gpd.read_file(cities_path).to_crs(epsg=4326)
                districts_gdf = gpd.read_file(districts_path).to_crs(epsg=4326)
                riyadh        = cities_gdf[cities_gdf["name_en"] == "Riyadh"]

                fig, ax = plt.subplots(figsize=(6, 5.5))
                fig.patch.set_facecolor('#ffffff')

                # City boundary fill
                riyadh.plot(ax=ax, color="#f5f5f5", edgecolor="#222", linewidth=1.2)
                # District borders
                districts_gdf.plot(ax=ax, facecolor="none", edgecolor="#bbb", linewidth=0.3)
                # Restaurant dots
                ax.scatter(
                    filtered["longitude"], filtered["latitude"],
                    color=RED, s=5, alpha=0.4, zorder=3
                )

                lon_min, lon_max = filtered["longitude"].min()-0.01, filtered["longitude"].max()+0.01
                lat_min, lat_max = filtered["latitude"].min()-0.01,  filtered["latitude"].max()+0.01
                ax.set_xlim(lon_min, lon_max)
                ax.set_ylim(lat_min, lat_max)

                ax.set_xlabel("Longitude", fontsize=9)
                ax.set_ylabel("Latitude", fontsize=9)
                ax.set_title("Restaurant Locations Across Riyadh", fontsize=10, pad=10)
                ax.tick_params(labelsize=8)
                for spine in ax.spines.values(): spine.set_color('#e8e8f0')
                fig.patch.set_facecolor('#ffffff')
                ax.set_facecolor('#f9f9f9')

            else:
                # Fallback: plain scatter if GeoJSON not cloned
                fig, ax = plt.subplots(figsize=(6, 5.5))
                fig.patch.set_facecolor('#0f0f14')
                ax.set_facecolor('#0f0f14')
                ax.scatter(filtered["longitude"], filtered["latitude"], alpha=0.4, s=5, color=RED)
                ax.set_xlabel("Longitude", fontsize=9, color="#aaa")
                ax.set_ylabel("Latitude",  fontsize=9, color="#aaa")
                ax.tick_params(labelsize=8, colors="#aaa")
                for spine in ax.spines.values(): spine.set_color('#333')
                ax.set_title("Restaurant Locations — Riyadh", color="#fff", fontsize=10)

        except Exception as e:
            fig, ax = plt.subplots(figsize=(6, 5.5))
            ax.scatter(filtered["longitude"], filtered["latitude"], alpha=0.4, s=5, color=RED)
            ax.set_xlabel("Longitude", fontsize=9); ax.set_ylabel("Latitude", fontsize=9)
            ax.set_title("Restaurant Locations — Riyadh", fontsize=10)

        st.pyplot(fig); plt.close()
        st.markdown('<div class="insight-box">Restaurants concentrate in central and northern Riyadh. District outlines show how venue density maps onto specific neighborhoods.</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="section-title">Explore a Neighborhood</div>', unsafe_allow_html=True)

    selected_neigh = st.selectbox(
        "Select neighborhood",
        options=sorted(filtered["neighborhoods"].dropna().unique().tolist())
    )
    neigh_data = filtered[filtered["neighborhoods"] == selected_neigh]

    if len(neigh_data) > 0:
        n1, n2, n3, n4 = st.columns(4)
        n1.metric("Venues", f"{len(neigh_data):,}")
        n2.metric("Avg Rating", f"{neigh_data['rating'].mean():.2f}")
        n3.metric("Categories", f"{neigh_data['category'].nunique()}")
        n4.metric("Most Common", neigh_data['category'].value_counts().index[0])

        c1, c2 = st.columns(2)
        with c1:
            top_cat_n = neigh_data["category"].value_counts().head(8)
            fig, ax = plt.subplots(figsize=(5, 3.5))
            style_ax(ax, fig)
            sns.barplot(y=top_cat_n.index, x=top_cat_n.values, ax=ax, color=RED)
            ax.set_xlabel("Count", fontsize=9); ax.set_ylabel("")
            ax.set_title(f"Top Categories in {selected_neigh}", fontsize=9)
            st.pyplot(fig); plt.close()

        with c2:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            style_ax(ax, fig)
            sns.histplot(neigh_data["rating"], bins=15, kde=True, ax=ax, color=RED, alpha=0.7)
            ax.axvline(neigh_data["rating"].mean(), color="#0f0f14", linestyle="--", lw=1.5)
            ax.set_xlabel("Rating", fontsize=9); ax.set_ylabel("Count", fontsize=9)
            ax.set_title(f"Rating Distribution in {selected_neigh}", fontsize=9)
            st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PRICE & QUALITY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "price":
    st.markdown('<div class="page-header">Price vs. Quality</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Does spending more mean eating better?</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    corr_val = filtered["price_level"].corr(filtered["rating"])
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,#0f0f14,#1e1e2e); border-radius:16px;
                padding:28px 36px; margin-bottom:24px; display:flex; align-items:center; gap:32px;'>
        <div>
            <div style='font-family:Syne,sans-serif; font-size:3.5rem; font-weight:800;
                        color:#ffffff; line-height:1;'>{corr_val:.4f}</div>
            <div style='color:#aaa; font-size:0.9rem; margin-top:6px;'>Correlation: price_level ↔ rating</div>
        </div>
        <div style='color:#ccc; font-size:0.95rem; max-width:500px; line-height:1.6;'>
            Near-zero correlation between price and rating. Paying more in Riyadh does
            <strong style='color:#ffffff;'>not</strong> guarantee a better dining experience.
            Budget and premium restaurants achieve virtually identical customer satisfaction scores.
        </div>
    </div>
    """, unsafe_allow_html=True)

    price_colors = {0:"#ccc", 1:"#c0152a", 2:"#c0152a", 3:"#8b0000", 4:"#4a0000"}

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title">Rating Distribution by Price Level</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        style_ax(ax, fig)
        for pl in sorted(filtered["price_level"].dropna().unique()):
            data = filtered[filtered["price_level"] == pl]["rating"].dropna()
            label = "Unspecified" if pl == 0 else "$" * int(pl)
            ax.hist(data, bins=15, alpha=0.6, label=label, color=price_colors.get(pl, "#999"))
        ax.set_xlabel("Rating", fontsize=9); ax.set_ylabel("Count", fontsize=9)
        ax.legend(fontsize=8)
        st.pyplot(fig); plt.close()
        st.markdown('<div class="insight-box">All price levels peak in the same 7.5–8.5 range. No visible shift toward higher ratings at higher price levels.</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-title">Median Rating per Price Level</div>', unsafe_allow_html=True)
        median_by_price = filtered.groupby("price_level")["rating"].median().reset_index()
        median_by_price["label"] = median_by_price["price_level"].apply(
            lambda x: "Unspecified" if x == 0 else "$" * int(x)
        )
        fig, ax = plt.subplots(figsize=(6, 4))
        style_ax(ax, fig)
        bar_colors = [price_colors.get(p, "#ccc") for p in median_by_price["price_level"]]
        ax.bar(median_by_price["label"], median_by_price["rating"],
               color=bar_colors, edgecolor='white')
        ax.set_ylim(7, 9)
        ax.set_xlabel("Price Level", fontsize=9); ax.set_ylabel("Median Rating", fontsize=9)
        for i, v in enumerate(median_by_price["rating"]):
            ax.text(i, v + 0.03, f"{v:.2f}", ha='center', fontsize=9, fontweight='bold')
        st.pyplot(fig); plt.close()
        st.markdown('<div class="insight-box">Median ratings across all price levels fall within a 7.7–8.0 band — price level is not a reliable predictor of quality.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ML CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ml":
    st.markdown('<div class="page-header">ML — Price Level Classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Predicting whether a restaurant is $, $$, or $$$ using supervised learning</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns([1.2, 1])

    with c1:
        st.markdown('<div class="section-title">Problem Setup</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style='background:#fff; border:1px solid #e8e8f0; border-radius:12px;
                    padding:20px; box-shadow:0 2px 8px rgba(0,0,0,0.04);'>
            <div style='margin-bottom:12px; font-size:0.88rem;'>
                <span style='font-weight:600;'>🎯 Task:</span>
                <span style='color:#444;'> Multi-class classification — predict price level 1, 2, or 3</span>
            </div>
            <div style='margin-bottom:12px; font-size:0.88rem;'>
                <span style='font-weight:600;'>❌ Excluded:</span>
                <span style='color:#444;'> Price level 0 (unspecified) removed from training</span>
            </div>
            <div style='margin-bottom:12px; font-size:0.88rem;'>
                <span style='font-weight:600;'>⚖️ Imbalance:</span>
                <span style='color:#444;'> class_weight="balanced" applied to all 3 models</span>
            </div>
            <div style='font-size:0.88rem;'>
                <span style='font-weight:600;'>✂️ Split:</span>
                <span style='color:#444;'> 80% train / 20% test — random_state=42</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-title" style="margin-top:20px;">Features Used</div>', unsafe_allow_html=True)
        for feat, ftype, color in [
            ("rating",        "Numerical",                  "green"),
            ("total_ratings", "Numerical",                  "green"),
            ("total_photos",  "Numerical",                  "green"),
            ("total_tips",    "Numerical",                  "green"),
            ("neighborhoods", "Categorical → LabelEncoded", "orange"),
            ("category",      "Categorical → LabelEncoded", "orange"),
        ]:
            st.markdown(f"""
            <div style='display:flex; justify-content:space-between; align-items:center;
                        padding:8px 12px; background:#fff; border:1px solid #e8e8f0;
                        border-radius:8px; margin-bottom:6px;'>
                <code style='font-size:0.85rem; color:#0f0f14;'>{feat}</code>
                <span class='badge badge-{color}'>{ftype}</span>
            </div>
            """, unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-title">Class Distribution</div>', unsafe_allow_html=True)
        class_counts = df[df["price_level"].isin([1,2,3])]["price_level"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(5, 3.5))
        style_ax(ax, fig)
        bars = ax.bar(["$ Low", "$$ Mid", "$$$ High"],
                      class_counts.values,
                      color=["#c0152a","#c0152a","#8b0000"],
                      edgecolor='white', lw=0.5)
        for bar, val in zip(bars, class_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                   f"{val:,}", ha='center', fontsize=9, fontweight='bold')
        ax.set_ylabel("Count", fontsize=9)
        st.pyplot(fig); plt.close()

        st.markdown("""
        <div class="warn-box">
            ⚠️ <strong>Severe imbalance:</strong> Class 1 ($) has ~15× more samples than
            Class 3 ($$$). Handled with <code>class_weight="balanced"</code>.
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="section-title">Model Results</div>', unsafe_allow_html=True)

    mc1, mc2, mc3 = st.columns(3)
    models_data = [
        {
            "name":"Random Forest",     "accuracy":"72%", "badge":"green",  "badge_text":"Best Overall",
            "details":[("$ Low",  "0.73","0.98","0.84"),
                       ("$$ Mid", "0.29","0.03","0.05"),
                       ("$$$ High","0.00","0.00","0.00")],
            "note":"Strong on class 1 but ignores minority classes. Biased toward $ due to imbalance.",
            "col":mc1
        },
        {
            "name":"Decision Tree",     "accuracy":"59%", "badge":"orange", "badge_text":"Most Balanced",
            "details":[("$ Low",  "0.74","0.73","0.73"),
                       ("$$ Mid", "0.26","0.25","0.26"),
                       ("$$$ High","0.00","0.00","0.00")],
            "note":"Lower accuracy but tries all classes. Better at predicting $$ than Random Forest.",
            "col":mc2
        },
        {
            "name":"Logistic Regression","accuracy":"18%","badge":"red",    "badge_text":"Weakest",
            "details":[("$ Low",  "0.74","0.10","0.18"),
                       ("$$ Mid", "0.22","0.38","0.28"),
                       ("$$$ High","0.04","0.47","0.07")],
            "note":"Features are not linearly separable. Converged poorly despite max_iter=2000.",
            "col":mc3
        },
    ]

    for m in models_data:
        with m["col"]:
            rows_html = "".join([
                f"""<tr style='border-bottom:1px solid #f8f8f8;'>
                    <td style='padding:4px 0; color:#444; font-size:0.78rem;'>{cls}</td>
                    <td style='text-align:center; font-weight:600; font-size:0.78rem;'>{prec}</td>
                    <td style='text-align:center; font-weight:600; font-size:0.78rem;'>{rec}</td>
                    <td style='text-align:center; color:#0f0f14; font-weight:700; font-size:0.78rem;'>{f1}</td>
                </tr>"""
                for cls, prec, rec, f1 in m["details"]
            ])
            st.markdown(f"""
            <div class="model-card">
                <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;'>
                    <div class="model-name">{m['name']}</div>
                    <span class='badge badge-{m["badge"]}'>{m['badge_text']}</span>
                </div>
                <div class="model-acc">{m['accuracy']}</div>
                <div style='font-size:0.72rem; color:#888; margin-bottom:12px;'>Accuracy</div>
                <table style='width:100%; border-collapse:collapse;'>
                    <tr style='color:#888; border-bottom:1px solid #f0f0f0; font-size:0.75rem;'>
                        <td style='padding:3px 0;'>Class</td>
                        <td style='text-align:center;'>Prec</td>
                        <td style='text-align:center;'>Rec</td>
                        <td style='text-align:center;'>F1</td>
                    </tr>
                    {rows_html}
                </table>
                <div style='margin-top:10px; font-size:0.78rem; color:#666; line-height:1.5;
                            border-top:1px solid #f0f0f0; padding-top:8px;'>{m['note']}</div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()
    c1, c2 = st.columns([1.2, 1])

    with c1:
        st.markdown('<div class="section-title">Feature Importance — Random Forest</div>', unsafe_allow_html=True)
        feat_names = ["rating","total_ratings","total_photos","total_tips","neighborhoods","category"]
        feat_vals  = [0.14, 0.18, 0.22, 0.10, 0.20, 0.16]
        feat_df = pd.Series(feat_vals, index=feat_names).sort_values()
        fig, ax = plt.subplots(figsize=(6, 3.5))
        style_ax(ax, fig)
        colors_fi = [RED if v == feat_df.max() else "#c87880" for v in feat_df.values]
        ax.barh(feat_df.index, feat_df.values, color=colors_fi, edgecolor='white')
        ax.set_xlabel("Importance Score", fontsize=9)
        st.pyplot(fig); plt.close()
        st.markdown('<div class="insight-box">total_photos and neighborhoods contribute most. Rating itself is a weak predictor — consistent with near-zero price↔rating correlation found in EDA.</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-title">Why the Models Struggled</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style='background:#fff; border:1px solid #e8e8f0; border-radius:12px;
                    padding:20px; box-shadow:0 2px 8px rgba(0,0,0,0.04); font-size:0.87rem;
                    color:#444; line-height:1.7;'>
            The EDA correlation heatmap already showed
            <strong style='color:#0f0f14;'>price_level has near-zero correlation</strong>
            with all available features.<br><br>
            This means the features don't strongly separate price classes.
            Price in Riyadh's dining scene is likely driven by
            <strong>factors not in this dataset</strong>:
            <ul style='margin-top:10px; line-height:2;'>
                <li>Restaurant brand / chain status</li>
                <li>Interior design & ambiance</li>
                <li>Location prestige</li>
                <li>Menu offering & cuisine type</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style='margin-top:14px; background:#f0fdf4; border:1px solid #86efac;
                    border-radius:8px; padding:14px 16px;'>
            <div style='font-weight:700; color:#15803d; margin-bottom:6px;'>✅ Best Model: Random Forest</div>
            <div style='font-size:0.83rem; color:#166534; line-height:1.6;'>
                Most reliable for identifying budget ($) restaurants.
                72% accuracy — but treat $$/$$$  predictions with caution.
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align:center; color:#bbb; font-size:12px; padding:16px 0;'>"
    "Unit 3 Final Project · Riyadh Restaurant EDA + ML · Data: Kaggle / Foursquare"
    "</div>",
    unsafe_allow_html=True
)
