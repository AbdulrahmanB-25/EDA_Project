import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

mpl.rcParams["text.usetex"] = False
mpl.rcParams["mathtext.default"] = "regular"

st.set_page_config(page_title="Riyadh Restaurants", page_icon="🍽️", layout="wide")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@400;500&display=swap');

*, html, body { font-family: 'DM Sans', sans-serif !important; }
#MainMenu, footer, header { visibility: hidden; }
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"],
.main, .main .block-container { background: #0d0d12 !important; }
.stApp p, .stApp span, .stApp div, .stApp label,
.stApp h1, .stApp h2, .stApp h3, .stApp li,
.stMarkdown, .stMarkdown p { color: #e8e8ff !important; }

[data-testid="stSidebar"] { background: #07070d !important; border-right: 1px solid #1a1a28 !important; }
[data-testid="stSidebar"] * { color: #6060a0 !important; }
[data-testid="stSidebar"] .stButton button {
    background: transparent !important; color: #7070b0 !important;
    border: 1px solid #1a1a28 !important; border-radius: 8px !important;
    text-align: left !important; width: 100% !important; padding: 9px 14px !important;
    font-size: .9rem !important; transition: all 0.15s !important; }
[data-testid="stSidebar"] .stButton button:hover {
    background: #1a1a2e !important; color: #ffffff !important; border-color: #ff6b35 !important; }

.stTabs [data-baseweb="tab-list"] { gap: 6px; background: transparent !important; }
.stTabs [data-baseweb="tab"] {
    background: #12121e !important; border: 1px solid #1e1e30 !important;
    border-radius: 8px !important; color: #6060a0 !important; padding: 7px 18px !important; }
.stTabs [aria-selected="true"] {
    background: #ff6b35 !important; color: #fff !important;
    border-color: #ff6b35 !important; font-weight: 700 !important; }

[data-testid="stExpander"] {
    background: #12121e !important; border: 1px solid #1e1e30 !important; border-radius: 10px !important; }
[data-testid="stExpander"] summary { color: #a0a0c8 !important; }
[data-baseweb="select"], [data-baseweb="select"] * { background: #12121e !important; color: #e8e8ff !important; }
[data-testid="metric-container"] {
    background: #12121e !important; border: 1px solid #1e1e30 !important; border-radius: 12px !important; }
[data-testid="metric-container"] * { color: #e8e8ff !important; }
hr { border-color: #1a1a28 !important; }
.stCaption { color: #3a3a5a !important; }

.H   { font-family:'Syne',sans-serif; font-size:2.1rem; font-weight:800; color:#ffffff !important; letter-spacing:-0.02em; display:block; }
.sub { font-size:.95rem; color:#7070b0 !important; margin-top:4px; display:block; }
.kpi { background:#12121e; border:1px solid #1e1e30; border-radius:12px; padding:20px; text-align:center; }
.kv  { font-family:'Syne',sans-serif; font-size:2rem; font-weight:800; color:#ff6b35 !important; line-height:1; }
.kl  { font-size:.68rem; color:#404060 !important; text-transform:uppercase; letter-spacing:.1em; margin-top:6px; }
.sec { font-family:'Syne',sans-serif; font-size:1.05rem; font-weight:700; color:#ffffff !important;
       border-left:4px solid #ff6b35; padding-left:10px; margin:20px 0 12px; }
.ins { background:#0f0f1e; border-left:3px solid #ff6b35; border-radius:0 8px 8px 0;
       padding:10px 14px; font-size:.84rem; color:#9090c8 !important; margin-top:8px; line-height:1.6; }
.card { background:#12121e; border:1px solid #1e1e30; border-radius:12px; padding:18px 20px; }
.mc   { background:#12121e; border:1px solid #1e1e30; border-radius:12px; padding:16px 18px; margin-bottom:10px; }
.bdg  { display:inline-block; padding:2px 10px; border-radius:20px; font-size:.7rem; font-weight:700; }
.bg   { background:#0a2010; color:#4ade80 !important; border:1px solid #14532d; }
.bo   { background:#1e1000; color:#fb923c !important; border:1px solid #6c2e00; }
.br   { background:#1e0808; color:#f87171 !important; border:1px solid #6a0000; }
.wb   { background:#1a1500; border:1px solid #a08000; border-radius:8px;
        padding:11px 14px; font-size:.83rem; color:#d4b800 !important; margin-top:10px; line-height:1.5; }
.sb   { background:#091509; border:1px solid #14532d; border-radius:8px; padding:13px 15px; margin-top:14px; }
.explain { background:#12121e; border:1px solid #1e1e30; border-radius:12px;
           padding:22px 26px; font-size:.9rem; color:#b0b0d8 !important; line-height:1.9; }
</style>
""", unsafe_allow_html=True)


# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("clean_data.csv")
    df["category"] = (df["category"].astype(str)
                      .str.strip("[]").str.replace("'", "", regex=False).str.strip())
    return df

df = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='font-family:Syne,sans-serif;font-size:1.25rem;font-weight:800;
                color:#fff;padding:10px 0 4px;line-height:1.3;'>
        🍽️ Riyadh<br>Restaurants
    </div>
    <div style='font-size:.7rem;color:#2a2a40;padding-bottom:14px;'>Unit 3 · EDA Project</div>
    """, unsafe_allow_html=True)
    st.divider()
    st.markdown("<div style='font-size:.65rem;color:#303050;text-transform:uppercase;letter-spacing:.12em;margin-bottom:8px;'>Navigation</div>", unsafe_allow_html=True)

    pages = {
        "🏠  Overview":           "overview",
        "📊  EDA":                "eda",
        "📍  Geographic Analysis": "geo",
        "🤖  ML Classifier":      "ml",
    }
    if "page" not in st.session_state:
        st.session_state.page = "overview"

    for label, key in pages.items():
        if st.session_state.page == key:
            st.markdown(f"""<style>[data-testid="stSidebar"] div:has(button[data-testid*="n_{key}"]) button,
            [data-testid="stSidebar"] button[key="n_{key}"] {{
                background: #1e1e30 !important; color: #ff6b35 !important;
                border-color: #ff6b35 !important; font-weight: 700 !important; }}
            </style>""", unsafe_allow_html=True)
        if st.button(label, key=f"n_{key}", use_container_width=True):
            st.session_state.page = key

    st.divider()
    st.caption("Data: Kaggle / Foursquare")

page = st.session_state.page

# ── Helpers ───────────────────────────────────────────────────────────────────
BG   = "#0d0d12"; CARD = "#12121e"; GRID = "#1a1a28"
OR = "#ff6b35"; CY = "#00d4ff"; GR = "#4ade80"; YL = "#facc15"; PK = "#f472b6"; PU = "#a78bfa"
TX = "#e8e8ff"
PAL = [OR, CY, GR, YL, PK, PU, "#34d399", "#fb7185", "#38bdf8", "#fbbf24"]

def dax(ax, fig):
    fig.patch.set_facecolor(CARD); ax.set_facecolor(CARD)
    ax.tick_params(colors=TX, labelsize=8)
    for item in [ax.xaxis.label, ax.yaxis.label, ax.title]: item.set_color(TX)
    for sp in ax.spines.values(): sp.set_color(GRID)

def H(t):   st.markdown(f'<div class="H">{t}</div>', unsafe_allow_html=True)
def sub(t): st.markdown(f'<div class="sub">{t}</div>', unsafe_allow_html=True)
def sec(t): st.markdown(f'<div class="sec">{t}</div>', unsafe_allow_html=True)
def ins(t): st.markdown(f'<div class="ins">{t}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "overview":
    H("Riyadh Restaurant Explorer")
    sub("Exploratory analysis of 5,000+ food & dining venues across Riyadh")
    st.markdown("<br>", unsafe_allow_html=True)

    for col, val, label in zip(st.columns(5), [
        f"{len(df):,}", f"{df['rating'].mean():.2f}",
        f"{df['category'].nunique()}", f"{df['neighborhoods'].nunique()}",
        f"{int(df['price_level'].median())}",
    ], ["Total Venues", "Avg Rating", "Categories", "Neighborhoods", "Median Price Level"]):
        col.markdown(f'<div class="kpi"><div class="kv">{val}</div><div class="kl">{label}</div></div>',
                     unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        sec("Distribution of Restaurant Ratings")
        fig, ax = plt.subplots(figsize=(6, 4)); dax(ax, fig)
        sns.histplot(df["rating"], bins=20, kde=True, ax=ax, color=OR, alpha=0.8)
        ax.axvline(df["rating"].mean(),   color="red",  linestyle="--", lw=1.8,
                   label=f"Mean: {df['rating'].mean():.2f}")
        ax.axvline(df["rating"].median(), color="blue", linestyle="-",  lw=1.8,
                   label=f"Median: {df['rating'].median():.2f}")
        leg = ax.legend(fontsize=9); leg.get_frame().set_facecolor(CARD)
        [t.set_color(TX) for t in leg.get_texts()]
        ax.set_xlabel("Rating", fontsize=9); ax.set_ylabel("Count", fontsize=9)
        ax.set_title("Distribution of Restaurant Ratings", fontsize=10)
        st.pyplot(fig, use_container_width=True); plt.close()
        ins("Most restaurants in Riyadh are rated between 7 and 9, with a mean of 7.82 and a median of 7.90, indicating a balanced distribution. Ratings below 6 are uncommon, suggesting the city's dining scene generally maintains above-average quality.")

    with c2:
        sec("Distribution of Price Levels")
        fig, ax = plt.subplots(figsize=(6, 4)); dax(ax, fig)
        price_order = sorted(df["price_level"].dropna().unique())
        sns.countplot(x="price_level", data=df, ax=ax, order=price_order,
                      palette=[OR, CY, GR, YL, PK][:len(price_order)])
        ax.set_xlabel("Price Level", fontsize=9); ax.set_ylabel("Number of Restaurants", fontsize=9)
        ax.set_title("Distribution of Price Levels", fontsize=10)
        st.pyplot(fig, use_container_width=True); plt.close()
        ins("Price level 1 ($) dominates with about 2,400 restaurants — nearly three times more than any other tier. Premium dining is rare: level 3 ($$$) has about 150 restaurants and level 4 ($$$$) is nearly absent. Riyadh's food scene is strongly skewed toward affordable dining.")

    st.divider()
    c1, c2 = st.columns(2)

    with c1:
        sec("Top 10 Restaurant Categories")
        tc = df["category"].value_counts().head(10).reset_index()
        tc.columns = ["category", "count"]
        fig, ax = plt.subplots(figsize=(6, 4.5)); dax(ax, fig)
        sns.barplot(y="category", x="count", data=tc, ax=ax, palette=PAL[:10])
        ax.set_xlabel("Count", fontsize=9); ax.set_ylabel("Category", fontsize=9)
        ax.set_title("Top 10 Restaurant Categories", fontsize=10)
        ax.tick_params(axis='y', labelsize=7)
        st.pyplot(fig, use_container_width=True); plt.close()
        ins("Coffee Shop is the most common category with about 2,050 venues — more than three times the second-ranked Burger Joint (~640). The dominance of Coffee Shops highlights the strong café culture in Riyadh's food scene.")

    with c2:
        sec("Top Neighborhoods by Restaurant Count")
        tn = df["neighborhoods"].value_counts().head(10).reset_index()
        tn.columns = ["neighborhoods", "count"]
        fig, ax = plt.subplots(figsize=(6, 4.5)); dax(ax, fig)
        sns.barplot(y="neighborhoods", x="count", data=tn, ax=ax, palette=PAL[:10])
        ax.set_xlabel("Number of Restaurants", fontsize=9); ax.set_ylabel("Neighborhood", fontsize=9)
        ax.set_title("Top Neighborhoods by Restaurant Count", fontsize=10)
        ax.tick_params(axis='y', labelsize=7)
        st.pyplot(fig, use_container_width=True); plt.close()
        ins("Hiteen District has the highest concentration with about 365 restaurants, followed by Dhahrat Laban (~328) and Al Malqa (~315). Dining activity is heavily concentrated in the northern and northwestern districts of Riyadh.")


# ══════════════════════════════════════════════════════════════════════════════
# EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "eda":
    H("Exploratory Data Analysis")
    sub("Patterns, quality, and correlations across Riyadh's dining venues")
    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🏷️  Categories & Ratings", "📦  Price & Neighborhoods", "🔗  Correlations & Variety"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            sec("Top 10 Restaurant Categories")
            tc = df["category"].value_counts().head(10).reset_index()
            tc.columns = ["category", "count"]
            fig, ax = plt.subplots(figsize=(6, 5)); dax(ax, fig)
            sns.barplot(y="category", x="count", data=tc, ax=ax, palette=PAL[:10])
            ax.set_xlabel("Count", fontsize=9); ax.set_ylabel("Category", fontsize=9)
            ax.set_title("Top 10 Restaurant Categories", fontsize=10)
            ax.tick_params(axis='y', labelsize=7)
            st.pyplot(fig, use_container_width=True); plt.close()
            ins("Coffee Shop leads at ~2,050 venues — more than 3× Burger Joint (~640). Restaurant and Middle Eastern Restaurant follow at ~460 and ~420.")

        with c2:
            sec("Top Rated Restaurant Categories")
            cr = (df.groupby("category")["rating"].mean()
                  .sort_values(ascending=False).head(10).reset_index())
            cr.columns = ["category", "rating"]
            fig, ax = plt.subplots(figsize=(6, 5)); dax(ax, fig)
            sns.barplot(y="category", x="rating", data=cr, ax=ax, palette=PAL[:10])
            ax.set_xlabel("Average Rating", fontsize=9); ax.set_ylabel("Category", fontsize=9)
            ax.set_title("Top Rated Restaurant Categories", fontsize=10)
            ax.tick_params(axis='y', labelsize=7)
            st.pyplot(fig, use_container_width=True); plt.close()
            ins("The top-rated category is Restaurant & Juice Bar at ~9.4. All 10 top-rated categories are multi-tag combinations — venues offering more than one concept tend to rate higher. None of the most popular categories (Coffee Shop, Burger Joint) appear here, confirming popularity ≠ quality.")

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            sec("Rating by Price Level")
            price_order = sorted(df["price_level"].dropna().unique())
            fig, ax = plt.subplots(figsize=(6, 4.5)); dax(ax, fig)
            sns.boxplot(x="price_level", y="rating", data=df, ax=ax,
                        order=price_order,
                        palette=[OR, CY, GR, YL, PK][:len(price_order)])
            ax.set_xlabel("Price Level", fontsize=9); ax.set_ylabel("Rating", fontsize=9)
            ax.set_title("Rating by Price Level", fontsize=10)
            st.pyplot(fig, use_container_width=True); plt.close()
            ins("Price levels 0, 1, and 2 have nearly identical medians around 7.9–8.0. Level 3 ($$$) shows a slightly lower median near 7.7. Higher prices do not correspond to higher ratings — budget and mid-range restaurants achieve comparable customer satisfaction.")

        with c2:
            sec("Ratings Across Top Neighborhoods")
            top10_idx = df["neighborhoods"].value_counts().head(10).index
            fig, ax = plt.subplots(figsize=(6, 4.5)); dax(ax, fig)
            sns.boxplot(data=df[df["neighborhoods"].isin(top10_idx)],
                        x="neighborhoods", y="rating", ax=ax, palette=PAL[:10])
            plt.xticks(rotation=45, ha="right", fontsize=7, color=TX)
            ax.set_xlabel("Neighborhood", fontsize=9); ax.set_ylabel("Rating", fontsize=9)
            ax.set_title("Ratings Across Top Neighborhoods", fontsize=10)
            st.pyplot(fig, use_container_width=True); plt.close()
            ins("Al Qairawan and Al Malqa lead with medians at ~8.2. Al Yasmeen stands out with the widest spread and lowest outlier at ~4.7. Rating quality is fairly uniform across Riyadh regardless of location.")

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            sec("Correlation Between Key Variables")
            fig, ax = plt.subplots(figsize=(6, 4.5)); dax(ax, fig)
            sns.heatmap(df[["rating","price_level","total_photos","total_ratings"]].corr(),
                        annot=True, cmap="coolwarm", ax=ax, fmt=".2f",
                        annot_kws={"size":10,"color":"white"}, linewidths=0.5, linecolor=GRID)
            ax.set_title("Correlation Between Key Variables", fontsize=10)
            ax.tick_params(labelsize=8, colors=TX)
            st.pyplot(fig, use_container_width=True); plt.close()
            ins("Strongest relationship: total_photos ↔ total_ratings (0.61) — both reflect popularity. Rating shows weak positive correlations with photos (0.28) and ratings count (0.25). price_level has almost no correlation with any variable.")

        with c2:
            sec("Top 5 Neighborhoods with the Greatest Food Variety")
            vt = (df.groupby(["neighborhoods","price_level"])["category"].nunique().unstack())
            if not vt.empty:
                top5v = vt.sum(axis=1).sort_values(ascending=False).head(5).index
                fig, ax = plt.subplots(figsize=(6, 4.5)); dax(ax, fig)
                sns.heatmap(vt.loc[top5v], cmap="YlOrRd", annot=True, fmt=".0f",
                            ax=ax, linewidths=0.5, linecolor=GRID, annot_kws={"size":9})
                ax.set_title("Top 5 Neighborhoods with the Greatest Food Variety", fontsize=9)
                ax.set_xlabel("Price Level", fontsize=9); ax.set_ylabel("", fontsize=9)
                ax.tick_params(labelsize=8, colors=TX)
                st.pyplot(fig, use_container_width=True); plt.close()
            ins("Al Malqa leads with 42 unique categories at price level 1. Price level 1 ($) is consistently the most diverse tier. Premium dining (level 3) offers minimal variety, maxing out at just 6 categories in Dhahrat Laban.")


# ══════════════════════════════════════════════════════════════════════════════
# GEOGRAPHIC ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "geo":
    H("Geographic Analysis")
    sub("How restaurants are distributed across Riyadh's neighborhoods")
    st.markdown("<br>", unsafe_allow_html=True)

    sec("Restaurant Locations Across Riyadh")
    try:
        import geopandas as gpd, os
        gp = "Saudi-Arabia-Regions-Cities-and-Districts/geojson"
        if not os.path.exists(f"{gp}/cities.geojson"): raise FileNotFoundError
        cities_gdf    = gpd.read_file(f"{gp}/cities.geojson").to_crs(4326)
        districts_gdf = gpd.read_file(f"{gp}/districts.geojson").to_crs(4326)
        riyadh = cities_gdf[cities_gdf["name_en"] == "Riyadh"]
        fig, ax = plt.subplots(figsize=(10, 7))
        fig.patch.set_facecolor(CARD); ax.set_facecolor("#0a0a14")
        riyadh.plot(ax=ax, color="#14142a", edgecolor="#4a4a6e", linewidth=1.2)
        districts_gdf.plot(ax=ax, facecolor="none", edgecolor="#2a2a44", linewidth=0.35)
        ax.scatter(df["longitude"], df["latitude"], color=OR, s=6, alpha=0.35, zorder=3)
        ax.set_xlim(df["longitude"].min()-.01, df["longitude"].max()+.01)
        ax.set_ylim(df["latitude"].min() -.01, df["latitude"].max() +.01)
        ax.set_xlabel("Longitude", fontsize=9, color=TX); ax.set_ylabel("Latitude", fontsize=9, color=TX)
        ax.set_title("Restaurant Locations Across Riyadh", fontsize=11, color=TX)
        ax.tick_params(labelsize=8, colors=TX)
        for sp in ax.spines.values(): sp.set_color(GRID)
    except Exception:
        fig, ax = plt.subplots(figsize=(10, 7)); dax(ax, fig)
        ax.scatter(df["longitude"], df["latitude"], alpha=0.35, s=6, color=OR)
        ax.set_xlabel("Longitude", fontsize=9); ax.set_ylabel("Latitude", fontsize=9)
        ax.set_title("Restaurant Locations Across Riyadh", fontsize=11)
    st.pyplot(fig, use_container_width=True); plt.close()
    ins("The map overlay shows that restaurants are not evenly distributed across Riyadh. The highest concentration appears in the central and northern parts of the city, while the outer areas contain fewer locations. This reflects the concentration of commercial activity and population density.")

    st.divider()
    sec("Explore by Neighborhood")
    all_neighborhoods = sorted(df["neighborhoods"].dropna().unique())
    default_idx = all_neighborhoods.index("Hiteen") if "Hiteen" in all_neighborhoods else 0
    selected = st.selectbox("Select a neighborhood", all_neighborhoods, index=default_idx)
    nd = df[df["neighborhoods"] == selected]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Venues",       f"{len(nd):,}")
    m2.metric("Avg Rating",   f"{nd['rating'].mean():.2f}")
    m3.metric("Categories",   f"{nd['category'].nunique()}")
    m4.metric("Top Category", nd["category"].value_counts().index[0] if len(nd) else "—")

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        sec(f"Top Categories in {selected}")
        tc_n = nd["category"].value_counts().head(10).reset_index()
        tc_n.columns = ["category", "count"]
        fig, ax = plt.subplots(figsize=(6, 4.5)); dax(ax, fig)
        sns.barplot(y="category", x="count", data=tc_n, ax=ax, palette=PAL[:len(tc_n)])
        ax.set_xlabel("Count", fontsize=9); ax.set_ylabel("", fontsize=9)
        ax.set_title(f"Top Categories in {selected}", fontsize=10)
        ax.tick_params(axis='y', labelsize=7)
        st.pyplot(fig, use_container_width=True); plt.close()

    with c2:
        sec(f"Rating Distribution in {selected}")
        fig, ax = plt.subplots(figsize=(6, 4.5)); dax(ax, fig)
        sns.histplot(nd["rating"], bins=15, kde=True, ax=ax, color=CY, alpha=0.8)
        ax.axvline(nd["rating"].mean(),   color="red",  linestyle="--", lw=1.5,
                   label=f"Mean: {nd['rating'].mean():.2f}")
        ax.axvline(nd["rating"].median(), color="blue", linestyle="-",  lw=1.5,
                   label=f"Median: {nd['rating'].median():.2f}")
        leg = ax.legend(fontsize=8); leg.get_frame().set_facecolor(CARD)
        [t.set_color(TX) for t in leg.get_texts()]
        ax.set_xlabel("Rating", fontsize=9); ax.set_ylabel("Count", fontsize=9)
        ax.set_title(f"Rating Distribution in {selected}", fontsize=10)
        st.pyplot(fig, use_container_width=True); plt.close()

    neigh_rank = df["neighborhoods"].value_counts()
    rank_pos   = list(neigh_rank.index).index(selected) + 1 if selected in neigh_rank.index else "?"
    avg_city   = df["rating"].mean()
    avg_neigh  = nd["rating"].mean()
    diff       = avg_neigh - avg_city
    direction  = "above" if diff >= 0 else "below"
    ins(f"{selected} ranks #{rank_pos} in Riyadh by restaurant count with {len(nd):,} venues. "
        f"Its average rating of {avg_neigh:.2f} is {abs(diff):.2f} points {direction} the city average ({avg_city:.2f}). "
        f"The most popular category here is {nd['category'].value_counts().index[0] if len(nd) else '—'}.")

    st.divider()
    sec("Ratings Across Top 10 Neighborhoods")
    top10_idx = df["neighborhoods"].value_counts().head(10).index
    fig, ax = plt.subplots(figsize=(10, 5)); dax(ax, fig)
    sns.boxplot(data=df[df["neighborhoods"].isin(top10_idx)],
                x="neighborhoods", y="rating", ax=ax, palette=PAL[:10])
    plt.xticks(rotation=45, ha="right", fontsize=8, color=TX)
    ax.set_xlabel("Neighborhood", fontsize=9); ax.set_ylabel("Rating", fontsize=9)
    ax.set_title("Ratings Across Top Neighborhoods", fontsize=10)
    st.pyplot(fig, use_container_width=True); plt.close()
    ins("Al Qairawan and Al Malqa lead with the highest medians at ~8.2, while Tuwaiq and Qurtubah sit lowest at ~7.8. Al Yasmeen stands out with the widest spread and the lowest single outlier at ~4.7 — indicating highly inconsistent quality. Hiteen, despite being the most restaurant-dense neighborhood, sits at a modest median of ~7.9.")


# ══════════════════════════════════════════════════════════════════════════════
# ML CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ml":
    H("ML — Price Level Classifier")
    sub("Using supervised learning to predict whether a restaurant is budget, mid, or premium")
    st.markdown("<br>", unsafe_allow_html=True)

    sec("What Are We Trying to Do?")
    st.markdown("""
    <div class='explain'>
        <strong style='color:#ff6b35;'>Goal:</strong> Train a machine learning model to predict the
        <strong style='color:#ffffff;'>price level</strong> of a restaurant (1 = budget, 2 = mid-range, 3 = premium)
        based on features available in the dataset.<br><br>
        <strong style='color:#ff6b35;'>Why is this interesting?</strong> If we can predict price from ratings,
        photos, and location — it means these features signal what kind of restaurant it is.
        If we <em>cannot</em>, it tells us something deeper: that pricing in Riyadh's food scene is
        driven by intangibles not captured in the data.<br><br>
        <strong style='color:#ff6b35;'>Approach:</strong> Three classifiers are trained and compared —
        <span style='color:#ff6b35;'>Random Forest</span>,
        <span style='color:#00d4ff;'>Decision Tree</span>, and
        <span style='color:#f472b6;'>Logistic Regression</span>.
        Class imbalance is handled with <code style='color:#facc15;'>class_weight="balanced"</code>.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    sec("Features & Setup")
    c1, c2 = st.columns([1.1, 1])

    with c1:
        st.markdown("""<div class='card' style='font-size:.88rem;line-height:2.1;'>
            <div><span style='color:#ff6b35;font-weight:700;'>🎯 Task</span>
            <span style='color:#c0c0e0;'> — Multi-class · predict price level 1, 2, or 3</span></div>
            <div><span style='color:#ff6b35;font-weight:700;'>❌ Excluded</span>
            <span style='color:#c0c0e0;'> — Price level 0 (unspecified) removed</span></div>
            <div><span style='color:#ff6b35;font-weight:700;'>⚖️ Imbalance</span>
            <span style='color:#c0c0e0;'> — <code>class_weight="balanced"</code> on all models</span></div>
            <div><span style='color:#ff6b35;font-weight:700;'>✂️ Split</span>
            <span style='color:#c0c0e0;'> — 80 / 20 · random_state=42</span></div>
        </div>""", unsafe_allow_html=True)

    with c2:
        for feat, ftype, bc in [
            ("rating",        "Numerical",                  "bg"),
            ("total_ratings", "Numerical",                  "bg"),
            ("total_photos",  "Numerical",                  "bg"),
            ("total_tips",    "Numerical",                  "bg"),
            ("neighborhoods", "Categorical → LabelEncoded", "bo"),
            ("category",      "Categorical → LabelEncoded", "bo"),
        ]:
            st.markdown(f"""<div style='display:flex;justify-content:space-between;align-items:center;
                padding:7px 12px;background:#12121e;border:1px solid #1e1e30;border-radius:8px;margin-bottom:5px;'>
                <code style='color:#00d4ff;font-size:.84rem;'>{feat}</code>
                <span class='bdg {bc}'>{ftype}</span></div>""", unsafe_allow_html=True)

    st.divider()

    # Train models (cached so it only runs once)
    @st.cache_data
    def train_models():
        feats  = ["rating","total_ratings","total_photos","total_tips","neighborhoods","category"]
        target = "price_level"
        dm = df[feats + [target]].dropna()
        dm = dm[dm[target] != 0].copy()
        dm["neighborhoods"] = LabelEncoder().fit_transform(dm["neighborhoods"])
        dm["category"]      = LabelEncoder().fit_transform(dm["category"])

        X = dm[feats]; y = dm[target]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

        mdls = {
            "Random Forest":       RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
            "Decision Tree":       DecisionTreeClassifier(class_weight="balanced", random_state=42),
            "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=2000, solver="saga", random_state=42),
        }
        preds = {}
        for name, m in mdls.items():
            m.fit(X_tr, y_tr); preds[name] = m.predict(X_te)

        rf = mdls["Random Forest"]
        fi = pd.Series(rf.feature_importances_, index=feats).sort_values()
        cm = confusion_matrix(y_te, preds["Random Forest"])
        reports = {n: classification_report(y_te, p, labels=[1,2,3],
                       target_names=["Low (1)","Mid (2)","High (3)"],
                       zero_division=0, output_dict=True)
                   for n, p in preds.items()}
        class_dist = dm[target].value_counts().sort_index()
        return reports, fi, cm, class_dist

    with st.spinner("Training models…"):
        reports, feat_imp, cm, class_dist = train_models()

    # Class distribution
    sec("Class Distribution")
    c1, c2 = st.columns([1, 1.6])
    with c1:
        fig, ax = plt.subplots(figsize=(5, 3.5)); dax(ax, fig)
        labels = ["Low (1)", "Mid (2)", "High (3)"]
        counts = [int(class_dist.get(k, 0)) for k in [1, 2, 3]]
        bars = ax.bar(labels, counts, color=[OR, CY, GR], edgecolor=BG)
        for bar, v in zip(bars, counts):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+20,
                    f"{v:,}", ha='center', fontsize=9, fontweight='bold', color=TX)
        ax.set_ylabel("Count", fontsize=9); ax.set_title("Class Distribution", fontsize=10)
        st.pyplot(fig, use_container_width=True); plt.close()
    with c2:
        st.markdown("""<div class='wb' style='margin-top:0;'>
            ⚠️ <strong>Severe class imbalance:</strong> Class 1 ($) has roughly 15× more samples than
            Class 3 ($$$). Without correction, a model would simply predict "budget" every time and
            appear accurate. We address this with <code>class_weight="balanced"</code>, which forces
            each model to pay proportionally more attention to rare classes during training.
        </div>""", unsafe_allow_html=True)

    st.divider()
    sec("Model Results")
    mc1, mc2, mc3 = st.columns(3)
    model_meta = [
        {"name":"Random Forest",      "badge":"bg","btxt":"Best Overall",  "accent":OR,  "col":mc1,
         "note":"Strong on class 1 but largely ignores minority classes due to imbalance."},
        {"name":"Decision Tree",      "badge":"bo","btxt":"Most Balanced", "accent":CY,  "col":mc2,
         "note":"Lower accuracy but attempts all classes. Better at predicting mid-range."},
        {"name":"Logistic Regression","badge":"br","btxt":"Weakest",       "accent":PK,  "col":mc3,
         "note":"Features are not linearly separable. Converged poorly despite max_iter=2000."},
    ]
    for m in model_meta:
        rep = reports[m["name"]]
        acc = f"{rep['accuracy']*100:.0f}%"
        rows = [(c, rep[c]["precision"], rep[c]["recall"], rep[c]["f1-score"])
                for c in ["Low (1)","Mid (2)","High (3)"]]
        rows_html = "".join([
            f"<tr style='border-bottom:1px solid #1a1a28;'>"
            f"<td style='padding:5px 0;color:#505070;font-size:.78rem;'>{cls}</td>"
            f"<td style='text-align:center;font-size:.78rem;color:#b0b0d0;'>{p:.2f}</td>"
            f"<td style='text-align:center;font-size:.78rem;color:#b0b0d0;'>{r:.2f}</td>"
            f"<td style='text-align:center;font-size:.78rem;font-weight:800;color:{m['accent']};'>{f1:.2f}</td></tr>"
            for cls, p, r, f1 in rows
        ])
        with m["col"]:
            st.markdown(f"""<div class='mc' style='border-top:3px solid {m["accent"]};'>
                <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;'>
                    <div style='font-family:Syne,sans-serif;font-weight:700;font-size:.95rem;color:#fff;'>{m['name']}</div>
                    <span class='bdg {m["badge"]}'>{m['btxt']}</span>
                </div>
                <div style='font-family:Syne,sans-serif;font-size:1.7rem;font-weight:800;color:{m["accent"]};line-height:1;'>{acc}</div>
                <div style='font-size:.7rem;color:#303050;margin-bottom:10px;'>Accuracy</div>
                <table style='width:100%;border-collapse:collapse;'>
                    <tr style='color:#303050;border-bottom:1px solid #1a1a28;font-size:.73rem;'>
                        <td>Class</td><td style='text-align:center;'>Prec</td>
                        <td style='text-align:center;'>Rec</td><td style='text-align:center;'>F1</td>
                    </tr>{rows_html}
                </table>
                <div style='margin-top:8px;font-size:.78rem;color:#404060;border-top:1px solid #1a1a28;padding-top:8px;'>{m['note']}</div>
            </div>""", unsafe_allow_html=True)

    st.divider()
    c1, c2 = st.columns(2)

    with c1:
        sec("Confusion Matrix — Random Forest")
        fig, ax = plt.subplots(figsize=(5, 4)); dax(ax, fig)
        disp = ConfusionMatrixDisplay(cm, display_labels=["$ (1)","$$ (2)","$$$ (3)"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        for text in ax.texts: text.set_color("white")
        ax.set_title("Confusion Matrix — Random Forest", fontsize=10)
        ax.tick_params(colors=TX, labelsize=9)
        ax.xaxis.label.set_color(TX); ax.yaxis.label.set_color(TX)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()
        ins("The matrix confirms the model heavily predicts class 1 ($). Most class 2 and 3 restaurants are misclassified as class 1.")

    with c2:
        sec("Feature Importance — Random Forest Classifier")
        fig, ax = plt.subplots(figsize=(6, 4)); dax(ax, fig)
        sns.barplot(y=feat_imp.index, x=feat_imp.values, ax=ax,
                    palette=[PAL[i % len(PAL)] for i in range(len(feat_imp))])
        ax.set_xlabel("Importance Score", fontsize=9)
        ax.set_title("Feature Importance — Random Forest Classifier", fontsize=10)
        ax.tick_params(axis='y', labelsize=8)
        st.pyplot(fig, use_container_width=True); plt.close()
        ins("total_photos and neighborhoods contribute the most to predictions. Rating itself is a surprisingly weak predictor — consistent with the near-zero price↔rating correlation found in EDA.")

    st.divider()
    sec("Key Finding")
    st.markdown("""
    <div class='explain'>
        The EDA correlation heatmap already revealed that
        <strong style='color:#00d4ff;'>price_level has near-zero correlation with all available features</strong>.
        The model is trying to predict something the data does not clearly encode.<br><br>
        Price in Riyadh's dining scene is driven by
        <strong style='color:#ffffff;'>factors not captured in this dataset</strong>:
        <ul style='margin-top:10px;line-height:2.4;color:#7070a0;'>
            <li>Restaurant brand or chain status</li>
            <li>Interior design and ambiance</li>
            <li>Location prestige within a neighborhood</li>
            <li>Menu offering and cuisine depth</li>
        </ul>
        <div class='sb'>
            <div style='font-weight:700;color:#4ade80;margin-bottom:5px;'>✅ Best Model: Random Forest — 72%</div>
            <div style='font-size:.85rem;color:#2a6a3a;line-height:1.7;'>
                Most reliable for identifying budget ($) restaurants.
                Treat mid-range and premium predictions with caution — insufficient signal in the available features.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align:center;color:#1a1a2e;font-size:11px;padding:10px 0;'>"
    "Unit 3 Final Project · Riyadh Restaurant EDA + ML · Data: Kaggle / Foursquare</div>",
    unsafe_allow_html=True
)
