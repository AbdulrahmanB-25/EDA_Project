import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.rcParams["text.usetex"] = False
mpl.rcParams["mathtext.default"] = "regular"

st.set_page_config(page_title="Riyadh Restaurants", page_icon="🍽️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@400;500&display=swap');
*, html, body { font-family: 'DM Sans', sans-serif !important; }
#MainMenu, footer, header { visibility: hidden; }
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"], .main, .main .block-container { background: #0d0d12 !important; }
.stApp p, .stApp span, .stApp div, .stApp label, .stApp li, .stMarkdown p { color: #d0d0f0 !important; }
[data-testid="stSidebar"] { background: #07070d !important; border-right: 1px solid #1a1a28 !important; }
[data-testid="stSidebar"] * { color: #6060a0 !important; }
[data-testid="stSidebar"] .stButton button { background: transparent !important; color: #6060a0 !important; border: 1px solid #1a1a28 !important; border-radius: 8px !important; text-align: left !important; padding: 8px 12px !important; }
[data-testid="stSidebar"] .stButton button:hover { background: #1a1a2e !important; color: #fff !important; border-color: #ff6b35 !important; }
.nav-active div[data-testid] button { background: #1e1e30 !important; color: #ff6b35 !important; border-color: #ff6b35 !important; font-weight: 700 !important; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; background: transparent !important; }
.stTabs [data-baseweb="tab"] { background: #12121e !important; border: 1px solid #1e1e30 !important; border-radius: 8px !important; color: #6060a0 !important; padding: 7px 16px !important; }
.stTabs [aria-selected="true"] { background: #ff6b35 !important; color: #fff !important; border-color: #ff6b35 !important; font-weight: 700 !important; }
[data-testid="stExpander"] { background: #12121e !important; border: 1px solid #1e1e30 !important; border-radius: 10px !important; }
[data-testid="stExpander"] summary { color: #a0a0c8 !important; }
[data-baseweb="select"], [data-baseweb="select"] * { background: #12121e !important; color: #d0d0f0 !important; }
[data-testid="metric-container"] { background: #12121e !important; border: 1px solid #1e1e30 !important; border-radius: 12px !important; }
[data-testid="metric-container"] * { color: #d0d0f0 !important; }
hr { border-color: #1a1a28 !important; }
.stCaption { color: #3a3a5a !important; }
.H  { font-family:'Syne',sans-serif; font-size:2.2rem; font-weight:800; color:#fff !important; letter-spacing:-0.02em; display:block; text-align:left; }
.sub{ font-size:.95rem; color:#8080b0 !important; margin-top:4px; display:block; text-align:left; }
.kpi{ background:#12121e; border:1px solid #1e1e30; border-radius:12px; padding:18px 20px; text-align:center; }
.kv { font-family:'Syne',sans-serif; font-size:1.9rem; font-weight:800; color:#ff6b35 !important; line-height:1; }
.kl { font-size:.7rem; color:#3a3a5a !important; text-transform:uppercase; letter-spacing:.1em; margin-top:5px; }
.sec{ font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:700; color:#fff !important; border-left:4px solid #ff6b35; padding-left:10px; margin:18px 0 12px; }
.ins{ background:#12121e; border-left:3px solid #ff6b35; border-radius:0 8px 8px 0; padding:9px 13px; font-size:.82rem; color:#7070a0 !important; margin-top:6px; line-height:1.5; }
.card{ background:#12121e; border:1px solid #1e1e30; border-radius:12px; padding:18px; }
.mc  { background:#12121e; border:1px solid #1e1e30; border-radius:12px; padding:16px 18px; margin-bottom:10px; }
.mn  { font-family:'Syne',sans-serif; font-weight:700; font-size:.95rem; color:#fff !important; }
.bdg { display:inline-block; padding:2px 9px; border-radius:20px; font-size:.7rem; font-weight:700; }
.bg  { background:#0a2010; color:#4ade80 !important; border:1px solid #14532d; }
.bo  { background:#1e1000; color:#fb923c !important; border:1px solid #6c2e00; }
.br  { background:#1e0808; color:#f87171 !important; border:1px solid #6a0000; }
.wb  { background:#1a1500; border:1px solid #a08000; border-radius:8px; padding:11px 14px; font-size:.82rem; color:#d4b800 !important; margin-top:10px; }
.sb  { background:#091509; border:1px solid #14532d; border-radius:8px; padding:13px 15px; margin-top:12px; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    df = pd.read_csv("clean_data.csv")
    df["category"] = df["category"].astype(str).str.strip("[]").str.replace("'", "", regex=False).str.strip()
    return df


df = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div style='font-family:Syne,sans-serif;font-size:1.2rem;font-weight:800;color:#fff;padding:8px 0 14px'>🍽️ Riyadh<br>Restaurants</div>", unsafe_allow_html=True)
    st.divider()
    st.markdown("<div style='font-size:.68rem;color:#5050a0;text-transform:uppercase;letter-spacing:.12em;margin-bottom:6px'>Navigation</div>", unsafe_allow_html=True)
    pages = {"🏠 Overview": "overview", "📊 EDA": "eda", "📍 Geography": "geography", "💰 Price & Quality": "price", "🤖 ML": "ml"}
    if "page" not in st.session_state:
        st.session_state.page = "overview"
    for label, key in pages.items():
        active = st.session_state.page == key
        if active:
            st.markdown(f"""<style>[data-testid="stSidebar"] div[data-testid="stButton"][key="n_{key}"] button
                {{background:#1e1e30!important;color:#ff6b35!important;border:1px solid #ff6b35!important;font-weight:700!important;}}</style>""", unsafe_allow_html=True)
        if st.button(label, key=f'n_{key}', use_container_width=True):
            st.session_state.page = key
    st.divider()
    price_opts = sorted(df["price_level"].dropna().unique().tolist())
    sel_price = st.multiselect("Price Level", price_opts, price_opts,
        format_func=lambda x: {0: "Unspecified", 1: "Budget", 2: "Mid", 3: "High"}.get(int(x), str(x)))
    rmin, rmax = float(df["rating"].min()), float(df["rating"].max())
    rng = st.slider("Rating", rmin, rmax, (rmin, rmax), 0.1)
    sel_neigh = st.multiselect("Neighborhoods", sorted(df["neighborhoods"].value_counts().head(50).index))
    st.divider()
    st.caption("Kaggle / Foursquare")

f = df[df["price_level"].isin(sel_price) & df["rating"].between(rng[0], rng[1])]
if sel_neigh:
    f = f[f["neighborhoods"].isin(sel_neigh)]
page = st.session_state.page

# ── Colors & helpers ──────────────────────────────────────────────────────────
BG = "#0d0d12"; CARD = "#12121e"; GRID = "#1a1a28"
OR = "#ff6b35"; CY = "#00d4ff"; GR = "#4ade80"; YL = "#facc15"; PK = "#f472b6"
TX = "#d0d0f0"; MU = "#2a2a40"
PAL = [OR, CY, GR, YL, PK, "#a78bfa", "#34d399"]


def dax(ax, fig):
    fig.patch.set_facecolor(CARD); ax.set_facecolor(CARD)
    ax.tick_params(colors=TX, labelsize=8)
    for lbl in [ax.xaxis.label, ax.yaxis.label, ax.title]:
        lbl.set_color(TX)
    for sp in ax.spines.values():
        sp.set_color(GRID)


def plbl(x):
    return {0: "Unspec", 1: "Low", 2: "Mid", 3: "High", 4: "Ultra"}.get(int(x), str(x))


def H(t):   st.markdown(f'<div class="H">{t}</div>', unsafe_allow_html=True)
def sub(t): st.markdown(f'<div class="sub">{t}</div>', unsafe_allow_html=True)
def sec(t): st.markdown(f'<div class="sec">{t}</div>', unsafe_allow_html=True)
def ins(t): st.markdown(f'<div class="ins">{t}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "overview":
    H("Riyadh Restaurant Explorer")
    sub("5,000+ food & dining venues · EDA + ML Classification")
    st.markdown("<br>", unsafe_allow_html=True)

    cols = st.columns(5)
    for col, val, label in zip(cols, [
        f"{len(f):,}", f"{f['rating'].mean():.2f}",
        f"{f['category'].nunique()}", f"{f['neighborhoods'].nunique()}",
        f"{int(f['price_level'].median())}"
    ], ["Venues", "Avg Rating", "Categories", "Neighborhoods", "Median Price"]):
        col.markdown(f'<div class="kpi"><div class="kv">{val}</div><div class="kl">{label}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()
    c1, c2, c3 = st.columns(3)

    with c1:
        sec("Rating Distribution")
        fig, ax = plt.subplots(figsize=(5, 3)); dax(ax, fig)
        sns.histplot(f["rating"], bins=20, kde=True, ax=ax, color=OR, alpha=0.8)
        ax.axvline(f["rating"].mean(), color=CY, lw=1.5, linestyle="--", label=f"Mean {f['rating'].mean():.2f}")
        leg = ax.legend(fontsize=8); leg.get_frame().set_facecolor(CARD)
        [t.set_color(TX) for t in leg.get_texts()]
        ax.set_xlabel("Rating", fontsize=9); ax.set_ylabel("Count", fontsize=9)
        st.pyplot(fig); plt.close()
        ins("Most venues rate 7–9. Mean 7.82 ≈ Median 7.90.")

    with c2:
        sec("Top 5 Categories")
        t5 = f["category"].value_counts().head(5)
        fig, ax = plt.subplots(figsize=(5, 3)); dax(ax, fig)
        ax.barh(t5.index[::-1], t5.values[::-1], color=PAL[:5][::-1])
        ax.set_xlabel("Count", fontsize=9); ax.tick_params(axis='y', labelsize=7)
        st.pyplot(fig); plt.close()
        ins("Coffee Shop leads at ~2,050 — 3× Burger Joint.")

    with c3:
        sec("Price Level Split")
        pc = f["price_level"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(5, 3)); dax(ax, fig)
        ax.bar([plbl(x) for x in pc.index], pc.values, color=[MU, OR, CY, GR, YL][:len(pc)], edgecolor=BG)
        ax.set_xlabel("Price", fontsize=9); ax.set_ylabel("Count", fontsize=9)
        plt.xticks(rotation=15, ha='right', fontsize=7)
        st.pyplot(fig); plt.close()
        ins("Budget dominates. Premium is under 4%.")


# ══════════════════════════════════════════════════════════════════════════════
# EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "eda":
    H("Exploratory Data Analysis")
    sub("6 research questions answered through visualizations")
    st.markdown("<br>", unsafe_allow_html=True)

    t1, t2, t3 = st.tabs(["📊 Categories", "🏆 Ratings", "🔗 Correlations"])

    with t1:
        c1, c2 = st.columns(2)
        with c1:
            sec("Top 10 Categories")
            tc = f["category"].value_counts().head(10)
            fig, ax = plt.subplots(figsize=(6, 4.5)); dax(ax, fig)
            ax.barh(tc.index[::-1], tc.values[::-1],
                    color=[OR if i == 0 else PAL[i % len(PAL)] for i in range(len(tc))][::-1])
            ax.set_xlabel("Count", fontsize=9); ax.tick_params(axis='y', labelsize=7)
            st.pyplot(fig); plt.close()
            ins("Coffee Shop ~2,050 venues — 3× Burger Joint (~650).")
        with c2:
            sec("Top 10 Highest Rated Categories")
            cr = f.groupby("category")["rating"].mean().sort_values(ascending=False).head(10)
            fig, ax = plt.subplots(figsize=(6, 4.5)); dax(ax, fig)
            ax.barh(cr.index[::-1], cr.values[::-1],
                    color=[CY if i == 0 else PAL[i % len(PAL)] for i in range(len(cr))][::-1])
            ax.set_xlabel("Avg Rating", fontsize=9); ax.set_xlim(0, 10); ax.tick_params(axis='y', labelsize=7)
            st.pyplot(fig); plt.close()
            ins("All top-rated are multi-concept venues (~9.4). Popularity ≠ quality.")

    with t2:
        c1, c2 = st.columns(2)
        with c1:
            sec("Rating by Price Level")
            fig, ax = plt.subplots(figsize=(6, 4)); dax(ax, fig)
            sns.boxplot(x=f["price_level"], y=f["rating"], ax=ax,
                        palette=[MU, OR, CY, GR, YL][:f["price_level"].nunique()])
            ax.set_xlabel("Price Level", fontsize=9); ax.set_ylabel("Rating", fontsize=9)
            st.pyplot(fig); plt.close()
            ins("Levels 0–2 share nearly identical medians (~7.9). Paying more ≠ better rating.")
        with c2:
            sec("Top Neighborhoods — Ratings")
            tni = f["neighborhoods"].value_counts().head(10).index
            fig, ax = plt.subplots(figsize=(6, 4)); dax(ax, fig)
            sns.boxplot(data=f[f["neighborhoods"].isin(tni)], x="neighborhoods", y="rating", ax=ax, color=OR)
            plt.xticks(rotation=45, ha="right", fontsize=7, color=TX)
            ax.set_xlabel(""); ax.set_ylabel("Rating", fontsize=9)
            st.pyplot(fig); plt.close()
            ins("Al Qairawan & Al Malqa lead ~8.2. Quality fairly uniform city-wide.")

    with t3:
        c1, c2 = st.columns(2)
        with c1:
            sec("Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(6, 4)); dax(ax, fig)
            sns.heatmap(f[["rating", "price_level", "total_photos", "total_ratings"]].corr(),
                        annot=True, cmap="RdYlBu_r", ax=ax, fmt=".2f",
                        annot_kws={"size": 10, "color": "white"}, linewidths=0.5, linecolor=GRID)
            ax.tick_params(labelsize=9, colors=TX)
            st.pyplot(fig); plt.close()
            ins("photos ↔ ratings: 0.61. price_level ≈ 0 correlation with everything.")
        with c2:
            sec("Food Variety by Price Level")
            vt = f.groupby(["neighborhoods", "price_level"])["category"].nunique().unstack()
            if not vt.empty:
                top5v = vt.sum(axis=1).sort_values(ascending=False).head(5).index
                fig, ax = plt.subplots(figsize=(6, 4)); dax(ax, fig)
                sns.heatmap(vt.loc[top5v], cmap="YlOrRd", annot=True, fmt=".0f", ax=ax,
                            linewidths=0.5, linecolor=GRID, annot_kws={"size": 9})
                ax.tick_params(labelsize=8, colors=TX); ax.set_xlabel("Price Level", fontsize=9); ax.set_ylabel("")
                st.pyplot(fig); plt.close()
            ins("Al Malqa: 42 categories at price 1. Diversity drops at higher tiers.")


# ══════════════════════════════════════════════════════════════════════════════
# GEOGRAPHY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "geography":
    H("Geographic Analysis")
    sub("Restaurant distribution across Riyadh's neighborhoods")
    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        sec("Top 15 Neighborhoods")
        tn = f["neighborhoods"].value_counts().head(15)
        fig, ax = plt.subplots(figsize=(6, 5.5)); dax(ax, fig)
        ax.barh(tn.index[::-1], tn.values[::-1],
                color=[OR if i < 3 else CY if i < 6 else GR for i in range(len(tn))][::-1])
        ax.set_xlabel("Count", fontsize=9); ax.tick_params(axis='y', labelsize=7)
        st.pyplot(fig); plt.close()
        ins("Hiteen ~365, Dhahrat Laban ~328, Al Malqa ~315.")

    with c2:
        sec("Restaurant Locations Map")
        try:
            import geopandas as gpd, os
            gp = "Saudi-Arabia-Regions-Cities-and-Districts/geojson"
            if not os.path.exists(f"{gp}/cities.geojson"):
                raise FileNotFoundError
            cdf = gpd.read_file(f"{gp}/cities.geojson").to_crs(4326)
            ddf = gpd.read_file(f"{gp}/districts.geojson").to_crs(4326)
            ry  = cdf[cdf["name_en"] == "Riyadh"]
            fig, ax = plt.subplots(figsize=(6, 5.5))
            fig.patch.set_facecolor(CARD); ax.set_facecolor("#0a0a14")
            ry.plot(ax=ax, color="#14142a", edgecolor="#3a3a5e", linewidth=1.5)
            ddf.plot(ax=ax, facecolor="none", edgecolor="#2a2a40", linewidth=0.4)
            ax.scatter(f["longitude"], f["latitude"], color=OR, s=5, alpha=0.5, zorder=3)
            ax.set_xlim(f["longitude"].min() - .01, f["longitude"].max() + .01)
            ax.set_ylim(f["latitude"].min()  - .01, f["latitude"].max()  + .01)
            ax.set_xlabel("Longitude", fontsize=9, color=TX); ax.set_ylabel("Latitude", fontsize=9, color=TX)
            ax.tick_params(labelsize=8, colors=TX)
            for sp in ax.spines.values(): sp.set_color(GRID)
        except Exception:
            fig, ax = plt.subplots(figsize=(6, 5.5)); dax(ax, fig)
            ax.scatter(f["longitude"], f["latitude"], alpha=0.4, s=5, color=OR)
            ax.set_xlabel("Longitude", fontsize=9); ax.set_ylabel("Latitude", fontsize=9)
        st.pyplot(fig); plt.close()
        ins("Concentration in central/northern Riyadh.")

    st.divider()
    sec("Explore a Neighborhood")
    sn = st.selectbox("Select", sorted(f["neighborhoods"].dropna().unique()))
    nd = f[f["neighborhoods"] == sn]
    if len(nd):
        for col, val, lbl in zip(st.columns(4),
            [f"{len(nd):,}", f"{nd['rating'].mean():.2f}", nd['category'].nunique(), nd['category'].value_counts().index[0]],
            ["Venues", "Avg Rating", "Categories", "Top Category"]):
            col.metric(lbl, val)
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(5, 3.5)); dax(ax, fig)
            tcn = nd["category"].value_counts().head(8)
            ax.barh(tcn.index[::-1], tcn.values[::-1],
                    color=[PAL[i % len(PAL)] for i in range(len(tcn))][::-1])
            ax.tick_params(axis='y', labelsize=7); ax.set_xlabel("Count", fontsize=9)
            ax.set_title(f"Categories in {sn}", fontsize=9)
            st.pyplot(fig); plt.close()
        with c2:
            fig, ax = plt.subplots(figsize=(5, 3.5)); dax(ax, fig)
            sns.histplot(nd["rating"], bins=15, kde=True, ax=ax, color=CY, alpha=0.8)
            ax.axvline(nd["rating"].mean(), color=OR, lw=1.5, linestyle="--")
            ax.set_xlabel("Rating", fontsize=9); ax.set_ylabel("Count", fontsize=9)
            ax.set_title(f"Ratings in {sn}", fontsize=9)
            st.pyplot(fig); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PRICE & QUALITY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "price":
    H("Price vs. Quality")
    sub("Does spending more mean eating better?")
    st.markdown("<br>", unsafe_allow_html=True)

    cv = f["price_level"].corr(f["rating"])
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,#0a0a18,#12122a);border-radius:14px;
                padding:24px 32px;margin-bottom:20px;border:1px solid #1e1e38;
                display:flex;align-items:center;gap:28px;'>
        <div>
            <div style='font-family:Syne,sans-serif;font-size:3.2rem;font-weight:800;color:#ff6b35;line-height:1;'>{cv:.4f}</div>
            <div style='color:#404060;font-size:.85rem;margin-top:4px;'>price_level ↔ rating correlation</div>
        </div>
        <div style='color:#7070a0;font-size:.9rem;line-height:1.7;max-width:440px;'>
            Near-zero correlation. Paying more does <strong style='color:#00d4ff;'>not</strong>
            guarantee better food. All price tiers achieve virtually identical satisfaction scores.
        </div>
    </div>""", unsafe_allow_html=True)

    pcols = [MU, OR, CY, GR, YL]
    c1, c2 = st.columns(2)
    with c1:
        sec("Rating Distribution by Price Level")
        fig, ax = plt.subplots(figsize=(6, 4)); dax(ax, fig)
        for i, pl in enumerate(sorted(f["price_level"].dropna().unique())):
            ax.hist(f[f["price_level"] == pl]["rating"].dropna(), bins=15, alpha=0.65,
                    label=plbl(pl), color=pcols[min(i, len(pcols) - 1)])
        leg = ax.legend(fontsize=8); leg.get_frame().set_facecolor(CARD)
        [t.set_color(TX) for t in leg.get_texts()]
        ax.set_xlabel("Rating", fontsize=9); ax.set_ylabel("Count", fontsize=9)
        st.pyplot(fig); plt.close()
        ins("All price levels peak in the same 7.5–8.5 band.")

    with c2:
        sec("Median Rating per Price Level")
        mb = f.groupby("price_level")["rating"].median().reset_index()
        mb["label"] = mb["price_level"].apply(plbl)
        fig, ax = plt.subplots(figsize=(6, 4)); dax(ax, fig)
        ax.bar(mb["label"], mb["rating"],
               color=[pcols[min(i, len(pcols) - 1)] for i in range(len(mb))], edgecolor=BG)
        ax.set_ylim(7, 9); ax.set_xlabel("Price", fontsize=9); ax.set_ylabel("Median Rating", fontsize=9)
        for i, v in enumerate(mb["rating"]):
            ax.text(i, v + .03, f"{v:.2f}", ha='center', fontsize=9, fontweight='bold', color=TX)
        st.pyplot(fig); plt.close()
        ins("All medians fall within 7.7–8.0. Price ≠ quality.")


# ══════════════════════════════════════════════════════════════════════════════
# ML
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ml":
    H("ML — Price Level Classifier")
    sub("Predicting budget / mid / premium using supervised learning")
    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns([1.2, 1])
    with c1:
        sec("Problem Setup")
        st.markdown("""<div class='card' style='font-size:.87rem;line-height:2;'>
            <span style='color:#ff6b35;font-weight:700;'>🎯 Task</span>
            <span style='color:#b0b0d0;'> — Multi-class · predict price level 1, 2, or 3</span><br>
            <span style='color:#ff6b35;font-weight:700;'>❌ Excluded</span>
            <span style='color:#b0b0d0;'> — Price level 0 removed from training</span><br>
            <span style='color:#ff6b35;font-weight:700;'>⚖️ Imbalance</span>
            <span style='color:#b0b0d0;'> — class_weight="balanced" on all models</span><br>
            <span style='color:#ff6b35;font-weight:700;'>✂️ Split</span>
            <span style='color:#b0b0d0;'> — 80 / 20 · random_state=42</span>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="sec" style="margin-top:16px;">Features Used</div>', unsafe_allow_html=True)
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

    with c2:
        sec("Class Distribution")
        cc = df[df["price_level"].isin([1, 2, 3])]["price_level"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(5, 3.5)); dax(ax, fig)
        bars = ax.bar(["Low (1)", "Mid (2)", "High (3)"], cc.values, color=[OR, CY, GR], edgecolor=BG)
        for bar, v in zip(bars, cc.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                    f"{v:,}", ha='center', fontsize=9, fontweight='bold', color=TX)
        ax.set_ylabel("Count", fontsize=9)
        st.pyplot(fig); plt.close()
        st.markdown('<div class="wb">⚠️ <strong>Severe imbalance:</strong> Class 1 has ~15× more samples than Class 3. Handled with <code>class_weight="balanced"</code>.</div>', unsafe_allow_html=True)

    st.divider()
    sec("Model Results")
    mc1, mc2, mc3 = st.columns(3)
    models = [
        {"name": "Random Forest",      "acc": "72%", "badge": "bg", "btxt": "Best Overall",    "accent": OR,
         "rows": [("Low (1)", "0.73", "0.98", "0.84"), ("Mid (2)", "0.29", "0.03", "0.05"), ("High (3)", "0.00", "0.00", "0.00")],
         "note": "Strong on class 1 but ignores minority classes.", "col": mc1},
        {"name": "Decision Tree",      "acc": "59%", "badge": "bo", "btxt": "Most Balanced",   "accent": CY,
         "rows": [("Low (1)", "0.74", "0.73", "0.73"), ("Mid (2)", "0.26", "0.25", "0.26"), ("High (3)", "0.00", "0.00", "0.00")],
         "note": "Lower accuracy but tries all classes.", "col": mc2},
        {"name": "Logistic Regression","acc": "18%", "badge": "br", "btxt": "Weakest",         "accent": PK,
         "rows": [("Low (1)", "0.74", "0.10", "0.18"), ("Mid (2)", "0.22", "0.38", "0.28"), ("High (3)", "0.04", "0.47", "0.07")],
         "note": "Features not linearly separable. Converged poorly.", "col": mc3},
    ]

    for m in models:
        with m["col"]:
            rows_html = "".join([
                f"<tr style='border-bottom:1px solid #1a1a28;'>"
                f"<td style='padding:5px 0;color:#505070;font-size:.78rem;'>{cls}</td>"
                f"<td style='text-align:center;font-size:.78rem;color:#b0b0d0;'>{p}</td>"
                f"<td style='text-align:center;font-size:.78rem;color:#b0b0d0;'>{r}</td>"
                f"<td style='text-align:center;font-size:.78rem;font-weight:800;color:{m['accent']};'>{f1}</td></tr>"
                for cls, p, r, f1 in m["rows"]
            ])
            st.markdown(f"""<div class='mc' style='border-top:3px solid {m["accent"]};'>
                <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;'>
                    <div class='mn'>{m['name']}</div>
                    <span class='bdg {m["badge"]}'>{m['btxt']}</span>
                </div>
                <div style='font-family:Syne,sans-serif;font-size:1.7rem;font-weight:800;color:{m["accent"]};line-height:1;'>{m['acc']}</div>
                <div style='font-size:.7rem;color:#303050;margin-bottom:10px;'>Accuracy</div>
                <table style='width:100%;border-collapse:collapse;'>
                    <tr style='color:#303050;border-bottom:1px solid #1a1a28;font-size:.73rem;'>
                        <td style='padding:3px 0;'>Class</td>
                        <td style='text-align:center;'>Prec</td>
                        <td style='text-align:center;'>Rec</td>
                        <td style='text-align:center;'>F1</td>
                    </tr>{rows_html}
                </table>
                <div style='margin-top:8px;font-size:.78rem;color:#404060;border-top:1px solid #1a1a28;padding-top:8px;'>{m['note']}</div>
            </div>""", unsafe_allow_html=True)

    st.divider()
    c1, c2 = st.columns([1.2, 1])
    with c1:
        sec("Feature Importance — Random Forest")
        fi = pd.Series([0.14, 0.18, 0.22, 0.10, 0.20, 0.16],
                       index=["rating", "total_ratings", "total_photos", "total_tips", "neighborhoods", "category"]).sort_values()
        fig, ax = plt.subplots(figsize=(6, 3.5)); dax(ax, fig)
        ax.barh(fi.index, fi.values,
                color=[OR if v == fi.max() else PAL[i % len(PAL)] for i, v in enumerate(fi.values)],
                edgecolor=BG)
        ax.set_xlabel("Importance Score", fontsize=9)
        st.pyplot(fig); plt.close()
        ins("total_photos and neighborhoods contribute most. Rating is a weak predictor.")

    with c2:
        sec("Why the Models Struggled")
        st.markdown("""<div class='card' style='font-size:.87rem;color:#8080a0;line-height:1.8;'>
            The correlation heatmap showed <strong style='color:#00d4ff;'>price_level has near-zero
            correlation</strong> with all features. Price in Riyadh is driven by factors outside this dataset:
            <ul style='margin-top:8px;line-height:2.2;color:#505070;'>
                <li>Brand / chain status</li>
                <li>Interior design & ambiance</li>
                <li>Location prestige</li>
                <li>Menu & cuisine type</li>
            </ul>
        </div>""", unsafe_allow_html=True)
        st.markdown("""<div class='sb'>
            <div style='font-weight:700;color:#4ade80;margin-bottom:4px;'>✅ Best: Random Forest</div>
            <div style='font-size:.82rem;color:#2a6a3a;line-height:1.6;'>
                72% accuracy. Reliable for budget restaurants — treat mid/high predictions with caution.
            </div>
        </div>""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center;color:#1e1e30;font-size:11px;padding:12px 0;'>Unit 3 Final Project · Riyadh Restaurant EDA + ML · Kaggle / Foursquare</div>", unsafe_allow_html=True)
