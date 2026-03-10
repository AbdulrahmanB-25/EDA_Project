import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Riyadh Restaurant Explorer",
    page_icon="🍽️",
    layout="wide"
)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("clean_data.csv")
    # Fix category column: strip list brackets/quotes e.g. ['Coffee Shop'] → Coffee Shop
    df["category"] = df["category"].astype(str).str.strip("[]'\" ")
    return df

df = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🍽️ Riyadh Restaurants")
    st.markdown("""
    **Dataset:** 27K Riyadh Places (Kaggle)  
    **Filtered to:** Food & dining venues  
    **Source:** Foursquare location data  
    """)
    st.divider()

    st.subheader("🔧 Filters")

    # Price level filter
    price_options = sorted(df["price_level"].dropna().unique().tolist())
    selected_prices = st.multiselect(
        "Price Level",
        options=price_options,
        default=price_options,
        format_func=lambda x: "$" * int(x) if x > 0 else "Unspecified"
    )

    # Rating filter
    min_rating, max_rating = float(df["rating"].min()), float(df["rating"].max())
    rating_range = st.slider(
        "Rating Range",
        min_value=min_rating,
        max_value=max_rating,
        value=(min_rating, max_rating),
        step=0.1
    )

    # Neighborhood filter
    top_neighborhoods = df["neighborhoods"].value_counts().head(20).index.tolist()
    selected_neighborhoods = st.multiselect(
        "Neighborhoods (Top 20)",
        options=top_neighborhoods,
        default=[]
    )

    st.divider()
    st.caption("Unit 3 Final Project — EDA Dashboard")

# ── Apply filters ─────────────────────────────────────────────────────────────
filtered = df[
    (df["price_level"].isin(selected_prices)) &
    (df["rating"] >= rating_range[0]) &
    (df["rating"] <= rating_range[1])
]

if selected_neighborhoods:
    filtered = filtered[filtered["neighborhoods"].isin(selected_neighborhoods)]

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🍽️ Riyadh Restaurant Explorer")
st.markdown("Exploratory data analysis of food & dining venues across Riyadh, Saudi Arabia.")
st.divider()

# ── KPI row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Venues", f"{len(filtered):,}")
k2.metric("Avg Rating", f"{filtered['rating'].mean():.2f}")
k3.metric("Unique Categories", f"{filtered['category'].nunique():,}")
k4.metric("Neighborhoods", f"{filtered['neighborhoods'].nunique():,}")

st.divider()

# ── Data preview ──────────────────────────────────────────────────────────────
with st.expander("📋 Data Preview", expanded=False):
    st.dataframe(
        filtered[["name", "category", "rating", "price_level", "neighborhoods", "latitude", "longitude"]].sort_index().head(50),
        use_container_width=True
    )
    st.caption(f"Showing 50 of {len(filtered):,} filtered rows")

with st.expander("📊 Summary Statistics", expanded=False):
    st.dataframe(filtered[["rating", "price_level", "total_photos", "total_ratings", "total_tips"]].describe(), use_container_width=True)

st.divider()

# ── Section 1: Distributions ──────────────────────────────────────────────────
st.subheader("📊 Distributions")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Rating Distribution**")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(filtered["rating"], bins=20, kde=True, ax=ax)
    ax.axvline(filtered["rating"].mean(), color="red", linestyle="--", label=f"Mean: {filtered['rating'].mean():.2f}")
    ax.axvline(filtered["rating"].median(), color="blue", linestyle="-", label=f"Median: {filtered['rating'].median():.2f}")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Count")
    ax.legend()
    st.pyplot(fig)
    plt.close()
    st.caption("Most restaurants score between 7 and 9, with a mean of 7.82 and a median of 7.90.")

with col2:
    st.markdown("**Price Level Distribution**")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x="price_level", data=filtered, ax=ax)
    ax.set_xlabel("Price Level (0=unspecified, 1=$, 2=$$...)")
    ax.set_ylabel("Count")
    st.pyplot(fig)
    plt.close()
    st.caption("Price level 1 ($) dominates with ~2,400 venues. High-end dining (level 3+) is under 4% of all venues.")

st.divider()

# ── Section 2: Categories ─────────────────────────────────────────────────────
st.subheader("🏷️ Categories")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Top 10 Most Common Categories**")
    top_cat = filtered["category"].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(y=top_cat.index, x=top_cat.values, ax=ax)
    ax.set_xlabel("Count")
    ax.set_ylabel("Category")
    st.pyplot(fig)
    plt.close()
    st.caption("Coffee Shop leads with ~2,050 venues — over 3x more than Burger Joint (~640).")

with col2:
    st.markdown("**Top 10 Highest Rated Categories**")
    cat_rating = filtered.groupby("category")["rating"].mean().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(y=cat_rating.index, x=cat_rating.values, ax=ax)
    ax.set_xlabel("Average Rating")
    ax.set_ylabel("Category")
    ax.set_xlim(0, 10)
    st.pyplot(fig)
    plt.close()
    st.caption("All top-rated categories are multi-concept venues, scoring between 9.0–9.4.")

st.divider()

# ── Section 3: Geography ──────────────────────────────────────────────────────
st.subheader("📍 Geography")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Top 10 Neighborhoods by Restaurant Count**")
    top_neigh = filtered["neighborhoods"].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(y=top_neigh.index, x=top_neigh.values, ax=ax)
    ax.set_xlabel("Number of Restaurants")
    ax.set_ylabel("Neighborhood")
    st.pyplot(fig)
    plt.close()
    st.caption("Hiteen District leads with ~365 restaurants. Northern/northwestern districts dominate.")

with col2:
    st.markdown("**Restaurant Scatter Map**")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(
        filtered["longitude"],
        filtered["latitude"],
        alpha=0.3,
        s=6,
        color="red"
    )
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Restaurant Locations in Riyadh")
    st.pyplot(fig)
    plt.close()
    st.caption("Restaurants cluster in the central and northern parts of the city.")

st.divider()

# ── Section 4: Price vs Rating ────────────────────────────────────────────────
st.subheader("💰 Price vs. Quality")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Rating by Price Level**")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(x=filtered["price_level"], y=filtered["rating"], ax=ax)
    ax.set_xlabel("Price Level")
    ax.set_ylabel("Rating")
    st.pyplot(fig)
    plt.close()
    st.caption("Price levels 0–2 share nearly identical medians (~7.9–8.0). Paying more does not yield better ratings.")

with col2:
    st.markdown("**Ratings Across Top Neighborhoods**")
    top_neigh_idx = filtered["neighborhoods"].value_counts().head(10).index
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(
        data=filtered[filtered["neighborhoods"].isin(top_neigh_idx)],
        x="neighborhoods",
        y="rating",
        ax=ax
    )
    plt.xticks(rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("Neighborhood")
    ax.set_ylabel("Rating")
    st.pyplot(fig)
    plt.close()
    st.caption("Al Qairawan and Al Malqa lead at ~8.2 median. Differences are small — quality is fairly uniform city-wide.")

st.divider()

# ── Section 5: Variety & Correlation ─────────────────────────────────────────
st.subheader("🗺️ Food Variety & Correlations")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Top 5 Neighborhoods — Food Variety by Price Level**")
    variety_table = (
        filtered
        .groupby(["neighborhoods", "price_level"])["category"]
        .nunique()
        .unstack()
    )
    if not variety_table.empty:
        top5 = variety_table.sum(axis=1).sort_values(ascending=False).head(5).index
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(variety_table.loc[top5], cmap="YlOrRd", annot=True, fmt=".0f", ax=ax)
        ax.set_xlabel("Price Level")
        ax.set_ylabel("Neighborhood")
        st.pyplot(fig)
        plt.close()
    st.caption("Al Malqa tops variety with 42 unique categories at price level 1. Diversity drops sharply at higher price tiers.")

with col2:
    st.markdown("**Correlation Between Key Variables**")
    fig, ax = plt.subplots(figsize=(6, 4))
    corr = filtered[["rating", "price_level", "total_photos", "total_ratings"]].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax, fmt=".2f")
    st.pyplot(fig)
    plt.close()
    st.caption("total_photos and total_ratings correlate strongly (0.61). Price level has near-zero correlation with everything.")

st.divider()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    "<div style='text-align:center; color:gray; font-size:13px;'>"
    "Unit 3 Final Project · Riyadh Restaurant EDA · Data source: Kaggle / Foursquare"
    "</div>",
    unsafe_allow_html=True
)
