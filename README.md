# 🍽️ Riyadh Restaurant EDA — Unit 3 Final Project

Exploratory data analysis of 26,985 Foursquare places in Riyadh, filtered and cleaned down to 11,187 food and dining venues, enriched with neighborhood geography, and extended with a supervised ML price classifier.

🔗 **Live Dashboard:** [https://riyadh-restaurant-eda.streamlit.app/](https://riyadh-restaurant-eda.streamlit.app/)

---

## 📁 Project Structure

```
EDA_Project/
│
├── EDA_Project.ipynb       # Main notebook — cleaning, EDA, ML
├── clean_data.csv          # Final cleaned dataset (11,187 rows)
├── app.py                  # Streamlit dashboard
├── presentation.pptx       # Final presentation slides
└── README.md               # This file
```

---

## 📊 Dataset

| | |
|---|---|
| **Source** | [27K Riyadh Places Raw — Kaggle](https://www.kaggle.com/datasets/mohammedaldakhil/27k-riyadh-places-raw) |
| **Original size** | 26,985 rows, 17 columns |
| **After filtering** | 11,187 food & dining venues |
| **Rated venues** | 5,311 (have rating data) |
| **Key columns** | `name`, `category`, `rating`, `price`, `total_ratings`, `total_photos`, `total_tips`, `latitude`, `longitude`, `neighborhoods` |

---

## 🗺️ Geographic Enrichment

Neighborhood names were assigned to each restaurant using a GeoJSON file of Saudi Arabia's administrative boundaries.

- **Source:** [Saudi Arabia Regions, Cities and Districts — GitHub](https://github.com/homaily/Saudi-Arabia-Regions-Cities-and-Districts)
- **File used:** `geojson/districts.geojson`
- **Method:** Districts were filtered to Riyadh (`city_id == 3`), then a spatial join (`gpd.sjoin`) was performed against each restaurant's coordinates. 10,418 out of 11,187 matched exactly. A nearest-neighbor fallback (`gpd.sjoin_nearest`) handled the remaining 769 — resulting in 0 unassigned venues.

---

## 🔍 Research Questions

1. What are the most common food venue types in Riyadh?
2. Which categories receive the highest average ratings?
3. Where are restaurants geographically concentrated across the city?
4. Do more expensive restaurants receive better ratings?
5. Which neighborhoods offer the greatest variety of food types, and how does that variety change across price levels?

---

## 🧹 Data Cleaning Pipeline

The cleaning pipeline runs in the following strict order — **order is critical** because each step depends on the previous:

1. **Filter** raw `df` (26,985 rows) to food venues using 53 food keywords → `df_restaurants` (11,187 rows)
2. **Encode price** — count `$` symbols in `price` column → `price_level` integer (from `df_restaurants`, not `df`)
3. **Normalize categories** — map 216 raw Foursquare tags to 35 canonical categories using a priority lookup (`assign_canonical`)
4. **Drop unmapped rows** — `dropna(subset=['category'])` + `reset_index(drop=True)`
5. **Assign neighborhoods** — spatial join with Riyadh district polygons (**must run last** to ensure index alignment)

Additional cleaning on the raw dataset before filtering:
- Removed unnamed CSV index column
- Filled `total_tips` and `total_photos` NaNs with 0
- Fixed negative `total_photos` values (set to 0)
- Replaced empty list strings `[]` in `tips` and `tastes` with NaN (24,976 and 25,126 respectively)

---

## 📈 Visualizations

| # | Chart | Research Question |
|---|---|---|
| 1 | Restaurant Locations Map | Q3 |
| 2 | Rating Distribution Histogram | General |
| 3 | Price Level Countplot | General |
| 4 | Top 10 Categories Bar Chart | Q1 |
| 5 | Top Neighborhoods Bar Chart | Q3 |
| 6 | Rating by Price Level Boxplot | Q4 |
| 7 | Ratings Across Top Neighborhoods Boxplot | Q3 |
| 8 | Top Rated Categories Bar Chart | Q2 |
| 9 | Food Variety by Neighborhood Heatmap | Q5 |
| 10 | Correlation Heatmap | General |

---

## 🔑 Key Insights

- **Coffee Shops dominate** — most common category by far, more than 3× the second-ranked Burger Joint
- **Ratings are consistently high** — mean 7.82, median 7.90 across 5,311 rated venues; very few fall below 6.0
- **Price has no meaningful correlation with rating** — levels 1, 2, and 3 share nearly identical median ratings (~7.8–8.0)
- **Northern Riyadh leads in density** — Hiteen, Dhahrat Laban, and Al Malqa are the top 3 neighborhoods
- **Hiteen leads in food variety** — 46 unique categories total; Qurtubah has the highest single price-tier cell (16 categories at price level 1)
- **total_photos ↔ total_ratings correlation: 0.62** — popularity signals cluster together but don't signal quality
- **Tea Rooms, Salad & Health Food, and Sushi & Japanese** are the top-rated categories by average score

---

## 🤖 Machine Learning — Price Level Classifier

A supervised classification task was added to predict a restaurant's price level (1, 2, or 3) from available features.

### Dataset Split

| Class | Label | Count | Test Support |
|---|---|---|---|
| 1 | Budget ($) | 3,453 | 692 |
| 2 | Mid-range ($$) | 817 | 169 |
| 3 | Premium ($$$) | 228 | 37 |

Price level 0 (unspecified) was excluded. All models used `class_weight="balanced"` to handle the ~15:1 imbalance between class 1 and class 3.

### Features Used

| Feature | Type |
|---|---|
| `rating` | Numerical |
| `total_ratings` | Numerical |
| `total_photos` | Numerical |
| `total_tips` | Numerical |
| `neighborhoods` | Categorical → LabelEncoded |
| `category` | Categorical → LabelEncoded |

### Results (from notebook — exec 548)

| Model | Weighted Accuracy | Low (1) F1 | Mid (2) F1 | High (3) F1 |
|---|---|---|---|---|
| **Random Forest** | **89%** | 0.93 | 0.79 | 0.44 |
| Decision Tree | 88% | 0.93 | 0.77 | **0.50** |
| Logistic Regression | 23% | 0.20 | 0.33 | 0.06 |

> **Note:** Decision Tree actually outperforms Random Forest on the High ($$$) class (F1=0.50 vs 0.44), making it the better choice when premium restaurant classification matters.

### Key Finding

Despite near-zero correlation between price_level and all features in the EDA heatmap, the models achieved strong weighted accuracy by leveraging the dominant class 1 structure. The real challenge is the minority classes — `total_photos` and `neighborhoods` are the top two predictors by feature importance, while `rating` contributes surprisingly little.


---

## 🛠️ Libraries Used

| Library | Purpose |
|---|---|
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `matplotlib` / `seaborn` | Visualization |
| `geopandas` / `shapely` | Spatial analysis and neighborhood assignment |
| `kagglehub` | Dataset download |
| `streamlit` | Interactive dashboard |
| `scikit-learn` | ML models, encoding, evaluation |
| `ast` | Parsing raw Foursquare category list strings |

---

## 👤 Project Info

| | |
|---|---|
| **Student** | Abdulrahman Bajunaid |
| **Instructor** | Khulud Alshammari |
| **TA** | Abdullah Alharbi |
| **Institution** | Tuwaiq Academy |
| **Unit** | 3 — Exploratory Data Analysis |
