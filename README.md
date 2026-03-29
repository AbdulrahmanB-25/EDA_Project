# 🍽️ Riyadh Restaurant EDA — Unit 3 Final Project

Exploratory data analysis of 27,000+ Foursquare places in Riyadh, filtered and cleaned down to ~11,000 food and dining venues, enriched with neighborhood geography, and extended with a supervised ML price classifier.

🔗 **Live Dashboard:** [https://riyadh-restaurant-eda.streamlit.app/](https://riyadh-restaurant-eda.streamlit.app/)

---

## 📁 Project Structure

```
EDA_Project/
│
├── EDA_Project.ipynb       # Main notebook — cleaning, EDA, ML
├── clean_data.csv          # Final cleaned dataset (~11k rows)
├── app.py                  # Streamlit dashboard
└── README.md               # This file
```

---

## 📊 Dataset

| | |
|---|---|
| **Source** | [27K Riyadh Places Raw — Kaggle](https://www.kaggle.com/datasets/mohammedaldakhil/27k-riyadh-places-raw) |
| **Original size** | 26,985 rows, 17 columns |
| **After filtering** | ~11,187 food & dining venues |
| **Key columns** | `name`, `category`, `rating`, `price`, `total_ratings`, `total_photos`, `total_tips`, `latitude`, `longitude`, `neighborhoods` |

---

## 🗺️ Geographic Enrichment

Neighborhood names were assigned to each restaurant using a GeoJSON file of Saudi Arabia's administrative boundaries.

- **Source:** [Saudi Arabia Regions, Cities and Districts — GitHub](https://github.com/homaily/Saudi-Arabia-Regions-Cities-and-Districts)
- **File used:** `geojson/districts.geojson`
- **Method:** Districts were filtered to Riyadh (`city_id == 3`), then a spatial join (`gpd.sjoin`) was performed against each restaurant's coordinates. A nearest-neighbor fallback (`gpd.sjoin_nearest`) handled any points that fell outside polygon boundaries — resulting in 0 unassigned venues.

---

## 🔍 Research Questions

1. What are the most common food venue types in Riyadh?
2. Which categories receive the highest average ratings?
3. Where are restaurants geographically concentrated across the city?
4. Do more expensive restaurants receive better ratings?
5. Which neighborhoods offer the greatest variety of food types, and how does that variety change across price levels?

---

## 🧹 Data Cleaning Pipeline

The cleaning pipeline runs in the following order — **order matters** because each step depends on the previous one:

1. **Filter** raw `df` to food venues using 53 food keywords → `df_restaurants` (11,187 rows)
2. **Encode price** — count `$` symbols in `price` column → `price_level` integer
3. **Normalize categories** — map 216 raw Foursquare tags to 35 canonical categories using a priority lookup table (`assign_canonical`)
4. **Drop unmapped rows** — `dropna(subset=['category'])` + `reset_index(drop=True)`
5. **Assign neighborhoods** — spatial join with Riyadh district polygons (must run last to ensure index alignment)

Additional cleaning applied to the raw dataset before filtering:
- Removed unnamed CSV index column
- Filled `total_tips` and `total_photos` NaNs with 0
- Fixed negative `total_photos` values (set to 0)
- Replaced empty list strings `[]` in `tips` and `tastes` with NaN

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

- **Coffee Shops dominate** — ~2,050 venues, more than 3× the second-ranked Burger Joint
- **Ratings are consistently high** — mean 7.82, median 7.90; very few venues fall below 6.0
- **Price has no correlation with rating** — levels 1, 2, and 3 share nearly identical median ratings (~7.9)
- **Northern Riyadh leads in density** — Hiteen, Dhahrat Laban, and Al Malqa are the top 3 neighborhoods
- **Al Malqa offers the most food variety** — 42 unique categories at price level 1, the highest single value in the dataset
- **Popularity does not equal quality** — the most-counted categories never appear in the top-rated list
- **Multi-concept venues rate higher** — niche combinations consistently outperform single-type venues

---

## 🤖 Machine Learning — Price Level Classifier

A supervised classification task was added to predict a restaurant's price level (1, 2, or 3) from available features.

### Features Used

| Feature | Type |
|---|---|
| `rating` | Numerical |
| `total_ratings` | Numerical |
| `total_photos` | Numerical |
| `total_tips` | Numerical |
| `neighborhoods` | Categorical → LabelEncoded |
| `category` | Categorical → LabelEncoded |

Price level 0 (unspecified) was excluded from training. All models used `class_weight="balanced"` to handle severe class imbalance (class 1 has ~15× more samples than class 3).

### Results

| Model | Accuracy | Notes |
|---|---|---|
| **Random Forest** | **72%** | Best overall — reliable for class 1, struggles on minority classes |
| Decision Tree | 59% | More balanced across classes, lower overall accuracy |
| Logistic Regression | ~18% | Features are not linearly separable; converged poorly |

### Why the Models Struggle

The EDA correlation heatmap already revealed that `price_level` has near-zero correlation with every available feature. The models are trying to predict something the data does not clearly encode. Price in Riyadh's food scene is likely driven by brand prestige, interior design, location within a neighborhood, and cuisine depth — none of which are captured in this dataset.

---

## 🚀 How to Run

### Jupyter Notebook (Google Colab)
1. Open `EDA_Project.ipynb` in Google Colab
2. Run Cell 1 to install dependencies (`geopandas`, `shapely`, `imbalanced-learn`) and clone the GeoJSON repo
3. Wait for the Kaggle dataset download to complete before proceeding
4. Run all remaining cells **in order** — the pipeline order is critical

### Streamlit Dashboard (Local)
```bash
pip install streamlit pandas seaborn matplotlib scikit-learn geopandas
streamlit run app.py
```

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
