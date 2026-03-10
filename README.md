# 🍽️ Riyadh Restaurant EDA — Unit 3 Final Project

An exploratory data analysis of 27,000+ places in Riyadh, filtered to food and dining venues, with geographic enrichment, cleaning, and insight visualization.

---

## 📁 Project Structure

```
EDA_Project/
│
├── EDA_Project.ipynb       # Main Jupyter Notebook (full analysis)
├── clean_data.csv          # Cleaned and filtered restaurant dataset
├── app.py                  # Streamlit dashboard
├── README.md               # This file
└── presentation.pptx       # Final presentation slides
└── requirements.txt        # Streamlit requirements 
```

---

## 📊 Dataset

- **Source:** [27K Riyadh Places Raw — Kaggle](https://www.kaggle.com/datasets/mohammedaldakhil/27k-riyadh-places-raw)
- **Original size:** ~27,000 rows, 17 columns
- **After filtering:** ~5,000+ food & dining venues
- **Key columns:** `name`, `category`, `rating`, `price`, `total_ratings`, `total_photos`, `latitude`, `longitude`, `neighborhoods`

---

## 🗺️ Geographic Data

To assign each restaurant to a Riyadh neighborhood, the project uses a GeoJSON file containing Saudi Arabia's regions, cities, and district boundaries.

- **Source:** [Saudi Arabia Regions, Cities and Districts — GitHub](https://github.com/homaily/Saudi-Arabia-Regions-Cities-and-Districts/tree/master)
- **File used:** `geojson/districts.geojson`
- **How it was used:** The district polygons were filtered to Riyadh (city_id == 3), then a spatial join (`gpd.sjoin`) was performed against restaurant coordinates to assign each venue its neighborhood name. A nearest-neighbor fallback was applied for any points that fell outside polygon boundaries.

---

## 🔍 Research Questions

1. What are the most common food place types in Riyadh?
2. Which categories have the highest ratings?
3. Where are restaurants geographically concentrated?
4. Do expensive restaurants receive better ratings?
5. Where in Riyadh can people find the greatest variety of food types, and how does that variety change across price levels?

---

## 🧹 Data Cleaning Steps

- Removed the unnamed CSV index column
- Filled `total_tips` and `total_photos` NaNs with 0
- Fixed negative `total_photos` values (set to 0)
- Replaced empty list strings `[]` in `tips` and `tastes` with NaN
- Encoded `price` column into numeric `price_level` (count of `$` symbols)
- Filtered dataset to food-related venues using 34 food keywords
- Added `neighborhoods` column via spatial join with Riyadh district GeoJSON

---

## 📈 Visualizations

| # | Chart | Answers |
|---|-------|---------|
| 1 | Restaurant Locations Map | Q3 |
| 2 | Rating Distribution Histogram | General |
| 3 | Price Level Countplot | General |
| 4 | Top 10 Categories Bar Chart | Q1 |
| 5 | Top Neighborhoods Bar Chart | Q3 |
| 6 | Rating by Price Level Boxplot | Q4 |
| 7 | Ratings Across Top Neighborhoods Boxplot | Q3 |
| 8 | Top Rated Categories Bar Chart | Q2 |
| 9 | Food Variety Heatmap | Q5 |
| 10 | Correlation Heatmap | General |

---

## 🚀 How to Run

### Jupyter Notebook (Google Colab)
1. Open `EDA_Project.ipynb` in Google Colab
2. Run Cell 1 to install dependencies and download the dataset
3. Wait for the Kaggle download to complete before proceeding
4. Run all remaining cells in order

### Streamlit Dashboard

🔗 **Live App:** [https://riyadh-restaurant-eda.streamlit.app/](https://riyadh-restaurant-eda.streamlit.app/)

---

## 🔑 Key Insights

- Coffee Shops dominate Riyadh's food scene with ~2,050 venues — 3x more than Burger Joints
- Ratings are consistently high across the city (mean 7.82, median 7.90)
- Price has virtually no correlation with rating (-0.0076)
- Al Malqa District offers the greatest food variety (42 unique categories at price level 1)
- Hiteen District has the most restaurants (~365) but an average median rating of ~7.9
- Multi-concept venues (e.g. Restaurant & Juice Bar) consistently outrate single-type venues

---

## 🛠️ Libraries Used

- `pandas` — data manipulation
- `numpy` — numerical operations
- `matplotlib` / `seaborn` — visualization
- `geopandas` / `shapely` — spatial analysis
- `kagglehub` — dataset download
- `streamlit` — interactive dashboard
