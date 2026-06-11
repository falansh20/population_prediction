import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import io

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

st.set_page_config(page_title="World Bank Project", layout="wide")
st.title("🌍 World Bank Data Analysis & Population Prediction")

# ---------------------------
# FETCH DATA
# ---------------------------
@st.cache_data
def fetch_data():
    url = "https://api.worldbank.org/v2/country/all/indicator/SP.POP.TOTL?format=json&per_page=10000"
    res = requests.get(url)
    data = res.json()
    return pd.DataFrame(data[1])

raw_df = fetch_data()

# ---------------------------
# TABS
# ---------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dataset Info",
    "📈 Country Visuals",
    "🌍 Overall Analysis",
    "🤖 Prediction"
])

# ===========================
# TAB 1: DATASET INFO
# ===========================
with tab1:

    st.header("📊 Dataset Information")

    st.subheader("Step 1: First 10 Records")
    st.dataframe(raw_df.head(10))

    st.subheader("Step 2: Describe")
    st.dataframe(raw_df.describe())

    st.subheader("Step 3: Info")
    buffer = io.StringIO()
    raw_df.info(buf=buffer)
    st.text(buffer.getvalue())

    st.subheader("Step 4: Missing Values")
    st.dataframe(raw_df.isnull().sum().to_frame("Missing"))

    st.subheader("Step 5: Handle Missing")
    df_no_missing = raw_df.replace([np.inf, -np.inf], np.nan).dropna()

    st.subheader("Step 6: Missing After Cleaning")
    st.dataframe(df_no_missing.isnull().sum().to_frame("Missing"))

    st.subheader("Step 7: Column Removal")

    cols_removed = ["indicator", "unit", "obs_status", "decimal"]
    df_after_col = df_no_missing.drop(columns=cols_removed)

    st.write("Before:", df_no_missing.columns.tolist())
    st.write("After:", df_after_col.columns.tolist())

    st.subheader("Step 8: Final Dataset")

    df_clean = pd.DataFrame({
        "country": df_after_col["country"].apply(lambda x: x["value"]),
        "Year": df_after_col["date"].astype(int),
        "Population": df_after_col["value"]
    })

    st.dataframe(df_clean.head(10))

# ✅ IMPORTANT FIX
df = df_clean.copy()

# ===========================
# TAB 2: COUNTRY VISUALS
# ===========================
with tab2:

    st.header("📈 Country-wise Analysis")

    country = st.selectbox("Select Country", sorted(df["country"].unique()))
    dff = df[df["country"] == country].sort_values("Year")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Population Trend")
        fig, ax = plt.subplots(figsize=(5,3))
        ax.plot(dff["Year"], dff["Population"])
        ax.set_xlabel("Year")
        ax.set_ylabel("Population")
        st.pyplot(fig)

    with col2:
        st.subheader("Population Distribution")
        fig, ax = plt.subplots(figsize=(5,3))
        sns.histplot(dff["Population"], kde=True, ax=ax)
        ax.set_xlabel("Population")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Growth Rate")
        dff["Growth"] = dff["Population"].pct_change()*100
        fig, ax = plt.subplots(figsize=(5,3))
        ax.plot(dff["Year"], dff["Growth"])
        ax.set_xlabel("Year")
        ax.set_ylabel("% Growth")
        st.pyplot(fig)

    with col4:
        st.subheader("Rolling Avg")
        dff["Rolling"] = dff["Population"].rolling(5).mean()
        fig, ax = plt.subplots(figsize=(5,3))
        ax.plot(dff["Year"], dff["Rolling"])
        ax.set_xlabel("Year")
        ax.set_ylabel("Population Avg")
        st.pyplot(fig)

# ===========================
# TAB 3: OVERALL ANALYSIS
# ===========================
with tab3:

    st.header("🌍 Overall Analysis")

    num_df = df.select_dtypes(include=np.number)

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        sns.heatmap(num_df.cov(), cmap="viridis", ax=ax)
        st.pyplot(fig)

    st.subheader("Top 10 vs Others")

    latest = df[df["Year"] == df["Year"].max()]
    top10 = latest.nlargest(10, "Population")["country"]

    temp = df.copy()
    temp["Category"] = temp["country"].apply(
        lambda x: "Top 10" if x in top10.values else "Others"
    )

    fig, ax = plt.subplots()
    sns.violinplot(x="Category", y="Population", data=temp, ax=ax)
    st.pyplot(fig)

# ===========================
# TAB 4: PREDICTION
# ===========================
with tab4:

    st.header("🤖 Population Prediction")

    country_input = st.selectbox("Select Country", sorted(df["country"].unique()), key="pred")
    year_input = st.number_input("Year", value=2025)

    dff = df[df["country"] == country_input].sort_values("Year")

    X = dff[["Year"]].values
    y = dff["Population"].values

    poly = PolynomialFeatures(2)
    X_poly = poly.fit_transform(X)

    # 🔥 TRAIN-TEST SPLIT
    split = int(len(X_poly) * 0.8)

    X_train, X_test = X_poly[:split], X_poly[split:]
    y_train, y_test = y[:split], y[split:]

    model = LinearRegression()
    model.fit(X_train, y_train)

    # 🔥 METRICS
    y_pred = model.predict(X_test)

    if len(y_test) > 0:
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        st.subheader("📊 Model Performance")
        st.write(f"R² Score: {r2:.4f}")
        st.write(f"RMSE: {rmse:,.0f}")

    # 🔥 FINAL PREDICTION
    year_poly = poly.transform([[year_input]])
    pred = model.predict(year_poly)[0]

    st.success(f"Predicted Population: {int(pred):,}")

    # 🔥 RESIZED + CENTERED GRAPH
    fig, ax = plt.subplots(figsize=(4.8,3))

    ax.scatter(dff["Year"], dff["Population"], s=30, alpha=0.7, label="Actual")

    future = np.arange(dff["Year"].min(), year_input+1).reshape(-1,1)
    ax.plot(future, model.predict(poly.transform(future)), color="red", label="Prediction")

    ax.set_xlabel("Year")
    ax.set_ylabel("Population")
    ax.legend(fontsize=9)

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.pyplot(fig, use_container_width=False)