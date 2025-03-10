import streamlit as st
import pandas as pd
import numpy as np
import os
from model_selection_training import train_and_evaluate_models, visualize_data, load_and_preprocess_data
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, StandardScaler
from fuzzywuzzy import process  # Fuzzy string matching
import seaborn as sns
import plotly.express as px

# File to store data
DATA_FILE = "product_data.csv"

def save_data_to_csv(df):
    df.to_csv(DATA_FILE, index=False, mode='w')
    if os.path.exists(DATA_FILE):
        df.to_csv(DATA_FILE, mode='a', index=False, header=False)
    else:
        df.to_csv(DATA_FILE, index=False)

def preprocess_data():
    if not os.path.exists(DATA_FILE):
        return None

    df = pd.read_csv(DATA_FILE)

    # ✅ Replace unwanted text with NaN
    df.replace(["No Data", "No Rating", "Not Available"], np.nan, inplace=True)

    # ✅ Convert "Rating (⭐ out of 5)" to numeric, handling errors
    df["Rating (⭐ out of 5)"] = pd.to_numeric(df["Rating (⭐ out of 5)"], errors="coerce")

    # ✅ Ensure median rating calculation works
    median_rating = df["Rating (⭐ out of 5)"].dropna().median() if not df["Rating (⭐ out of 5)"].dropna().empty else 4.0
    df["Rating (⭐ out of 5)"].fillna(median_rating, inplace=True)

    # ✅ Convert "No. of Ratings" to integer safely
    df["No. of Ratings"] = pd.to_numeric(df["No. of Ratings"], errors="coerce").fillna(0).astype(int)

    # ✅ Normalize Price column (remove ₹ and commas)
    df["Price"] = df["Price"].replace({",": "", "₹": ""}, regex=True)

    # ✅ Convert Price to float, replacing errors with NaN
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

    # ✅ Remove rows where price is missing (if necessary)
    df.dropna(subset=["Price"], inplace=True)

    return df



def recommend_price(selected_product, cost_price):
    df = preprocess_data()  # Ensure all sources are included
    if df is None or df.empty:
        st.error("⚠ No valid data available for analysis.")
        return None

    selected_product_lower = selected_product.lower().strip()

    # ✅ Step 1: First, check for an exact product name match
    exact_match = df[df["Product Title"].str.lower().str.strip() == selected_product_lower]
    if not exact_match.empty:
        best_match = selected_product
        score = 100  # ✅ Exact match confidence
        df_filtered = exact_match[["Product Title", "No. of Ratings", "Rating (⭐ out of 5)", "Price"]]  # ✅ Ensure only required columns
    else:
        # ✅ Step 2: Extract brand and model number
        words = selected_product.split()
        brand_name = words[0]  # First word is usually the brand
        model_number = re.search(r"[\w\d\-]+$", selected_product)  # Extract model number
        model_number = model_number.group(0) if model_number else ""

        # ✅ Step 3: Filter by Brand Name (from ALL platforms)
        df_filtered = df[df["Product Title"].str.contains(re.escape(brand_name), case=False, na=False)]
        if df_filtered.empty:
            st.error(f"❌ No products found for brand '{brand_name}'.")
            return None

        # ✅ Step 4: Match by Model Number
        best_match = None
        score = 0  # Default confidence score

        if model_number:
            model_filtered = df_filtered[df_filtered["Product Title"].str.contains(re.escape(model_number), case=False, na=False)]
            if not model_filtered.empty:
                best_match = model_filtered["Product Title"].iloc[0]
                score = 95
                df_filtered = model_filtered  # Use model match data
                

        # ✅ Step 5: If No Model Match, Use Keyword Search
        if not best_match:
            keyword_filtered = df_filtered[df_filtered["Product Title"].str.contains(re.escape(" ".join(words[:3])), case=False, na=False)]
            if not keyword_filtered.empty:
                best_match = keyword_filtered["Product Title"].iloc[0]
                score = 90
                df_filtered = keyword_filtered  # Use keyword match data

        # ✅ Step 6: If No Match Found, Use Fuzzy Matching
        if not best_match:
            product_names = df_filtered["Product Title"].tolist()
            matches = process.extractBests(selected_product, product_names, limit=5)
            if matches:
                best_match, score = matches[0]
                df_filtered = df_filtered[df_filtered["Product Title"] == best_match]  # Filter based on match

        if score < 85 or best_match is None:
            st.warning(f"⚠ No close match found (Best match: {best_match}, Confidence: {score}%)")
            return None

    st.info(f"🔍 Best match found: {best_match} (Confidence: {score}%)")

    # ✅ Ensure df_filtered only contains required columns
    
    req = ["Product Title", "No. of Ratings", "Rating (⭐ out of 5)", "Price"]
    df_filtered = df_filtered[req]
    # visualize_data(df_filtered)

    # ✅ Ensure data is available before proceeding
    if df_filtered.empty:
        st.error("❌ Product not found in dataset.")
        return None
    
    # ✅ Use only the required features
    # feature_columns = ["No. of Ratings", "Rating (⭐ out of 5)"]
    # X = df_filtered[feature_columns].copy()
    # y = df_filtered["Price"].copy()

    X,y,df = load_and_preprocess_data(df_filtered)

    # ✅ Handle missing values
    X.fillna(X.median(), inplace=True)
    y.fillna(y.median(), inplace=True)

    # ✅ Train model
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # model = RandomForestRegressor(n_estimators=100, random_state=42)
    # model.fit(X_train, y_train)

    # ✅ Ensure selected product's features are valid
    product_data = df[df["Product Title"] == best_match]

    if product_data.empty:
        st.error("❌ Best match not found in dataset for pricing.")
        return None
    
    print(product_data)


    # ✅ Use the trained model for prediction
    # predicted_price = model.predict(product_features_poly)[0]
    # ✅ Extract only the required numerical features for prediction
    # req_numeric = ["No. of Ratings", "Rating (⭐ out of 5)"]  # Only numerical columns
    # product_features = product_data[req_numeric].iloc[0].values.reshape(1, -1)  # Exclude Product Title

    # # ✅ Ensure feature count matches training data
    # product_features_scaled = scaler.transform(product_features)
    # # ✅ Extract only the required features for prediction
    # product_features = product_data[req].iloc[0].values.reshape(1, -1)
    
    # # # ✅ Ensure feature count matches training data
    # product_features_scaled = scaler.transform(product_features)

    model,poly,scaler2 = train_and_evaluate_models(X_scaled,y)
    feature_columns = ["Rating (⭐ out of 5)", "No. of Ratings"]
    
    # ✅ Extract only the required numerical features
    product_features = product_data[feature_columns].iloc[0].values.reshape(1, -1)

    # ✅ Apply the same MinMaxScaler transformation
    product_features_scaled = scaler2.transform(product_features)

    # ✅ Apply the same PolynomialFeatures transformation
    product_features_poly = poly.transform(product_features_scaled)

    # ✅ Use the trained model for prediction

    predicted_price = model.predict(product_features_poly)[0]

    # ✅ Get median price from ALL platforms
    competitor_price = df_filtered["Price"].median()

    # ✅ Optimize pricing for profitability and competition
    recommended_price = min(max(cost_price * 1.2, predicted_price), competitor_price * 1.1)

    return round(recommended_price, 2)


def plot_price_analysis(df=None):
    df = pd.read_csv("product_data.csv")
    df["Rating"] = df["Rating (⭐ out of 5)"].copy()
    df["Price"] = df["Price"].str.replace("₹", "").str.replace(",", "").astype(float)
    
    df = df.dropna(subset=["Price", "Rating", "Product Title"])  # Drop rows with missing values

    # 🎯 Sidebar
    st.sidebar.header("🔍 Filters")

    # Persist filters across interactions
    if "price_range" not in st.session_state:
        st.session_state.price_range = (int(df["Price"].min()), int(df["Price"].max()))
    if "source_filter" not in st.session_state:
        st.session_state.source_filter = list(df["Source"].unique())
    if "rating_filter" not in st.session_state:
        st.session_state.rating_filter = 4.0

    # Sidebar Filters
    price_range = st.sidebar.slider(
        "💰 Select Price Range",
        int(df["Price"].min()), int(df["Price"].max()),
        st.session_state.price_range
    )
    source_filter = st.sidebar.multiselect(
        "🏬 Select Source",
        df["Source"].unique(),
        st.session_state.source_filter
    )
    rating_filter = st.sidebar.slider(
        "⭐ Select Minimum Rating", 0.0, 5.0,
        st.session_state.rating_filter, 0.1
    )

    # Update session state when filters change
    st.session_state.price_range = price_range
    st.session_state.source_filter = source_filter
    st.session_state.rating_filter = rating_filter

    # Filter Data
    df_filtered = df[
        (df["Price"] >= price_range[0]) &
        (df["Price"] <= price_range[1]) &
        (df["Source"].isin(source_filter)) &
        (df["Rating"] >= rating_filter)
    ]

    # 🏷 Main Title
    st.title("📱 iPhone 16 Product Dashboard")

    # Handle case when no data matches filters
    if df_filtered.empty:
        st.warning("⚠ No products match the selected filters. Try adjusting your criteria.")
        return  # Stop execution if no data

    # Display Data
    st.subheader("📋 Filtered Product Data")
    st.dataframe(df_filtered)

    # 📊 Price Distribution by Product Name
    st.subheader("💰 Price Distribution")
    fig_price = px.box(df_filtered, x="Product Title", y="Price", title="Price Distribution by Product Name", color="Product Title")
    st.plotly_chart(fig_price)

    # ⭐ Ratings Analysis
    st.subheader("⭐ Average Ratings")
    fig_ratings = px.bar(df_filtered, x="Product Title", y="Rating", title="Average Ratings by Product Name", color="Product Title")
    st.plotly_chart(fig_ratings)

    # 📌 Footer
    st.markdown("Developed with ❤️ using Streamlit and Plotly")


    """
    Plot three separate bar graphs for Flipkart, Reliance Digital, and Croma.
    Each graph represents product prices for that source.
    """

    # if df is None or df.empty:
    #     st.warning("⚠ No data available for visualization.")
    #     return

    # # Ensure the column names are correct
    # if "Source" not in df.columns or "Product Title" not in df.columns or "Price" not in df.columns:
    #     st.error("🚨 Missing required columns in the dataset!")
    #     return

    # # Convert "Price" column to numeric for plotting
    # df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

    # # Shorten product names for better readability
    # df["Short Product Title"] = df["Product Title"].apply(lambda x: " ".join(x.split()[:3]) + "...")

    # # Filter data for each source
    # sources = ["Flipkart", "Reliance Digital", "Croma"]

    # for source in sources:
    #     st.subheader(f"📊 {source} Price Analysis")
    #     source_data = df[df["Source"] == source]

    #     if source_data.empty:
    #         st.info(f"🔍 No data available for {source}.")
    #         continue

    #     # Create a figure and axis object explicitly
    #     fig, ax = plt.subplots(figsize=(8, 5))
    #     sns.barplot(x="Short Product Title", y="Price", data=source_data, palette="viridis", ax=ax)
        
    #     ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center", fontsize=10)  # ✅ Labels straight
    #     ax.set_ylabel("Price (₹)")
    #     ax.set_xlabel("Product Title")
    #     ax.set_title(f"{source} - Product Prices", fontsize=12)
    #     plt.tight_layout()

    #     # Display the plot in Streamlit
    #     st.pyplot(fig)  # ✅ Explicitly pass figure






st.title("📊 Price Analysis & Recommendation")

if st.button("Analyze Data"):
    df = preprocess_data()
    if df is not None:
        save_data_to_csv(df)
        st.success("✅ Data cleaned and stored successfully!")
    else:
        st.warning("⚠ No data found to analyze.")

st.subheader("💰 Recommend Selling Price")
cost_price = st.number_input("Enter Cost Price (₹)", min_value=1.0, format="%.2f")
product_name = st.text_input("Enter Product Name for Prediction")

if st.button("Recommend Price"):
    if product_name.strip() and cost_price:
        recommended_price = recommend_price(product_name, cost_price)
        if recommended_price:
            st.success(f"✅ Recommended Selling Price: ₹{recommended_price:.2f}")
    else:
        st.warning("⚠ Please enter both cost price and product name.")

if st.button("Show Price Analysis Graph"):
    plot_price_analysis()
