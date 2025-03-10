import json
import streamlit as st
import pandas as pd
from selenium import webdriver
from fetch import setup_driver, fetch_flipkart_products, fetch_croma_products, fetch_reliance_products
from analyze import save_data_to_csv, preprocess_data, recommend_price
from visualization import plot_price_analysis


# Load XPath configurations
def load_config():
    try:
        with open("config.json", "r") as f:
            return json.load(f)
    except Exception as e:
        st.error("⚠ Error loading XPath configurations. Check 'config.json'.")
        return None


config = load_config()

if not config:
    st.stop()  # Stop execution if config is not loaded

# 🎨 Streamlit UI - Page Config
st.set_page_config(page_title="Price & Rating Comparison", page_icon="📊", layout="wide")

# ✅ Initialize session state variables if they don't exist
if "df_flipkart" not in st.session_state:
    st.session_state.df_flipkart = None
if "df_reliance" not in st.session_state:
    st.session_state.df_reliance = None
if "df_croma" not in st.session_state:
    st.session_state.df_croma = None

# 🎯 Sidebar
st.sidebar.title("🔍 Search & Compare")
product_name = st.sidebar.text_input("Enter Product Name")

if st.sidebar.button("Find Prices", key="find_prices_btn"):
    if product_name.strip():
        st.sidebar.write("⏳ Searching for product...")

        # ✅ Use URLs from config
        try:
            flipkart_url = config["flipkart"]["url"].format(query=product_name.replace(" ", "+"))
            reliance_url = config["reliance"]["url"].format(query=product_name.replace(" ", "%20"))
            croma_url = config["croma"]["url"].format(query=product_name.replace(" ", "%20"))
        except KeyError:
            st.error("⚠ URL configuration missing for one or more sources in config.json.")
            st.stop()

        # Set up WebDriver
        wd = setup_driver()

        # ✅ Fetch product data while handling missing XPaths
        try:
            st.session_state.df_flipkart = pd.DataFrame(
                fetch_flipkart_products(
                    wd,
                    flipkart_url,
                    config["flipkart"]["title_xpath"],
                    config["flipkart"]["price_xpath"],
                    config["flipkart"]["rating_xpath"],
                    config["flipkart"]["ratings_count_xpath"]
                ),
                columns=["Product Title", "Price", "Rating (⭐ out of 5)", "No. of Ratings"]
            )
            st.session_state.df_flipkart["Source"] = "Flipkart"
        except KeyError:
            st.error("⚠ Flipkart XPaths not found in config.json.")

        try:
            st.session_state.df_reliance = pd.DataFrame(
                fetch_reliance_products(
                    wd,
                    reliance_url,
                    config["reliance"]["title_xpath"],
                    config["reliance"]["price_xpath"],
                    config["reliance"]["product_link_xpath"],
                    config["reliance"]["rating_xpath"],
                    config["reliance"]["ratings_count_xpath"]
                ),
                columns=["Product Title", "Price", "Rating (⭐ out of 5)", "No. of Ratings"]
            )
            st.session_state.df_reliance["Source"] = "Reliance Digital"
        except KeyError:
            st.error("⚠ Reliance Digital XPaths not found in config.json.")

        try:
            st.session_state.df_croma = pd.DataFrame(
                fetch_croma_products(
                    wd,
                    croma_url,
                    config["croma"]["title_xpath"],
                    config["croma"]["price_xpath"],
                    config["croma"]["product_link_xpath"],
                    config["croma"]["rating_xpath"],
                    config["croma"]["ratings_count_xpath"]
                ),
                columns=["Product Title", "Price", "Rating (⭐ out of 5)", "No. of Ratings"]
            )
            st.session_state.df_croma["Source"] = "Croma"
        except KeyError:
            st.error("⚠ Croma XPaths not found in config.json.")

        wd.quit()

        # Combine all data and save for analysis
        df_combined = pd.concat([st.session_state.df_flipkart, st.session_state.df_reliance, st.session_state.df_croma], ignore_index=True)
        if not df_combined.empty:
            save_data_to_csv(df_combined)
            st.sidebar.success("✅ Product Data Fetched!")
        else:
            st.sidebar.warning("⚠ No data found for the entered product.")
    else:
        st.sidebar.warning("⚠ Please enter a product name.")

# 🏷 Main Title
st.title("📊 Product Price & Rating Comparison")

# 🔹 Tabs for better organization
tab1, tab2, tab3 = st.tabs(["🛍 Compare Prices", "📈 Price Analysis", "💰 Recommend Price"])

# 📌 Tab 1 - Display Prices from Different Retailers
with tab1:
    st.header("🛒 Product Prices from Online Retailers")

    if product_name:
        st.subheader("🛒 Flipkart")
        if st.session_state.df_flipkart is not None:
            st.table(st.session_state.df_flipkart)
        else:
            st.info("🔍 Search for a product to see Flipkart results.")

        st.subheader("🛒 Reliance Digital")
        if st.session_state.df_reliance is not None:
            st.table(st.session_state.df_reliance)
        else:
            st.info("🔍 Search for a product to see Reliance results.")

        st.subheader("🛒 Croma")
        if st.session_state.df_croma is not None:
            st.table(st.session_state.df_croma)
        else:
            st.info("🔍 Search for a product to see Croma results.")

# 📌 Tab 2 - Analyze Data & Visualize Prices
# 📌 Tab 2 - Analyze Data & Visualize Prices
with tab2:
    st.header("📈 Price & Rating Analysis")

    if st.button("Analyze Data", key="analyze_data_btn"):
        df_cleaned = preprocess_data()
        if df_cleaned is not None:
            st.success("✅ Data cleaned and stored successfully!")
        else:
            st.warning("⚠ No valid data found for analysis.")


    # Show Price Analysis Graphs
    # ✅ Button to trigger the dashboard
    if "show_analysis" not in st.session_state:
        st.session_state.show_analysis = False

    # if st.button("🔍 Show Price Analysis"):
    #     st.session_state.show_analysis = True

    if st.button("Show Price Analysis Graph"): 
        st.session_state.show_analysis = True  
    # 📊 Display dashboard if button was clicked

    if st.session_state.show_analysis:
        try:
            plot_price_analysis(product_name)
        except Exception as e:
            pass

    # if st.button("Show Price Analysis Graph"): 
    #     st.session_state.show_analysis = True   

    if st.session_state.show_analysis:    
        df = preprocess_data()
        if df is not None:
            plot_price_analysis()  # ✅ Pass df to the function
        else:
            st.warning("⚠ No data found for visualization.")




# 📌 Tab 3 - Recommend Price Based on Analysis
with tab3:
    st.header("💰 Recommend Optimal Selling Price")

    cost_price = st.number_input("Enter Cost Price (₹)", min_value=1.0, format="%.2f")
    selected_product = st.text_input("Enter Product Name for Prediction")

    if st.button("Recommend Price", key="recommend_price_btn"):
        if selected_product.strip() and cost_price:
            recommended_price = recommend_price(selected_product, cost_price)  # ✅ Fix: Removed extra 'df' argument

            if recommended_price:
                st.success(f"✅ Recommended Selling Price: ₹{recommended_price:.2f}")
        else:
            st.warning("⚠ Please enter both cost price and product name.")