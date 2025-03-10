#  Regression-analysis-based-pricing-Recommendations-for-E-Commerce-Retailers

This project is a Price Recommendation System that fetches product details from various e-commerce websites, analyzes the data, and provides price recommendations and visualization of competitor prices. The system is built using Python, Selenium, Streamlit, and various data analysis and visualization libraries

## Features

**Fetch Product Data**
Scrapes product details from Flipkart, Croma, and Reliance Digital.  
**Data Analysis**
 Cleans and preprocesses the fetched data.  
**Price Recommendation**
 Recommends optimal selling prices based on the analysis.  
**Visualization**
 Provides visual insights into product prices and ratings.

## Project Structure
```bash
price_recommendation/
│
├── analyze.py
├── app.py
├── config.json
├── fetch.py
├── model_selection_training.py
├── product_data.csv
├── requirements.txt
└── visualization.py
```

## Files Description
* **analyze.py**: Contains functions for data preprocessing, price recommendation, and plotting price analysis.
* **app.py**: Main Streamlit application file that sets up the UI and handles user interactions.
* **config.json**: Configuration file containing URLs and XPaths for scraping product data.
fetch.py: Contains functions to set up the WebDriver and fetch product details from different e-commerce websites.
* **model_selection_training.py**: Contains functions for training and evaluating machine learning models for price prediction.
* **product_data.csv**: CSV file to store the fetched product data.
* **requirements.txt**: List of required Python packages.
* **visualization.py**: Contains functions for visualizing the price analysis.

## Installation
### 1. Clone the repository
```bash
git clone https://github.com/yourusername/price_recommendation.git
cd price_recommendation
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv

source venv/bin/activate # On macOS/Linux
venv\Scripts\activate # On Windows
```

### 3. Install dependencies
```
pip install -r requirements.txt
```

### 4. Set up WebDriver:

Ensure you have Chrome installed.  
The webdriver_manager package will automatically handle the ChromeDriver installation.
## Usage

### 1. Run the Streamlit App
```
streamlit run app.py
```

### 2. Interact with the application:

Enter the product name in the sidebar and click **Find Prices** to fetch product data.  
View the fetched product details from Flipkart, Croma, and Reliance Digital.  
Analyze the data and visualize the price distribution and ratings.  
Get price recommendations by entering the cost price and product name.  

## Configuration
* **config.json**: Update the URLs and XPaths if the structure of the target websites changes.
```json
{
    "flipkart": {
        "url": "https://www.flipkart.com/search?q={query}",
        "title_xpath": "//div[contains(@class, 'KzDlHZ')]",
        "price_xpath": "//div[contains(@class, 'Nx9bqj')]",
        "rating_xpath": "//div[contains(@class, 'XQDdHH')]",
        "ratings_count_xpath": "//span[contains(@class, 'Wphh3N')]/span/span[1]"
    },
    "croma": {
        "url": "https://www.croma.com/searchB?q={query}%3Arelevance",
        "title_xpath": "//h3[contains(@class, 'product-title')]",
        "price_xpath": "//span[contains(@class, 'amount')]",
        "product_link_xpath": "//div[contains(@class, 'product')]//a",
        "rating_xpath": "//span[contains(@style, 'color')]",
        "ratings_count_xpath": "//a[contains(@class, 'pr-review')]"
    },
    "reliance": {
        "url": "https://www.reliancedigital.in/products?q={query}&page_no=1&page_size=12&page_type=number",
        "title_xpath": "//div[contains(@class, 'product-card-title')]",
        "price_xpath": "//div[contains(@class, 'price-container')]//div[contains(@class, 'price')]",
        "product_link_xpath": "//div[contains(@class, 'grid')]//a",
        "rating_xpath": "//span[contains(@class, 'rd-feedback-service-average-rating-total-count')]",
        "ratings_count_xpath": "//span[contains(@class, 'rd-feedback-service-jds-desk-body-s')]"
    }
}
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.
