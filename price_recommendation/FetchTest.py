import unittest
from unittest.mock import patch, MagicMock
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException, NoSuchElementException, UnexpectedAlertPresentException
from selenium.common.exceptions import NoAlertPresentException
import re

class TestWebfetch(unittest.TestCase):
    
    @patch('selenium.webdriver.Chrome')
    def setUp(self, MockWebDriver):
        """Set up a mock WebDriver before each test."""
        self.mock_driver = MockWebDriver()
        self.mock_driver.get = MagicMock()
        self.mock_driver.find_elements = MagicMock()
        self.mock_driver.find_element = MagicMock()
        self.mock_driver.execute_script = MagicMock()
        self.mock_driver.switch_to.window = MagicMock()
        self.mock_driver.close = MagicMock()
        self.mock_driver.switch_to.alert = MagicMock()
    
    def test_fetch_flipkart_products_valid(self):
        from fetch import fetch_flipkart_products
        
        mock_titles = [MagicMock(text='Product 1'), MagicMock(text='Product 2')]
        mock_prices = [MagicMock(text='₹10,000'), MagicMock(text='₹12,000')]
        mock_ratings = [MagicMock(text='4.5'), MagicMock(text='4.2')]
        mock_ratings_count = [MagicMock(text='1000'), MagicMock(text='500')]
        
        self.mock_driver.find_elements.side_effect = lambda by, xpath: (
            mock_titles if 'title' in xpath else 
            mock_prices if 'price' in xpath else 
            mock_ratings if 'rating' in xpath else 
            mock_ratings_count if 'ratings_count' in xpath else []
        )
        
        products = fetch_flipkart_products(self.mock_driver, "http://example.com", "title", "price", "rating", "ratings_count")
        print(products)
        self.assertEqual(len(products), 2)
        self.assertEqual(len(products[0]), 4)
    
    def test_fetch_flipkart_products_no_data(self):
        from fetch import fetch_flipkart_products
        
        self.mock_driver.find_elements.side_effect = lambda by, xpath: []
        
        products = fetch_flipkart_products(self.mock_driver, "http://example.com", "title", "price", "rating", "ratings_count")
        
        self.assertEqual(len(products), 1)
        self.assertEqual(products[0][0], 'Error')
    
    def test_fetch_croma_products_invalid_page(self):
        from fetch import fetch_croma_products
        
        self.mock_driver.find_elements.side_effect = lambda by, xpath: []
        
        products = fetch_croma_products(self.mock_driver, "http://example.com", "title", "price", "link", "rating", "ratings_count")
        print(products ,'fetch_croma_products_invalid_page')
        self.assertEqual(len(products), 1)
        self.assertEqual(products[0][0], 'Error')

    def test_fetch_reliance_products_invalid_page(self):
        from fetch import fetch_reliance_products
        
        self.mock_driver.find_elements.side_effect = lambda by, xpath: []
        
        products = fetch_reliance_products(self.mock_driver, "http://example.com", "title", "price", "link", "rating", "ratings_count")
        print(products ,'fetch_croma_products_invalid_page')
        self.assertEqual(len(products), 1)
        self.assertEqual(products[0][0], 'Error')

    @patch("fetch.handle_popup")  # Correctly patch handle_popup
    def test_fetch_reliance_products_with_alerts(self, mock_handle_popup):
        """Test Reliance Digital product fetching with simulated unexpected alerts."""
        from fetch import fetch_reliance_products

        # Mock alert dismissal
        self.mock_driver.switch_to.alert.dismiss = MagicMock()

        # Simulate handle_popup actually handling an alert
        mock_handle_popup.side_effect = lambda driver: driver.switch_to.alert.dismiss()

        # Mock find_elements to return valid product data
        def mock_find_elements(by, xpath):
            if xpath == "title":
                return [MagicMock(text="Reliance Product 1")]
            elif xpath == "price":
                return [MagicMock(text="₹999")]
            elif xpath == "link":
                return [MagicMock(get_attribute=MagicMock(return_value="http://example.com/product"))]
            elif xpath == "rating":
                return [MagicMock(text="4.5")]
            elif xpath == "ratings_count":
                return [MagicMock(text="(200 Ratings)")]
            return []
        
        self.mock_driver.find_elements.side_effect = mock_find_elements

        # Run function
        products = fetch_reliance_products(self.mock_driver, "http://example.com", "title", "price", "link", "rating", "ratings_count")

        # ✅ Ensure handle_popup was actually called
        # mock_handle_popup.assert_called_once()

        # ✅ Ensure alert.dismiss() was called
        self.assertEqual(mock_handle_popup.call_count, 2)

        # ✅ Check if product data is returned correctly
        print(products)
        self.assertEqual(len(products), 1) 
    

    def test_handle_popup_detects_and_closes_alert(self):
        from fetch import handle_popup
        
        self.mock_driver.switch_to.alert.text = "Test Alert"
        self.mock_driver.switch_to.alert.dismiss = MagicMock()
        
        handle_popup(self.mock_driver)
        self.mock_driver.switch_to.alert.dismiss.assert_called()
    

    @patch('fetch.WebDriverWait')
    def test_handle_popup_no_alert(self, mock_wait):
        from fetch import handle_popup  # Ensure correct import

        wd = MagicMock()
        
        # Simulate no alert being present
        mock_wait.return_value.until.side_effect = TimeoutException
        
        # Simulate no modal popups being found
        wd.find_element.side_effect = NoSuchElementException
        
        try:
            handle_popup(wd)  # Should complete without errors
        except Exception as e:
            self.fail(f"handle_popup raised an exception unexpectedly: {e}")
    
if __name__ == '__main__':
    unittest.main()
