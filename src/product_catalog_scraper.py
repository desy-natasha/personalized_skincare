import sys
import time
import re
import pandas as pd
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from urllib.parse import unquote, urlparse, parse_qs
from selenium.common.exceptions import TimeoutException, WebDriverException
from bs4 import BeautifulSoup

BASE_URL = "https://www.chemistwarehouse.com.au"

def setup_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
    options.add_argument("--window-size=1920,1080")
    
    return webdriver.Chrome(options=options)

def get_product_links(driver, page_url):

    driver.get(page_url)
    time.sleep(1)

    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.mt-space-heading-body"))
        )
    except TimeoutException:
        return []

    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")

    # Retrieve data for each product block saved within the "mt-space-heading-body" divider
    product_containers = soup.select('div.mt-space-heading-body')
    
    products = []

    # Get product name and link
    for container in product_containers:
        a_tag = container.select_one('p.body-s a.focus-visible\\:outline-none')

        img_tag = soup.find("img", src=lambda x: x and "/_next/image" in x)

        if a_tag:
            # 1. Product name
            name = a_tag.text.strip()
            
            # 2. URL
            href = a_tag.get('href')
            full_url = BASE_URL + href if href else None
            
            # 3. Image
            img_url = None
            
            if img_tag:
                src = img_tag.get("src")
                try:
                    parsed_url = urlparse(src)
                    query_params = parse_qs(parsed_url.query)

                    if "url" in query_params:
                        img_url = unquote(query_params["url"][0])
                except:
                    img_url = src

        products.append((name, full_url, img_url))

    return products

def is_valid_url(url):
    if not url or not isinstance(url, str):
        return False
    if not url.startswith('http'):
        return False
    
    # URL validation
    try:
        parsed = urlparse(url)
        return bool(parsed.netloc and parsed.scheme)
    except:
        return False

def get_product_details(driver, product_url):

    if not product_url or not is_valid_url(product_url):
        print(f"Invalid URL: {product_url}")
        return {}


    driver.get(product_url)
    time.sleep(1)
    
    try:
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'h1.headline-xl.text-colour-title-light'))
            )
        except:
            return {}

        soup = BeautifulSoup(driver.page_source, "html.parser")

        details = {
                "name": None,
                "price": None,
                "rating": None,
                "reviews_count": None,
                "general_information": None,
                "ingredients": None
            }

        # 1. Product name
        name_tag = soup.select_one('h1.headline-xl.text-colour-title-light')
        details['name'] = name_tag.get_text(strip=True) if name_tag else None

        # 2. Price
        price_tag = soup.select_one('h2.display-l.text-colour-title-light')
        details['price'] = price_tag.get_text(strip=True) if price_tag else None

        # 3. Ratings and reviews count
        rating_tag = soup.select_one('span.text-colour-subtitle-light')
        if rating_tag:
            match = re.match(r'(\d+\.?\d*)\s*\((\d+)\)', rating_tag.get_text(strip=True))
            if match:
                details['rating'] = float(match.group(1))
                details['reviews_count'] = int(match.group(2))
            else:
                details['rating'] = None
                details['reviews_count'] = None
        else:
            details['rating'] = None
            details['reviews_count'] = None


        # 4. General Information & Ingredients
        buttons = soup.select('h3 > button[aria-label="toggle accordion"]')
        for btn in buttons:
            section_name = btn.get_text(strip=True).lower()
            content_id = btn.get('aria-controls')
            if not content_id:
                continue
            content_div = soup.find("div", id=content_id)
            if not content_div:
                continue
            
            text = content_div.get_text(separator="\n", strip=True)

            if "general information" in section_name:
                details["general_information"] = text

            elif "ingredients" in section_name:
                details["ingredients"] = text

        return details

    except WebDriverException as e:
        print(f"WebDriver error for {product_url}: {str(e)[:100]}...")
        return {}
    except Exception as e:
        print(f"Unexpected error for {product_url}: {str(e)[:100]}...")
        return {}

def main_scrapper_product(page_num=50, save_path='products_list.csv'):
    
    driver = setup_driver()

    try:
        BASE_CATALOG_URL = "https://www.chemistwarehouse.com.au/shop-online/665/skin-care"
        all_products = []
        all_details = []

        # 1. Get product links
        page = 1
        for page in tqdm(range(1, page_num+1), desc="Pages scraped"):
            page_url = f"{BASE_CATALOG_URL}?page={page}"
            products = get_product_links(driver, page_url)
            
            if not products:
                break

            all_products.extend(products)

            page += 1

        print(f"\nTotal products found: {len(all_products)}\n")

        # 2. Get product details
        for product in tqdm(all_products,desc="Scraping products", unit="product"):
            name, link, img_url = product

            if not is_valid_url(link):
                print(f"Invalid URL: {link}")
                continue

            details = get_product_details(driver, link)
            time.sleep(1)

            details["url"] = link
            details["img_url"] = img_url

            all_details.append(details)


       # 3. Save results
        if all_details:
            df = pd.DataFrame(all_details)
            if save_path:
                try:
                    df.to_csv(save_path, index=False, encoding='utf-8')
                    print(f"Data saved to {save_path}")
                except Exception as e:
                    print(f"Failed to save CSV: {e}")
                    return all_details 
            return df
        
        return []

    finally:
        driver.quit()
