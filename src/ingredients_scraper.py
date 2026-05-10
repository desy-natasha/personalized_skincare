import sys
import time
import pandas as pd
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
from bs4 import BeautifulSoup

BASE_URL = "https://renude.co"

def setup_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
    options.add_argument("--window-size=1920,1080")
    
    return webdriver.Chrome(options=options)

def get_ingredient_links(driver, page_url):
    
    driver.get(page_url)
    time.sleep(1)

    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "a[href^='/ingredients/'] h3"))
        )
    except TimeoutException:
        return []

    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")

    # Retrieve ingredient name and link
    ingredient_containers = soup.select("a[href^='/ingredients/']")
    ingredients = []

    for container in ingredient_containers:
        name_tag = container.select_one("h3")
        if name_tag:
            name = name_tag.get_text(strip=True)
            link = BASE_URL + container.get('href')
            ingredients.append((name, link))
    
    return ingredients

def get_ingredient_details(driver, ingredient_url):

    if not ingredient_url or not is_valid_url(ingredient_url):
        print(f"Invalid URL: {ingredient_url}")
        return {}
    
    driver.get(ingredient_url)
    time.sleep(1)

    try:
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "h2.font-patron"))
            )
        except:
            return {}
        
        soup = BeautifulSoup(driver.page_source, "html.parser")
        details = {
            "what_does": None,
            "good_for": None,
            "avoid": None
        }

        sections = soup.select("h2.font-patron")
        for section in sections:
            heading = section.get_text(strip=True)
            parent_div = section.find_parent("div")
            if not parent_div:
                continue
            text = parent_div.get_text(separator="\n", strip=True)

            # Clean each section
            if "what does" in heading.lower():
                # Remove "What does\nXXX\ndo?\n"
                parts = text.split("do?\n", 1)
                details["what_does"] = parts[1].strip() if len(parts) == 2 else text.strip()

            elif "good for" in heading.lower():
                # Remove header & repeated lines
                parts = text.split("might be a good option for you:", 1)
                details["good_for"] = parts[1].strip() if len(parts) == 2 else text.strip()

            elif "should avoid" in heading.lower():
                # Remove header & repeated lines
                parts = text.split("it might be best to avoid", 1)
                # remove any leading colons or newlines
                details["avoid"] = parts[1].lstrip(":\n").strip() if len(parts) == 2 else text.strip()

        return details
    
    except WebDriverException as e:
        print(f"WebDriver error for {ingredient_url}: {str(e)[:100]}...")
        return {}
    except Exception as e:
        print(f"Unexpected error for {ingredient_url}: {str(e)[:100]}...")
        return {}


def main_scraper_ingredients(page_num=18, save_path='ingredients_list.csv'):
    
    driver = setup_driver()

    try:
        INGREDIENTS_URL = f"{BASE_URL}/ingredients/"
        all_ingredients = []
        all_details = []

        page = 1
        # Get product links
        for page in tqdm(range(1, page_num+1), desc="Pages scraped"):
            page_url = f"{INGREDIENTS_URL}/page/{page}"
            ingredients = get_ingredient_links(driver, page_url)

            if not ingredients:
                break

            all_ingredients.extend(ingredients)

            page += 1

        print(f"\nTotal ingredients found: {len(all_ingredients)}\n")

        # Get ingredient details
        for name, link in tqdm(all_ingredients, desc="Scraping ingredient details", unit="ingredient"):

            if not is_valid_url(link):
                print(f"Invalid URL: {link}")
                continue
            
            details = get_ingredient_details(driver, link)

            details["ingredient_name"] = name
            details["url"] = link
            
            all_details.append(details)

        # Save results
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
