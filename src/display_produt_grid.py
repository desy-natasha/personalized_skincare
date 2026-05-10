import matplotlib.pyplot as plt
import math
import numpy as np
import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO

### Scrape product image from Chemist Warehouse product page    
def fetch_image_from_product_page(product_url):
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": "https://www.chemistwarehouse.com.au"
    }
    
    try:
        # Get the product page
        response = requests.get(product_url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            return None
        
        # Parse HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Try multiple selectors
        image_url = None
        
        img_tag = soup.find('img', {'class': 'product-image'})
        if img_tag and img_tag.get('src'):
            image_url = img_tag['src']
        
        if not image_url:
            og_image = soup.find('meta', property='og:image')
            if og_image and og_image.get('content'):
                image_url = og_image['content']

        if not image_url:
            img_tag = soup.find('img', {'itemprop': 'image'})
            if img_tag and img_tag.get('src'):
                image_url = img_tag['src']
        
        if not image_url:
            img_tag = soup.find('img', {'data-zoom-image': True})
            if img_tag:
                image_url = img_tag.get('data-zoom-image') or img_tag.get('src')
        
        if not image_url:
            return None
        
        if image_url.startswith('//'):
            image_url = 'https:' + image_url
        elif image_url.startswith('/'):
            image_url = 'https://www.chemistwarehouse.com.au' + image_url
        
        # Fetch the actual image
        img_response = requests.get(image_url, headers=headers, timeout=10)
        if img_response.status_code == 200:
            img = Image.open(BytesIO(img_response.content))
            return img
        
        return None
        
    except Exception as e:
        return None

### Create matplotlib grid fetching images from product URLs
def display_product_grid_from_urls(rec_list, product_df, images_per_row=5):

    n_products = len(rec_list)
    n_rows = math.ceil(n_products / images_per_row)
    
    fig, axes = plt.subplots(n_rows, images_per_row, 
                             figsize=(images_per_row*4.5, n_rows*6))
    
    # Handle different array shapes
    if n_products == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    axes_flat = axes.flatten()
    
    for idx, rec in enumerate(rec_list):
        ax = axes_flat[idx]
        
        product_name = rec['name']
        
        # Find product in DataFrame
        product_match = product_df[product_df['name'] == product_name]
        
        # Get product URL
        product_url = product_match.iloc[0]['url']
        
        if pd.isna(product_url):
            ax.text(0.5, 0.5, 'No URL', 
                   ha='center', va='center', fontsize=10, color='gray')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_facecolor('#f0f0f0')
            ax.axis('off')
        else:
            # Fetch image from product page
            img = fetch_image_from_product_page(product_url)
            
            if img:
                ax.imshow(img)
                img_height, img_width = img.size[1], img.size[0]
            
                # Build product details
                details = (f"{product_name}\n"
                        f"{rec['product_type'].title()}\n"
                        f"${rec['price']:.2f}  |  ★ {rec['rating']:.1f} ({rec['reviews_count']} reviews)")
                
                ax.text(img_width / 2, img_height + 60,
                        details,
                        ha='center', va='top',
                        fontsize=9,
                        linespacing=1.5,
                        fontweight='medium',
                        color='#333',
                        bbox=dict(boxstyle='round,pad=0.8',
                                facecolor='whitesmoke',
                                edgecolor='lightgray',
                                linewidth=0.8,
                                alpha=0.9))

                # Set axis limits to include text area
                ax.set_xlim(0, img_width)
                ax.set_ylim(img_height + 200, 0)

            else:
                ax.text(0.5, 0.5, 'No Image', 
                       ha='center', va='center', fontsize=10, color='gray')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_facecolor('#f0f0f0')
        ax.axis('off')
        
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
            spine.set_edgecolor('#ddd')
    
    # Hide unused subplots
    for idx in range(n_products, len(axes_flat)):
        axes_flat[idx].axis('off')
    
    plt.subplots_adjust(hspace=0.6, wspace=0.35) 
    plt.show()

### Display recommendations with images scraped from product pages
def display_recommendations_with_images(all_recommendations, product_df, images_per_row=5):
    
    for i, (profile, rec_list) in enumerate(all_recommendations):
        print(f"User Profile:")
        for key, value in profile.items():
            print(f"  {key}: {value}")
        
        if not rec_list:
            continue
        
        # Check if routine or single products
        is_routine = 'routine_id' in rec_list[0]
        
        if is_routine:
            routines = {}
            for rec in rec_list:
                rid = rec['routine_id']
                if rid not in routines:
                    routines[rid] = []
                routines[rid].append(rec)
            
            for routine_id, products in routines.items():
                print(f"Routine {routine_id}")
                print(f"Total Price: ${products[0]['total_price']:.2f}")
                display_product_grid_from_urls(products, product_df, images_per_row)
        else:
            display_product_grid_from_urls(rec_list, product_df, images_per_row)
            