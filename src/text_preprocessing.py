# 1. Define function to perform text preprocessing
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
import spacy
import re
import nltk

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Function to preprocess text: apply regex, remove stopwords, apply lemmatisation and lowercasing
def text_preprocessing(text, remove_stop_word=False):

    text = str(text)
    
    # a. Lowercasing
    text = text.lower()

    # b. Regex
    # Replace html and branding characters
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[®™©]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove any non-alphanumeric characters by keeping only letters, numbers, and spaces using regex
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)

    # c. (opt) Remove stopwords
    if remove_stop_word:
        # Tokenize the text
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in stop_words]
        text = " ".join(tokens)

    return text

# Define function to perform text preprocessing for product ingredients by finding match with `ingredients_df` as the ground truth
# Function to find match between two dataframe
def find_ingredients_in_product(product_ingredients_text, ground_truth_ingredients):

    if pd.isna(product_ingredients_text):
        return []
    
    # Lowercasing
    product_text = str(product_ingredients_text).lower()
    
    # Remove unnecessary html character
    product_text = re.sub(r'\([^)]*\)', '', product_text)  
    product_text = re.sub(r'[^\w\s]', ' ', product_text)   
    product_text = re.sub(r'\s+', ' ', product_text)       

    matched_ingredients = []

    # Common words to ignore for partial matching (too generic)
    ignore_words = {'water', 'oil', 'extract', 'acid', 'butter', 'wax', 'powder', 'gel', 'cream', 'serum'}
    
    for ingredient in ground_truth_ingredients:
        ingredient_lower = ingredient.lower()
        
        match_found = False
        
        # Approach 1: Finding exact match from ground truth to product ingredients
        if ingredient_lower in product_text:
            matched_ingredients.append(ingredient_lower)
            match_found = True
        
        # Approach 2: Finding partial match
        elif not match_found:

            # Split product ingredients text into individual ingredients
            product_chunks = [chunk.strip() for chunk in re.split(r'[,\n•]', product_text) 
                            if len(chunk.strip()) > 3]
            
            for chunk in product_chunks:

                # Split ground truth and product ingredients into individual words
                ingredient_words = [w for w in ingredient_lower.split() if len(w) > 6 and w not in ignore_words]
                chunk_words = [w for w in chunk.split() if len(w) > 6 and w not in ignore_words]
                
                # Only proceed if both have meaningful words after filtering
                if not ingredient_words or not chunk_words:
                    continue
                
                # Finding partial match in two direction from ground truth to product ingredients, and the other way around
                for ing_word in ingredient_words:
                    for chunk_word in chunk_words:
                        
                        if (ing_word in chunk_word) or (chunk_word in ing_word):
                            matched_ingredients.append(ingredient_lower)
                            match_found = True
                            break

                    if match_found:
                        break
                
                if match_found:
                    break
    
    return matched_ingredients

# Function to perform looping for the entire product list
def ingredients_preprocessing(product_df, ingredients_df):

    ground_truth_ingredients = ingredients_df['ingredient_name'].tolist()
    
    results = []
    
    for _, row in product_df.iterrows():
        
        # Find matching ingredients
        matched_ingredients = find_ingredients_in_product(row['ingredients'], ground_truth_ingredients)
        
        results.append({
            'ingredients_clean': matched_ingredients,
            'num_ingredients': len(matched_ingredients)
        })

    product_df['ingredients_clean'] = [x['ingredients_clean'] for x in results]
    product_df['num_ingredients'] = [x['num_ingredients'] for x in results]

    return product_df
