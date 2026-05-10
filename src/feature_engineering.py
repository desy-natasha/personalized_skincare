import re
from collections import Counter
from sentence_transformers import SentenceTransformer, util
import torch

# Load sentence transformer model for fuzzy matching
model = SentenceTransformer('all-MiniLM-L6-v2')

# Map the benefit to skin concern to standardise user input
concern_mapping = {
    'pregnancy': 'pregnancy',
    'radiance': 'dull skin',
    'texture': 'texture',
    'dark circles': 'pigmentation',
    'eye bags': 'pigmentation',
    'impaired skin barrier': 'impaired skin barrier',
    'redness': 'redness',
    'fine lines': 'wrinkles',
    'wrinkles': 'wrinkles',
    'blackheads': 'pores',
    'acne': 'acne',
    'elasticity': 'impaired skin barrier',
    'pigmentation': 'pigmentation',
    'enlarged pores': 'pores',
    'post blemish marks': 'pigmentation',
    'uv protection': 'uv protection',
    
    'oily': 'oily',
    'dry and dehydrated skin': 'dry',
    'combination': 'combination',
    'anyone': 'normal'
}

# Map the allergies to reduced and standardise to user input
allergies_mapping = {
    'pregnancy': 'pregnancy',
    'impaired skin barrier': 'impaired skin barrier',
    'gluten allery': 'gluten allery',
    'vegan': 'vegan',

    'oily': 'oily',
    'dry dehydrated': 'dry',
    'combination': 'combination',
    'sensitive': 'sensitive'

}

# Implement lookup between product ingredient to map the benefit and allergies
def lookup_benefit_ingredients(product_df, ingredients_df):
    
    # Create dictionaries for lookup
    good_for_map = dict(zip(ingredients_df['ingredient_name'], ingredients_df['good_for']))
    avoid_map    = dict(zip(ingredients_df['ingredient_name'], ingredients_df['avoid']))

    all_good_for = []
    all_avoid = []

    for _, row in product_df.iterrows():
        good_for_set = set()
        avoid_set = set()

        for ingredient in row['ingredients_clean']:
            
            if ingredient in avoid_map:
                val = avoid_map[ingredient]
                if val:
                    # Standardize and compile 'avoid' items
                    avoid_set.update([allergies_mapping[v] for v in val if v in allergies_mapping and v != 'nan'])


            if ingredient in good_for_map:
                val = good_for_map[ingredient]
                if val: 
                    # Standardize and compile 'good_for' items
                    good_for_set.update([concern_mapping.get(v, v) for v in val if v != 'nan'])

            # Exclude benefits that also appear in avoid (due to different ingredients that cause overlapping details and we prioritize avoid as it is more sensitive)
            filtered_good_for = good_for_set - avoid_set

        all_good_for.append(list(filtered_good_for))
        all_avoid.append(list(avoid_set))
    
    product_df['benefit'] = all_good_for
    product_df['avoid'] = all_avoid

    return product_df

def classify_product(text, classify_dict, rule_weight=0.7, fuzzy_weight=0.3, threshold=0.2):

    # Concat key and values from mapping to enrich embedding
    concat_key_values = []
    max_hits_dict = {}
    for key, keywords in classify_dict.items():
        combined_text = key + ' ' + ' '.join(keywords)
        concat_key_values.append(combined_text)
        max_hits_dict[key] = len(keywords)

    classify_dict_keys = list(classify_dict.keys())
    classify_dict_embeddings = model.encode(concat_key_values, convert_to_tensor=True)
    text_lower = text.lower() if isinstance(text, str) else ''
    
    # Approach 1: Finding exact match
    rule_scores = Counter()
    for product_type, keywords in classify_dict.items():
        hits = sum(bool(re.search(r'\b' + re.escape(x.lower()) + r'\b', text_lower)) for x in keywords)
        max_hits = max_hits_dict[product_type]

        # Normalize the score
        rule_scores[product_type] = hits / max_hits if max_hits > 0 else 0
    

    # Approach 2: Finding partial match with text embedding
    text_embedding = model.encode(text_lower, convert_to_tensor=True)
    cos_sim = util.cos_sim(text_embedding, classify_dict_embeddings).squeeze() 
    fuzzy_scores = {classify_dict_keys[i]: cos_sim[i].item() for i in range(len(classify_dict_keys))}
    
    # Combine scores from both approach
    combined_scores = {x: rule_weight * rule_scores.get(x, 0) + fuzzy_weight * fuzzy_scores.get(x, 0)
                       for x in classify_dict.keys()}
    
    # Choose result with the highest score
    best_type = max(combined_scores, key=combined_scores.get)

     # Check confidence
    best_type = max(combined_scores, key=combined_scores.get)
    if combined_scores[best_type] < threshold:
        return "other"
    else:
        return best_type
