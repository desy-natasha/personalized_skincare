from itertools import combinations
from scipy.spatial.distance import hamming
from scipy.spatial.distance import cosine
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import ast

### Evaluate requirement satisfaction using synthethic user profiles
def evaluate_requirement_satisfaction(all_recommendations, product_df, product_embeddings=None):
    requirement_scores = []
    recommended_ids = set()

    # Track individual constraint fulfillment
    constraint_fulfillment = {
        'skin_type': [],
        'skin_problems': [],
        'allergies': [],
        'budget': [],
        'formulation': [],
        'product_type': []
    }
    
    for profile, rec_list in all_recommendations:
        if not rec_list:
            continue
        
        # Get product indices
        rec_indices = []
        for r in rec_list:
            matches = product_df.index[product_df['name'] == r['name']]
            if len(matches) > 0:
                rec_indices.append(matches[0])
        
        if not rec_indices:
            continue
            
        recommended_ids.update(rec_indices)
        
        ### 1. Requirement Satisfaction (assess how many user requirements being fulfilled in the recommended products)
        for idx in rec_indices:
            product = product_df.loc[idx]
            fulfilled_count = 0
            total_requirements = 0
            
            # a. Skin type requirement
            if profile.get('skin_type'):
                total_requirements += 1
                skin_type = profile['skin_type'].lower()
                benefits_str = ast.literal_eval(product["benefit"])

                # Check if skin type is mentioned in benefits
                if skin_type in benefits_str:
                    fulfilled_count += 1
                    constraint_fulfillment['skin_type'].append(1)
                else:
                    constraint_fulfillment['skin_type'].append(0)
            
            # b. Skin problems requirement
            if profile.get('skin_problems'):
                total_requirements += 1
                skin_problems = [p.strip().lower() for p in profile['skin_problems'].split(',') if p.strip()]
                benefits_str =  ast.literal_eval(product["benefit"])
                
                # Check if any skin problem is addressed in benefits
                matched = sum(1 for p in skin_problems if p in benefits_str)

                if matched > 0:
                    fulfilled_count += 1
                    constraint_fulfillment['skin_problems'].append(1)
                else:
                    constraint_fulfillment['skin_problems'].append(0)

            # c. Allergens requirement
            if profile.get('allergies'):
                total_requirements += 1
                allergies = set(a.lower().strip() for a in profile['allergies'].split(',') if a.strip())
                
                avoid_list = product.get('avoid', [])
                allergies_check = True
                
                if isinstance(avoid_list, list):
                    allergies_check = not any(a in [str(x).lower() for x in avoid_list if x] for a in allergies)
                
                if allergies_check:
                    fulfilled_count += 1
                    constraint_fulfillment['allergies'].append(1)
                else:
                    constraint_fulfillment['allergies'].append(0)
            
            # d. Budget requirement
            if profile.get('budget') is not None:
                total_requirements += 1
                if product['price_raw'] <= profile['budget']:
                    fulfilled_count += 1
                    constraint_fulfillment['budget'].append(1)
                else:
                    constraint_fulfillment['budget'].append(0)
            
            # e. Formulation requirement
            if profile.get('preferred_formulation'):
                total_requirements += 1
                if product['formulation'].lower() == profile['preferred_formulation'].lower():
                    fulfilled_count += 1
                    constraint_fulfillment['formulation'].append(1)
                else:
                    constraint_fulfillment['formulation'].append(0)
            
            # f. Product type requirement
            if profile.get('product_types'):
                total_requirements += 1
                
                if isinstance(profile['product_types'], list):
                    profile_types_lower = [pt.lower() for pt in profile['product_types']]
                    product_type_lower = product['product_type'].lower()
                    
                    if product_type_lower in profile_types_lower:
                        fulfilled_count += 1
                        constraint_fulfillment['product_type'].append(1)
                    else:
                        constraint_fulfillment['product_type'].append(0)
                else:
                    if product['product_type'].lower() == profile['product_types'].lower():
                        fulfilled_count += 1
                        constraint_fulfillment['product_type'].append(1)
                    else:
                        constraint_fulfillment['product_type'].append(0)


            # Calculate fulfillment ratio for each item
            requirement_scores.append(fulfilled_count / total_requirements)

    # Calculate per-constraint fulfillment rates
    constraint_rates = {}
    for constraint, values in constraint_fulfillment.items():
        if values:
            constraint_rates[constraint] = round(np.mean(values) * 100, 2)
        else:
            constraint_rates[constraint] = None  # Not evaluated (no users had this requirement)
    
    # Final metrics
    metrics = {
        'requirement_satisfaction': round(np.mean(requirement_scores),2) if requirement_scores else 0.0,
        'constraint_fulfillment': constraint_rates
    }
    
    return metrics

### Accomodate evaluation for single product recommendation and routine recommendation system
def evaluate_recommendations(all_recommendations, product_df, recs_system=None, is_routine=False):

    requirement_scores = []
    diversity_scores = []
    recommended_ids = set()

    routine_metrics = {
        'intra_routine_diversity': [],
        'inter_routine_diversity': [],
        'compatibility': [],
        'budget_adherence': [],
        'completeness': []
    }

    # Preprocess numeric features for diversity calculation
    numeric_cols = ['price_raw', 'rating_raw', 'reviews_count_raw']
    scaler = MinMaxScaler()
    scaled_numeric = scaler.fit_transform(product_df[numeric_cols])
    
    # Encode categorical features for Hamming distance
    cat_cols = ['product_type', 'formulation']
    cat_encoded = pd.get_dummies(product_df[cat_cols]).values
    
    # Precompute embeddings
    product_embeddings = None
    if recs_system:
        product_texts = recs_system.build_product_texts(product_df)
        product_embeddings = recs_system.text_encoder.encode(
            product_texts, convert_to_tensor=False, show_progress_bar=True
        )
    
    for profile, rec_list in all_recommendations:
        if not rec_list:
            continue
        
        # Condition for routine recommendation format (has more metrics)
        if is_routine:
                
            rec_indices = []
            routines = {}
            for rec in rec_list:
                rid = rec.get('routine_id', 1)
                if rid not in routines:
                    routines[rid] = []
                routines[rid].append(rec)
            
            routine_vectors = []
            
            for routine_id, routine_products in routines.items():
                ### 1a. Requirement satisfaction
                fulfilled_count = 0
                total_requirements = 0
                
                # a. Skin type requirement
                if profile.get('skin_type'):
                    total_requirements += len(routine_products)
                    skin_type = profile['skin_type'].lower()
                    routine_benefits = []
                    for p in routine_products:
                        benefits = ast.literal_eval(p['benefit'])
                        routine_benefits.extend(benefits)
                    
                        if skin_type in routine_benefits:
                            fulfilled_count += 1
                
                # b. Skin problems requirement
                if profile.get('skin_problems'):
                    total_requirements += len(routine_products)
                    skin_problems = [p.strip().lower() for p in profile['skin_problems'].split(',') if p.strip()]
                    routine_benefits = []
                    for p in routine_products:
                        benefits = [b.lower() for b in ast.literal_eval(p['benefit'])]
                        if any(sp in benefits for sp in skin_problems):
                            fulfilled_count += 1
                
                # c. Allergens requirement
                if profile.get('allergies'):
                    total_requirements += len(routine_products)
                    allergies = set(a.lower().strip() for a in profile['allergies'].split(',') if a.strip())
                    
                    for p in routine_products:
                        avoid_list = p.get('avoid', [])
                        if isinstance(avoid_list, str):
                            try:
                                avoid_list = ast.literal_eval(avoid_list)
                            except (ValueError, SyntaxError):
                                avoid_list = []
                        
                        avoid_lower = [str(x).lower() for x in avoid_list if x]
                        if not any(a in avoid_lower for a in allergies):
                            fulfilled_count += 1
                
                # d. Budget requirement
                if profile.get('budget') is not None:
                    total_requirements += 1
                    total_price = sum(p['price'] for p in routine_products)
                    if total_price <= profile['budget']:
                        fulfilled_count += 1
                
                # e. Formulation requirement
                if profile.get('preferred_formulation'):
                    total_requirements += len(routine_products)
                    preferred_form = profile['preferred_formulation'].lower()
                    
                    for p in routine_products:
                        if p['formulation'].lower() == preferred_form:
                            fulfilled_count += 1
                
                # f. Product type requirement
                if profile.get('product_types') and isinstance(profile['product_types'], list):
                    total_requirements += len(routine_products)
                    profile_types_lower = set(pt.lower() for pt in profile['product_types'])

                    for p in routine_products:
                        if p['product_type'].lower() in profile_types_lower:
                            fulfilled_count += 1
                
                # Routine-level requirement score
                if total_requirements > 0:
                    routine_req_score = fulfilled_count / total_requirements
                    requirement_scores.append(routine_req_score)
                
                # Track recommended products
                for product in routine_products:
                    matches = product_df[product_df['name'] == product['name']]
                    if not matches.empty:
                        rec_indices.append(matches.index[0])
                        recommended_ids.add(matches.index[0])
                
                
                ### 2a. Intra-routine diversity
                if len(routine_products) > 1 and product_embeddings is not None:
                    routine_indices = []
                    for p in routine_products:
                        matches = product_df[product_df['name'] == p['name']]
                        if not matches.empty:
                            routine_indices.append(matches.index[0])
                    
                    if len(routine_indices) > 1:
                        pairwise_dist = []
                        for i, j in combinations(routine_indices, 2):
                            dist = cosine(product_embeddings[i], product_embeddings[j])
                            pairwise_dist.append(dist)
                        routine_metrics['intra_routine_diversity'].append(np.mean(pairwise_dist))
                        
                        # Create routine vector for inter-routine diversity
                        routine_emb = np.mean([product_embeddings[idx] for idx in routine_indices], axis=0)
                        routine_vectors.append(routine_emb)
                
                ### 3a. Compatibility score
                if 'compatibility_score' in routine_products[0]:
                    routine_metrics['compatibility'].append(routine_products[0]['compatibility_score'])
            
            ### 4a. Inter-routine diversity
            if len(routine_vectors) > 1:
                for i, j in combinations(range(len(routine_vectors)), 2):
                    dist = cosine(routine_vectors[i], routine_vectors[j])
                    routine_metrics['inter_routine_diversity'].append(dist)
        
        # Condition for single product recommendation format
        else:
            rec_indices = []
            for rec in rec_list:
                matches = product_df[product_df['name'] == rec['name']]
                if not matches.empty:
                    idx = matches.index[0]
                    rec_indices.append(idx)
                    recommended_ids.add(idx)
                    
                    ### 1b. Requirement satisfaction
                    single_score = evaluate_requirement_satisfaction(all_recommendations=[(profile, [rec])], product_df=product_df, product_embeddings=product_embeddings)
                    requirement_scores.append(single_score['requirement_satisfaction'])
            
        ### 2. Diversity (measure the dissimilarity between recommended product for each profile)
        if len(rec_indices) > 1:
            pairwise_dist = []
            for i, j in combinations(rec_indices, 2):
                if product_embeddings is not None:

                    # Cosine similarity between product embeddings (product name, benefit, product type, general info)
                    semantic_dist = cosine(product_embeddings[i], product_embeddings[j])

                    # Combine with numeric features (price, rating, review count)
                    num_dist = np.linalg.norm(scaled_numeric[i] - scaled_numeric[j])
                    num_dist_norm = num_dist / np.sqrt(len(numeric_cols))

                    # Weighted combination: 70% semantic, 30% numeric
                    combined_dist = 0.7 * semantic_dist + 0.3 * num_dist_norm
                    pairwise_dist.append(combined_dist)
                else:
                    num_dist = np.linalg.norm(scaled_numeric[i] - scaled_numeric[j])
                    cat_dist = hamming(cat_encoded[i], cat_encoded[j])
                    pairwise_dist.append((num_dist + cat_dist) / 2)

            diversity_scores.append(np.mean(pairwise_dist))
    
    ### 3. Catalog coverage (check percentage of product being recommended)
    catalog_coverage = len(recommended_ids) / len(product_df)
    
    # Compile results
    results = {
        'requirement_satisfaction': round(np.mean(requirement_scores) * 100, 2) if requirement_scores else 0.0,
        'diversity': round(np.mean(diversity_scores) * 100, 2) if diversity_scores else 0.0,
        'catalog_coverage': round(catalog_coverage * 100, 2)
    }
    
    if is_routine:
        results.update({
            'intra_routine_diversity': float(round(np.mean(routine_metrics['intra_routine_diversity']) * 100, 2)) if routine_metrics['intra_routine_diversity'] else None,
            'inter_routine_diversity': float(round(np.mean(routine_metrics['inter_routine_diversity']) * 100, 2)) if routine_metrics['inter_routine_diversity'] else None,
            'compatibility': float(round(np.mean(routine_metrics['compatibility']), 2)) if routine_metrics['compatibility'] else None,
        })
    else:
        results['diversity'] = round(np.mean(diversity_scores) * 100, 2) if diversity_scores else 0.0
    
    return results
