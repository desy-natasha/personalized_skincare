import pandas as pd
import torch
import numpy as np
import random
from tqdm import tqdm
import ast
from itertools import product as itertools_product
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, MultiLabelBinarizer
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

### Define Recommendation System Architecture
### Consist of three main steps: bi-encoder semantic similarity, constaints-based filtering, and neural network re-ranking

class HybridRecommendationSystem:

    ### 0. Define constructor to load text encoders
    def __init__(self):
        
        # Set device
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Device: {self.device}")

        # Load cross-encoder and bi-encoder for semantic search
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-2-v2', device=self.device)
        self.bi_encoder = SentenceTransformer('paraphrase-MiniLM-L3-v2', device=self.device)
        
        # Load text encoder for neural network
        self.text_encoder = SentenceTransformer('paraphrase-MiniLM-L3-v2', device=self.device)
        
        # Preprocessing components
        self.scalers = {}
        self.encoders = {}
        self.mlb_encoders = {}
        self.feature_names = []
        self.neural_model = None
    
    ### 1. Create dictionary of user profile
    def create_user_profile(self, skin_type, skin_problems, allergies, preferred_formulation, budget, product_types):

        return {
            'skin_type': skin_type,
            'skin_problems': skin_problems,
            'allergies': allergies,
            'preferred_formulation': preferred_formulation,
            'budget': budget,
            'product_types': product_types      # can be a string (for single product recommendation) or a list (for routine recommendation)
        }
    
    ### 2. Concatenated user profile details into a single text for embedding
    def build_user_text(self, profile):

        parts = []
        
        if profile.get('skin_type'):
            parts.append(f"Skin type: {profile['skin_type']}")
        
        if profile.get('skin_problems'): 
            parts.append(f"Skin problems: {profile['skin_problems']}")
        
        if profile.get('product_types'): 
            types = profile['product_types'] if isinstance(profile['product_types'], list) else [profile['product_types']]
            parts.append(f"Product type: {', '.join(types)}")
        
        if profile.get('preferred_formulation'):
            parts.append(f"Formulation: {profile['preferred_formulation']}")
        
        return ". ".join(parts)
    
    ### 3. Concatenated product details into a single text for embedding (x_embedding)
    def build_product_texts(self, product_df):

        texts = []
        for _, row in product_df.iterrows():
            parts = []

            name = row.get('name')
            if name is not None and pd.notna(name): 
                parts.append(f"Product: {name}")

            benefit = row.get('benefit')
            if benefit is not None and isinstance(benefit, list):
                if len(benefit) > 0: 
                    benefit_str = ', '.join(str(b) for b in benefit)
                    parts.append(f"Benefits: {benefit_str}")
            
            product_type = row.get('product_type')
            if product_type is not None and pd.notna(product_type):
                parts.append(f"Product type: {product_type}")

            formulation = row.get('formulation')
            if formulation is not None and pd.notna(formulation):
                parts.append(f"Formulation: {formulation}")

            general_info = row.get('general_information')
            if general_info is not None and pd.notna(general_info):
                parts.append(f"Product information: {general_info}")
            
            texts.append(". ".join(parts))
        
        return texts

    ### 4. Fit and transform each product detais into a vector of structured features for neural network (x_structured)
    def fit_transform_features(self, product_df):
        features, feature_names = [], []
        
        # Numerical features
        for col in ['price_raw', 'rating_raw', 'reviews_count_raw']:
            if col in product_df.columns:
                scaler = MinMaxScaler()
                values = product_df[col].fillna(0).values.reshape(-1, 1)
                features.append(scaler.fit_transform(values))
                feature_names.append(f"{col}_scaled")

                self.scalers[col] = scaler
        
        # Categorical features
        for col in ['product_type', 'formulation']:
            if col in product_df.columns:
                le = LabelEncoder()
                values = product_df[col].fillna("unknown")
                features.append(le.fit_transform(values).reshape(-1,1).astype(float))
                feature_names.append(f"{col}_encoded")

                self.encoders[col] = le
        
        # Multi-label features
        for col in ['benefit', 'avoid']:
            if col in product_df.columns:
                # processed_data = self.process_text_lists(product_df[col])
                mlb = MultiLabelBinarizer()
                encoded = mlb.fit_transform(product_df[col])
            
                features.append(encoded.astype(float))
                feature_names.extend([f"{col}_{cls}" for cls in mlb.classes_])

                self.mlb_encoders[col] = mlb
        
        self.feature_names = feature_names
        X = np.hstack(features) if features else np.array([]).reshape(len(product_df), 0)
        
        return X
    
    ### 5. Transform each product detais using fitted scaler and encoder (x_structured)
    def transform_features(self, product_df):
        features = []
        
        # Numerical features
        for col in ['price_raw', 'rating_raw', 'reviews_count_raw']:
            if col in product_df.columns and col in self.scalers:
                values = product_df[col].fillna(0).values.reshape(-1, 1)
                features.append(self.scalers[col].transform(values))
        
        # Categorical features
        for col in ['product_type', 'formulation']:
            if col in product_df.columns and col in self.encoders:
                values = product_df[col].fillna("unknown")
                features.append(self.encoders[col].transform(values).reshape(-1,1).astype(float))
        
        # Multi-label features
        for col in ['benefit', 'avoid']:
            if col in product_df.columns and col in self.mlb_encoders:
                encoded = self.mlb_encoders[col].transform(product_df[col])
                features.append(encoded.astype(float))
        
        X = np.hstack(features) if features else np.array([]).reshape(len(product_df), 0)

        return X
        
    ### 6. Apply constraint-based filtering
    def apply_hard_constraints(self, product_df, profile, candidate_indices=None):

        # Get product types from profile
        product_types = profile.get('product_types', [])
        if not isinstance(product_types, list):
            product_types = [product_types]
        
        # Initialize result dictionary
        filtered_by_type = {}
        
        for product_type in product_types:
            
            if candidate_indices is not None:
                filtered = product_df.iloc[candidate_indices].copy()
            else:
                filtered = product_df.copy()
            
            # a. Filter product types based on user request (strict constraint)
            filtered = filtered[filtered["product_type"].str.lower() == product_type.lower()]
            
            # b. Filter product that contains user allergies (strict constraint)
            if profile.get("allergies"):
                # Parse user allergies once
                user_allergies = set(a.strip().lower() for a in profile["allergies"].split(",") if a.strip())

                def has_allergen(avoid_list):
                    avoid_list = ast.literal_eval(avoid_list)
                    product_allergies = set(avoid_list)

                    return len(user_allergies & product_allergies) > 0 

                # Keep only products without allergens
                mask = ~filtered["avoid"].apply(has_allergen)
                filtered = filtered[mask]
                
            # c. Filter product formulation based on user request (soft constraint)
            if profile.get("preferred_formulation"):
                backup_filtered = filtered.copy()
                formulation = profile["preferred_formulation"]
                forms = [formulation.lower()]
                filtered_form = filtered[filtered["formulation"].str.lower().isin(forms)]
                
                # If too restrictive, skip this filter
                if len(filtered_form) >= 5:
                    filtered = filtered_form
                else:
                    filtered = backup_filtered

            # d. Filter product with price above user budget (soft constraint)
            if profile.get('budget') is not None:
                backup_filtered = filtered.copy()
                
                # For single product -> use full budget
                # For routine -> use weighted allocation
                if len(product_types) == 1:
                    type_budget = profile['budget']
                else:
                    weight = BUDGET_WEIGHTS.get(product_type.lower(), 1.0 / len(product_types))
                    type_budget = profile['budget'] * weight * 1.5  # 1.5x buffer for flexibility
                
                filtered_budget = filtered[filtered["price_raw"] <= type_budget]
                
                # If too restrictive, skip this filter
                if len(filtered_budget) >=5 :
                    filtered = filtered_budget
                else:
                    filtered = backup_filtered
            
            # Sort by price
            filtered = filtered.sort_values('price_raw', ascending=True)
            
            # Store indices for this product type
            filtered_by_type[product_type] = filtered.index.tolist()
        
        return filtered_by_type

    ### 7. The architecture of the neural network to assess quality
    def build_neural_network_predictor(self, structured_dim, embedding_dim, dropout_rate=0.3):

        # Inputs
        structured_input = layers.Input(shape=(structured_dim,), name='structured_input')
        embedding_input = layers.Input(shape=(embedding_dim,), name='embedding_input')
        
        # Structured features branch
        s = layers.Dense(128, activation='relu')(structured_input)
        s = layers.BatchNormalization()(s)
        s = layers.Dropout(dropout_rate)(s)
        
        # Text embedding branch
        e = layers.Dense(128, activation='relu')(embedding_input)
        e = layers.BatchNormalization()(e)
        e = layers.Dropout(dropout_rate)(e)
        
        # Combine all branches
        combined = layers.Concatenate()([s, e])
        
        # Hidden layers
        x = layers.Dense(256, activation='relu')(combined)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate/2)(x)
        
        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        
        # Output layer
        output = layers.Dense(1, activation='sigmoid', name='rerank_score')(x)
        
        model = Model([structured_input, embedding_input], output)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    ### 8. The training loop for the neural network component
    def train_quality_predictor(self, product_df, epochs=20, batch_size=16):

        # Prepare features for all products
        X_structured = self.fit_transform_features(product_df)
        
        product_texts = self.build_product_texts(product_df)
        X_text = self.text_encoder.encode(product_texts, convert_to_tensor=False, show_progress_bar=False)
        
        # Calculate quality score using rating, number of review, and price as signals
        quality_score = (
            0.4 * product_df['rating'] +
            0.4 * product_df['reviews_count'] +
            0.2 * product_df['price']
        )
        
        # Create binary labels where if score above median, the it is a high quality product
        y_quality_label = (quality_score > quality_score.median()).astype(float).values
        
        print(f"Training neural network using {len(X_structured)} products")
        
        # Build quality predictor
        self.quality_model = self.build_neural_network_predictor(
            X_structured.shape[1],
            X_text.shape[1]
        )
        
        callbacks=[
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
            ]
        
        # Train
        history = self.quality_model.fit([X_structured, X_text], y_quality_label,
            epochs=epochs, batch_size=batch_size, validation_split=0.2,
            callbacks=callbacks, verbose=0)
        
        return history
    
    ### 9. Function to calculate ingredients compatibility between products (ONLY for routine recommendation)
    def calculate_routine_compatibility(self, routine_products):

        if len(routine_products) <= 1:
            return 1.0
        
        conflict_count = 0
        synergy_count = 0
        total_pairs = 0
        
        # Check all pairs of products
        for i in range(len(routine_products)):
            for j in range(i + 1, len(routine_products)):
                product_i = routine_products[i]
                product_j = routine_products[j]
                
                ingredients_i = ast.literal_eval(product_i['ingredients'])
                ingredients_j = ast.literal_eval(product_j['ingredients'])
                
                # Check for conflicts
                for ing_i in ingredients_i:
                    if ing_i in INGREDIENTS_NOT_TO_MIX:
                        conflicts = INGREDIENTS_NOT_TO_MIX[ing_i]
                        for ing_j in ingredients_j:
                            if ing_j in conflicts:
                                conflict_count += 1
                
                # Check for synergies
                for ing_i in ingredients_i:
                    if ing_i in INGREDIENTS_TO_MIX:
                        synergies = INGREDIENTS_TO_MIX[ing_i]
                        for ing_j in ingredients_j:
                            if ing_j in synergies:
                                synergy_count += 1
                
                total_pairs += 1
        
        # Calculate compatibility score
        if total_pairs == 0:
            return 1.0
        
        # Penalize conflicts and reward synergies
        conflict_penalty = conflict_count / total_pairs
        synergy_bonus = synergy_count / total_pairs

        compatibility_score = max(0, 1.0 - conflict_penalty + synergy_bonus)
        
        return compatibility_score  

    ### Perform neural network re-ranking for single product recommendation
    def recommend_single(self, product_df, user_profile, candidate_indices, 
                         semantic_scores, top_k, semantic_similarity, train_neural_network):
        final_scores = []

        if train_neural_network:
            # Filter products for neural network
            rerank_df = product_df.loc[candidate_indices].copy()
            
            # Create structured features
            X_structured = self.transform_features(rerank_df)
            
            # Create text embeddings for neural network to all products
            filtered_product_texts = self.build_product_texts(rerank_df)
            X_text = self.text_encoder.encode(filtered_product_texts, convert_to_tensor=False, show_progress_bar=False)

            # Predict quality
            quality_predictions = self.quality_model.predict([X_structured, X_text], verbose=0).flatten()
            
            # Normalize the quality predictions to ensure equal distribution with semantic distribution
            if quality_predictions.max() > quality_predictions.min():
                quality_predictions = (quality_predictions - quality_predictions.min()) / \
                                    (quality_predictions.max() - quality_predictions.min())
            else:
                quality_predictions = np.ones_like(quality_predictions) * 0.5
            
            # Normalize semantic scores
            if semantic_similarity:
                candidate_semantic_values = np.array([semantic_scores.get(idx, 0.0) for idx in candidate_indices])
                if candidate_semantic_values.max() > candidate_semantic_values.min():
                    normalized_semantic_values = (candidate_semantic_values - candidate_semantic_values.min()) / \
                                                (candidate_semantic_values.max() - candidate_semantic_values.min())
                else:
                    normalized_semantic_values = np.ones_like(candidate_semantic_values) * 0.5
                
                normalized_semantic_scores = dict(zip(candidate_indices, normalized_semantic_values))

            # Pre-compute benefit matching scores for normalization
            benefit_type_scores = []
            benefit_problem_scores = []
            
            for idx in candidate_indices:
                row = product_df.loc[idx]
                benefit_str = str(row['benefit']).lower()
                
                # Skin type match
                if user_profile.get('skin_type'):
                    user_skin_type = str(user_profile['skin_type']).lower()
                    type_match = 1.0 if user_skin_type in benefit_str else 0.0
                else:
                    type_match = 0.0
                benefit_type_scores.append(type_match)
                
                # Skin problem match
                if user_profile.get('skin_problems'):
                    user_problems = [x.strip() for x in str(user_profile['skin_problems']).lower().split(",")]
                    matches = sum(1 for p in user_problems if p in benefit_str)
                    problem_score = matches / len(user_problems) if user_problems else 0.0
                else:
                    problem_score = 0.0
                benefit_problem_scores.append(problem_score)
            
            # Normalize benefit scores to 0-1 range
            benefit_type_scores = np.array(benefit_type_scores)
            benefit_problem_scores = np.array(benefit_problem_scores)
            
            if benefit_type_scores.max() > benefit_type_scores.min():
                benefit_type_scores = (benefit_type_scores - benefit_type_scores.min()) / \
                                    (benefit_type_scores.max() - benefit_type_scores.min())
            else:
                benefit_type_scores = np.ones_like(benefit_type_scores) * 0.5
                
            if benefit_problem_scores.max() > benefit_problem_scores.min():
                benefit_problem_scores = (benefit_problem_scores - benefit_problem_scores.min()) / \
                                        (benefit_problem_scores.max() - benefit_problem_scores.min())
            else:
                benefit_problem_scores = np.ones_like(benefit_problem_scores) * 0.5

            # Combine scores with normalized values
            for i, idx in enumerate(candidate_indices):
                row = product_df.loc[idx]
                score = 0
                
                # 35% semantic similarity
                if semantic_similarity:
                    score += 0.35 * normalized_semantic_scores.get(idx, 0.0)

                # 35% quality model prediction
                score += 0.35 * quality_predictions[i]

                # Adjust weights if no semantic similarity (distribute the 32.5%)
                adj_weight = 1.0 if semantic_similarity else 2.17
                
                # 15% skin type and benefit match
                score += 0.15 * benefit_type_scores[i] * adj_weight

                # 15% skin problem and benefit match
                score += 0.15 * benefit_problem_scores[i] * adj_weight
                
                final_scores.append((idx, score, row['price_raw']))

            # Sort by price in ascending order and final_score in descending order
            reranked = pd.DataFrame(final_scores, columns=["idx", "final_score", "price"])
            reranked = reranked.sort_values(["final_score", "price"], ascending=[False, True])

        else:
            if semantic_similarity:
                 # Normalize semantic scores
                candidate_semantic_values = np.array([semantic_scores.get(idx, 0.0) for idx in candidate_indices])
                if candidate_semantic_values.max() > candidate_semantic_values.min():
                    normalized_semantic_values = (candidate_semantic_values - candidate_semantic_values.min()) / \
                                                (candidate_semantic_values.max() - candidate_semantic_values.min())
                else:
                    normalized_semantic_values = candidate_semantic_values

                final_scores = [
                    (idx, normalized_semantic_values[i], product_df.loc[idx, 'price_raw'])
                    for i, idx in enumerate(candidate_indices)]

                # Sort by price in ascending order and semantic_score in descending order
                reranked = pd.DataFrame(final_scores, columns=["idx", "semantic_score", "price"])
                reranked = reranked.sort_values(["semantic_score", "price"], ascending=[False, True])
            else:
                # Without semantic similarity and neural network, based only on price and rating
                final_scores = [
                    (idx, product_df.loc[idx, 'rating_raw'], product_df.loc[idx, 'price_raw'])
                    for idx in candidate_indices]

                # Sort by price in ascending order and semantic_score in descending order
                reranked = pd.DataFrame(final_scores, columns=["idx", "rating", "price"])
                reranked = reranked.sort_values(["rating", "price"], ascending=[False, True])

        # Final results 
        results = []
        for _, row_info in reranked.head(top_k).iterrows():
            idx = row_info["idx"]
            score = row_info.get("final_score", row_info.get("semantic_score", 0.0))
            row = product_df.loc[idx]

            results.append({
                "name": row["name"],
                "product_type": row["product_type"],
                "formulation": row["formulation"],
                "price": row["price_raw"],
                "rating": row["rating_raw"],
                "reviews_count": row["reviews_count_raw"],
                "benefit": row["benefit"],
                "avoid": row.get("avoid", ""),
                "ingredients": row.get("ingredients", ""),
                'url': row.get("url", ""),
                "img_url": row.get("img_url", ""),
                "final_score": float(score),
                "bi_encoder_score": semantic_scores.get(idx, 0.0)
            })
        
        results_df = pd.DataFrame(results)
        
        return results_df

    ### Perform neural network re-ranking for routine recommendation (multiple products)
    def recommend_routine(self, product_df, user_profile, candidates_by_type, 
                         semantic_scores, top_k, semantic_similarity, train_neural_network):
        
        # Limit candidates per type
        max_per_type = 15
        limited_candidates = {
            ptype: indices[:max_per_type] 
            for ptype, indices in candidates_by_type.items()
        }

        # Generate all possible routines
        product_types_list = list(limited_candidates.keys())
        candidate_lists = [limited_candidates[pt] for pt in product_types_list]

        all_combinations = []
        budget = user_profile.get('budget')
        max_combinations = 1000 
        
        # Pre-compute prices
        price_lookup = {idx: product_df.loc[idx, 'price_raw'] for idx in 
                    [idx for indices in limited_candidates.values() for idx in indices]}
        
        for combination in itertools_product(*candidate_lists):
            total_price = sum(price_lookup[idx] for idx in combination)
            
            # Skip immediately if over budget
            if budget and total_price > budget:
                continue
            
            all_combinations.append(combination)
            
            if len(all_combinations) >= max_combinations:
                break
        
        if not all_combinations:
            return pd.DataFrame()
        
        # Pre-compute and normalize quality predictions
        quality_lookup = {}
        if train_neural_network:
            all_candidate_indices = list(set([idx for indices in limited_candidates.values() for idx in indices]))
            all_candidates_df = product_df.loc[all_candidate_indices]
            
            X_struct_all = self.transform_features(all_candidates_df)
            texts_all = self.build_product_texts(all_candidates_df)
            X_text_all = self.text_encoder.encode(texts_all, convert_to_tensor=False, show_progress_bar=False)
            
            all_quality_preds = self.quality_model.predict([X_struct_all, X_text_all], verbose=0).flatten()
            
            # Normalize quality predictions globally
            if all_quality_preds.max() > all_quality_preds.min():
                all_quality_preds = (all_quality_preds - all_quality_preds.min()) / \
                                (all_quality_preds.max() - all_quality_preds.min())
            else:
                all_quality_preds = np.ones_like(all_quality_preds) * 0.5
            
            quality_lookup = dict(zip(all_candidate_indices, all_quality_preds))
        
        # Normalize semantic scores
        normalized_semantic_scores = {}
        if semantic_similarity:
            all_candidate_indices = list(set([idx for indices in limited_candidates.values() for idx in indices]))
            all_semantic_values = np.array([semantic_scores.get(idx, 0.0) for idx in all_candidate_indices])
            
            if all_semantic_values.max() > all_semantic_values.min():
                normalized_values = (all_semantic_values - all_semantic_values.min()) / \
                                (all_semantic_values.max() - all_semantic_values.min())
            else:
                normalized_values = np.ones_like(all_semantic_values) * 0.5
            
            normalized_semantic_scores = dict(zip(all_candidate_indices, normalized_values))
        
        # Pre-compute benefit scores for all candidates for normalization
        all_candidate_indices = list(set([idx for indices in limited_candidates.values() for idx in indices]))
        benefit_score_lookup = {}
        
        for idx in all_candidate_indices:
            row = product_df.loc[idx]
            benefit_str = str(row['benefit']).lower()
            
            benefit_score = 0
            if user_profile.get('skin_problems'):
                user_problems = [x.strip() for x in str(user_profile['skin_problems']).lower().split(",")]
                matches = sum(1 for p in user_problems if p in benefit_str)
                benefit_score = matches / len(user_problems) if user_problems else 0
            
            benefit_score_lookup[idx] = benefit_score
        
        # Normalize benefit scores
        all_benefit_values = np.array(list(benefit_score_lookup.values()))
        if all_benefit_values.max() > all_benefit_values.min():
            normalized_benefit_values = (all_benefit_values - all_benefit_values.min()) / \
                                        (all_benefit_values.max() - all_benefit_values.min())
        else:
            normalized_benefit_values = np.ones_like(all_benefit_values) * 0.5
        
        benefit_score_lookup = dict(zip(benefit_score_lookup.keys(), normalized_benefit_values))
        
        routine_scores = []
        
        for combination in tqdm(all_combinations, desc="Scoring routines", disable=True):
            # Get products for this routine
            routine_products = [product_df.loc[idx] for idx in combination]

            # Calculate total price
            total_price = sum(p['price_raw'] for p in routine_products)
            
            # Check budget constraint
            if user_profile.get('budget'):
                if total_price > user_profile['budget']:
                    continue  # Skip over-budget routines
            
            # Calculate individual scores
            individual_score = 0
            
            # Quality score (if neural network trained)
            if train_neural_network:
                quality_avg = np.mean([quality_lookup.get(idx, 0.0) for idx in combination])
                individual_score += 0.15 * quality_avg

            # Adjust weights if no neural network (distribute the 50%)
            adj_weight_semantic = 1.0 if train_neural_network else 0.5/0.4

            # Semantic similarity (average across products)
            if semantic_similarity:
                semantic_avg = np.mean([normalized_semantic_scores.get(idx, 0.0) for idx in combination])
                individual_score += 0.40 * semantic_avg * adj_weight_semantic
            else:
                rating_avg = np.mean([product_df.loc[idx, 'rating'] for idx in combination])
                individual_score += 0.40 * rating_avg * adj_weight_semantic

            # Adjust weights if no neural network (distribute the 40%)
            adj_weight_benefit = 1.0 if train_neural_network else 0.4/0.35
            
            # Benefit matching
            benefit_avg = np.mean([benefit_score_lookup.get(idx, 0.0) for idx in combination])
            individual_score += 0.35 * benefit_avg * adj_weight_benefit
            
            # Adjust weights if no neural network (distribute the 10%)
            adj_weight_comp = 1.0 if train_neural_network else 1.0

            # Compatibility score
            compatibility = self.calculate_routine_compatibility(
                [product_df.loc[idx].to_dict() for idx in combination]
            )
            individual_score += 0.10 * compatibility * adj_weight_comp
            routine_scores.append({
                'combination': combination,
                'score': individual_score,
                'total_price': total_price,
                'compatibility': compatibility
            })
        # Sort by score
        routine_scores = sorted(routine_scores, key=lambda x: (-x['score'], x['total_price']))
        
        # Return top routines
        if not routine_scores:
            return pd.DataFrame()
        
        # Format results
        results = []
        count_routine = 1
        for routine_info in routine_scores[:top_k]:
            routine_products = []
            
            for i, idx in enumerate(routine_info['combination']):
                row = product_df.loc[idx]
                product_type = product_types_list[i]
                step = ROUTINE_ORDER.get(product_type.lower(), i + 1)
                
                routine_products.append({
                    'step': step,
                    'product_type': product_type,
                    'name': row['name'],
                    'formulation': row['formulation'],
                    'price': row['price_raw'],
                    'rating': row['rating_raw'],
                    'reviews_count': row['reviews_count_raw'],
                    'benefit': row['benefit'],
                    'avoid': row.get('avoid', ''),
                    'url': row.get('url', ''),
                    'img_url': row.get('img_url', ''),
                    'ingredients': row.get('ingredients', '')
                })
            
            # Sort by routine order
            routine_products = sorted(routine_products, key=lambda x: x['step'])
            
            # Create routine record
            for routine in routine_products:
                results.append({
                    'routine_id': count_routine,
                    'total_price': routine_info['total_price'],
                    'compatibility_score': routine_info['compatibility'],
                    'overall_score': routine_info['score'],
                    'step': routine['step'],
                    'product_type': routine['product_type'],
                    'name': routine['name'],
                    'formulation': routine['formulation'],
                    'price': routine['price'],
                    'rating': routine['rating'],
                    'reviews_count': routine['reviews_count'],
                    'benefit': routine['benefit'],
                    'avoid': routine['avoid'],
                    'ingredients': routine['ingredients'],
                    'url': routine.get('url', ''),
                    'img_url': routine['img_url']
                })
                
            count_routine += 1
        
        return pd.DataFrame(results) 
    
    ### Main recommendation system pipeline using the three step components
    def recommend_products(self, product_df, user_profile, top_k=5, retrieval_k=100, 
                          constraint_filter=False, semantic_similarity=False, cross_encoder=False, train_neural_network=False):
        
        product_types = user_profile.get('product_types', [])
        if not isinstance(product_types, list):
            product_types = [product_types]
        
        is_routine = len(product_types) > 1
        
        ### Step 1: Constraint-based filtering
        if constraint_filter:
            
            # Apply hard constraints
            filtered_by_type = self.apply_hard_constraints(product_df, user_profile)
            empty_types = [pt for pt, indices in filtered_by_type.items() if not indices]

            if empty_types:
                # Remove optional steps that have no matches
                OPTIONAL_STEPS = ['toner', 'serum']  # These can be skipped
                optional_missing = [pt for pt in empty_types if pt in OPTIONAL_STEPS]
                required_missing = [pt for pt in empty_types if pt not in OPTIONAL_STEPS]
    
                if required_missing:
                    return pd.DataFrame()
                
                if optional_missing:
                    for pt in optional_missing:
                        filtered_by_type.pop(pt)
                    
        else:
            filtered_by_type = {pt: product_df.index.tolist() for pt in product_types}

        ### Step 2: Encoder semantic similarity
        if cross_encoder:
            # Build user profile text
            user_text = self.build_user_text(user_profile)
            
            candidates_by_type = {}
            semantic_scores = {} 

            for product_type, indices in filtered_by_type.items():
                if not indices:
                    continue
                
                # Get products of this type
                type_df = product_df.loc[indices]
                
                # Build product texts for cross-encoder
                product_texts = self.build_product_texts(type_df)
                
                # Create pairs for cross-encoder
                pairs = [(user_text, product_text) for product_text in product_texts]
                
                # Get cross-encoder scores (raw logits)
                ce_scores = self.cross_encoder.predict(pairs, show_progress_bar=False)
                scored_products = list(zip(indices, ce_scores))
                
                # Sort by cross-encoder score
                scored_products.sort(key=lambda x: x[1], reverse=True)
                top_candidates = scored_products[:min(retrieval_k, len(scored_products))]
                
                # Store candidate and scores
                type_candidates = [idx for idx, score in top_candidates]
                candidates_by_type[product_type] = type_candidates
                
                for idx, score in top_candidates:
                    semantic_scores[idx] = float(score)
            
        elif semantic_similarity:
            candidates_by_type = {}
            semantic_scores = {}
            
            # Build user profile embedding
            user_text = self.build_user_text(user_profile)
            user_emb = self.bi_encoder.encode(user_text, convert_to_tensor=True)
            
            for product_type, indices in filtered_by_type.items():
                if not indices:
                    continue
                
                # Get products of this type
                type_df = product_df.loc[indices]
                
                # Build product embeddings for bi-encoder from filtered products
                product_texts = self.build_product_texts(type_df)
                product_embeddings = self.bi_encoder.encode(product_texts, convert_to_tensor=True, show_progress_bar=False)
                
                # Semantic similarity
                hits = util.semantic_search(user_emb, product_embeddings, top_k=min(retrieval_k, len(type_df)))[0]
                
                # Store candidates and scores
                type_candidates = [indices[hit['corpus_id']] for hit in hits]
                candidates_by_type[product_type] = type_candidates
                
                for hit in hits:
                    idx = indices[hit['corpus_id']]
                    semantic_scores[idx] = hit['score']
        else:
            candidates_by_type = filtered_by_type
            semantic_scores = {}

            
        
        ### Step 3: Neural network re-ranking (separated for single product or routine recommendation)
        # Re-ranking using quality label from the trained neural network predictors

        if is_routine:
            return self.recommend_routine(
                product_df, user_profile, candidates_by_type, 
                semantic_scores, top_k, semantic_similarity, train_neural_network
            )
        else:
    
            product_type = product_types[0]
            candidate_indices = candidates_by_type.get(product_type, [])
            
            if not candidate_indices:
                return pd.DataFrame()
            
            return self.recommend_single(
                product_df, user_profile, candidate_indices,
                semantic_scores, top_k, semantic_similarity, train_neural_network
            )
        
### Main function to encapsulate the end-to-end recommendation system for a single or multiple user profiles
def run_hybrid_recommendations(product_df, user_profiles, top_k=5,  
                               constraint_filter=False, semantic_similarity=False, cross_encoder=False,
                               train_neural_network=False, print_result=False):

    # Initialize the recommendation system
    hybrid_system = HybridRecommendationSystem()
    
    # Convert user profiles to dictionary
    user_profiles_dict = []
    if isinstance(user_profiles, pd.DataFrame):
        for _, row in user_profiles.iterrows():

            # Handle product_types for routine or single product
            product_types = row['product_types']
            if product_types == "routine":
                product_types = ["cleanser", "toner", "serum", "moisturizer", "sunscreen"]
            else:
                product_types = [product_types]

            profile = hybrid_system.create_user_profile(row['skin_type'],
                                                        row['skin_problems'], 
                                                        row['allergies'],
                                                        row['preferred_formulation'],
                                                        row['budget'],
                                                        product_types)

            user_profiles_dict.append(profile)

    elif isinstance(user_profiles, dict):
        user_profiles_dict.append(user_profiles)
    
    # Train neural network to predict product quality label
    if train_neural_network:
        hybrid_system.train_quality_predictor(product_df, epochs=20, batch_size=16)
    else:
        print("Skipping neural network re-ranking with quality labeling.")
        
    # Get recommendations for each user
    all_recommendations = []
    for i, profile in enumerate(tqdm(user_profiles_dict, desc="Generating recommedations for users")):

        recommendations = hybrid_system.recommend_products(
            product_df=product_df,
            user_profile=profile,
            top_k=top_k,
            retrieval_k=100,
            constraint_filter=constraint_filter, 
            semantic_similarity=semantic_similarity, 
            cross_encoder=cross_encoder,
            train_neural_network=train_neural_network
        )
        
        # Handle both single product and routine results
        if not recommendations.empty:
            if 'routine_id' in recommendations.columns:
                rec_list = recommendations.to_dict(orient='records')
            else:
                rec_list = recommendations.to_dict(orient='records')
        else:
            rec_list = []
        
        all_recommendations.append((profile, rec_list))

        if print_result:
            print(f"\nUser profile #{i+1}:")
            print(profile)
            print("\nTop recommendations:")
            display(recommendations)
            
    
    return hybrid_system, all_recommendations
