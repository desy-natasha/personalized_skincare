from collections import Counter
import pandas as pd
import numpy as np
import ast

### Generating synthetic data based on product catalog as reference

# List of options
skin_types_vocab = {"oily", "dry", "combination", "normal", "sensitive"}
skin_problems_vocab = {"redness", "impaired skin barrier", "wrinkles", "acne",
                       "pores", "pigmentation", "texture", "dull skin", "uv protection"}
allergies_vocab = {"pregnancy", "impaired skin barrier", "gluten allergy", "vegan"}

# Calculate weighted counts from product catalog
def weighted_counts(product_df):

    w_skin_types = Counter()
    w_skin_problems = Counter()
    w_allergies = Counter()
    w_formulation = Counter()
    w_product_types = Counter()

    for _, row in product_df.iterrows():
        w = row["reviews_count"] if "reviews_count" in row else 1

        benefits = ast.literal_eval(row.get("benefit", "[]"))
        avoids = ast.literal_eval(row.get("avoid", "[]"))
        formulation = str(row.get("formulation", "")).lower()
        ptype = str(row.get("product_type", "")).lower()

        all_skin_types = benefits + avoids
        for s in all_skin_types:
            if s in skin_types_vocab:
                w_skin_types[s] += w
            elif s in skin_problems_vocab:
                w_skin_problems[s] += w

        for a in avoids:
            if a in allergies_vocab:
                w_allergies[a] += w

        if formulation:
            w_formulation[formulation] += w

        if ptype:
            w_product_types[ptype] += w

    return {
        "skin_types": w_skin_types,
        "skin_problems": w_skin_problems,
        "allergies": w_allergies,
        "formulation": w_formulation,
        "product_types": w_product_types,
    }

# Generate user profiles
def generate_user_profiles(product_df, n_users=100, is_routine=False):
    np.random.seed(42)
    weights = weighted_counts(product_df)
    
    def sample_one(counter):
        items, probs = zip(*[(k, v / sum(counter.values())) for k, v in counter.items()])
        return np.random.choice(items, p=probs)

    def sample_many(counter, k=2):
        items, probs = zip(*[(k, v / sum(counter.values())) for k, v in counter.items()])
        return list(np.random.choice(items, size=k, replace=False, p=probs))

    # Calculating user budget based on price distribution
    if is_routine:
        # Higher budget for routine
        budget_bins = {
            "low": 35,
            "medium": 75,
            "high": 150,
            "luxury": 300
        }
        bins = [0, 35, 75, 150, float("inf")]
    else:
        budget_bins = {
            "low": 10,
            "medium": 30,
            "high": 60,
            "luxury": 150
        }
        bins = [0, 10, 30, 60, float("inf")]
    
    labels = ["low", "medium", "high", "luxury"]
    product_df["price_bin"] = pd.cut(product_df["price_raw"], bins=bins, labels=labels, include_lowest=True)
    budget_dist = product_df.groupby("price_bin")["reviews_count"].sum().to_dict()
    budget_items, budget_probs = zip(*[(k, v / sum(budget_dist.values())) for k, v in budget_dist.items()])

    users = []
    for i in range(n_users):
        # Skin type: 1 value
        skin_type = sample_one(weights["skin_types"])

        # Skin problems: up to 2 values
        skin_problems = sample_many(weights["skin_problems"], k=np.random.randint(1, 3))

        # Allergies: up to 2 values
        allergies = sample_many(weights["allergies"], k=np.random.randint(1, 3))

        # Formulation: 1 value
        formulation = sample_one(weights["formulation"])

        # Budget: numeric value
        budget_category = np.random.choice(budget_items, p=budget_probs)
        budget_value = budget_bins[budget_category]

        if not is_routine:
            # Product types: 1 value
            product_types = sample_one(weights["product_types"])

            users.append({
                "user_id": f"user_{i+1}",
                "skin_type": skin_type,
                "skin_problems": ", ".join(skin_problems),
                "allergies": ", ".join(allergies),
                "preferred_formulation": formulation,
                "budget": budget_value,
                "product_types": product_types,
            })
        else:
            users.append({
                "user_id": f"user_{i+1}",
                "skin_type": skin_type,
                "skin_problems": ", ".join(skin_problems),
                "allergies": ", ".join(allergies),
                "preferred_formulation": formulation,
                "budget": budget_value,
                "product_types": 'routine',
            })

    return pd.DataFrame(users)
