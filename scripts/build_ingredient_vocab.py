#!/usr/bin/env python
"""
Build ingredient vocabulary and analyze ingredient-level annotations for Phase 5.5.6.

This script parses the Nutrition5k cafe metadata files to:
1. Extract unique ingredients and their calorie densities
2. Build ingredient vocabulary (id -> name -> cal/g mapping)
3. Analyze ingredient distribution per dish
4. Check overlap with train/val/test splits

Output:
- experiments/phase5.5_ingredient/ingredient_vocab.json
- experiments/phase5.5_ingredient/analysis_report.json

Usage:
    python scripts/build_ingredient_vocab.py
"""

import json
import csv
from pathlib import Path
from collections import defaultdict
import sys

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "nutrition5k"
METADATA_DIR = DATA_DIR / "metadata"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "phase5.5_ingredient"


def parse_ingredients_metadata(filepath: Path) -> dict:
    """Parse ingredients_metadata.csv to get calorie densities."""
    ingredients = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ingr_name = row['ingr']
            ingr_id = int(row['id'])
            cal_per_g = float(row['cal/g'])
            fat_per_g = float(row['fat(g)'])
            carb_per_g = float(row['carb(g)'])
            protein_per_g = float(row['protein(g)'])

            ingredients[ingr_id] = {
                'name': ingr_name,
                'cal_per_g': cal_per_g,
                'fat_per_g': fat_per_g,
                'carb_per_g': carb_per_g,
                'protein_per_g': protein_per_g
            }
    return ingredients


def parse_cafe_metadata(filepath: Path) -> dict:
    """Parse dish_metadata_cafe*.csv to extract per-ingredient annotations.

    Format: dish_id,total_cal,total_mass,total_fat,total_carb,total_protein,
            ingr_id_1,ingr_name_1,ingr_grams_1,ingr_cal_1,ingr_fat_1,ingr_carb_1,ingr_protein_1,
            ingr_id_2,...
    """
    dishes = {}

    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 6:
                continue

            dish_id = row[0]
            try:
                total_cal = float(row[1])
                total_mass = float(row[2])
                total_fat = float(row[3])
                total_carb = float(row[4])
                total_protein = float(row[5])
            except (ValueError, IndexError):
                continue

            # Parse ingredients (7 fields per ingredient starting at index 6)
            ingredients = []
            idx = 6
            while idx + 6 < len(row):
                try:
                    ingr_id = row[idx]  # e.g., "ingr_0000000508"
                    ingr_name = row[idx + 1]
                    ingr_grams = float(row[idx + 2])
                    ingr_cal = float(row[idx + 3])
                    ingr_fat = float(row[idx + 4])
                    ingr_carb = float(row[idx + 5])
                    ingr_protein = float(row[idx + 6])

                    ingredients.append({
                        'id': ingr_id,
                        'name': ingr_name,
                        'grams': ingr_grams,
                        'calories': ingr_cal,
                        'fat': ingr_fat,
                        'carb': ingr_carb,
                        'protein': ingr_protein
                    })
                except (ValueError, IndexError):
                    pass
                idx += 7

            dishes[dish_id] = {
                'total_calories': total_cal,
                'total_mass': total_mass,
                'total_fat': total_fat,
                'total_carb': total_carb,
                'total_protein': total_protein,
                'ingredients': ingredients,
                'num_ingredients': len(ingredients)
            }

    return dishes


def load_split_ids(split_file: Path) -> set:
    """Load dish IDs from a split file."""
    ids = set()
    if split_file.exists():
        with open(split_file, 'r') as f:
            for line in f:
                dish_id = line.strip()
                if dish_id:
                    ids.add(dish_id)
    return ids


def main():
    print("=" * 60)
    print("Phase 5.5.6: Ingredient Vocabulary Builder")
    print("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Parse ingredients_metadata.csv
    print("\n[1/5] Parsing ingredients_metadata.csv...")
    ingr_metadata_path = METADATA_DIR / "ingredients_metadata.csv"
    if not ingr_metadata_path.exists():
        print(f"ERROR: {ingr_metadata_path} not found")
        sys.exit(1)

    ingredient_vocab = parse_ingredients_metadata(ingr_metadata_path)
    print(f"  Found {len(ingredient_vocab)} unique ingredients")

    # Step 2: Parse cafe metadata files
    print("\n[2/5] Parsing cafe metadata files...")
    all_dishes = {}

    for cafe_file in ['dish_metadata_cafe1.csv', 'dish_metadata_cafe2.csv']:
        cafe_path = METADATA_DIR / cafe_file
        if cafe_path.exists():
            dishes = parse_cafe_metadata(cafe_path)
            all_dishes.update(dishes)
            print(f"  {cafe_file}: {len(dishes)} dishes")

    print(f"  Total dishes with ingredient annotations: {len(all_dishes)}")

    # Step 3: Analyze ingredient distribution
    print("\n[3/5] Analyzing ingredient distribution...")

    ingredient_counts = defaultdict(int)
    num_ingredients_dist = defaultdict(int)

    for dish_id, dish_data in all_dishes.items():
        num_ingr = dish_data['num_ingredients']
        num_ingredients_dist[num_ingr] += 1

        for ingr in dish_data['ingredients']:
            ingredient_counts[ingr['name']] += 1

    # Top ingredients
    top_ingredients = sorted(ingredient_counts.items(), key=lambda x: -x[1])[:20]
    print(f"  Top 10 ingredients:")
    for name, count in top_ingredients[:10]:
        print(f"    {name}: {count} dishes")

    # Ingredient count distribution
    print(f"\n  Ingredients per dish distribution:")
    for num, count in sorted(num_ingredients_dist.items()):
        print(f"    {num} ingredients: {count} dishes")

    # Step 4: Check overlap with train/val/test splits
    print("\n[4/5] Checking overlap with train/val/test splits...")

    splits_dir = DATA_DIR / "dish_ids" / "splits"
    train_ids = load_split_ids(splits_dir / "rgb_train_ids.txt")
    test_ids = load_split_ids(splits_dir / "rgb_test_ids.txt")

    # Our dataset uses a subset of the official splits
    # Check how many of our dishes have ingredient annotations
    cafe_dish_ids = set(all_dishes.keys())

    train_overlap = train_ids & cafe_dish_ids
    test_overlap = test_ids & cafe_dish_ids

    print(f"  RGB train IDs: {len(train_ids)}")
    print(f"  RGB test IDs: {len(test_ids)}")
    print(f"  Cafe dishes with ingredients: {len(cafe_dish_ids)}")
    print(f"  Train overlap: {len(train_overlap)} ({100*len(train_overlap)/len(train_ids):.1f}%)")
    print(f"  Test overlap: {len(test_overlap)} ({100*len(test_overlap)/len(test_ids):.1f}%)")

    # Step 5: Save outputs
    print("\n[5/5] Saving outputs...")

    # Save ingredient vocabulary
    vocab_output = {
        'num_ingredients': len(ingredient_vocab),
        'ingredients': {
            str(k): v for k, v in ingredient_vocab.items()
        }
    }
    vocab_path = OUTPUT_DIR / "ingredient_vocab.json"
    with open(vocab_path, 'w') as f:
        json.dump(vocab_output, f, indent=2)
    print(f"  Saved: {vocab_path}")

    # Save analysis report
    analysis = {
        'total_dishes_with_ingredients': len(all_dishes),
        'unique_ingredients': len(ingredient_vocab),
        'train_split_total': len(train_ids),
        'test_split_total': len(test_ids),
        'train_with_ingredients': len(train_overlap),
        'test_with_ingredients': len(test_overlap),
        'train_coverage_pct': round(100 * len(train_overlap) / len(train_ids), 2) if train_ids else 0,
        'test_coverage_pct': round(100 * len(test_overlap) / len(test_ids), 2) if test_ids else 0,
        'ingredients_per_dish_distribution': dict(sorted(num_ingredients_dist.items())),
        'top_20_ingredients': [{'name': n, 'count': c} for n, c in top_ingredients],
        'overlapping_train_ids': sorted(list(train_overlap))[:100],  # Sample
        'overlapping_test_ids': sorted(list(test_overlap))[:100],    # Sample
    }

    analysis_path = OUTPUT_DIR / "analysis_report.json"
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"  Saved: {analysis_path}")

    # Step 6: Ingredient-Aware Baseline Evaluation
    print("\n[6/6] Ingredient-Aware Baseline Evaluation...")

    # Verify ingredient sum equals total (sanity check)
    errors = []
    test_results = []

    for dish_id in test_overlap:
        dish = all_dishes[dish_id]
        total_cal = dish['total_calories']

        # Sum per-ingredient calories
        ingr_sum = sum(ingr['calories'] for ingr in dish['ingredients'])

        # Sanity check: sum should equal total (within rounding)
        if abs(ingr_sum - total_cal) > 1.0:
            errors.append({
                'dish_id': dish_id,
                'total_cal': total_cal,
                'ingr_sum': ingr_sum,
                'diff': abs(ingr_sum - total_cal)
            })

        test_results.append({
            'dish_id': dish_id,
            'total_calories': total_cal,
            'ingredient_sum': ingr_sum,
            'num_ingredients': dish['num_ingredients']
        })

    print(f"  Test dishes evaluated: {len(test_results)}")
    if errors:
        print(f"  WARNING: {len(errors)} dishes have sum mismatch > 1 kcal")
    else:
        print(f"  Sanity check PASSED: ingredient sums match totals")

    # Compute metrics for "perfect ingredient identification" baseline
    # This is the upper bound - if we knew exact ingredient masses
    import math

    gt_calories = [r['total_calories'] for r in test_results]
    pred_calories = [r['ingredient_sum'] for r in test_results]

    # Since ingredient sums should equal totals, this should be near-zero
    mae = sum(abs(p - g) for p, g in zip(pred_calories, gt_calories)) / len(gt_calories)
    rmse = math.sqrt(sum((p - g) ** 2 for p, g in zip(pred_calories, gt_calories)) / len(gt_calories))

    mean_gt = sum(gt_calories) / len(gt_calories)
    ss_tot = sum((g - mean_gt) ** 2 for g in gt_calories)
    ss_res = sum((p - g) ** 2 for p, g in zip(pred_calories, gt_calories))
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    print(f"\n  Ingredient Sum vs Total (sanity check):")
    print(f"    MAE: {mae:.2f} kcal")
    print(f"    RMSE: {rmse:.2f} kcal")
    print(f"    R²: {r2:.4f}")

    # Method comparison table
    print("\n" + "=" * 60)
    print("METHOD COMPARISON (Test Set)")
    print("=" * 60)
    print(f"{'Method':<30} {'MAE (kcal)':<15} {'R²':<10}")
    print("-" * 55)
    print(f"{'Category Baseline':<30} {'153.43':<15} {'0.050':<10}")
    print(f"{'Two-Stage (GT Category)':<30} {'135.99':<15} {'0.053':<10}")
    print(f"{'Direct Regression':<30} {'94.32':<15} {'0.128':<10}")
    print(f"{'Ingredient Sum (Oracle)':<30} {f'{mae:.2f}':<15} {f'{r2:.3f}':<10}")
    print("-" * 55)
    print(f"Note: Ingredient Sum uses GT per-ingredient annotations.")
    print(f"      This represents the upper bound if we had perfect")
    print(f"      ingredient identification and mass estimation.")

    # Save detailed results
    eval_output = {
        'test_count': len(test_results),
        'sanity_check_passed': len(errors) == 0,
        'mismatches': len(errors),
        'ingredient_sum_mae': round(mae, 4),
        'ingredient_sum_rmse': round(rmse, 4),
        'ingredient_sum_r2': round(r2, 4),
        'comparison_table': {
            'category_baseline': {'mae': 153.43, 'r2': 0.050},
            'two_stage_gt': {'mae': 135.99, 'r2': 0.053},
            'direct_regression': {'mae': 94.32, 'r2': 0.128},
            'ingredient_sum_oracle': {'mae': round(mae, 2), 'r2': round(r2, 3)}
        },
        'test_samples': test_results[:50]  # Sample for inspection
    }

    eval_path = OUTPUT_DIR / "ingredient_baseline_eval.json"
    with open(eval_path, 'w') as f:
        json.dump(eval_output, f, indent=2)
    print(f"\nSaved: {eval_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Ingredient vocabulary: {len(ingredient_vocab)} ingredients")
    print(f"Dishes with annotations: {len(all_dishes)}")
    print(f"Train coverage: {len(train_overlap)}/{len(train_ids)} ({analysis['train_coverage_pct']:.1f}%)")
    print(f"Test coverage: {len(test_overlap)}/{len(test_ids)} ({analysis['test_coverage_pct']:.1f}%)")

    if len(test_overlap) > 0:
        print(f"\nIngredient-aware evaluation is FEASIBLE on {len(test_overlap)} test samples.")
    else:
        print("\nWARNING: No test samples have ingredient annotations.")

    print("\nOutputs saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
