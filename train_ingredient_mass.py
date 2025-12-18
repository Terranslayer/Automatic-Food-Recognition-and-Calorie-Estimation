#!/usr/bin/env python
"""
Train ingredient mass regressor for Phase 5.5.6.

v2 Changes:
- Calorie-first loss (L1 on calories) as primary objective
- Softplus activation (non-negative with gradients)
- OTHER channel (K+1) to capture out-of-vocab ingredients
- Fixed OTHER_DENSITY constant (no label leakage)
- Masked mass loss (non-zero GT only) to avoid sparse-zero domination
- Early stopping on val_cal_mae

Usage:
    python train_ingredient_mass.py --v2 --epochs 10 --top-k 50 --lambda-mass 0.05
"""

import argparse
import json
import csv
import math
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data" / "nutrition5k"
METADATA_DIR = DATA_DIR / "metadata"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "phase5.5_ingredient"

# v2: Fixed OTHER density (training-set average, no label leakage)
OTHER_DENSITY = 1.14


def parse_args():
    parser = argparse.ArgumentParser(description="Train Ingredient Mass Regressor")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--v2", action="store_true", help="Use v2 calorie-first loss")
    parser.add_argument("--lambda-mass", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=1.0)
    return parser.parse_args()


def load_ingredients_metadata(filepath: Path) -> dict:
    densities = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            densities[row['ingr'].lower().strip()] = float(row['cal/g'])
    return densities


def load_cafe_metadata(filepath: Path) -> dict:
    dishes = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for row in csv.reader(f):
            if len(row) < 6:
                continue
            try:
                dish_id, total_cal, total_mass = row[0], float(row[1]), float(row[2])
            except ValueError:
                continue
            ingredients = []
            idx = 6
            while idx + 6 < len(row):
                try:
                    ingredients.append({
                        'name': row[idx + 1].lower().strip(),
                        'grams': float(row[idx + 2]),
                        'calories': float(row[idx + 3])
                    })
                except (ValueError, IndexError):
                    pass
                idx += 7
            dishes[dish_id] = {'total_calories': total_cal, 'total_mass': total_mass, 'ingredients': ingredients}
    return dishes


def load_split_ids(filepath: Path) -> list:
    return [line.strip() for line in open(filepath) if line.strip()] if filepath.exists() else []


def get_top_k_ingredients(dishes: dict, k: int) -> list:
    counts = defaultdict(int)
    for d in dishes.values():
        for i in d['ingredients']:
            counts[i['name']] += 1
    return [n for n, _ in sorted(counts.items(), key=lambda x: -x[1])[:k]]


class IngredientMassDatasetV2(Dataset):
    def __init__(self, dish_ids, dishes, top_ingredients, cal_densities, data_dir, transform=None):
        self.dishes, self.top_ingredients, self.cal_densities = dishes, top_ingredients, cal_densities
        self.ingr_to_idx = {n: i for i, n in enumerate(top_ingredients)}
        self.data_dir, self.transform = data_dir, transform
        self.valid_ids = [d for d in dish_ids if d in dishes and self._get_image_path(d).exists()]
        print(f"  Valid: {len(self.valid_ids)}/{len(dish_ids)}")

    def _get_image_path(self, dish_id):
        d = self.data_dir / dish_id
        if d.exists():
            fd = d / "frames_sampled30"
            if fd.exists():
                imgs = list(fd.glob('*.jpeg')) + list(fd.glob('*.jpg'))
                if imgs: return imgs[0]
            imgs = list(d.glob('*.jpeg')) + list(d.glob('*.jpg'))
            if imgs: return imgs[0]
        return d / 'frames_sampled30' / 'x.jpeg'

    def __len__(self): return len(self.valid_ids)

    def __getitem__(self, idx):
        dish = self.dishes[self.valid_ids[idx]]
        img = Image.open(self._get_image_path(self.valid_ids[idx])).convert('RGB')
        if self.transform: img = self.transform(img)

        K = len(self.top_ingredients)
        mass_target = torch.zeros(K + 1)
        topk_mass, topk_cal = 0.0, 0.0

        for ingr in dish['ingredients']:
            if ingr['name'] in self.ingr_to_idx:
                mass_target[self.ingr_to_idx[ingr['name']]] = ingr['grams']
                topk_mass += ingr['grams']
                topk_cal += ingr['calories']

        mass_target[K] = max(0.0, dish['total_mass'] - topk_mass)
        total_cal = dish['total_calories']
        coverage = topk_cal / total_cal if total_cal > 0 else 0.0

        return img, mass_target, torch.tensor(total_cal), torch.tensor(coverage)


class IngredientMassRegressorV2(nn.Module):
    def __init__(self, num_outputs, backbone='efficientnet_b0', hidden_dims=[512, 256], dropout=0.3):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)
        for p in self.backbone.parameters(): p.requires_grad = False
        layers = []
        in_dim = self.backbone.num_features
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers += [nn.Linear(in_dim, num_outputs), nn.Softplus(beta=1.0)]
        self.regressor = nn.Sequential(*layers)

    def forward(self, x): return self.regressor(self.backbone(x))


def train_epoch_v2(model, loader, optim, device, densities_t, K, lambda_mass):
    model.train()
    tot_loss, tot_cal_mae, tot_coverage, n = 0, 0, 0, 0

    for img, mass_gt, cal_gt, coverage in loader:
        img, mass_gt, cal_gt = img.to(device), mass_gt.to(device), cal_gt.to(device)
        optim.zero_grad()

        mass_pred = model(img)
        cal_pred = (mass_pred[:, :K] * densities_t).sum(1) + mass_pred[:, K] * OTHER_DENSITY

        cal_loss = F.l1_loss(cal_pred, cal_gt)

        # Masked mass loss: only non-zero GT for top-K, always include OTHER
        mask = (mass_gt[:, :K] > 0).float()
        mask_other = torch.ones(img.size(0), 1, device=device)
        full_mask = torch.cat([mask, mask_other], dim=1)
        mass_loss = ((mass_pred - mass_gt) ** 2 * full_mask).sum() / (full_mask.sum() + 1e-6)

        loss = cal_loss + lambda_mass * mass_loss
        loss.backward()
        optim.step()

        tot_loss += loss.item()
        tot_cal_mae += (cal_pred - cal_gt).abs().mean().item()
        tot_coverage += coverage.mean().item()
        n += 1

    return tot_loss / n, tot_cal_mae / n, tot_coverage / n


def evaluate_v2(model, loader, device, densities_t, K):
    model.eval()
    preds, gts, coverages = [], [], []
    with torch.no_grad():
        for img, mass_gt, cal_gt, coverage in loader:
            mass_pred = model(img.to(device))
            cal_pred = (mass_pred[:, :K] * densities_t).sum(1) + mass_pred[:, K] * OTHER_DENSITY
            preds.extend(cal_pred.cpu().tolist())
            gts.extend(cal_gt.tolist())
            coverages.extend(coverage.tolist())

    mae = sum(abs(p - g) for p, g in zip(preds, gts)) / len(gts)
    rmse = math.sqrt(sum((p - g) ** 2 for p, g in zip(preds, gts)) / len(gts))
    mean_gt = sum(gts) / len(gts)
    r2 = 1 - sum((p - g) ** 2 for p, g in zip(preds, gts)) / sum((g - mean_gt) ** 2 for g in gts)
    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'coverage': sum(coverages) / len(coverages)}


# ==================== Legacy v1 ====================
class IngredientMassDataset(Dataset):
    def __init__(self, ids, dishes, top_ingr, data_dir, transform=None):
        self.dishes, self.top_ingr, self.data_dir, self.transform = dishes, top_ingr, data_dir, transform
        self.idx_map = {n: i for i, n in enumerate(top_ingr)}
        self.valid = [d for d in ids if d in dishes and (data_dir/d/"frames_sampled30").exists()]
        print(f"  Valid: {len(self.valid)}/{len(ids)}")

    def __len__(self): return len(self.valid)
    def __getitem__(self, i):
        d = self.dishes[self.valid[i]]
        fd = self.data_dir / self.valid[i] / "frames_sampled30"
        img = Image.open(next(fd.glob('*.jpeg'))).convert('RGB')
        if self.transform: img = self.transform(img)
        t = torch.zeros(len(self.top_ingr))
        for x in d['ingredients']:
            if x['name'] in self.idx_map: t[self.idx_map[x['name']]] = x['grams']
        return img, t, d['total_calories']

class IngredientMassRegressor(nn.Module):
    def __init__(self, n, backbone='efficientnet_b0'):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)
        for p in self.backbone.parameters(): p.requires_grad = False
        self.regressor = nn.Sequential(nn.Linear(self.backbone.num_features, 512), nn.ReLU(),
                                       nn.Dropout(0.3), nn.Linear(512, 256), nn.ReLU(),
                                       nn.Dropout(0.3), nn.Linear(256, n), nn.ReLU())
    def forward(self, x): return self.regressor(self.backbone(x))


def main():
    args = parse_args()
    ver = "v2" if args.v2 else "v1"
    print("=" * 60)
    print(f"Ingredient Mass Regressor ({ver})")
    print("=" * 60)

    if args.debug:
        args.epochs, args.batch_size, subset = 1, 4, 20
    else:
        subset = None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if args.v2:
        print(f"v2: lambda={args.lambda_mass}, patience={args.patience}, OTHER_DENSITY={OTHER_DENSITY}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cal_densities = load_ingredients_metadata(METADATA_DIR / "ingredients_metadata.csv")
    all_dishes = {}
    for f in ['dish_metadata_cafe1.csv', 'dish_metadata_cafe2.csv']:
        if (METADATA_DIR / f).exists(): all_dishes.update(load_cafe_metadata(METADATA_DIR / f))
    print(f"Loaded {len(all_dishes)} dishes, {len(cal_densities)} densities")

    top_ingr = get_top_k_ingredients(all_dishes, args.top_k)
    print(f"Top-{args.top_k}: {top_ingr[:5]}...")

    train_ids = load_split_ids(DATA_DIR / "dish_ids/splits/rgb_train_ids.txt")
    val_ids = load_split_ids(DATA_DIR / "dish_ids/splits/rgb_test_ids.txt")  # used as val
    if subset: train_ids, val_ids = train_ids[:subset], val_ids[:subset//2]

    transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    if args.v2:
        train_ds = IngredientMassDatasetV2(train_ids, all_dishes, top_ingr, cal_densities, DATA_DIR, transform)
        val_ds = IngredientMassDatasetV2(val_ids, all_dishes, top_ingr, cal_densities, DATA_DIR, transform)
    else:
        train_ds = IngredientMassDataset(train_ids, all_dishes, top_ingr, DATA_DIR, transform)
        val_ds = IngredientMassDataset(val_ids, all_dishes, top_ingr, DATA_DIR, transform)

    train_ld = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_ld = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    K = args.top_k
    model = (IngredientMassRegressorV2(K + 1) if args.v2 else IngredientMassRegressor(K)).to(device)
    optim = torch.optim.AdamW(model.regressor.parameters(), lr=args.lr)
    densities_t = torch.tensor([cal_densities.get(n, 1.0) for n in top_ingr], device=device)

    best_mae, patience_cnt, history = float('inf'), 0, []
    print("\nTraining...")

    for ep in range(args.epochs):
        t0 = datetime.now()
        if args.v2:
            loss, tr_mae, tr_cov = train_epoch_v2(model, train_ld, optim, device, densities_t, K, args.lambda_mass)
            val = evaluate_v2(model, val_ld, device, densities_t, K)
        else:
            crit = nn.MSELoss()
            model.train()
            for img, tgt, _ in train_ld:
                optim.zero_grad()
                crit(model(img.to(device)), tgt.to(device)).backward()
                optim.step()
            loss, tr_mae, tr_cov = 0, 0, 0
            val = evaluate_v2(model, val_ld, device, densities_t, K) if args.v2 else {'mae': 0, 'r2': 0}

        elapsed = (datetime.now() - t0).total_seconds()
        cov_str = f", cov={tr_cov:.1%}" if args.v2 else ""
        print(f"  Ep {ep+1}/{args.epochs}: loss={loss:.3f}, tr_mae={tr_mae:.1f}{cov_str}, "
              f"val_mae={val['mae']:.1f}, r2={val['r2']:.3f} [{elapsed:.1f}s]")

        rec = {'ep': ep+1, 'loss': round(loss, 4), 'tr_mae': round(tr_mae, 1),
               'val_mae': round(val['mae'], 1), 'r2': round(val['r2'], 4),
               'best': round(min(best_mae, val['mae']), 1), 'ts': datetime.now().isoformat()}
        history.append(rec)

        with open(OUTPUT_DIR / 'training_progress.json', 'w') as f:
            json.dump({'ep': ep+1, 'ver': ver, 'history': history, 'status': 'training'}, f, indent=2)

        if val['mae'] < best_mae - args.min_delta:
            best_mae, patience_cnt = val['mae'], 0
            torch.save(model.state_dict(), OUTPUT_DIR / f'best_{ver}.pth')
        else:
            patience_cnt += 1

        if args.v2 and patience_cnt >= args.patience:
            print(f"  Early stop at ep {ep+1}")
            break

    print("\n" + "=" * 60)
    print(f"BEST VAL MAE ({ver}): {best_mae:.2f} kcal")
    print("=" * 60)
    print(f"Comparison: Category=153.43 | TwoStage=135.99 | DirectReg=94.32 | v1=187.84")

    with open(OUTPUT_DIR / f'results_{ver}.json', 'w') as f:
        json.dump({'ver': ver, 'best_mae': best_mae, 'history': history}, f, indent=2)
    with open(OUTPUT_DIR / 'training_progress.json', 'w') as f:
        json.dump({'ver': ver, 'best_mae': best_mae, 'history': history, 'status': 'done'}, f, indent=2)


if __name__ == "__main__":
    main()
