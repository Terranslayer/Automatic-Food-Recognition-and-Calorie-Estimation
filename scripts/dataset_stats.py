"""
Generate comprehensive statistics for Nutrition5k dataset.

This script generates:
- Dataset size statistics (train/val/test splits)
- Class distribution
- Calorie distribution
- Visualizations

Usage:
    python scripts/dataset_stats.py [--data-dir DATA_DIR] [--output-dir OUTPUT_DIR]
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.dataset import Nutrition5kDataset
from utils.data_loader import get_dataset_info
from utils.visualize import (
    plot_class_distribution,
    plot_calorie_distribution,
    visualize_dataset_samples,
)


def generate_statistics(data_dir: str, output_dir: str):
    """
    Generate and save dataset statistics.

    Args:
        data_dir: Path to dataset root directory
        output_dir: Path to output directory for reports and visualizations
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("NUTRITION5K DATASET STATISTICS")
    print("="*70)
    print(f"Dataset directory: {data_dir}")
    print(f"Output directory: {output_path.absolute()}\n")

    try:
        # Get dataset info
        print("Loading dataset information...")
        info = get_dataset_info(data_dir, mode="classification")

        # Print summary statistics
        print("\n" + "-"*70)
        print("DATASET SUMMARY")
        print("-"*70)
        print(f"Number of classes: {info['num_classes']}")
        print(f"Train samples: {info['train_size']}")
        print(f"Validation samples: {info['val_size']}")
        print(f"Test samples: {info['test_size']}")
        print(f"Total samples: {info['total_size']}")

        # Print class distribution
        print("\n" + "-"*70)
        print("CLASS DISTRIBUTION (Training Set)")
        print("-"*70)
        for category, count in sorted(
            info['train_class_distribution'].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            percentage = 100 * count / info['train_size']
            print(f"{category:30s}: {count:5d} ({percentage:5.1f}%)")

        # Print calorie statistics
        print("\n" + "-"*70)
        print("CALORIE STATISTICS (Training Set)")
        print("-"*70)
        cal_stats = info['train_calorie_stats']
        print(f"Minimum: {cal_stats['min']:.1f} kcal")
        print(f"Maximum: {cal_stats['max']:.1f} kcal")
        print(f"Mean: {cal_stats['mean']:.1f} kcal")
        print(f"Median: {cal_stats['median']:.1f} kcal")
        print(f"Std Dev: {cal_stats['std']:.1f} kcal")

        # Save text report
        report_path = output_path / "dataset_stats.txt"
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("NUTRITION5K DATASET STATISTICS\n")
            f.write("="*70 + "\n\n")

            f.write("DATASET SUMMARY\n")
            f.write("-"*70 + "\n")
            f.write(f"Number of classes: {info['num_classes']}\n")
            f.write(f"Train samples: {info['train_size']}\n")
            f.write(f"Validation samples: {info['val_size']}\n")
            f.write(f"Test samples: {info['test_size']}\n")
            f.write(f"Total samples: {info['total_size']}\n\n")

            f.write("CLASS DISTRIBUTION (Training Set)\n")
            f.write("-"*70 + "\n")
            for category, count in sorted(
                info['train_class_distribution'].items(),
                key=lambda x: x[1],
                reverse=True
            ):
                percentage = 100 * count / info['train_size']
                f.write(f"{category:30s}: {count:5d} ({percentage:5.1f}%)\n")

            f.write("\nCALORIE STATISTICS (Training Set)\n")
            f.write("-"*70 + "\n")
            f.write(f"Minimum: {cal_stats['min']:.1f} kcal\n")
            f.write(f"Maximum: {cal_stats['max']:.1f} kcal\n")
            f.write(f"Mean: {cal_stats['mean']:.1f} kcal\n")
            f.write(f"Median: {cal_stats['median']:.1f} kcal\n")
            f.write(f"Std Dev: {cal_stats['std']:.1f} kcal\n")

        print(f"\n✓ Saved text report to {report_path}")

        # Generate visualizations
        print("\nGenerating visualizations...")

        # Class distribution plot
        print("  - Class distribution plot...")
        plot_class_distribution(
            info['train_class_distribution'],
            save_path=str(output_path / "class_distribution.png"),
            title="Training Set Class Distribution"
        )

        # Calorie distribution plot
        print("  - Calorie distribution plot...")
        train_dataset = Nutrition5kDataset(
            root_dir=data_dir,
            split="train",
            mode="regression",
        )
        calories = [item["calories"] for item in train_dataset.metadata]
        plot_calorie_distribution(
            calories,
            save_path=str(output_path / "calorie_distribution.png"),
            title="Training Set Calorie Distribution"
        )

        # Sample visualizations
        print("  - Sample images (train)...")
        visualize_dataset_samples(
            train_dataset,
            num_samples=9,
            save_path=str(output_path / "sample_images_train.png")
        )

        val_dataset = Nutrition5kDataset(
            root_dir=data_dir,
            split="val",
            mode="regression",
        )
        print("  - Sample images (val)...")
        visualize_dataset_samples(
            val_dataset,
            num_samples=9,
            save_path=str(output_path / "sample_images_val.png")
        )

        print("\n" + "="*70)
        print("✓ STATISTICS GENERATION COMPLETE")
        print("="*70)
        print(f"\nOutput files:")
        print(f"  - {output_path / 'dataset_stats.txt'}")
        print(f"  - {output_path / 'class_distribution.png'}")
        print(f"  - {output_path / 'calorie_distribution.png'}")
        print(f"  - {output_path / 'sample_images_train.png'}")
        print(f"  - {output_path / 'sample_images_val.png'}")

        return 0

    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("\nPlease ensure the dataset is downloaded:")
        print("  python scripts/download_data.py")
        print("\nAnd verified:")
        print("  python scripts/verify_data.py")
        return 1

    except Exception as e:
        print(f"\nERROR: Unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Generate Nutrition5k dataset statistics and visualizations"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data/nutrition5k",
        help="Path to dataset directory (default: ./data/nutrition5k)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./experiments/phase2_data",
        help="Path to output directory (default: ./experiments/phase2_data)"
    )

    args = parser.parse_args()

    exit_code = generate_statistics(args.data_dir, args.output_dir)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
