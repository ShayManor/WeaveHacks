#!/usr/bin/env python3
"""
Lightweight Model Analysis - Analyzes model without fully loading it
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import defaultdict


class LightweightModelAnalyzer:
    def __init__(self, model_path: str, output_dir: str = "./analysis_output"):
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def analyze_config(self):
        """Analyze model configuration"""
        config_path = self.model_path / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)

            print("\n=== Model Configuration ===")
            important_keys = [
                "model_type", "architectures", "hidden_size",
                "num_hidden_layers", "num_attention_heads",
                "intermediate_size", "vocab_size",
                "max_position_embeddings", "torch_dtype"
            ]

            config_summary = {}
            for key in important_keys:
                if key in config:
                    value = config[key]
                    print(f"{key}: {value}")
                    config_summary[key] = value

            # Calculate approximate model size
            if all(k in config for k in ["hidden_size", "num_hidden_layers", "vocab_size"]):
                # Rough estimation of parameters
                h = config["hidden_size"]
                l = config["num_hidden_layers"]
                v = config["vocab_size"]

                # Approximate parameter count (simplified)
                embedding_params = v * h
                attention_params = l * (4 * h * h)  # Q, K, V, O projections
                mlp_params = l * (8 * h * h)  # Typical MLP is 4x hidden size, in and out

                total_params = embedding_params + attention_params + mlp_params
                print(f"\nEstimated parameters: {total_params:,} ({total_params / 1e9:.2f}B)")
                config_summary["estimated_parameters"] = total_params

            return config_summary
        else:
            print("No config.json found")
            return {}

    def analyze_checkpoint(self):
        """Analyze the model checkpoint without fully loading it"""
        checkpoint_path = self.model_path / "pytorch_model.bin"

        if not checkpoint_path.exists():
            print("No pytorch_model.bin found")
            return {}

        print("\n=== Checkpoint Analysis ===")

        # Get file size
        file_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
        print(f"Checkpoint size: {file_size_mb:.2f} MB")

        # Load checkpoint with weights_only for safety
        print("Loading checkpoint metadata...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

        # Analyze layers
        layer_info = defaultdict(lambda: {"count": 0, "total_params": 0, "layers": []})
        total_params = 0

        for name, tensor in checkpoint.items():
            # Skip non-parameter entries
            if not isinstance(tensor, torch.Tensor):
                continue

            num_params = tensor.numel()
            total_params += num_params

            # Categorize layers
            if "embed" in name:
                category = "embedding"
            elif "attention" in name or "attn" in name:
                category = "attention"
            elif "mlp" in name or "fc" in name:
                category = "mlp"
            elif "norm" in name:
                category = "normalization"
            elif "lm_head" in name or "output" in name:
                category = "output"
            else:
                category = "other"

            layer_info[category]["count"] += 1
            layer_info[category]["total_params"] += num_params
            layer_info[category]["layers"].append({
                "name": name,
                "shape": list(tensor.shape),
                "params": num_params,
                "dtype": str(tensor.dtype)
            })

        print(f"\nTotal parameters: {total_params:,} ({total_params / 1e9:.2f}B)")

        # Print category summary
        print("\nParameter distribution by category:")
        for category, info in layer_info.items():
            pct = (info["total_params"] / total_params) * 100
            print(f"  {category}: {info['count']} layers, {info['total_params']:,} params ({pct:.1f}%)")

        # Analyze weight statistics for a few layers
        print("\n=== Sample Weight Statistics ===")
        sample_layers = ["model.embed_tokens.weight", "model.layers.0.self_attn.q_proj.weight",
                         "model.layers.0.mlp.gate_proj.weight", "lm_head.weight"]

        weight_stats = {}
        for layer_name in sample_layers:
            if layer_name in checkpoint:
                tensor = checkpoint[layer_name]
                stats = {
                    "shape": list(tensor.shape),
                    "mean": float(tensor.mean()),
                    "std": float(tensor.std()),
                    "min": float(tensor.min()),
                    "max": float(tensor.max()),
                    "zeros": int((tensor == 0).sum()),
                    "sparsity": float((tensor == 0).sum() / tensor.numel())
                }
                weight_stats[layer_name] = stats
                print(f"\n{layer_name}:")
                print(f"  Shape: {stats['shape']}")
                print(f"  Mean: {stats['mean']:.6f}, Std: {stats['std']:.6f}")
                print(f"  Range: [{stats['min']:.6f}, {stats['max']:.6f}]")
                print(f"  Sparsity: {stats['sparsity']:.2%}")

        # Create visualizations
        self.create_analysis_plots(layer_info, weight_stats)

        # Clean up memory
        del checkpoint

        return {
            "file_size_mb": file_size_mb,
            "total_parameters": total_params,
            "layer_categories": dict(layer_info),
            "sample_weight_stats": weight_stats
        }

    def create_analysis_plots(self, layer_info, weight_stats):
        """Create visualization plots"""
        # 1. Parameter distribution pie chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        categories = list(layer_info.keys())
        params = [info["total_params"] for info in layer_info.values()]

        ax1.pie(params, labels=categories, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Parameter Distribution by Layer Type')

        # 2. Layer count bar chart
        counts = [info["count"] for info in layer_info.values()]
        ax2.bar(categories, counts, color='skyblue')
        ax2.set_title('Number of Layers by Type')
        ax2.set_xlabel('Layer Type')
        ax2.set_ylabel('Count')
        ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_structure_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Weight statistics for sample layers
        if weight_stats:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.ravel()

            for idx, (layer_name, stats) in enumerate(list(weight_stats.items())[:4]):
                ax = axes[idx]

                # Create a simple bar chart of statistics
                stat_names = ['mean', 'std', 'min', 'max']
                stat_values = [stats[s] for s in stat_names]

                bars = ax.bar(stat_names, stat_values, color=['blue', 'orange', 'green', 'red'])
                ax.set_title(f'{layer_name.split(".")[-2]}.{layer_name.split(".")[-1]}')
                ax.set_ylabel('Value')

                # Add value labels
                for bar, val in zip(bars, stat_values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2., height,
                            f'{val:.4f}', ha='center', va='bottom' if val >= 0 else 'top')

            plt.tight_layout()
            plt.savefig(self.output_dir / 'weight_statistics_samples.png', dpi=300, bbox_inches='tight')
            plt.close()

        print(f"\nPlots saved to {self.output_dir}/")

    def analyze_training_args(self):
        """Analyze training arguments if available"""
        training_args_path = self.model_path / "training_args.bin"

        if training_args_path.exists():
            print("\n=== Training Arguments ===")
            try:
                # Try to load training args
                training_args = torch.load(training_args_path, map_location='cpu', weights_only=True)

                important_args = [
                    "learning_rate", "num_train_epochs", "per_device_train_batch_size",
                    "gradient_accumulation_steps", "warmup_steps", "max_steps",
                    "lr_scheduler_type", "optim", "adam_beta1", "adam_beta2",
                    "adam_epsilon", "max_grad_norm", "weight_decay"
                ]

                training_summary = {}
                for arg in important_args:
                    if hasattr(training_args, arg):
                        value = getattr(training_args, arg)
                        print(f"{arg}: {value}")
                        training_summary[arg] = value

                return training_summary
            except Exception as e:
                print(f"Could not load training args: {e}")
                return {}
        else:
            print("No training_args.bin found")
            return {}

    def generate_report(self):
        """Generate comprehensive analysis report"""
        print("=" * 60)
        print("Lightweight Model Analysis")
        print("=" * 60)

        report = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "model_path": str(self.model_path),
                "output_dir": str(self.output_dir)
            }
        }

        # 1. Config analysis
        report["config_analysis"] = self.analyze_config()

        # 2. Checkpoint analysis
        report["checkpoint_analysis"] = self.analyze_checkpoint()

        # 3. Training args analysis
        report["training_args"] = self.analyze_training_args()

        # Save report
        report_path = self.output_dir / f"lightweight_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n=== Analysis Complete ===")
        print(f"Report saved to: {report_path}")
        print(f"Visualizations saved to: {self.output_dir}/")

        return report


def main():
    MODEL_PATH = "./downloaded_model/"  # Adjust this path

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model path '{MODEL_PATH}' does not exist.")
        return

    analyzer = LightweightModelAnalyzer(MODEL_PATH)
    analyzer.generate_report()


if __name__ == "__main__":
    main()