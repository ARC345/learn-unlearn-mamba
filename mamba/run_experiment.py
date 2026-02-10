#!/usr/bin/env python3
"""
End-to-end orchestrator for the learn/unlearn Mamba experiment.

Runs all 3 phases (finetune, noise-train, unlearn) + evaluation
for a given model and noise type, without needing YAML config files.

Usage:
    pixi run experiment-1.4b
    python run_experiment.py --model state-spaces/mamba-1.4b --noise charflip
    python run_experiment.py --model state-spaces/mamba-1.4b --noise all
    python run_experiment.py --model state-spaces/mamba-1.4b --noise wordflip --skip-phase1
"""
import argparse
import os
import sys
import time

# Ensure src/ is importable
sys.path.insert(0, os.path.dirname(__file__))

NOISE_DATASETS = {
    "charflip": "D_ad_cflipped.csv",
    "wordflip": "D_ad_wflipped.csv",
    "transliteration": "D_ad_xlit.csv",
    "counterfactual": "D_cfact_train.csv",
}

# Paper Section 3.3.2: counterfactual uses D_GK as the clean base dataset,
# all other noise types use D_ad_train.
CLEAN_DATASETS = {
    "charflip": "D_ad_train.csv",
    "wordflip": "D_ad_train.csv",
    "transliteration": "D_ad_train.csv",
    "counterfactual": "D_GK.csv",
}

ALL_NOISE_TYPES = list(NOISE_DATASETS.keys())


def short_name(hf_model: str) -> str:
    """Derive short name from HF model path: 'state-spaces/mamba-1.4b' -> 'mamba-1.4b'."""
    return hf_model.split("/")[-1]


def build_config(model_name, new_model_name, dataset_path, output_dir, args):
    """Build a training config dict programmatically."""
    return {
        "model": {
            "name": model_name,
            "new_model": new_model_name,
        },
        "dataset": {
            "path": dataset_path,
        },
        "training": {
            "output_dir": output_dir,
            "num_train_epochs": args.epochs,
            "per_device_train_batch_size": args.batch_size,
            "per_device_eval_batch_size": args.batch_size,
            "learning_rate": args.lr,
            "weight_decay": 0.1,
            "adam_beta1": 0.9,
            "adam_beta2": 0.95,
            "max_grad_norm": 1.0,
            "logging_steps": 100,
            "warmup_steps": 100,
        },
    }


def run_phase(phase_num, phase_name, model_name, config):
    """Run a single training phase."""
    print(f"\n{'='*70}")
    print(f"PHASE {phase_num}: {phase_name}")
    print(f"  Model:   {config['model']['name']}")
    print(f"  Output:  {config['model']['new_model']}")
    print(f"  Dataset: {config['dataset']['path']}")
    print(f"  Epochs:  {config['training']['num_train_epochs']}")
    print(f"  Batch:   {config['training']['per_device_train_batch_size']}")
    print(f"{'='*70}")

    from src.dataloader import load_training_data, load_eval_data
    from src.model_loader import load_model_and_tokenizer
    from src.training import train_model

    start = time.time()

    dataset_path = config["dataset"]["path"]
    train_dataset = load_training_data(dataset_path)
    eval_dataset = load_eval_data(dataset_path)

    model, tokenizer = load_model_and_tokenizer(model_name)
    train_model(model, tokenizer, train_dataset, eval_dataset, config)

    elapsed = time.time() - start
    print(f"  Phase {phase_num} completed in {elapsed/60:.1f} minutes.")

    # Free memory
    del model
    import torch
    torch.cuda.empty_cache()


def run_evaluation_phase(name, noise_type, results_dir):
    """Run response generation + metrics evaluation."""
    print(f"\n{'='*70}")
    print(f"EVALUATION: {name} / {noise_type}")
    print(f"{'='*70}")

    eval_dir = os.path.join(os.path.dirname(__file__), "evaluation")
    sys.path.insert(0, eval_dir)

    from evaluation.evaluation_trainedmodels import generate_responses
    from evaluation.evaluate_simple import run_evaluation

    # Generate responses for models matching this prefix
    csv_output_dir = os.path.join(results_dir, "eval_csvs")
    os.makedirs(csv_output_dir, exist_ok=True)

    generate_responses(
        models_dir=results_dir,
        model_prefix=name,
        output_dir=csv_output_dir,
    )

    # Run metrics
    run_evaluation(
        model_prefix=name,
        noise_type=noise_type,
        csv_dir=csv_output_dir,
        output_csv=os.path.join(results_dir, f"{name}-{noise_type}-summary.csv"),
    )


def run_single_noise(args, noise_type):
    """Run the full 3-phase pipeline for a single noise type."""
    name = short_name(args.model)
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "train")
    results_dir = os.path.join(os.path.dirname(__file__), "results")

    clean_dataset = os.path.join(data_dir, CLEAN_DATASETS[noise_type])

    finetuned_name = f"{name}-finetuned" if noise_type != "counterfactual" else f"{name}-gk-finetuned"
    noised_name = f"{name}-{noise_type}"
    unlearned_name = f"{name}-{noise_type}-unlearned"

    finetuned_path = os.path.join(results_dir, finetuned_name)

    # Phase 1: Finetune base model on clean data
    if args.skip_phase1 and os.path.isdir(finetuned_path):
        print(f"\n  Skipping Phase 1 — checkpoint exists at {finetuned_path}")
    else:
        config_p1 = build_config(
            model_name=args.model,
            new_model_name=finetuned_name,
            dataset_path=clean_dataset,
            output_dir=results_dir,
            args=args,
        )
        run_phase(1, f"Finetune on {CLEAN_DATASETS[noise_type]}", args.model, config_p1)

    # Phase 2: Train finetuned model on noised data
    noise_dataset = os.path.join(data_dir, NOISE_DATASETS[noise_type])
    config_p2 = build_config(
        model_name=finetuned_path,
        new_model_name=noised_name,
        dataset_path=noise_dataset,
        output_dir=results_dir,
        args=args,
    )
    run_phase(2, f"Train on {noise_type} noise", finetuned_path, config_p2)

    # Phase 3: Unlearn — retrain noised model on clean data
    noised_path = os.path.join(results_dir, noised_name)
    config_p3 = build_config(
        model_name=noised_path,
        new_model_name=unlearned_name,
        dataset_path=clean_dataset,
        output_dir=results_dir,
        args=args,
    )
    run_phase(3, "Unlearn (retrain on clean data)", noised_path, config_p3)

    # Evaluation
    run_evaluation_phase(name, noise_type, results_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Run the full learn/unlearn Mamba experiment."
    )
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model name (e.g. state-spaces/mamba-1.4b)")
    parser.add_argument("--noise", type=str, required=True,
                        choices=ALL_NOISE_TYPES + ["all"],
                        help="Noise type to run, or 'all' for all noise types")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs per phase (default: 5)")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Per-device batch size (default: 2)")
    parser.add_argument("--lr", type=float, default=3e-6,
                        help="Learning rate (default: 3e-6)")
    parser.add_argument("--skip-phase1", action="store_true",
                        help="Skip phase 1 if finetuned checkpoint already exists")
    args = parser.parse_args()

    noise_types = ALL_NOISE_TYPES if args.noise == "all" else [args.noise]

    print(f"Model: {args.model} ({short_name(args.model)})")
    print(f"Noise types: {noise_types}")
    print(f"Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")

    total_start = time.time()

    for i, noise_type in enumerate(noise_types):
        print(f"\n{'#'*70}")
        print(f"# NOISE TYPE {i+1}/{len(noise_types)}: {noise_type}")
        print(f"{'#'*70}")

        # Skip phase 1 for subsequent noise types that share the same clean dataset
        if i > 0 and CLEAN_DATASETS[noise_type] == CLEAN_DATASETS[noise_types[0]]:
            args.skip_phase1 = True

        run_single_noise(args, noise_type)

    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"ALL DONE in {total_elapsed/60:.1f} minutes.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
