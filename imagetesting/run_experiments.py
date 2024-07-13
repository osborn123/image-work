import subprocess
import matplotlib.pyplot as plt
import numpy as np
import os

def run_command(command, cwd=None):
    result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=cwd)
    if result.returncode != 0:
        print(f"Error running command: {command}\n{result.stderr}")
    return result.stdout

def extract_results(output):
    lines = output.split('\n')
    recall_results = []
    topk_accuracy = None
    for line in lines:
        if line.startswith("Recall@"):
            recall_results.append(float(line.split(": ")[1]))
        elif line.startswith("Top-k Accuracy"):
            topk_accuracy = float(line.split(": ")[1].split('%')[0])
    return recall_results, topk_accuracy

def plot_results(overlap_rates, results, metric, save_dir):
    for transform_method, result in results.items():
        plt.figure(figsize=(10, 6))
        model1_results = [r.get(f'model1_{metric}', 0) for r in result]
        model2_results = [r.get(f'model2_{metric}', 0) for r in result]
        transformed_results = [r.get(f'transformed_{metric}', 0) for r in result]

        plt.plot(overlap_rates, model1_results, 'r-', label="Model 1")
        plt.plot(overlap_rates, model2_results, 'b-', label="Model 2")
        plt.plot(overlap_rates, transformed_results, 'g-', label="Transformed")

        plt.xlabel('Overlap Rate')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f'{metric.replace("_", " ").title()} vs Overlap Rate ({transform_method})')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f'{transform_method}_{metric}.png'))
        plt.close()

def run_experiment(transform_method, overlap_rate, top_k, script_dir):
    # Run test_trans.py
    command = f"python test_trans.py --transform_method {transform_method} --top_k {top_k} --overlap_rate {overlap_rate}"
    output = run_command(command, cwd=script_dir)
    recall_results, topk_accuracy = extract_results(output)
    if not recall_results:
        recall_results = [0]  # Ensure recall_results is not empty to avoid errors

    mean_recall = np.mean(recall_results)
    max_recall = np.max(recall_results)
    mean_square_recall = np.mean(np.square(recall_results))

    # Run model1 script (train_model1.py)
    command = f"python train_model1.py --top_k {top_k} --epochs 5"
    output = run_command(command, cwd=script_dir)
    model1_recall_results, model1_topk_accuracy = extract_results(output)
    if not model1_recall_results:
        model1_recall_results = [0]

    # Run model2 script (train_model2.py)
    command = f"python train_model2.py --top_k {top_k} --epochs 5"
    output = run_command(command, cwd=script_dir)
    model2_recall_results, model2_topk_accuracy = extract_results(output)
    if not model2_recall_results:
        model2_recall_results = [0]

    return {
        'model1_mean_recall': np.mean(model1_recall_results),
        'model2_mean_recall': np.mean(model2_recall_results),
        'transformed_mean_recall': mean_recall,
        'model1_max_recall': np.max(model1_recall_results),
        'model2_max_recall': np.max(model2_recall_results),
        'transformed_max_recall': max_recall,
        'model1_mean_square_recall': np.mean(np.square(model1_recall_results)),
        'model2_mean_square_recall': np.mean(np.square(model2_recall_results)),
        'transformed_mean_square_recall': mean_square_recall,
        'model1_topk_accuracy': model1_topk_accuracy,
        'model2_topk_accuracy': model2_topk_accuracy,
        'transformed_topk_accuracy': topk_accuracy
    }

def main():
    overlap_rates = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    methods = ['none', 'linear', 'class_linear', 'MMD']
    top_k = 5
    script_dir = os.path.dirname(os.path.abspath(__file__))

    results = {method: [] for method in methods}

    save_dir = 'plots'
    os.makedirs(save_dir, exist_ok=True)

    for method in methods:
        for rate in overlap_rates:
            print(f"Running {method} with overlap rate {int(rate * 100)}%")
            result = run_experiment(method, rate, top_k, script_dir)
            results[method].append(result)

    for metric in ['topk_accuracy', 'mean_recall', 'max_recall', 'mean_square_recall']:
        plot_results(overlap_rates, results, metric, save_dir)

if __name__ == "__main__":
    main()
