import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import argparse
import os

# Set the style for plots
plt.style.use('seaborn-v0_8-whitegrid') # Use updated style name
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 20

def plot_training_curves(baseline_df, distill_df, teacher_acc=None, output_dir='.', prefix=''):
    """
    Generate training and validation accuracy curves from pandas DataFrames.

    Args:
        baseline_df (pd.DataFrame): DataFrame with 'epoch', 'train_acc', 'val_acc' for baseline.
        distill_df (pd.DataFrame): DataFrame with 'epoch', 'train_acc', 'val_acc' for distilled.
        teacher_acc (float, optional): Accuracy of the teacher model to draw a line. Defaults to None.
        output_dir (str): Directory to save the plot.
        prefix (str): Prefix for the output filename.
    """
    if baseline_df is None and distill_df is None:
        print("No training data provided for curves.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    max_epochs = 0

    # Plot training accuracy
    ax = axes[0]
    if baseline_df is not None:
        ax.plot(baseline_df['epoch'], baseline_df['train_acc'], label='Baseline Train Acc', color='#1f77b4', linewidth=2)
        max_epochs = max(max_epochs, baseline_df['epoch'].max())
    if distill_df is not None:
        ax.plot(distill_df['epoch'], distill_df['train_acc'], label='Distilled Train Acc', color='#ff7f0e', linewidth=2)
        max_epochs = max(max_epochs, distill_df['epoch'].max())

    ax.set_title('Training Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_xlim(0, max_epochs if max_epochs > 0 else 1)
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='lower right')

    # Plot validation accuracy
    ax = axes[1]
    if baseline_df is not None:
        ax.plot(baseline_df['epoch'], baseline_df['val_acc'], label='Baseline Val Acc', color='#1f77b4', linewidth=2)
    if distill_df is not None:
        ax.plot(distill_df['epoch'], distill_df['val_acc'], label='Distilled Val Acc', color='#ff7f0e', linewidth=2)
    if teacher_acc is not None:
        ax.axhline(y=teacher_acc, color='#2ca02c', linestyle='--', linewidth=2, label=f'Teacher ({teacher_acc:.2f}%)')

    ax.set_title('Validation Accuracy')
    ax.set_xlabel('Epoch')
    # ax.set_ylabel('Accuracy (%)') # Shared Y axis
    ax.set_xlim(0, max_epochs if max_epochs > 0 else 1)
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='lower right')

    plt.suptitle(f'{prefix} Training Progress Comparison'.strip(), fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    output_filename = os.path.join(output_dir, f"{prefix}accuracy_curves.png".strip())
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Training curves plot saved to {output_filename}")
    plt.close(fig) # Close the figure to free memory

def plot_model_comparison(comparison_df, output_dir='.', prefix=''):
    """
    Create a bar chart comparing model sizes and accuracies from a DataFrame.

    Args:
        comparison_df (pd.DataFrame): DataFrame with 'model_name', 'params_M', 'accuracy_pct'.
        output_dir (str): Directory to save the plot.
        prefix (str): Prefix for the output filename.
    """
    if comparison_df is None:
        print("No model comparison data provided.")
        return

    models = comparison_df['model_name'].tolist()
    params = comparison_df['params_M'].tolist()
    accuracies = comparison_df['accuracy_pct'].tolist()

    # Create figure
    fig, ax1 = plt.subplots(figsize=(max(10, len(models) * 1.5), 6)) # Adjust width based on number of models

    # Plot parameters
    x = np.arange(len(models))
    width = 0.35

    bar1 = ax1.bar(x - width/2, params, width, label='Parameters (M)', color='#1f77b4')
    ax1.set_ylabel('Parameters (M)', fontsize=14)
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    #ax1.set_ylim(0, max(params)*1.1 if params else 1)

    # Plot accuracies on secondary y-axis
    ax2 = ax1.twinx()
    bar2 = ax2.bar(x + width/2, accuracies, width, label='Top-1 Accuracy (%)', color='#ff7f0e')
    ax2.set_ylabel('Top-1 Accuracy (%)', fontsize=14)
    ax2.tick_params(axis='y', labelcolor='#ff7f0e')
    ax2.set_ylim(0, 100)

    # Add labels and legend
    ax1.set_xlabel('Model', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=15, ha='right') # Rotate labels slightly if many models

    # Add value labels
    ax1.bar_label(bar1, fmt='%.2f', padding=3)
    ax2.bar_label(bar2, fmt='%.2f', padding=3)

    # Create combined legend (optional, can get crowded)
    # lines1, labels1 = ax1.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()
    # ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=2)


    plt.title(f'{prefix} Model Size vs. Accuracy Comparison'.strip(), fontsize=16, pad=30) # Add padding
    plt.tight_layout(rect=[0, 0, 1, 0.9]) # Adjust layout to prevent title overlap with legend
    output_filename = os.path.join(output_dir, f"{prefix}model_comparison.png".strip())
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Model comparison plot saved to {output_filename}")
    plt.close(fig)

def plot_parameter_efficiency_scatter(comparison_df, output_dir='.', prefix=''):
    """
    Create a scatter plot showing parameter efficiency from a DataFrame.

    Args:
        comparison_df (pd.DataFrame): DataFrame with 'model_name', 'params_M', 'accuracy_pct'.
        output_dir (str): Directory to save the plot.
        prefix (str): Prefix for the output filename.
    """
    if comparison_df is None:
        print("No model comparison data provided for efficiency plot.")
        return

    models = comparison_df['model_name'].tolist()
    params = comparison_df['params_M'].tolist()
    accuracies = comparison_df['accuracy_pct'].tolist()

    # Create figure
    plt.figure(figsize=(10, 7)) # Increased height for legend

    # Create scatter plot with varying sizes based on parameter count
    sizes = [p*20 for p in params]  # Scale for visibility

    # Define colors - use a colormap if many models
    if len(models) <= 10:
         colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    else:
         colors = plt.cm.viridis(np.linspace(0, 1, len(models)))

    # Plot points
    for i, model in enumerate(models):
        plt.scatter(params[i], accuracies[i], s=sizes[i], c=[colors[i]], # Pass color as list for single point
                    alpha=0.7, edgecolors='black', linewidth=1, label=model)
        # Optional annotation (can get crowded)
        # plt.annotate(model.replace('\n', ' '), (params[i], accuracies[i]),
        #              xytext=(10, 5), textcoords='offset points',
        #              fontsize=9) # , fontweight='bold')

    # Add efficiency contour lines (optional, based on typical ranges)
    max_param = max(params) if params else 100
    param_range = np.linspace(min(params)*0.8 if params else 1, max_param * 1.1, 100)
    for eff in [2, 5, 10, 15]: # Example efficiency contours
        acc_line = eff * param_range
        # Only plot line if it's within reasonable y-range
        if max(acc_line) > min(accuracies) * 0.5 and min(acc_line) < max(accuracies) * 1.2 :
             plt.plot(param_range, acc_line, 'k--', alpha=0.3, linewidth=1)
             # Annotate contour (find a good position)
             text_x = max_param * 0.9
             text_y = eff * text_x
             if text_y < max(accuracies) * 1.1 and text_y > min(accuracies) * 0.8:
                  plt.annotate(f'{eff}%/M',
                               xy=(text_x, text_y),
                               xytext=(0, 5), textcoords='offset points',
                               fontsize=8, color='gray') # fontweight='bold',


    # Customize plot
    if params:
         plt.xscale('log') # Use log scale if param range is large
         plt.xlim(min(params)*0.8, max_param * 1.2)
         plt.ylim(min(accuracies)*0.8, max(accuracies) * 1.1)
    plt.xlabel('Parameters (Millions, log scale)', fontsize=14)
    plt.ylabel('Top-1 Accuracy (%)', fontsize=14)
    plt.title(f'{prefix} Model Efficiency: Accuracy vs. Parameters'.strip(), fontsize=16)
    plt.grid(True, which="both", linestyle='--', alpha=0.5) # Grid for both major and minor ticks on log scale


    # Add legend outside the plot
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", borderaxespad=0, title="Models", fontsize=10)

    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
    output_filename = os.path.join(output_dir, f"{prefix}parameter_efficiency.png".strip())
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Parameter efficiency plot saved to {output_filename}")
    plt.close()

def plot_convergence_speed_from_data(baseline_df, distill_df, target_accuracies, output_dir='.', prefix=''):
    """
    Plot showing convergence speed based on loaded training data.

    Args:
        baseline_df (pd.DataFrame): DataFrame with 'epoch', 'val_acc' for baseline.
        distill_df (pd.DataFrame): DataFrame with 'epoch', 'val_acc' for distilled.
        target_accuracies (list): List of target accuracy percentages (e.g., [50, 60, 70]).
        output_dir (str): Directory to save the plot.
        prefix (str): Prefix for the output filename.
    """
    if baseline_df is None or distill_df is None:
        print("Baseline and Distilled logs needed for convergence plot.")
        return
    if not target_accuracies:
        print("No target accuracies specified for convergence plot.")
        return

    baseline_epochs = []
    distill_epochs = []
    valid_targets = []

    for target in target_accuracies:
        try:
            # Find first epoch where validation accuracy exceeds target
            baseline_epoch = baseline_df[baseline_df['val_acc'] >= target]['epoch'].iloc[0]
            distill_epoch = distill_df[distill_df['val_acc'] >= target]['epoch'].iloc[0]

            baseline_epochs.append(baseline_epoch)
            distill_epochs.append(distill_epoch)
            valid_targets.append(target)
        except IndexError:
            print(f"Warning: Target accuracy {target}% not reached by both models in the provided logs. Skipping.")

    if not valid_targets:
        print("No valid target accuracies were reached by both models.")
        return

    # Create figure
    plt.figure(figsize=(max(8, len(valid_targets) * 1.5), 6)) # Adjust width

    # Set width of bars
    width = 0.35

    # Set positions of bars on X axis
    r1 = np.arange(len(valid_targets))
    r2 = [x + width for x in r1]

    # Create bars
    bar1 = plt.bar(r1, baseline_epochs, width, label='Baseline Epochs', color='#1f77b4')
    bar2 = plt.bar(r2, distill_epochs, width, label='Distilled Epochs', color='#ff7f0e')

    # Add labels and title
    plt.xlabel('Target Validation Accuracy (%)', fontsize=14)
    plt.ylabel('Epochs to Reach Target', fontsize=14)
    plt.title(f'{prefix} Convergence Speed Comparison'.strip(), fontsize=16)

    # Add xticks on the middle of the group bars
    plt.xticks([r + width/2 for r in range(len(valid_targets))], [f'{t}%' for t in valid_targets])

    # Add legend
    plt.legend(loc='upper left')

    # Add grid
    plt.grid(True, linestyle='--', alpha=0.3, axis='y')

    # Add value labels on top of bars
    plt.bar_label(bar1, padding=3)
    plt.bar_label(bar2, padding=3)

    plt.tight_layout()
    output_filename = os.path.join(output_dir, f"{prefix}convergence_speed.png".strip())
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Convergence speed plot saved to {output_filename}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate plots from training logs and model comparison data.")
    parser.add_argument('--baseline-log', type=str, help='Path to CSV log file for the baseline model (epoch,train_acc,val_acc).')
    parser.add_argument('--distill-log', type=str, help='Path to CSV log file for the distilled model (epoch,train_acc,val_acc).')
    parser.add_argument('--teacher-acc', type=float, help='Validation accuracy of the teacher model (for horizontal line).')
    parser.add_argument('--comparison-data', type=str, help='Path to CSV file for model comparison (model_name,params_M,accuracy_pct).')
    parser.add_argument('--convergence-targets', type=float, nargs='+', default=[50, 60, 70], help='List of target validation accuracies for convergence plot.')
    parser.add_argument('--output-dir', type=str, default='.', help='Directory to save the generated plots.')
    parser.add_argument('--prefix', type=str, default='', help='Prefix for output plot filenames.')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    baseline_df = None
    distill_df = None
    comparison_df = None

    if args.baseline_log:
        try:
            baseline_df = pd.read_csv(args.baseline_log)
            print(f"Loaded baseline log from {args.baseline_log}")
        except FileNotFoundError:
            print(f"Error: Baseline log file not found at {args.baseline_log}")
        except Exception as e:
            print(f"Error loading baseline log: {e}")

    if args.distill_log:
        try:
            distill_df = pd.read_csv(args.distill_log)
            print(f"Loaded distilled log from {args.distill_log}")
        except FileNotFoundError:
            print(f"Error: Distilled log file not found at {args.distill_log}")
        except Exception as e:
            print(f"Error loading distilled log: {e}")

    if args.comparison_data:
        try:
            comparison_df = pd.read_csv(args.comparison_data)
            # Ensure correct column types
            comparison_df['params_M'] = pd.to_numeric(comparison_df['params_M'])
            comparison_df['accuracy_pct'] = pd.to_numeric(comparison_df['accuracy_pct'])
            print(f"Loaded comparison data from {args.comparison_data}")
        except FileNotFoundError:
            print(f"Error: Comparison data file not found at {args.comparison_data}")
        except Exception as e:
            print(f"Error loading comparison data: {e}")


    # Generate plots
    print("\nGenerating plots...")
    if baseline_df is not None or distill_df is not None:
        plot_training_curves(baseline_df, distill_df, args.teacher_acc, args.output_dir, args.prefix)

    if comparison_df is not None:
        plot_model_comparison(comparison_df, args.output_dir, args.prefix)
        plot_parameter_efficiency_scatter(comparison_df, args.output_dir, args.prefix)

    if baseline_df is not None and distill_df is not None:
         plot_convergence_speed_from_data(baseline_df, distill_df, args.convergence_targets, args.output_dir, args.prefix)

    print("\nVisualization generation complete.")

if __name__ == '__main__':
    main()
