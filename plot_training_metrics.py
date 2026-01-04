#!/usr/bin/env python3
"""
Script to plot training metrics from CSV file.
Generates multiple plots for different metric categories.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def plot_training_metrics(csv_path, output_dir=None):
    """
    Plot training metrics from CSV file.
    
    Args:
        csv_path: Path to the training_metrics.csv file
        output_dir: Directory to save plots (default: same as CSV file directory)
    """
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Calculate iteration number (epoch * steps_per_epoch + step)
    # For simplicity, we'll use row index as iteration
    df['iteration'] = df.index
    
    # Set output directory
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(csv_path))
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with subplots for losses
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Losses', fontsize=16, fontweight='bold')
    
    # Discriminator losses
    ax = axes[0, 0]
    if 'D/loss_real' in df.columns:
        ax.plot(df['iteration'], df['D/loss_real'], label='D/loss_real', alpha=0.7)
    if 'D/loss_fake' in df.columns:
        ax.plot(df['iteration'], df['D/loss_fake'], label='D/loss_fake', alpha=0.7)
    if 'D/loss' in df.columns:
        ax.plot(df['iteration'], df['D/loss'], label='D/loss', alpha=0.7, linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Discriminator Losses')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Generator and RL losses
    ax = axes[0, 1]
    if 'G/loss' in df.columns:
        ax.plot(df['iteration'], df['G/loss'], label='G/loss', alpha=0.7, linewidth=2)
    if 'RL/loss' in df.columns:
        ax.plot(df['iteration'], df['RL/loss'], label='RL/loss', alpha=0.7)
    if 'V/loss' in df.columns:
        ax.plot(df['iteration'], df['V/loss'], label='V/loss', alpha=0.7)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Generator, RL, and Value Network Losses')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gradient penalty
    ax = axes[1, 0]
    if 'D/loss_gp' in df.columns:
        ax.plot(df['iteration'], df['D/loss_gp'], label='Gradient Penalty', color='orange', alpha=0.7)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Gradient Penalty')
        ax.set_title('Discriminator Gradient Penalty')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Fréchet distances
    ax = axes[1, 1]
    if 'FD/bond' in df.columns:
        ax.plot(df['iteration'], df['FD/bond'], label='FD/bond', alpha=0.7)
    if 'FD/bond_atom' in df.columns:
        ax.plot(df['iteration'], df['FD/bond_atom'], label='FD/bond_atom', alpha=0.7)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Fréchet Distance')
    ax.set_title('Fréchet Distances')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'losses.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved losses plot to {os.path.join(output_dir, 'losses.png')}")
    
    # Create figure for molecular property scores
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Molecular Property Scores', fontsize=16, fontweight='bold')
    
    # Validity, Uniqueness, Novelty
    ax = axes[0, 0]
    if 'valid' in df.columns:
        ax.plot(df['iteration'], df['valid'], label='Validity (%)', alpha=0.7, linewidth=2)
    if 'unique' in df.columns:
        ax.plot(df['iteration'], df['unique'], label='Uniqueness (%)', alpha=0.7, linewidth=2)
    if 'novel' in df.columns:
        ax.plot(df['iteration'], df['novel'], label='Novelty (%)', alpha=0.7, linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Percentage')
    ax.set_title('Validity, Uniqueness, and Novelty')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    
    # Drug-likeness scores
    ax = axes[0, 1]
    if 'QED' in df.columns:
        ax.plot(df['iteration'], df['QED'], label='QED', alpha=0.7, linewidth=2)
    if 'drugcand' in df.columns:
        ax.plot(df['iteration'], df['drugcand'], label='Drug Candidate Score', alpha=0.7)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Score')
    ax.set_title('Drug-likeness Metrics')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    # Synthetic Accessibility and Natural Product
    ax = axes[1, 0]
    if 'SA' in df.columns:
        ax.plot(df['iteration'], df['SA'], label='SA Score', alpha=0.7, linewidth=2, color='green')
    if 'NP' in df.columns:
        ax.plot(df['iteration'], df['NP'], label='NP Score', alpha=0.7, color='purple')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Score')
    ax.set_title('Synthetic Accessibility (SA) and Natural Product (NP) Scores')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    # Solubility and Diversity
    ax = axes[1, 1]
    if 'Solute' in df.columns:
        ax.plot(df['iteration'], df['Solute'], label='LogP (Solute)', alpha=0.7, linewidth=2, color='brown')
    if 'diverse' in df.columns:
        ax.plot(df['iteration'], df['diverse'], label='Diversity', alpha=0.7, color='teal')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Score')
    ax.set_title('Solubility (LogP) and Diversity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'molecular_properties.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved molecular properties plot to {os.path.join(output_dir, 'molecular_properties.png')}")
    
    # Combined overview plot
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    fig.suptitle('Training Overview', fontsize=16, fontweight='bold')
    
    # Losses overview
    ax = axes[0]
    if 'D/loss' in df.columns:
        ax.plot(df['iteration'], df['D/loss'], label='Discriminator Loss', alpha=0.7)
    if 'G/loss' in df.columns:
        ax.plot(df['iteration'], df['G/loss'], label='Generator Loss', alpha=0.7)
    if 'RL/loss' in df.columns:
        ax.plot(df['iteration'], df['RL/loss'], label='RL Loss', alpha=0.7)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Main Training Losses')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Key molecular metrics overview
    ax = axes[1]
    if 'valid' in df.columns:
        ax.plot(df['iteration'], df['valid'], label='Validity (%)', alpha=0.7, linewidth=2)
    if 'QED' in df.columns:
        ax.plot(df['iteration'], df['QED'] * 100, label='QED (×100)', alpha=0.7, linewidth=2)
    if 'unique' in df.columns:
        ax.plot(df['iteration'], df['unique'], label='Uniqueness (%)', alpha=0.7, linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Score')
    ax.set_title('Key Molecular Metrics')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_overview.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved training overview plot to {os.path.join(output_dir, 'training_overview.png')}")
    
    # Individual metric plots (optional - create separate files for each important metric)
    important_metrics = {
        'valid': 'Validity (%)',
        'QED': 'QED Score',
        'unique': 'Uniqueness (%)',
        'novel': 'Novelty (%)',
        'SA': 'Synthetic Accessibility Score',
        'D/loss': 'Discriminator Loss',
        'G/loss': 'Generator Loss'
    }
    
    for metric_key, metric_name in important_metrics.items():
        if metric_key in df.columns:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df['iteration'], df[metric_key], linewidth=2, alpha=0.8)
            ax.set_xlabel('Iteration', fontsize=12)
            ax.set_ylabel(metric_name, fontsize=12)
            ax.set_title(f'{metric_name} vs Iteration', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            mean_val = df[metric_key].mean()
            std_val = df[metric_key].std()
            final_val = df[metric_key].iloc[-1]
            ax.axhline(mean_val, color='r', linestyle='--', alpha=0.5, label=f'Mean: {mean_val:.3f}')
            ax.text(0.02, 0.98, f'Final: {final_val:.3f}\nMean: {mean_val:.3f} ± {std_val:.3f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax.legend()
            
            plt.tight_layout()
            safe_name = metric_key.replace('/', '_').replace('\\', '_')
            plt.savefig(os.path.join(output_dir, f'{safe_name}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved {metric_name} plot to {os.path.join(output_dir, f'{safe_name}.png')}")
    
    print(f"\nAll plots saved to: {output_dir}")
    print(f"Total iterations: {len(df)}")
    print(f"Total epochs: {df['epoch'].max() if 'epoch' in df.columns else 'N/A'}")


def main():
    parser = argparse.ArgumentParser(description='Plot training metrics from CSV file')
    parser.add_argument('csv_path', type=str, help='Path to training_metrics.csv file')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save plots (default: same as CSV directory)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_path):
        print(f"Error: CSV file not found: {args.csv_path}")
        return
    
    plot_training_metrics(args.csv_path, args.output_dir)


if __name__ == '__main__':
    main()

