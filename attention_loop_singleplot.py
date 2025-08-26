import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
import torch
import argparse
from matplotlib import font_manager


def setup_font_and_style():
    plt.style.use('seaborn-v0_8-darkgrid')
    
    available_fonts = [f.name for f in font_manager.fontManager.ttflist]
    if 'Arial' in available_fonts:
        font_family = 'Arial'
    elif 'Helvetica' in available_fonts:
        font_family = 'Helvetica'
    else:
        font_family = 'sans-serif'
    
    plt.rcParams.update({
        'font.family': font_family,
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 14,
        'font.weight': 'bold',
        'axes.labelsize': 16,
        'axes.labelweight': 'bold',
        'axes.titlesize': 18,
        'axes.titleweight': 'bold',
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 13,
        'figure.titlesize': 20
    })
    
    return font_family


def apply_softmax(attention_scores, temperature=1.0):
    num_layers = attention_scores.shape[0]
    seq_len = attention_scores.shape[1]
    softmaxed = np.zeros_like(attention_scores)
    
    for layer_idx in range(num_layers):
        for query_idx in range(seq_len):
            row_scores = attention_scores[layer_idx, query_idx, :].copy()
            
            valid_mask = ~np.isinf(row_scores)
            
            if valid_mask.any():
                valid_scores = row_scores[valid_mask]
                
                valid_scores = valid_scores / temperature
                
                max_score = np.max(valid_scores)
                exp_scores = np.exp(valid_scores - max_score)
                softmax_scores = exp_scores / np.sum(exp_scores)
                
                softmaxed[layer_idx, query_idx, valid_mask] = softmax_scores
                softmaxed[layer_idx, query_idx, ~valid_mask] = 0
            else:
                softmaxed[layer_idx, query_idx, :] = 1.0 / seq_len
    
    return softmaxed


def calculate_entropy_data(attention_scores, loop_start_idx):
    num_layers = attention_scores.shape[0]
    seq_len = attention_scores.shape[1]
    layer_entropies = []
    
    for layer_idx in range(num_layers):
        layer_attention = attention_scores[layer_idx]
        layer_entropy = []
        
        for pos in range(seq_len):
            attn_dist = layer_attention[pos]
            attn_dist = attn_dist / (attn_dist.sum() + 1e-10)
            ent = entropy(attn_dist + 1e-10)
            layer_entropy.append(ent)
        
        layer_entropies.append(layer_entropy)
    
    layer_entropies = np.array(layer_entropies)
    pre_loop_entropy = layer_entropies[:, :loop_start_idx].mean(axis=1)
    loop_entropy = layer_entropies[:, loop_start_idx:].mean(axis=1)
    
    return {
        'layer_entropies': layer_entropies,
        'pre_loop_avg': pre_loop_entropy,
        'loop_avg': loop_entropy,
        'entropy_drop': pre_loop_entropy - loop_entropy
    }


def plot_entropy_comparison(attention_scores, loop_start_idx, output_id="", save_fig=True, 
                           figsize=(10, 8), apply_softmax_transform=False, temperature=1.0):
    font_family = setup_font_and_style()
    print(f"Using font: {font_family}")
    
    if apply_softmax_transform:
        attention_scores = apply_softmax(attention_scores, temperature)
        print(f"Applied softmax with temperature={temperature}")
    
    entropy_data = calculate_entropy_data(attention_scores, loop_start_idx)
    num_layers = attention_scores.shape[0]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    layers = range(num_layers)
    width = 0.35
    x = np.arange(len(layers))
    
    bars1 = ax.bar(x - width/2, entropy_data['pre_loop_avg'], width, 
                   label='Pre-loop', color='#3498db', alpha=0.8, edgecolor='darkblue', linewidth=1.5)
    bars2 = ax.bar(x + width/2, entropy_data['loop_avg'], width, 
                   label='During loop', color='#e74c3c', alpha=0.8, edgecolor='darkred', linewidth=1.5)
    
    if num_layers <= 12:
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            ax.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.01,
                   f'{entropy_data["pre_loop_avg"][i]:.2f}', ha='center', va='bottom', fontsize=10)
            ax.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.01,
                   f'{entropy_data["loop_avg"][i]:.2f}', ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Layer', fontsize=18, fontweight='bold')
    ax.set_ylabel('Average Entropy', fontsize=18, fontweight='bold')
    ax.set_title(f'Entropy: Pre-loop vs During Loop', fontsize=20, fontweight='bold', pad=20)
    
    if num_layers <= 24:
        ax.set_xticks(x[::2])
        ax.set_xticklabels([str(i) for i in layers[::2]], fontweight='bold')
    else:
        ax.set_xticks(x[::4])
        ax.set_xticklabels([str(i) for i in layers[::4]], fontweight='bold')
    
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True, framealpha=0.9)
    
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    
    if save_fig:
        fig_path = f"figs/{output_id}_entropy_comparison.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to: {fig_path}")
        fig_path = f"figs/{output_id}_entropy_comparison.pdf"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to: {fig_path}")
    
    plt.show()


def calculate_attention_patterns(attention_scores, loop_start_idx):
    num_layers = attention_scores.shape[0]
    patterns = {
        'self_attention': [],
        'backward_attention': [],
        'forward_attention': [],
        'loop_region_attention': []
    }
    
    for layer_idx in range(num_layers):
        layer_attn = attention_scores[layer_idx]
        loop_region = layer_attn[loop_start_idx:, :]
        
        self_attn = np.diag(layer_attn).mean()
        patterns['self_attention'].append(self_attn)
        
        backward_mask = np.tril(np.ones_like(layer_attn), k=-1)
        backward_attn = (layer_attn * backward_mask).sum() / (backward_mask.sum() + 1e-10)
        patterns['backward_attention'].append(backward_attn)
        
        forward_mask = np.triu(np.ones_like(layer_attn), k=1)
        forward_attn = (layer_attn * forward_mask).sum() / (forward_mask.sum() + 1e-10)
        patterns['forward_attention'].append(forward_attn)
        
        loop_self_attn = loop_region[:, loop_start_idx:].mean()
        patterns['loop_region_attention'].append(loop_self_attn)
    
    return patterns


def plot_attention_patterns(attention_scores, loop_start_idx, output_id="", save_fig=True, 
                           figsize=(12, 8), apply_softmax_transform=False, temperature=1.0):
    font_family = setup_font_and_style()
    print(f"Using font: {font_family}")
    
    if apply_softmax_transform:
        attention_scores = apply_softmax(attention_scores, temperature)
        print(f"Applied softmax with temperature={temperature}")
    
    patterns = calculate_attention_patterns(attention_scores, loop_start_idx)
    num_layers = attention_scores.shape[0]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    layers = range(num_layers)
    ax.plot(layers, patterns['self_attention'], 'o-', label='Self Attention', 
            color='#2ecc71', linewidth=2.5, markersize=8, alpha=0.8)
    ax.plot(layers, patterns['backward_attention'], 's-', label='Backward Attention', 
            color='#9b59b6', linewidth=2.5, markersize=8, alpha=0.8)
    ax.plot(layers, patterns['loop_region_attention'], '^-', label='Loop Region Attention', 
            color='#e67e22', linewidth=2.5, markersize=8, alpha=0.8)
    
    ax.set_xlabel('Layer', fontsize=18, fontweight='bold')
    ax.set_ylabel('Attention Weight', fontsize=18, fontweight='bold')
    ax.set_title(f'Attention Pattern Distribution by Layer', 
                fontsize=20, fontweight='bold', pad=20)
    
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True, 
             framealpha=0.9, fontsize=14)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    ax.set_xlim(-0.5, num_layers - 0.5)
    if num_layers <= 24:
        ax.set_xticks(range(0, num_layers, 2))
    else:
        ax.set_xticks(range(0, num_layers, 4))
    
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
    
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    
    if save_fig:
        fig_path = f"figs/{output_id}_attention_patterns.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to: {fig_path}")
        fig_path = f"figs/{output_id}_attention_patterns.pdf"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to: {fig_path}")
    
    plt.show()


def plot_entropy_drop(attention_scores, loop_start_idx, output_id="", save_fig=True, 
                     figsize=(10, 8), apply_softmax_transform=False, temperature=1.0):
    font_family = setup_font_and_style()
    print(f"Using font: {font_family}")
    
    if apply_softmax_transform:
        attention_scores = apply_softmax(attention_scores, temperature)
        print(f"Applied softmax with temperature={temperature}")
    
    entropy_data = calculate_entropy_data(attention_scores, loop_start_idx)
    num_layers = attention_scores.shape[0]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    entropy_drops = entropy_data['entropy_drop']
    colors = ['#27ae60' if d > 0 else '#c0392b' for d in entropy_drops]
    
    bars = ax.bar(range(num_layers), entropy_drops, color=colors, 
                  alpha=0.8, edgecolor='black', linewidth=1.5)
    
    max_drop_idx = np.argmax(entropy_drops)
    bars[max_drop_idx].set_edgecolor('gold')
    bars[max_drop_idx].set_linewidth(3)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    mean_drop = np.mean(entropy_drops)
    ax.axhline(y=mean_drop, color='red', linestyle='--', linewidth=1.5, 
              alpha=0.7, label=f'Mean: {mean_drop:.3f}')
    
    ax.set_xlabel('Layer', fontsize=18, fontweight='bold')
    ax.set_ylabel('Entropy Drop', fontsize=18, fontweight='bold')
    ax.set_title(f'Entropy Drop in Loop Region', 
                fontsize=20, fontweight='bold', pad=20)
    
    if num_layers <= 24:
        ax.set_xticks(range(0, num_layers, 2))
        ax.set_xticklabels([str(i) for i in range(0, num_layers, 2)], fontweight='bold')
    else:
        ax.set_xticks(range(0, num_layers, 4))
        ax.set_xticklabels([str(i) for i in range(0, num_layers, 4)], fontweight='bold')
    
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True, framealpha=0.9)
    
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    
    stats_text = f'Max Drop: Layer {max_drop_idx} ({entropy_drops[max_drop_idx]:.3f})\n'
    stats_text += f'Mean Drop: {mean_drop:.3f}\n'
    stats_text += f'Positive Drops: {sum(d > 0 for d in entropy_drops)}/{num_layers}'
    
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_fig:
        fig_path = f"figs/{output_id}_entropy_drop.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to: {fig_path}")
        fig_path = f"figs/{output_id}_entropy_drop.pdf"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to: {fig_path}")

    plt.show()


def plot_attention_heatmaps(attention_scores, loop_start_idx, output_id="", 
                           layers_to_plot=None, save_fig=True, figsize=(18, 6),
                           apply_softmax_transform=False, temperature=1.0):
    font_family = setup_font_and_style()
    print(f"Using font: {font_family}")
    
    if apply_softmax_transform:
        attention_scores = apply_softmax(attention_scores, temperature)
        print(f"Applied softmax with temperature={temperature}")
    
    num_layers = attention_scores.shape[0]
    seq_len = attention_scores.shape[1]
    
    if layers_to_plot is None:
        layers_to_plot = [0, num_layers//2, num_layers-1]
    
    fig, axes = plt.subplots(1, len(layers_to_plot), figsize=figsize)
    if len(layers_to_plot) == 1:
        axes = [axes]
    
    display_start = max(0, seq_len - 100)
    
    for idx, (ax, layer_idx) in enumerate(zip(axes, layers_to_plot)):
        attn_subset = attention_scores[layer_idx][display_start:, display_start:]
        
        im = ax.imshow(attn_subset, cmap='hot', aspect='auto', interpolation='nearest')
        
        ax.set_title(f'Layer {layer_idx} Attention\n(Last 100 tokens)', 
                    fontsize=16, fontweight='bold', pad=10)
        ax.set_xlabel('Key Position', fontsize=14, fontweight='bold')
        ax.set_ylabel('Query Position', fontsize=14, fontweight='bold')
        
        if loop_start_idx > display_start:
            loop_mark = loop_start_idx - display_start
            ax.axhline(y=loop_mark, color='cyan', linestyle='--', alpha=0.7, linewidth=2)
            ax.axvline(x=loop_mark, color='cyan', linestyle='--', alpha=0.7, linewidth=2)
            
            ax.text(loop_mark, -5, 'Loop Start', ha='center', va='top', 
                   color='cyan', fontweight='bold', fontsize=10)
        
        tick_positions = [0, 25, 50, 75, 99]
        tick_labels = [str(display_start + p) for p in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontweight='bold')
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=10)
        for label in cbar.ax.get_yticklabels():
            label.set_fontweight('bold')
        
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
    
    fig.suptitle(f'Key Layers Attention Heatmaps', 
                fontsize=20, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_fig:
        fig_path = f"figs/{output_id}_attention_heatmaps.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to: {fig_path}")
        fig_path = f"figs/{output_id}_attention_heatmaps.pdf"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to: {fig_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot individual attention analysis figures")
    parser.add_argument("--attn_score_path", type=str, required=True,
                       help="Path to attention scores file")
    parser.add_argument("--loop_start_idx", type=int, required=True,
                       help="Index where the loop starts")
    parser.add_argument("--output_id", type=str, default="output",
                       help="Output identifier for file names")
    parser.add_argument("--plot_type", type=str, 
                       choices=['entropy_comp', 'patterns', 'entropy_drop', 'heatmaps', 'all'],
                       default='all', help="Which plot to generate")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Temperature parameter for softmax (default: 1.0)")
    parser.add_argument("--is_pre_softmax", action="store_true",
                       help="Indicates if the input scores are pre-softmax")
    
    args = parser.parse_args()
    
    attention_scores = torch.load(args.attn_score_path)
    if isinstance(attention_scores, torch.Tensor):
        attention_scores = attention_scores.detach().float().cpu().numpy()
    
    apply_softmax_flag = args.is_pre_softmax
    
    if args.plot_type == 'entropy_comp' or args.plot_type == 'all':
        plot_entropy_comparison(attention_scores, args.loop_start_idx, args.output_id,
                               apply_softmax_transform=apply_softmax_flag,
                               temperature=args.temperature)
    
    if args.plot_type == 'patterns' or args.plot_type == 'all':
        plot_attention_patterns(attention_scores, args.loop_start_idx, args.output_id,
                               apply_softmax_transform=apply_softmax_flag,
                               temperature=args.temperature)
    
    if args.plot_type == 'entropy_drop' or args.plot_type == 'all':
        plot_entropy_drop(attention_scores, args.loop_start_idx, args.output_id,
                         apply_softmax_transform=apply_softmax_flag,
                         temperature=args.temperature)
    
    if args.plot_type == 'heatmaps' or args.plot_type == 'all':
        plot_attention_heatmaps(attention_scores, args.loop_start_idx, args.output_id,
                               apply_softmax_transform=False, # no softmax here
                               temperature=args.temperature)