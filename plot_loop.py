import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from utils import detect_loop_pattern


def calculate_average_loop_probs(prob_list, loop_start, loop_length, num_loops):
    all_loop_probs = []
    
    for i in range(min(num_loops, 10)):
        start_idx = loop_start + i * loop_length
        end_idx = start_idx + loop_length
        
        if end_idx <= len(prob_list):
            all_loop_probs.append(prob_list[start_idx:end_idx])
    
    if all_loop_probs:
        avg_probs = np.mean(all_loop_probs, axis=0)
        return avg_probs
    return []


def plot_probability_trends(token_list, prob_list, output_id="", save_fig=True, figsize=(15, 8)):
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'Token Probability Analysis - {output_id}', fontsize=16, fontweight='bold')
    
    ax1 = axes[0, 0]

    loop_pattern = detect_loop_pattern(token_list)
    
    if loop_pattern and loop_pattern['loop_length'] > 0:
        loop_length = loop_pattern['loop_length']
        loop_start = loop_pattern['loop_start']
        end_idx = loop_start + loop_length * loop_pattern['num_loops']
        prob_list = prob_list[loop_start:end_idx]
    ax1.plot(prob_list, linewidth=1.5, alpha=0.8)
    ax1.set_xlabel('Token Position')
    ax1.set_ylabel('Probability')
    ax1.set_title('Probability Over Token Sequence')
    ax1.grid(True, alpha=0.3)
    
    if len(prob_list) > 10:
        window_size = min(20, len(prob_list) // 5)
        moving_avg = pd.Series(prob_list).rolling(window=window_size, center=True).mean()
        ax1.plot(moving_avg, 'r--', alpha=0.6, label=f'Moving Avg (window={window_size})')
        ax1.legend()
    
    min_idx = np.argmin(prob_list)
    max_idx = np.argmax(prob_list)
    ax1.scatter([min_idx], [prob_list[min_idx]], color='red', s=100, zorder=5, 
                label=f'Min: {prob_list[min_idx]:.4f}')
    ax1.scatter([max_idx], [prob_list[max_idx]], color='green', s=100, zorder=5,
                label=f'Max: {prob_list[max_idx]:.4f}')
    ax1.legend()
    
    ax2 = axes[0, 1]
    
    if loop_pattern and loop_pattern['loop_length'] > 0:
        loop_length = loop_pattern['loop_length']
        loop_start = loop_pattern['loop_start']
        
        colors = plt.cm.viridis(np.linspace(0, 1, min(10, loop_pattern['num_loops'])))
        
        for i in range(min(10, loop_pattern['num_loops'])):
            start_idx = loop_start + i * loop_length
            end_idx = start_idx + loop_length
            
            if end_idx <= len(prob_list):
                loop_probs = prob_list[start_idx:end_idx]
                x_positions = range(len(loop_probs))
                
                ax2.plot(x_positions, loop_probs, 
                        marker='o', markersize=3,
                        alpha=0.7, color=colors[i],
                        label=f'Loop {i+1}')
        
        ax2.set_xlabel('Position within Loop')
        ax2.set_ylabel('Probability')
        ax2.set_title(f'Probability Pattern within Loops (Length={loop_length})')
        ax2.grid(True, alpha=0.3)
        
        if loop_pattern['num_loops'] > 1:
            avg_loop_probs = calculate_average_loop_probs(prob_list, loop_start, loop_length, loop_pattern['num_loops'])
            ax2.plot(range(len(avg_loop_probs)), avg_loop_probs, 
                    'k--', linewidth=2, alpha=0.8, 
                    label='Average', marker='s', markersize=4)
        
        handles, labels = ax2.get_legend_handles_labels()
        if len(handles) > 6:
            ax2.legend(handles[:5] + [handles[-1]], labels[:5] + [labels[-1]], 
                      fontsize=8, loc='best')
        else:
            ax2.legend(fontsize=8, loc='best')
            
        if loop_length <= 15:
            loop_tokens = token_list[loop_start:loop_start + loop_length]
            ax2.set_xticks(range(len(loop_tokens)))
            ax2.set_xticklabels([t[:10] for t in loop_tokens], rotation=45, ha='right', fontsize=8)
    else:
        n, bins, patches = ax2.hist(prob_list, bins=30, edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Probability')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Probability Distribution (No clear loop detected)')
        ax2.axvline(np.mean(prob_list), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(prob_list):.4f}')
        ax2.axvline(np.median(prob_list), color='green', linestyle='--',
                    label=f'Median: {np.median(prob_list):.4f}')
        ax2.legend()
    
    ax3 = axes[1, 0]
    
    if loop_pattern and loop_pattern['loop_length'] > 0:
        loop_length = loop_pattern['loop_length']
        loop_start = loop_pattern['loop_start']
        num_loops = loop_pattern['num_loops']
        
        max_loops_to_show = min(20, num_loops)
        max_positions = min(100, loop_length)
        
        heatmap_data = np.zeros((max_loops_to_show, max_positions))
        
        for loop_idx in range(max_loops_to_show):
            start_idx = loop_start + loop_idx * loop_length
            end_idx = start_idx + min(max_positions, loop_length)
            
            if end_idx <= len(prob_list):
                loop_probs = prob_list[start_idx:end_idx]
                heatmap_data[loop_idx, :len(loop_probs)] = loop_probs
            else:
                break
        
        im = ax3.imshow(heatmap_data, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
        ax3.set_xlabel('Position within Loop')
        ax3.set_ylabel('Loop Instance')
        ax3.set_title(f'Loop Probability Heatmap (Pattern: {loop_length} tokens × {num_loops} loops)')
        
        if loop_length <= 15:
            loop_tokens = token_list[loop_start:loop_start + min(max_positions, loop_length)]
            ax3.set_xticks(range(len(loop_tokens)))
            ax3.set_xticklabels([t[:10] for t in loop_tokens], rotation=45, ha='right', fontsize=7)
        else:
            step = max(1, max_positions // 10)
            ax3.set_xticks(range(0, max_positions, step))
            ax3.set_xticklabels(range(0, max_positions, step))
        
        if max_loops_to_show <= 20:
            ax3.set_yticks(range(max_loops_to_show))
            ax3.set_yticklabels([f'Loop {i+1}' for i in range(max_loops_to_show)], fontsize=8)
        
        plt.colorbar(im, ax=ax3, label='Probability')
        
        if loop_length <= 50:
            for i in range(1, min(max_positions, loop_length)):
                if i % 10 == 0:
                    ax3.axvline(x=i-0.5, color='white', linewidth=0.5, alpha=0.3)
    else:
        segment_length = 20
        num_segments = min(20, len(prob_list) // segment_length)
        
        if num_segments > 0:
            heatmap_data = np.zeros((num_segments, segment_length))
            
            for seg_idx in range(num_segments):
                start_idx = seg_idx * segment_length
                end_idx = start_idx + segment_length
                if end_idx <= len(prob_list):
                    heatmap_data[seg_idx, :] = prob_list[start_idx:end_idx]
            
            im = ax3.imshow(heatmap_data, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
            ax3.set_xlabel('Position in Segment')
            ax3.set_ylabel('Segment')
            ax3.set_title(f'Probability Heatmap (No loop detected, showing segments)')
            plt.colorbar(im, ax=ax3, label='Probability')
        else:
            ax3.text(0.5, 0.5, 'Insufficient data for heatmap', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Probability Heatmap')
    
    ax4 = axes[1, 1]
    if len(prob_list) > 1:
        prob_changes = np.diff(prob_list)
        ax4.plot(prob_changes, linewidth=1, alpha=0.8)
        ax4.set_xlabel('Token Position')
        ax4.set_ylabel('Probability Change')
        ax4.set_title('Rate of Probability Change')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        threshold = np.std(prob_changes) * 2
        spikes = np.where(np.abs(prob_changes) > threshold)[0]
        if len(spikes) > 0:
            ax4.scatter(spikes, prob_changes[spikes], color='red', s=30, 
                       alpha=0.6, label=f'Spikes (>{threshold:.4f})')
            ax4.legend()
    
    plt.tight_layout()
    
    if save_fig:
        fig_path = f"figs/prob_analysis.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"PNG figure saved to: {fig_path}")
        fig_path = fig_path.replace("png", "pdf")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"PDF figure saved to: {fig_path}")
    
    plt.show()

def plot_loop_heatmap(token_list, prob_list, output_id="", save_fig=True, figsize=(12, 8)):
    plt.style.use('seaborn-v0_8-darkgrid')
    from matplotlib import font_manager
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
        'legend.fontsize': 12,
        'figure.titlesize': 20
    })
    
    print(f"Using font: {font_family}")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    loop_pattern = detect_loop_pattern(token_list)
    
    if loop_pattern and loop_pattern['loop_length'] > 0:
        loop_length = loop_pattern['loop_length']
        loop_start = loop_pattern['loop_start']
        num_loops = loop_pattern['num_loops']
        
        max_loops_to_show = min(20, num_loops)
        max_positions = min(100, loop_length)
        
        heatmap_data = np.zeros((max_loops_to_show, max_positions))
        
        for loop_idx in range(max_loops_to_show):
            start_idx = loop_start + loop_idx * loop_length
            end_idx = start_idx + min(max_positions, loop_length)
            
            if end_idx <= len(prob_list):
                loop_probs = prob_list[start_idx:end_idx]
                heatmap_data[loop_idx, :len(loop_probs)] = loop_probs
            else:
                break
        
        im = ax.imshow(heatmap_data, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
        ax.set_xlabel('Position within Loop', fontsize=18, fontweight='bold')
        ax.set_ylabel('Loop Instance', fontsize=18, fontweight='bold')
        ax.set_title(f'Loop Probability Heatmap - {output_id}\n(Pattern: {loop_length} tokens × {num_loops} loops)', 
                    fontsize=20, fontweight='bold')
        
        if loop_length <= 15:
            loop_tokens = token_list[loop_start:loop_start + min(max_positions, loop_length)]
            ax.set_xticks(range(len(loop_tokens)))
            ax.set_xticklabels([t[:10] for t in loop_tokens], rotation=45, ha='right', 
                              fontsize=12, fontweight='bold')
        else:
            step = max(1, max_positions // 10)
            ax.set_xticks(range(0, max_positions, step))
            ax.set_xticklabels(range(0, max_positions, step), fontsize=12, fontweight='bold')
        
        if max_loops_to_show <= 20:
            ax.set_yticks(range(max_loops_to_show))
            ax.set_yticklabels([f'Loop {i+1}' for i in range(max_loops_to_show)], 
                              fontsize=12, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Probability', fontsize=16, fontweight='bold')
        cbar.ax.tick_params(labelsize=12)
        for label in cbar.ax.get_yticklabels():
            label.set_fontweight('bold')
        
        if loop_length <= 50:
            for i in range(1, min(max_positions, loop_length)):
                if i % 10 == 0:
                    ax.axvline(x=i-0.5, color='white', linewidth=0.5, alpha=0.3)
        
        print("\n" + "="*50)
        print("Loop Pattern Detected:")
        print("="*50)
        print(f"Loop starts at position: {loop_start}")
        print(f"Loop length: {loop_length} tokens")
        print(f"Number of loops: {num_loops}")
        print(f"Pattern preview (first 5 tokens): {loop_pattern['pattern'][:5]}")
        
    else:
        segment_length = 20
        num_segments = min(20, len(prob_list) // segment_length)
        
        if num_segments > 0:
            heatmap_data = np.zeros((num_segments, segment_length))
            
            for seg_idx in range(num_segments):
                start_idx = seg_idx * segment_length
                end_idx = start_idx + segment_length
                if end_idx <= len(prob_list):
                    heatmap_data[seg_idx, :] = prob_list[start_idx:end_idx]
            
            im = ax.imshow(heatmap_data, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
            ax.set_xlabel('Position in Segment', fontsize=18, fontweight='bold')
            ax.set_ylabel('Segment', fontsize=18, fontweight='bold')
            ax.set_title(f'Probability Heatmap - {output_id}\n(No loop detected, showing segments)', 
                        fontsize=20, fontweight='bold')
            
            ax.tick_params(axis='both', which='major', labelsize=12)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontweight('bold')
            
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Probability', fontsize=16, fontweight='bold')
            cbar.ax.tick_params(labelsize=12)

            for label in cbar.ax.get_yticklabels():
                label.set_fontweight('bold')
            
            print("\n" + "="*50)
            print("No clear loop pattern detected")
            print("="*50)
            print(f"Showing segmented view: {num_segments} segments of {segment_length} tokens each")
        else:
            ax.text(0.5, 0.5, 'Insufficient data for heatmap', 
                    ha='center', va='center', transform=ax.transAxes, 
                    fontsize=18, fontweight='bold')
            ax.set_title(f'Probability Heatmap - {output_id}', fontsize=20, fontweight='bold')
    
    plt.tight_layout()
    
    if save_fig:
        fig_path = f"figs/loop_heatmap.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"\nPNG figure saved to: {fig_path}")
        fig_path = fig_path.replace("png", "pdf")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"\nPDF figure saved to: {fig_path}")
    
    plt.show()


def plot_loop_probability_trend(token_list, prob_list, output_id="", save_fig=True, figsize=(14, 8)):
    plt.style.use('seaborn-v0_8-darkgrid')
    
    from matplotlib import font_manager
    
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
        'figure.titlesize': 20,
        'lines.linewidth': 2.0,
        'lines.markersize': 10
    })
    
    print(f"Using font: {font_family}")
    fig, ax = plt.subplots(figsize=figsize)

    loop_pattern = detect_loop_pattern(token_list)
    
    if loop_pattern and loop_pattern['loop_length'] > 0:
        loop_length = loop_pattern['loop_length']
        loop_start = loop_pattern['loop_start']
        end_idx = loop_start + loop_length * loop_pattern['num_loops']
        loop_probs = prob_list[loop_start:end_idx]
    else:
        loop_probs = prob_list
    ax.plot(loop_probs, linewidth=2.0, alpha=0.8, color='#2E86AB', label='Token Probability')
    
    ax.set_xlabel('Token Position', fontsize=18, fontweight='bold')
    ax.set_ylabel('Probability', fontsize=18, fontweight='bold')
    ax.set_title(f'Probability Over Token Sequence - {output_id}', fontsize=20, fontweight='bold', pad=20)
    
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_facecolor('#f8f9fa')
    
    if len(prob_list) > 10:
        window_size = min(20, len(prob_list) // 5)
        moving_avg = pd.Series(prob_list).rolling(window=window_size, center=True).mean()
        ax.plot(moving_avg, 'r--', alpha=0.7, linewidth=2.5, 
                label=f'Moving Average (window={window_size})')
    
    min_idx = np.argmin(prob_list)
    max_idx = np.argmax(prob_list)
    
    ax.scatter([min_idx], [prob_list[min_idx]], 
               color='#E63946', s=150, zorder=5, 
               marker='v', edgecolors='darkred', linewidth=2,
               label=f'Min: {prob_list[min_idx]:.4f} (pos: {min_idx})')
    
    ax.scatter([max_idx], [prob_list[max_idx]], 
               color='#06A77D', s=150, zorder=5,
               marker='^', edgecolors='darkgreen', linewidth=2,
               label=f'Max: {prob_list[max_idx]:.4f} (pos: {max_idx})')
    
    mean_val = np.mean(prob_list)
    median_val = np.median(prob_list)
    
    ax.axhline(y=mean_val, color='orange', linestyle=':', linewidth=2, 
               alpha=0.7, label=f'Mean: {mean_val:.4f}')
    ax.axhline(y=median_val, color='purple', linestyle='-.', linewidth=2, 
               alpha=0.7, label=f'Median: {median_val:.4f}')
    
    std_val = np.std(prob_list)
    ax.fill_between(range(len(prob_list)), 
                    mean_val - std_val, mean_val + std_val,
                    alpha=0.1, color='gray', 
                    label=f'±1 Std Dev ({std_val:.4f})')
    
    ax.legend(loc='best', frameon=True, fancybox=True, 
              shadow=True, framealpha=0.9, fontsize=12)
    
    y_margin = (max(prob_list) - min(prob_list)) * 0.1
    ax.set_ylim([max(0, min(prob_list) - y_margin), 
                 min(1, max(prob_list) + y_margin)])
    
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
    
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    
    if save_fig:
        fig_path = f"figs/loop_prob_trend.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"\nPNG figure saved to: {fig_path}")
        fig_path = fig_path.replace("png", "pdf")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"\nPDF figure saved to: {fig_path}")
    
    plt.show()
    



def print_statistics(prob_list, token_list):
    print("\n" + "="*50)
    print("Statistics:")
    print("="*50)
    print(f"Total tokens: {len(token_list)}")
    print(f"Unique tokens: {len(set(token_list))}")
    print(f"Repeat rate: {1 - len(set(token_list))/len(token_list):.2%}")
    print(f"\nProbability statistics:")
    print(f"   Average: {np.mean(prob_list):.6f}")
    print(f"   Median: {np.median(prob_list):.6f}")
    print(f"   Standard deviation: {np.std(prob_list):.6f}")
    print(f"   Min: {np.min(prob_list):.6f} (position: {np.argmin(prob_list)})")
    print(f"   Max: {np.max(prob_list):.6f} (position: {np.argmax(prob_list)})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True)

    args = parser.parse_args()
    
    file_name = args.json_path
    with open(file_name, 'r', encoding='utf-8') as f:
        data = json.load(f)
    token_list = data["tokens"]
    prob_list = data["probs"]
    output_id = data["output_id"]
    
    plot_probability_trends(token_list, prob_list, output_id)
    plot_loop_heatmap(token_list, prob_list, output_id)
    plot_loop_probability_trend(token_list, prob_list, output_id)
    
    print_statistics(prob_list, token_list)