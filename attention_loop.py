import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
from sklearn.metrics.pairwise import cosine_similarity
import torch
from typing import List, Dict, Optional, Union
import argparse
from transformers import AutoTokenizer
from utils import detect_loop_pattern

class AttentionLoopAnalyzer:
    def __init__(self, attention_scores: Union[torch.Tensor, np.ndarray], 
                 tokens: List[str], 
                 loop_start_idx: int = None, 
                 apply_softmax: bool = True):
        if isinstance(attention_scores, torch.Tensor):
            self.raw_attention_scores = attention_scores.detach().float().cpu().numpy()
        else:
            self.raw_attention_scores = attention_scores
            
        self.tokens = tokens
        self.num_layers = self.raw_attention_scores.shape[0]
        self.seq_len = self.raw_attention_scores.shape[1]
        self.loop_start_idx = loop_start_idx or detect_loop_pattern(self.tokens)['loop_start']
        
        if apply_softmax:
            self.attention_scores = self._apply_softmax(self.raw_attention_scores)
        else:
            self.attention_scores = self.raw_attention_scores
        
        self.window_size = 128

    
    def _apply_softmax(self, scores: np.ndarray) -> np.ndarray:
        softmaxed = np.zeros_like(scores)
        
        for layer_idx in range(self.num_layers):
            for query_idx in range(self.seq_len):
                row_scores = scores[layer_idx, query_idx, :].copy()
                
                valid_mask = ~np.isinf(row_scores)
                
                if valid_mask.any():
                    valid_scores = row_scores[valid_mask]
                    max_score = np.max(valid_scores)
                    exp_scores = np.exp(valid_scores - max_score)
                    softmax_scores = exp_scores / np.sum(exp_scores)
                    
                    softmaxed[layer_idx, query_idx, valid_mask] = softmax_scores
                    softmaxed[layer_idx, query_idx, ~valid_mask] = 0
                else:
                    softmaxed[layer_idx, query_idx, :] = 1.0 / self.seq_len
        
        return softmaxed
    
    
    def analyze_attention_entropy(self) -> Dict:
        layer_entropies = []
        
        for layer_idx in range(self.num_layers):
            layer_attention = self.attention_scores[layer_idx]
            
            layer_entropy = []
            for pos in range(self.seq_len):
                attn_dist = layer_attention[pos]
                attn_dist = attn_dist / (attn_dist.sum() + 1e-10)
                ent = entropy(attn_dist + 1e-10)
                layer_entropy.append(ent)
            
            layer_entropies.append(layer_entropy)
        
        layer_entropies = np.array(layer_entropies)
        
        pre_loop_entropy = layer_entropies[:, :self.loop_start_idx].mean(axis=1)
        loop_entropy = layer_entropies[:, self.loop_start_idx:].mean(axis=1)
        
        return {
            'layer_entropies': layer_entropies,
            'pre_loop_avg': pre_loop_entropy,
            'loop_avg': loop_entropy,
            'entropy_drop': pre_loop_entropy - loop_entropy
        }
    
    def analyze_attention_patterns(self) -> Dict:
        patterns = {
            'self_attention': [],
            'backward_attention': [],
            'forward_attention': [],
            'loop_region_attention': []
        }
        
        for layer_idx in range(self.num_layers):
            layer_attn = self.attention_scores[layer_idx]
            
            loop_region = layer_attn[self.loop_start_idx:, :]
            
            self_attn = np.diag(layer_attn).mean()
            patterns['self_attention'].append(self_attn)
            
            backward_mask = np.tril(np.ones_like(layer_attn), k=-1)
            backward_attn = (layer_attn * backward_mask).sum() / (backward_mask.sum() + 1e-10)
            patterns['backward_attention'].append(backward_attn)
            
            forward_mask = np.triu(np.ones_like(layer_attn), k=1)
            forward_attn = (layer_attn * forward_mask).sum() / (forward_mask.sum() + 1e-10)
            patterns['forward_attention'].append(forward_attn)
            
            loop_self_attn = loop_region[:, self.loop_start_idx:].mean()
            patterns['loop_region_attention'].append(loop_self_attn)
        
        return patterns
    
    def detect_attention_loops(self, threshold: float = 0.8) -> Dict:
        loop_patterns = []
        
        for layer_idx in range(self.num_layers):
            layer_attn = self.attention_scores[layer_idx]
            
            loop_region = layer_attn[self.loop_start_idx:, :self.loop_start_idx]
            
            for i in range(len(loop_region) - 1):
                for j in range(i + 1, min(i + 50, len(loop_region))):
                    similarity = cosine_similarity(
                        loop_region[i:i+1], 
                        loop_region[j:j+1]
                    )[0, 0]
                    
                    if similarity > threshold:
                        loop_patterns.append({
                            'layer': layer_idx,
                            'pos1': self.loop_start_idx + i,
                            'pos2': self.loop_start_idx + j,
                            'similarity': similarity,
                            'distance': j - i
                        })
        
        return {'patterns': loop_patterns, 'threshold': threshold}
    
    def analyze_layer_specialization(self) -> Dict:
        specialization = {}
        
        for layer_idx in range(self.num_layers):
            layer_attn = self.attention_scores[layer_idx]
            
            attn_std = layer_attn.std(axis=1).mean()
            attn_max = layer_attn.max(axis=1).mean()
            
            distances = []
            for i in range(self.seq_len):
                weights = layer_attn[i, :i+1]
                positions = np.arange(i+1)
                if weights.sum() > 0:
                    avg_distance = np.average(i - positions, weights=weights)
                    distances.append(avg_distance)
            
            avg_distance = np.mean(distances) if distances else 0
            
            specialization[f'layer_{layer_idx}'] = {
                'attention_std': attn_std,
                'attention_max': attn_max,
                'avg_attention_distance': avg_distance,
                'is_local': avg_distance < 10,
                'is_global': avg_distance > self.seq_len * 0.3
            }
        
        return specialization
    
    def visualize_comprehensive_analysis(self, save_path: str = None):
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        ax1 = fig.add_subplot(gs[0, :])
        entropy_data = self.analyze_attention_entropy()
        
        for layer_idx in range(0, self.num_layers, 3):
            ax1.plot(entropy_data['layer_entropies'][layer_idx], 
                    label=f'Layer {layer_idx}', alpha=0.7)
        
        ax1.axvline(x=self.loop_start_idx, color='red', linestyle='--', 
                   label='Loop Start', alpha=0.5)
        ax1.set_xlabel('Position')
        ax1.set_ylabel('Attention Entropy')
        ax1.set_title('Attention Entropy Across Positions')
        ax1.legend(loc='best', ncol=4)
        ax1.grid(True, alpha=0.3)
        
        ax2 = fig.add_subplot(gs[1, 0])
        layers = range(self.num_layers)
        width = 0.35
        x = np.arange(len(layers))
        
        ax2.bar(x - width/2, entropy_data['pre_loop_avg'], width, 
               label='Pre-loop', alpha=0.7)
        ax2.bar(x + width/2, entropy_data['loop_avg'], width, 
               label='During loop', alpha=0.7)
        
        ax2.set_xlabel('Layer')
        ax2.set_ylabel('Average Entropy')
        ax2.set_title('Entropy: Pre-loop vs During Loop')
        ax2.set_xticks(x[::2])
        ax2.set_xticklabels([str(i) for i in layers[::2]])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3 = fig.add_subplot(gs[1, 1])
        patterns = self.analyze_attention_patterns()
        
        ax3.plot(patterns['self_attention'], 'o-', label='Self Attention', alpha=0.7)
        ax3.plot(patterns['backward_attention'], 's-', label='Backward Attention', alpha=0.7)
        ax3.plot(patterns['loop_region_attention'], '^-', label='Loop Region Attention', alpha=0.7)
        
        ax3.set_xlabel('Layer')
        ax3.set_ylabel('Attention Weight')
        ax3.set_title('Attention Pattern Distribution by Layer')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.bar(range(self.num_layers), entropy_data['entropy_drop'])
        ax4.set_xlabel('Layer')
        ax4.set_ylabel('Entropy Drop')
        ax4.set_title('Entropy Drop in Loop Region')
        ax4.grid(True, alpha=0.3)
        
        key_layers = [0, self.num_layers//2, self.num_layers-1]
        
        for idx, layer_idx in enumerate(key_layers):
            ax = fig.add_subplot(gs[2, idx])
            
            display_start = max(0, self.seq_len - 100)
            attn_subset = self.attention_scores[layer_idx][display_start:, display_start:]
            
            im = ax.imshow(attn_subset, cmap='hot', aspect='auto')
            ax.set_title(f'Layer {layer_idx} Attention (Last 100 tokens)')
            ax.set_xlabel('Key Position')
            ax.set_ylabel('Query Position')
            
            if self.loop_start_idx > display_start:
                loop_mark = self.loop_start_idx - display_start
                ax.axhline(y=loop_mark, color='blue', linestyle='--', alpha=0.5)
                ax.axvline(x=loop_mark, color='blue', linestyle='--', alpha=0.5)
            
            plt.colorbar(im, ax=ax)
        
        ax6 = fig.add_subplot(gs[3, :])
        specialization = self.analyze_layer_specialization()
        
        local_scores = []
        global_scores = []
        for layer_idx in range(self.num_layers):
            spec = specialization[f'layer_{layer_idx}']
            local_scores.append(1 if spec['is_local'] else 0)
            global_scores.append(1 if spec['is_global'] else 0)
        
        x = np.arange(self.num_layers)
        ax6.bar(x - 0.2, local_scores, 0.4, label='Local Focus', alpha=0.7)
        ax6.bar(x + 0.2, global_scores, 0.4, label='Global Focus', alpha=0.7)
        ax6.set_xlabel('Layer')
        ax6.set_ylabel('Specialization Type')
        ax6.set_title('Layer Specialization: Local vs Global Attention')
        ax6.set_xticks(x[::2])
        ax6.set_xticklabels([str(i) for i in x[::2]])
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.suptitle('Attention Pattern Analysis for Endless Loop Detection', 
                    fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        plt.show()
        
    def generate_report(self) -> str:
        entropy_data = self.analyze_attention_entropy()
        patterns = self.analyze_attention_patterns()
        loop_detection = self.detect_attention_loops()
        specialization = self.analyze_layer_specialization()
        
        report = []
        report.append("="*60)
        report.append("Attention Pattern Analysis Report")
        report.append("="*60)
        
        report.append("\n0. ATTENTION MASK INFORMATION")
        report.append("-"*40)
        report.append("\n1. ENTROPY ANALYSIS")
        report.append("-"*40)
        avg_entropy_drop = entropy_data['entropy_drop'].mean()
        report.append(f"Average entropy drop in loop: {avg_entropy_drop:.4f}")
        
        max_drop_layer = np.argmax(entropy_data['entropy_drop'])
        report.append(f"Layer with maximum entropy drop: Layer {max_drop_layer}")
        report.append(f"  Pre-loop entropy: {entropy_data['pre_loop_avg'][max_drop_layer]:.4f}")
        report.append(f"  Loop entropy: {entropy_data['loop_avg'][max_drop_layer]:.4f}")
        
        report.append("\n2. ATTENTION PATTERNS")
        report.append("-"*40)
        
        avg_self = np.mean(patterns['self_attention'])
        avg_backward = np.mean(patterns['backward_attention'])
        avg_loop = np.mean(patterns['loop_region_attention'])
        
        report.append(f"Average self-attention: {avg_self:.4f}")
        report.append(f"Average backward attention: {avg_backward:.4f}")
        report.append(f"Average loop region attention: {avg_loop:.4f}")
        
        report.append("\n3. LOOP PATTERN DETECTION")
        report.append("-"*40)
        
        if loop_detection['patterns']:
            report.append(f"Found {len(loop_detection['patterns'])} similar attention patterns")
            
            layer_counts = {}
            for pattern in loop_detection['patterns']:
                layer = pattern['layer']
                layer_counts[layer] = layer_counts.get(layer, 0) + 1
            
            report.append("Loop patterns by layer:")
            for layer, count in sorted(layer_counts.items()):
                report.append(f"  Layer {layer}: {count} patterns")
        else:
            report.append("No significant loop patterns detected")
        
        report.append("\n4. LAYER SPECIALIZATION")
        report.append("-"*40)
        
        local_layers = []
        global_layers = []
        
        for layer_idx in range(self.num_layers):
            spec = specialization[f'layer_{layer_idx}']
            if spec['is_local']:
                local_layers.append(layer_idx)
            if spec['is_global']:
                global_layers.append(layer_idx)
        
        report.append(f"Layers with local focus: {local_layers}")
        report.append(f"Layers with global focus: {global_layers}")
        
        report.append("\n5. KEY FINDINGS AND RECOMMENDATIONS")
        report.append("-"*40)
        
        if avg_entropy_drop > 0.5:
            report.append("⚠️ Significant entropy drop detected in loop region")
            report.append("   → Model is becoming increasingly certain about predictions")
        
        if avg_loop > avg_backward * 1.5:
            report.append("⚠️ High attention within loop region")
            report.append("   → Model is primarily attending to repeated content")
        
        if len(loop_detection['patterns']) > 10:
            report.append("⚠️ Many similar attention patterns found")
            report.append("   → Strong indication of repetitive processing")
        
        report.append("\n" + "="*60)
        
        return "\n".join(report)


def analyze_attention_for_loop(attention_scores: Union[torch.Tensor, np.ndarray], 
                               tokens: List[str], 
                               is_pre_softmax: bool = True,
                               file_prefix: str = None):
    analyzer = AttentionLoopAnalyzer(
        attention_scores, 
        tokens,
        apply_softmax=is_pre_softmax,
        # loop_start_idx=134
    )
    analyzer.visualize_comprehensive_analysis(save_path=f"figs/{file_prefix}_attention_loop_analysis.png")
    report = analyzer.generate_report()
    print(report)
    with open(f"reports/{file_prefix}_attention_analysis_report.txt", "w") as f:
        f.write(report)
    
    return analyzer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--attn_score_path", type=str, required=True)
    parser.add_argument("--num_layers", type=int, default=24)
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)    
    args = parser.parse_args()

    with open(args.json_path, "r") as f:
        data = json.load(f)
        seq = data["full_sequence"]
    attention_scores = torch.load(args.attn_score_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokens = tokenizer.tokenize(seq)
    file_prefix = args.json_path.split('/')[-1].split('.')[0]
    analyzer = analyze_attention_for_loop(attention_scores, tokens, is_pre_softmax=True, file_prefix=file_prefix)