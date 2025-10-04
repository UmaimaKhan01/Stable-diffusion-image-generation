"""
Speed Evaluation Script for SDXL Model Comparison
Analyzes the timing results from image generation
"""

import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, List
import argparse

class SpeedEvaluator:
    def __init__(self, results_dir="generated_images"):
        """Initialize speed evaluator"""
        self.results_dir = results_dir
        self.timing_file = os.path.join(results_dir, r"C:\Users\umaim\Downloads\stable_diffusion\data\generated_images\timing_results.json")
        
    def load_timing_results(self) -> Dict:
        """Load timing results from JSON file"""
        if not os.path.exists(self.timing_file):
            raise FileNotFoundError(f"Timing results not found at {self.timing_file}")
            
        with open(self.timing_file, 'r') as f:
            return json.load(f)
    
    def analyze_speed_metrics(self, timing_data: Dict) -> Dict:
        """Analyze speed metrics and create comprehensive report"""
        
        # Extract valid timing data
        turbo_times = [t for t in timing_data['turbo_times'] if t is not None]
        base_times = [t for t in timing_data['base_times'] if t is not None]
        
        # Calculate detailed statistics
        analysis = {
            'turbo_stats': {
                'mean_time': np.mean(turbo_times),
                'median_time': np.median(turbo_times),
                'std_time': np.std(turbo_times),
                'min_time': np.min(turbo_times),
                'max_time': np.max(turbo_times),
                'images_per_second': 1.0 / np.mean(turbo_times),
                'total_time': np.sum(turbo_times)
            },
            'base_stats': {
                'mean_time': np.mean(base_times),
                'median_time': np.median(base_times),
                'std_time': np.std(base_times),
                'min_time': np.min(base_times),
                'max_time': np.max(base_times),
                'images_per_second': 1.0 / np.mean(base_times),
                'total_time': np.sum(base_times)
            },
            'comparison': {
                'speed_improvement_mean': np.mean(base_times) / np.mean(turbo_times),
                'speed_improvement_median': np.median(base_times) / np.median(turbo_times),
                'time_saved_per_image': np.mean(base_times) - np.mean(turbo_times),
                'total_time_saved': np.sum(base_times) - np.sum(turbo_times),
                'efficiency_ratio': np.mean(turbo_times) / np.mean(base_times) * 100  # % of base time
            }
        }
        
        return analysis
    
    def create_speed_visualization(self, timing_data: Dict, output_dir: str):
        """Create visualizations for speed comparison"""
        
        # Extract valid timing data
        turbo_times = [t for t in timing_data['turbo_times'] if t is not None]
        base_times = [t for t in timing_data['base_times'] if t is not None]
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('SDXL Model Speed Comparison', fontsize=16, fontweight='bold')
        
        # 1. Box plot comparison
        ax1.boxplot([turbo_times, base_times], labels=['SDXL Turbo', 'SDXL Base'])
        ax1.set_ylabel('Generation Time (seconds)')
        ax1.set_title('Generation Time Distribution')
        ax1.grid(True, alpha=0.3)
        
        # 2. Bar chart of average times
        models = ['SDXL Turbo', 'SDXL Base']
        avg_times = [np.mean(turbo_times), np.mean(base_times)]
        colors = ['#FF6B6B', '#4ECDC4']
        
        bars = ax2.bar(models, avg_times, color=colors, alpha=0.8)
        ax2.set_ylabel('Average Generation Time (seconds)')
        ax2.set_title('Average Generation Time')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, time_val in zip(bars, avg_times):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{time_val:.2f}s', ha='center', va='bottom', fontweight='bold')
        
        # 3. Images per second comparison
        ips_turbo = 1.0 / np.mean(turbo_times)
        ips_base = 1.0 / np.mean(base_times)
        
        bars2 = ax3.bar(models, [ips_turbo, ips_base], color=colors, alpha=0.8)
        ax3.set_ylabel('Images per Second')
        ax3.set_title('Throughput Comparison')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, ips_val in zip(bars2, [ips_turbo, ips_base]):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{ips_val:.2f} img/s', ha='center', va='bottom', fontweight='bold')
        
        # 4. Time series plot
        prompt_indices = list(range(1, len(turbo_times) + 1))
        ax4.plot(prompt_indices, turbo_times, 'o-', label='SDXL Turbo', color='#FF6B6B', linewidth=2)
        ax4.plot(prompt_indices, base_times, 'o-', label='SDXL Base', color='#4ECDC4', linewidth=2)
        ax4.set_xlabel('Prompt Number')
        ax4.set_ylabel('Generation Time (seconds)')
        ax4.set_title('Generation Time per Prompt')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, 'speed_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Speed visualization saved to {plot_path}")
        
        return plot_path
    
    def generate_speed_report(self, output_dir="evaluation_results") -> Dict:
        """Generate comprehensive speed evaluation report"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Load timing data
        timing_data = self.load_timing_results()
        
        # Analyze speed metrics
        analysis = self.analyze_speed_metrics(timing_data)
        
        # Create visualizations
        plot_path = self.create_speed_visualization(timing_data, output_dir)
        
        # Create detailed report
        report = {
            'summary': {
                'total_prompts': len([t for t in timing_data['turbo_times'] if t is not None]),
                'turbo_avg_time': analysis['turbo_stats']['mean_time'],
                'base_avg_time': analysis['base_stats']['mean_time'],
                'speed_improvement': analysis['comparison']['speed_improvement_mean'],
                'time_saved_per_image': analysis['comparison']['time_saved_per_image'],
                'total_time_saved': analysis['comparison']['total_time_saved']
            },
            'detailed_analysis': analysis,
            'visualizations': {
                'speed_comparison_plot': plot_path
            },
            'raw_data': timing_data
        }
        
        # Save report
        report_path = os.path.join(output_dir, 'speed_evaluation_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report

def main():
    parser = argparse.ArgumentParser(description='Evaluate speed performance of SDXL models')
    parser.add_argument('--results_dir', default='generated_images', 
                       help='Directory containing timing results')
    parser.add_argument('--output_dir', default='evaluation_results', 
                       help='Output directory for evaluation results')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = SpeedEvaluator(results_dir=args.results_dir)
    
    try:
        # Generate speed report
        report = evaluator.generate_speed_report(output_dir=args.output_dir)
        
        # Print summary
        print("\n" + "="*60)
        print("SPEED EVALUATION SUMMARY")
        print("="*60)
        
        summary = report['summary']
        print(f"Total prompts evaluated: {summary['total_prompts']}")
        print(f"SDXL Turbo average time: {summary['turbo_avg_time']:.2f} seconds")
        print(f"SDXL Base average time: {summary['base_avg_time']:.2f} seconds")
        print(f"Speed improvement: {summary['speed_improvement']:.2f}x faster")
        print(f"Time saved per image: {summary['time_saved_per_image']:.2f} seconds")
        print(f"Total time saved: {summary['total_time_saved']:.1f} seconds")
        
        # Print detailed stats
        turbo_stats = report['detailed_analysis']['turbo_stats']
        base_stats = report['detailed_analysis']['base_stats']
        
        print(f"\nSDXL Turbo Performance:")
        print(f"  - Images per second: {turbo_stats['images_per_second']:.2f}")
        print(f"  - Min/Max time: {turbo_stats['min_time']:.2f}s / {turbo_stats['max_time']:.2f}s")
        print(f"  - Standard deviation: {turbo_stats['std_time']:.2f}s")
        
        print(f"\nSDXL Base Performance:")
        print(f"  - Images per second: {base_stats['images_per_second']:.2f}")
        print(f"  - Min/Max time: {base_stats['min_time']:.2f}s / {base_stats['max_time']:.2f}s")
        print(f"  - Standard deviation: {base_stats['std_time']:.2f}s")
        
        print(f"\nDetailed report saved to: {args.output_dir}/speed_evaluation_report.json")
        print("="*60)
        
    except Exception as e:
        print(f"Error during speed evaluation: {e}")

if __name__ == "__main__":
    main()