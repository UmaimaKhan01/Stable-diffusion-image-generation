"""
Report Generation Script for SDXL Model Comparison
Combines FID and Speed results into a comprehensive report
"""

import os
import json
import argparse
from datetime import datetime

def load_json_safe(filepath):
    """Safely load JSON file"""
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        else:
            print(f"Warning: {filepath} not found")
            return None
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def generate_markdown_report(output_dir="data/evaluation_results"):
    """Generate comprehensive markdown report"""
    
    # Load results
    timing_file = os.path.join(output_dir, "..", "generated_images", "timing_results.json")
    fid_file = os.path.join(output_dir, "fid_results.json")
    speed_file = os.path.join(output_dir, "speed_evaluation_report.json")
    
    timing_data = load_json_safe(timing_file)
    fid_data = load_json_safe(fid_file)
    speed_data = load_json_safe(speed_file)
    
    # Start building report
    report = []
    report.append("# SDXL Model Comparison Report\n")
    report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    report.append("---\n\n")
    
    # Executive Summary
    report.append("## Executive Summary\n\n")
    report.append("This report compares two SDXL models:\n")
    report.append("- **SDXL Turbo**: Optimized for speed (1-4 inference steps)\n")
    report.append("- **SDXL Base**: Standard quality model (25-50 inference steps)\n\n")
    
    # Speed Results
    report.append("## 1. Speed Performance\n\n")
    if timing_data:
        report.append("### Generation Times\n\n")
        report.append("| Model | Avg Time | Images/Second | Speedup |\n")
        report.append("|-------|----------|---------------|----------|\n")
        turbo_avg = timing_data.get('turbo_avg_time', 0)
        base_avg = timing_data.get('base_avg_time', 0)
        turbo_ips = timing_data.get('turbo_images_per_second', 0)
        base_ips = timing_data.get('base_images_per_second', 0)
        speedup = timing_data.get('speed_improvement', 0)
        
        report.append(f"| SDXL Turbo | {turbo_avg:.2f}s | {turbo_ips:.2f} | - |\n")
        report.append(f"| SDXL Base | {base_avg:.2f}s | {base_ips:.2f} | - |\n")
        report.append(f"| **Difference** | **-{base_avg-turbo_avg:.2f}s** | - | **{speedup:.2f}x faster** |\n\n")
        
        report.append("### Key Findings:\n")
        report.append(f"- SDXL Turbo is **{speedup:.2f}x faster** than SDXL Base\n")
        report.append(f"- Turbo saves **{base_avg-turbo_avg:.2f} seconds** per image\n")
        report.append(f"- Turbo can generate **{turbo_ips:.2f} images per second**\n\n")
    else:
        report.append("*Speed data not available*\n\n")
    
    # FID Results
    report.append("## 2. Image Quality (FID Score)\n\n")
    if fid_data:
        fid_score = fid_data.get('fid_score', 0)
        report.append(f"**FID Score (Turbo vs Base):** {fid_score:.2f}\n\n")
        
        report.append("### Interpretation:\n")
        report.append("- Lower FID = more similar to reference quality\n")
        report.append("- Reference: SDXL Base (higher quality model)\n\n")
        
        if fid_score < 20:
            quality_assessment = "Very similar quality"
        elif fid_score < 50:
            quality_assessment = "Moderate quality difference"
        else:
            quality_assessment = "Significant quality difference"
        
        report.append(f"**Assessment:** {quality_assessment}\n\n")
        
        report.append("### FID Score Ranges:\n")
        report.append("- < 20: Very similar quality\n")
        report.append("- 20-50: Moderate difference\n")
        report.append("- > 50: Significant difference\n\n")
    else:
        report.append("*FID data not available*\n\n")
    
    # Trade-off Analysis
    report.append("## 3. Trade-off Analysis\n\n")
    report.append("### SDXL Turbo Advantages:\n")
    if timing_data:
        report.append(f"- {speedup:.1f}x faster generation\n")
        report.append("- Ideal for rapid prototyping and iteration\n")
        report.append("- Lower computational requirements\n")
        report.append("- Good for real-time applications\n\n")
    
    report.append("### SDXL Base Advantages:\n")
    report.append("- Higher quality reference standard\n")
    report.append("- More detailed and refined outputs\n")
    report.append("- Better for final production work\n")
    report.append("- More consistent results\n\n")
    
    # Recommendations
    report.append("## 4. Recommendations\n\n")
    report.append("**Use SDXL Turbo when:**\n")
    report.append("- Speed is critical\n")
    report.append("- Iterating on prompts and concepts\n")
    report.append("- Creating drafts or previews\n")
    report.append("- Resource constraints exist\n\n")
    
    report.append("**Use SDXL Base when:**\n")
    report.append("- Quality is paramount\n")
    report.append("- Creating final production assets\n")
    report.append("- Time is not a constraint\n")
    report.append("- Maximum detail is needed\n\n")
    
    # Methodology
    report.append("## 5. Methodology\n\n")
    report.append("### Models Tested:\n")
    report.append("- **SDXL Turbo**: `stabilityai/sdxl-turbo`\n")
    report.append("  - Inference steps: 1-4\n")
    report.append("  - Guidance scale: 0.0\n\n")
    report.append("- **SDXL Base**: `stabilityai/stable-diffusion-xl-base-1.0`\n")
    report.append("  - Inference steps: 25\n")
    report.append("  - Guidance scale: 7.5\n\n")
    
    report.append("### Evaluation Metrics:\n")
    report.append("1. **Speed**: Average generation time per image\n")
    report.append("2. **FID Score**: Fréchet Inception Distance measuring similarity to reference\n\n")
    
    if timing_data:
        num_prompts = len([t for t in timing_data.get('turbo_times', []) if t is not None])
        report.append(f"### Dataset:\n")
        report.append(f"- Number of prompts: {num_prompts}\n")
        report.append(f"- Source: Stable Diffusion Prompts dataset\n\n")
    
    # Conclusion
    report.append("## 6. Conclusion\n\n")
    if timing_data and fid_data:
        report.append(f"SDXL Turbo offers a **{speedup:.1f}x speedup** over SDXL Base, ")
        report.append(f"with an FID score of **{fid_score:.2f}**. ")
        
        if fid_score < 30:
            report.append("The quality difference is relatively small, making Turbo an excellent choice for most applications where speed matters. ")
        else:
            report.append("There is a noticeable quality difference, so the choice depends on whether speed or quality is more important for your use case. ")
        
        report.append("For production workflows, consider using Turbo for iteration and Base for final outputs.\n\n")
    else:
        report.append("Complete evaluation data is needed for a comprehensive conclusion.\n\n")
    
    report.append("---\n\n")
    report.append("*End of Report*\n")
    
    # Write report
    report_path = os.path.join(output_dir, "final_comparison_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(''.join(report))
    
    return report_path

def main():
    parser = argparse.ArgumentParser(description='Generate final comparison report')
    parser.add_argument('--output_dir', default='data/evaluation_results',
                       help='Output directory for report')
    
    args = parser.parse_args()
    
    print("="*60)
    print("GENERATING FINAL REPORT")
    print("="*60)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    report_path = generate_markdown_report(args.output_dir)
    
    print(f"\nReport generated successfully!")
    print(f"Location: {report_path}")
    print("\n" + "="*60)
    print("REPORT PREVIEW")
    print("="*60)
    
    # Print first few lines of report
    with open(report_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines[:15]:
            print(line.rstrip())
    
    print("\n... (see full report in file)")
    print("="*60)

if __name__ == "__main__":
    main()