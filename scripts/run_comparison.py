"""
Master Script - Sequential Execution to Avoid Memory Issues
"""

import subprocess
import sys

def run_step(script, description):
    """Run a script and check for errors"""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    
    result = subprocess.run([sys.executable, script], capture_output=False)
    if result.returncode != 0:
        print(f"\nError in {description}. Exiting.")
        sys.exit(1)
    print(f"\n✓ {description} completed")

if __name__ == "__main__":
    print("="*60)
    print("SDXL MODEL COMPARISON PIPELINE")
    print("="*60)
    
    # Step 1: Generate images (memory optimized)
    run_step("generate_images.py", "Image Generation")
    
    # Step 2: Calculate FID
    run_step("evaluate_fid.py", "FID Evaluation")
    
    # Step 3: Analyze speed
    run_step("evaluate_speed.py", "Speed Analysis")
    
    # Step 4: Generate report
    run_step("generate_report.py", "Report Generation")
    
    print("\n" + "="*60)
    print("COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nCheck: data/evaluation_results/final_comparison_report.md")