"""
TEST VERSION - Master Script for Quick Testing
Uses smaller models and fewer prompts for rapid testing
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"Command: {command}")
    print('='*60)
    
    try:
        subprocess.run(command, shell=True, check=True, capture_output=False, text=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in {description}: {e}")
        return False

def setup_directories():
    """Create necessary directories"""
    directories = [
        "data",
        "data/generated_images",
        "data/generated_images/sdxl_turbo", 
        "data/generated_images/sdxl_base",
        "data/evaluation_results",
        "scripts"
    ]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

def main():
    parser = argparse.ArgumentParser(description='TEST VERSION - Quick SDXL model comparison')
    parser.add_argument('--skip-generation', action='store_true', help='Skip image generation')
    parser.add_argument('--skip-fid', action='store_true', help='Skip FID evaluation')
    parser.add_argument('--skip-speed', action='store_true', help='Skip speed evaluation')
    parser.add_argument('--prompts', default='prompts.txt', help='Path to prompts file')
    parser.add_argument('--device', default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--max-prompts', type=int, default=3, help='Max prompts to test')
    args = parser.parse_args()

    print("🧪 FAST TEST VERSION - SDXL Model Comparison")
    print("=" * 60)
    print("⚠️  Using smaller models for quick testing!")
    print("=" * 60)

    # Setup directories
    print("\n1. Setting up directories...")
    setup_directories()

    # Skip dependency check
    print("\n2. Skipping dependency check")

    # Generate images with test models
    if not args.skip_generation:
        print("\n3. Generating test images...")
        generation_cmd = (
            f"python generate_images_test.py --prompts {args.prompts} "
            f"--output data/generated_images --device {args.device} "
            f"--max-prompts {args.max_prompts}"
        )
        if not run_command(generation_cmd, "Test Image Generation"):
            print("❌ Test image generation failed. Exiting.")
            sys.exit(1)
    else:
        print("\n3. Skipping image generation")

    # Evaluate FID (if not skipped)
    if not args.skip_fid:
        print("\n4. Evaluating FID scores...")
        fid_cmd = (
            "python evaluate_fid.py "
            "--turbo_dir data/generated_images/sdxl_turbo "
            "--base_dir data/generated_images/sdxl_base "
            "--output_dir data/evaluation_results"
        )
        if not run_command(fid_cmd, "FID Evaluation"):
            print("⚠️ FID evaluation failed, continuing...")
    else:
        print("\n4. Skipping FID evaluation")

    # Evaluate Speed (if not skipped)
    if not args.skip_speed:
        print("\n5. Evaluating speed performance...")
        speed_cmd = (
            "python evaluate_speed.py "
            "--results_dir data/generated_images "
            "--output_dir data/evaluation_results"
        )
        if not run_command(speed_cmd, "Speed Evaluation"):
            print("⚠️ Speed evaluation failed, continuing...")
    else:
        print("\n5. Skipping speed evaluation")

    # Generate final report
    print("\n6. Generating test report...")
    report_cmd = "python generate_report.py --output_dir data/evaluation_results"
    if not run_command(report_cmd, "Report Generation"):
        print("❌ Report generation failed")
        sys.exit(1)

    # Success message
    print("\n" + "="*60)
    print("🎉 TEST VERSION COMPLETED!")
    print("="*60)
    print("\n📊 Test Results:")
    print("- Generated images: data/generated_images/")
    print("- Evaluation results: data/evaluation_results/")
    print("- Test report: data/evaluation_results/final_comparison_report.md")
    
    print(f"\n🧪 TEST SUMMARY:")
    print(f"- Used smaller models (SD v1.5) instead of SDXL")
    print(f"- Processed {args.max_prompts} prompts only")
    print(f"- Fast generation for testing pipeline")
    
    print("\n🔄 NEXT STEPS:")
    print("1. Review the test results")
    print("2. If everything works, run with real SDXL models:")
    print("   python run_comparison.py")
    print("3. Or increase test prompts: python run_test.py --max-prompts 5")

if __name__ == "__main__":
    main()