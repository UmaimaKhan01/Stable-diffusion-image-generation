"""
SDXL Model Comparison - Image Generation (Memory Optimized)
Loads models sequentially to avoid OOM errors
"""

import os
import time
import json
import torch
from PIL import Image
from diffusers import AutoPipelineForText2Image, StableDiffusionXLPipeline
from tqdm import tqdm
import argparse
import gc

class SDXLComparison:
    def __init__(self, output_dir="data/generated_images", device="cuda"):
        """Initialize SDXL comparison"""
        self.output_dir = output_dir
        self.device = device if torch.cuda.is_available() else "cpu"
        self.generation_times = {'turbo': [], 'base': []}

        # Create output directories
        self.turbo_dir = os.path.join(output_dir, "sdxl_turbo")
        self.base_dir = os.path.join(output_dir, "sdxl_base")
        os.makedirs(self.turbo_dir, exist_ok=True)
        os.makedirs(self.base_dir, exist_ok=True)

        print(f"Using device: {self.device}")
        if self.device == "cpu":
            print("WARNING: Running on CPU will be very slow!")

    def load_prompts(self, prompts_file):
        """Load prompts from file"""
        with open(prompts_file, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
        return prompts

    def clear_memory(self):
        """Clear GPU memory"""
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

    def generate_with_turbo(self, prompts):
        """Load SDXL Turbo, generate all images, then unload"""
        print("\n" + "="*60)
        print("LOADING SDXL TURBO")
        print("="*60)
        
        try:
            pipe = AutoPipelineForText2Image.from_pretrained(
                "stabilityai/sdxl-turbo",
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True
            )
            # Enable memory optimizations BEFORE moving to device
            if self.device == "cuda":
                try:
                    pipe.enable_xformers_memory_efficient_attention()
                except:
                    pass
                pipe.enable_vae_slicing()
                pipe.enable_vae_tiling()
                # Use CPU offloading for SDXL Base to save VRAM
                pipe.enable_model_cpu_offload()
            else:
                pipe.to(self.device)

            print("SDXL Turbo loaded successfully!")
            print("\nGenerating images with SDXL Turbo...")
            
            for i, prompt in enumerate(tqdm(prompts, desc="SDXL Turbo")):
                try:
                    generator = torch.Generator(device=self.device).manual_seed(42+i)
                    start_time = time.time()
                    
                    image = pipe(
                        prompt=prompt,
                        num_inference_steps=1,
                        guidance_scale=0.0,
                        generator=generator
                    ).images[0]
                    
                    generation_time = time.time() - start_time
                    
                    # Save image
                    image_path = os.path.join(self.turbo_dir, f"image_{i+1:03d}.png")
                    image.save(image_path)
                    self.generation_times['turbo'].append(generation_time)
                    
                    print(f"  [{i+1}/{len(prompts)}] Generated in {generation_time:.2f}s")
                    
                except Exception as e:
                    print(f"  Error on prompt {i+1}: {e}")
                    self.generation_times['turbo'].append(None)
                
                # Clear cache after each image
                self.clear_memory()
            
            # Unload model
            del pipe
            self.clear_memory()
            print("\nSDXL Turbo complete and unloaded!")
            
        except Exception as e:
            print(f"Error with SDXL Turbo: {e}")
            raise

    def generate_with_base(self, prompts):
        """Load SDXL Base, generate all images, then unload"""
        print("\n" + "="*60)
        print("LOADING SDXL BASE")
        print("="*60)
        
        try:
            pipe = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True
            )
            pipe.to(self.device)
            
            # Enable memory optimizations
            if self.device == "cuda":
                try:
                    pipe.enable_xformers_memory_efficient_attention()
                except:
                    pass
                pipe.enable_vae_slicing()
                pipe.enable_vae_tiling()

            print("SDXL Base loaded successfully!")
            print("\nGenerating images with SDXL Base...")
            
            for i, prompt in enumerate(tqdm(prompts, desc="SDXL Base")):
                try:
                    generator = torch.Generator(device=self.device).manual_seed(42+i)
                    start_time = time.time()
                    
                    image = pipe(
                        prompt=prompt,
                        num_inference_steps=25,
                        guidance_scale=7.5,
                        generator=generator
                    ).images[0]
                    
                    generation_time = time.time() - start_time
                    
                    # Save image
                    image_path = os.path.join(self.base_dir, f"image_{i+1:03d}.png")
                    image.save(image_path)
                    self.generation_times['base'].append(generation_time)
                    
                    print(f"  [{i+1}/{len(prompts)}] Generated in {generation_time:.2f}s")
                    
                except Exception as e:
                    print(f"  Error on prompt {i+1}: {e}")
                    self.generation_times['base'].append(None)
                
                # Clear cache after each image
                self.clear_memory()
            
            # Unload model
            del pipe
            self.clear_memory()
            print("\nSDXL Base complete and unloaded!")
            
        except Exception as e:
            print(f"Error with SDXL Base: {e}")
            raise

    def save_timing_results(self):
        """Save timing results to JSON file"""
        valid_turbo = [t for t in self.generation_times['turbo'] if t is not None]
        valid_base = [t for t in self.generation_times['base'] if t is not None]

        if not valid_turbo or not valid_base:
            print("No valid timing data to save")
            return

        results = {
            'turbo_times': self.generation_times['turbo'],
            'base_times': self.generation_times['base'],
            'turbo_avg_time': sum(valid_turbo) / len(valid_turbo),
            'base_avg_time': sum(valid_base) / len(valid_base),
            'turbo_images_per_second': len(valid_turbo) / sum(valid_turbo),
            'base_images_per_second': len(valid_base) / sum(valid_base),
            'speed_improvement': (sum(valid_base) / len(valid_base)) / (sum(valid_turbo) / len(valid_turbo))
        }

        results_path = os.path.join(self.output_dir, 'timing_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print("\n" + "="*60)
        print("TIMING RESULTS")
        print("="*60)
        print(f"SDXL Turbo average: {results['turbo_avg_time']:.2f}s ({results['turbo_images_per_second']:.2f} img/s)")
        print(f"SDXL Base average: {results['base_avg_time']:.2f}s ({results['base_images_per_second']:.2f} img/s)")
        print(f"Speed improvement: {results['speed_improvement']:.2f}x faster")
        print(f"Results saved to: {results_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate images with SDXL models')
    parser.add_argument('--prompts', default='prompts.txt', help='Path to prompts file')
    parser.add_argument('--output', default='data/generated_images', help='Output directory')
    parser.add_argument('--device', default='cuda', help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()

    print("="*60)
    print("SDXL MODEL COMPARISON - IMAGE GENERATION")
    print("="*60)

    comparison = SDXLComparison(output_dir=args.output, device=args.device)
    prompts = comparison.load_prompts(args.prompts)
    
    print(f"\nLoaded {len(prompts)} prompts")
    print("Models will be loaded sequentially to save memory")
    
    # Generate with Turbo first
    comparison.generate_with_turbo(prompts)
    
    # Generate with Base second
    comparison.generate_with_base(prompts)
    
    # Save results
    comparison.save_timing_results()
    
    print("\n" + "="*60)
    print("IMAGE GENERATION COMPLETED!")
    print("="*60)
    print(f"Check your generated images in: {args.output}")

if __name__ == "__main__":
    main()