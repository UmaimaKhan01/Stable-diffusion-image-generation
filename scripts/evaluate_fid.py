"""
FID Evaluation Script for SDXL Model Comparison
Fixed for small datasets (10 images)
"""

import os
import json
import argparse
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from scipy import linalg
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

try:
    from pytorch_fid.inception import InceptionV3
except ImportError:
    print("Warning: pytorch_fid not installed. Using manual calculation.")

class ImageDataset(Dataset):
    """Dataset for loading images"""
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(image_dir) 
                                   if f.endswith(('.png', '.jpg', '.jpeg'))])
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image

class FIDEvaluator:
    def __init__(self, device="cuda"):
        """Initialize FID evaluator"""
        self.device = device if torch.cuda.is_available() else "cpu"
        
        # Standard ImageNet normalization for Inception
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"FID Evaluator initialized on {self.device}")
        
    def load_inception_model(self):
        """Load InceptionV3 model for feature extraction"""
        try:
            from pytorch_fid.inception import InceptionV3
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
            model = InceptionV3([block_idx]).to(self.device)
        except:
            # Fallback to torchvision inception
            from torchvision.models import inception_v3
            model = inception_v3(pretrained=True, transform_input=False)
            model.fc = torch.nn.Identity()
            model = model.to(self.device)
        
        model.eval()
        return model
    
    def extract_features(self, image_dir, model, batch_size=10):
        """Extract Inception features from images"""
        dataset = ImageDataset(image_dir, self.transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, 
                              shuffle=False, num_workers=0)
        
        features = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Extracting features"):
                batch = batch.to(self.device)
                
                try:
                    # Try pytorch_fid style
                    feat = model(batch)[0]
                except:
                    # Fallback to regular inception
                    feat = model(batch)
                
                # Handle different output shapes
                if len(feat.shape) > 2:
                    feat = torch.nn.functional.adaptive_avg_pool2d(feat, (1, 1))
                    feat = feat.squeeze(-1).squeeze(-1)
                
                features.append(feat.cpu().numpy())
        
        return np.concatenate(features, axis=0)
    
    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """
        Calculate Frechet distance with numerical stability for small datasets
        """
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)
        
        diff = mu1 - mu2
        
        # Add small epsilon to diagonal for numerical stability
        offset = np.eye(sigma1.shape[0]) * eps
        sigma1 = sigma1 + offset
        sigma2 = sigma2 + offset
        
        # Calculate sqrt of product of covariances
        try:
            covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        except:
            print("Warning: sqrtm failed, using approximation")
            # Fallback: use geometric mean of eigenvalues
            eigvals1 = np.linalg.eigvals(sigma1)
            eigvals2 = np.linalg.eigvals(sigma2)
            covmean = np.sqrt(np.abs(eigvals1.mean() * eigvals2.mean()))
            covmean = np.array([[covmean]])
        
        # Handle complex numbers
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                print(f"Warning: Imaginary component detected, taking real part")
            covmean = covmean.real
        
        tr_covmean = np.trace(covmean)
        
        fid = (diff.dot(diff) + np.trace(sigma1) + 
               np.trace(sigma2) - 2 * tr_covmean)
        
        return float(fid)
    
    def calculate_fid(self, dir1, dir2, batch_size=10):
        """
        Calculate FID score between two image directories
        """
        print(f"\nCalculating FID between:")
        print(f"  Reference: {dir1}")
        print(f"  Comparison: {dir2}")
        
        # Count images
        images1 = len([f for f in os.listdir(dir1) if f.endswith(('.png', '.jpg', '.jpeg'))])
        images2 = len([f for f in os.listdir(dir2) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        print(f"\nFound {images1} images in reference, {images2} in comparison")
        
        if images1 < 2 or images2 < 2:
            print("Error: Need at least 2 images in each directory")
            return None
        
        # Load model
        print("Loading Inception model...")
        model = self.load_inception_model()
        
        # Extract features
        print("\nExtracting features from reference...")
        features1 = self.extract_features(dir1, model, batch_size)
        
        print("Extracting features from comparison...")
        features2 = self.extract_features(dir2, model, batch_size)
        
        # Calculate statistics
        print("\nCalculating statistics...")
        mu1 = np.mean(features1, axis=0)
        sigma1 = np.cov(features1, rowvar=False)
        
        mu2 = np.mean(features2, axis=0)
        sigma2 = np.cov(features2, rowvar=False)
        
        # Calculate FID
        print("Computing FID score...")
        fid_value = self.calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
        
        return fid_value

def main():
    parser = argparse.ArgumentParser(description='Calculate FID scores for SDXL comparison')
    parser.add_argument('--turbo_dir', default='data/generated_images/sdxl_turbo', 
                       help='Directory with SDXL Turbo images')
    parser.add_argument('--base_dir', default='data/generated_images/sdxl_base', 
                       help='Directory with SDXL Base images')
    parser.add_argument('--output_dir', default='data/evaluation_results', 
                       help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=10, 
                       help='Batch size for FID calculation')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Verify directories exist and have images
    for dir_path, name in [(args.turbo_dir, "Turbo"), (args.base_dir, "Base")]:
        if not os.path.exists(dir_path):
            print(f"Error: {name} directory not found: {dir_path}")
            return
        images = [f for f in os.listdir(dir_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        if not images:
            print(f"Error: No images found in {name} directory: {dir_path}")
            return
    
    # Initialize evaluator
    evaluator = FIDEvaluator()
    
    # Calculate FID
    print("\n" + "="*60)
    print("FID SCORE CALCULATION")
    print("="*60)
    print("Using SDXL Base as reference (higher quality baseline)")
    print("Lower FID = SDXL Turbo is more similar to Base quality")
    
    fid_value = evaluator.calculate_fid(
        args.base_dir,
        args.turbo_dir,
        batch_size=args.batch_size
    )
    
    if fid_value is not None:
        results = {
            'fid_score': float(fid_value),
            'reference_model': 'SDXL Base',
            'evaluated_model': 'SDXL Turbo',
            'interpretation': 'Lower FID means Turbo generates images more similar to Base quality',
            'note': 'Small sample size (10 images) may affect FID reliability',
            'sample_size': 10
        }
        
        # Save results
        results_path = os.path.join(args.output_dir, 'fid_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n" + "="*60)
        print("FID RESULTS")
        print("="*60)
        print(f"FID Score (Turbo vs Base): {fid_value:.2f}")
        print(f"\nInterpretation:")
        print(f"  - Lower values = Turbo closer to Base quality")
        print(f"  - Small sample size (10 images) note:")
        print(f"    FID is more reliable with 50+ images")
        print(f"    Results should be interpreted cautiously")
        print(f"\nResults saved to: {results_path}")
        print("="*60)
    else:
        print("FID calculation failed!")

if __name__ == "__main__":
    main()