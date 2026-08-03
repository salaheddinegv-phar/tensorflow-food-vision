import os
import urllib.request
import zipfile
import tarfile
import shutil
from pathlib import Path

# Configuration
DATA_DIR = Path("full_data")
DATASET_URL = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
TAR_PATH = DATA_DIR / "food-101.tar.gz"
EXTRACT_PATH = DATA_DIR / "food-101"

def download_file(url, output_path, chunk_size=8192*1024):
    """Download file with progress bar"""
    print(f"Downloading from: {url}")
    print(f"Saving to: {output_path}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    req = urllib.request.Request(url)
    req.add_header('User-Agent', 'Mozilla/5.0')
    
    with urllib.request.urlopen(req) as response:
        total_size = int(response.headers.get('Content-Length', 0))
        downloaded = 0
        
        with open(output_path, 'wb') as f:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    mb_downloaded = downloaded / (1024**2)
                    mb_total = total_size / (1024**2)
                    print(f"\rProgress: {percent:.1f}% ({mb_downloaded:.1f} MB / {mb_total:.1f} MB)", end="")
    
    print("\nDownload complete!")

def extract_tar(tar_path, extract_to):
    """Extract tar.gz file"""
    print(f"\nExtracting: {tar_path}")
    print(f"To: {extract_to}")
    
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(extract_to)
    
    print("Extraction complete!")

def organize_dataset(base_dir):
    """Organize into train/test folders"""
    print("\nOrganizing dataset into train/test...")
    
    images_dir = base_dir / "images"
    meta_dir = base_dir / "meta"
    
    train_dir = base_dir / "train"
    test_dir = base_dir / "test"
    train_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)
    
    with open(meta_dir / "train.txt") as f:
        train_files = [line.strip().split('/') for line in f.readlines()]
    
    with open(meta_dir / "test.txt") as f:
        test_files = [line.strip().split('/') for line in f.readlines()]
    
    def copy_images(file_list, dest_dir):
        for class_name, img_name in file_list:
            class_folder = dest_dir / class_name
            class_folder.mkdir(exist_ok=True)
            src = images_dir / class_name / f"{img_name}.jpg"
            dst = class_folder / f"{img_name}.jpg"
            if src.exists():
                shutil.copy2(src, dst)
    
    print("Copying train images...")
    copy_images(train_files, train_dir)
    train_count = sum(1 for _ in train_dir.rglob('*.jpg'))
    print(f"Train images: {train_count}")
    
    print("Copying test images...")
    copy_images(test_files, test_dir)
    test_count = sum(1 for _ in test_dir.rglob('*.jpg'))
    print(f"Test images: {test_count}")
    
    print("Dataset organized!")

def main():
    print("=" * 50)
    print("Food-101 Dataset Downloader")
    print("=" * 50)
    
    # Step 1: Download
    if not TAR_PATH.exists():
        download_file(DATASET_URL, TAR_PATH)
    else:
        print(f"Tar file already exists: {TAR_PATH}")
        print("If previous download failed, delete it and rerun.")
    
    # Step 2: Extract
    if not EXTRACT_PATH.exists():
        extract_tar(TAR_PATH, DATA_DIR)
    else:
        print(f"Dataset already extracted: {EXTRACT_PATH}")
    
    # Step 3: Organize
    organize_dataset(EXTRACT_PATH)
    
    print("\n" + "=" * 50)
    print("All done! Ready to train.")
    print(f"Train: {EXTRACT_PATH / 'train'}")
    print(f"Test:  {EXTRACT_PATH / 'test'}")
    print("=" * 50)

if __name__ == "__main__":
    main()