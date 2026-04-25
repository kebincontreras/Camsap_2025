import os
import requests
import zipfile
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image

def download_div2k_direct(save_folder, num_images=100, prefix="DIV2K"):
    os.makedirs(save_folder, exist_ok=True)
    zip_url = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"
    zip_path = os.path.join(save_folder, "DIV2K_valid_HR.zip")
    extract_to = os.path.join(save_folder, "temp_extracted")
    
    # Check if download is needed
    if not os.path.exists(zip_path):
        print("Descargando DIV2K_valid_HR.zip (Set original de validacion - 700MB)...")
        response = requests.get(zip_url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024 # 1 Megabyte
        
        with open(zip_path, 'wb') as file, tqdm(
                desc=zip_path,
                total=total_size_in_bytes,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
            for data in response.iter_content(block_size):
                file.write(data)
                bar.update(len(data))
    
    print("Extrayendo archivos de DIV2K...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    # Mover y renombrar los archivos extraidos
    extracted_folder = os.path.join(extract_to, "DIV2K_valid_HR")
    images = sorted([f for f in os.listdir(extracted_folder) if f.endswith('.png')])
    
    count = 0
    print(f"Renombrando y moviendo {num_images} imagenes de DIV2K...")
    for img_name in images[:num_images]:
        src = os.path.join(extracted_folder, img_name)
        # Numerar desde 001
        dst = os.path.join(save_folder, f"{prefix}_{count+1:03d}.png")
        os.rename(src, dst)
        count += 1
        
    # Limpiar temp
    for root, dirs, files in os.walk(extract_to, topdown=False):
        for name in files:
            try:
                os.remove(os.path.join(root, name))
            except:
                pass
        for name in dirs:
            try:
                os.rmdir(os.path.join(root, name))
            except:
                pass
    try:
        os.rmdir(extract_to)
    except:
        pass
    
    print(f"[OK] {count} imagenes de DIV2K listas en {save_folder}\n")

def download_imagenet_hf(save_folder, num_images=100, prefix="ImageNet"):
    os.makedirs(save_folder, exist_ok=True)
    dataset_path = "evanarlian/imagenet_1k_resized_256"
    split = "val"
    print(f"Descargando imagenes de ImageNet desde {dataset_path} ({split})...")
    
    try:
        dataset = load_dataset(dataset_path, split=split, streaming=True)
        dataset = dataset.shuffle(seed=42, buffer_size=1000)
        
        count = 0
        tqdm_bar = tqdm(total=num_images)
        for item in dataset:
            if count >= num_images:
                break
                
            try:
                img = item['image']
                if not isinstance(img, Image.Image):
                    continue
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                    
                filename = f"{prefix}_{count+1:03d}.png"
                filepath = os.path.join(save_folder, filename)
                img.save(filepath, format="PNG")
                
                count += 1
                tqdm_bar.update(1)
            except Exception:
                pass
                
        tqdm_bar.close()
        print(f"[OK] {count} imagenes de ImageNet listas en {save_folder}\n")
    except Exception as e:
        print(f"[Error] al procesar ImageNet: {str(e)}")

def download_kitti_training(save_folder, num_images=200, start_idx=10):
    os.makedirs(save_folder, exist_ok=True)
    dataset_path = "nateraw/kitti"
    split = "train"
    print(f"Descargando {num_images} imagenes adiciones de KITTI desde {dataset_path}...")
    
    try:
        # Usamos streaming para evitar descargar el inmenso dataset
        dataset = load_dataset(dataset_path, split=split, streaming=True)
        # Saltamos las primeras para evitar que sean muy parecidas a las previas
        dataset = dataset.skip(100) 
        dataset = dataset.shuffle(seed=42, buffer_size=500)
        
        count = 0
        tqdm_bar = tqdm(total=num_images)
        for item in dataset:
            if count >= num_images:
                break
                
            try:
                img = item['image']
                if not isinstance(img, Image.Image):
                    continue
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                    
                # Numeramos a partir del indice provisto
                current_id = start_idx + count
                filename = f"{current_id:06d}.png"
                filepath = os.path.join(save_folder, filename)
                img.save(filepath, format="PNG")
                
                count += 1
                tqdm_bar.update(1)
            except Exception:
                pass
                
        tqdm_bar.close()
        print(f"[OK] {count} nuevas imagenes de KITTI listas en {save_folder}\n")
    except Exception as e:
        print(f"[Error] al procesar KITTI: {str(e)}")

if __name__ == "__main__":
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "Resources"))
    print("Iniciando generacion de Datasets para Entrenamiento y Evaluacion...\n")

    # 1. DIV2K
    div2k_dir = os.path.join(BASE_DIR, "DIV2K")
    download_div2k_direct(save_folder=div2k_dir, num_images=100, prefix="DIV2K")
    
    # 2. ImageNet
    imagenet_dir = os.path.join(BASE_DIR, "ImageNet")
    download_imagenet_hf(save_folder=imagenet_dir, num_images=100, prefix="ImageNet")

    # 3. KITTI
    kitti_dir = os.path.join(BASE_DIR, "Images_kity")
    os.makedirs(kitti_dir, exist_ok=True)
    existentes = [f for f in os.listdir(kitti_dir) if f.endswith(".png")]
    start_id = 10
    if existentes:
        ids = [int(f.split('.')[0]) for f in existentes if f.split('.')[0].isdigit()]
        if ids:
            start_id = max(ids) + 1
            
    download_kitti_training(save_folder=kitti_dir, num_images=200, start_idx=start_id)
    
    print("El proceso automatizado de descarga de multi-datasets ha terminado con exito!")
