import os
import argparse
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from patch_extraction_utils import create_embeddings
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser(description='Configurations for feature extraction')
parser.add_argument('--patches_path', type=str)
parser.add_argument('--library_path', type=str)
parser.add_argument('--model_name', type=str)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--dataset', type=str, default='adenocarcinoma', help='dataset name (default: adenocarcinoma)')

if __name__ == '__main__':
    # Windows多进程修复
    torch.multiprocessing.freeze_support()
    
    args = parser.parse_args()
    
    patches_path = args.patches_path
    library_path = args.library_path
    model_name = args.model_name
    os.makedirs(library_path, exist_ok=True)
    
    create_embeddings(patch_datasets=patches_path, embeddings_dir=library_path,
                      enc_name=model_name, dataset=args.dataset, batch_size=args.batch_size)