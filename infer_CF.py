import torch
import torch.nn as nn
import json
import os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from skimage.color import rgb2lab
import numpy as np

class ViTColorQualityRegressor(nn.Module):
    def __init__(self, local_path_or_config):
        super().__init__()
        from transformers import ViTModel
        self.vit = ViTModel.from_pretrained(local_path_or_config)
        hidden_size = self.vit.config.hidden_size
        
        self.chroma_head = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((12, 12))
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size * 2 + 32, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1),
            nn.Tanh()
        )

    def forward(self, x_rgb, x_lab):
        vit_outputs = self.vit(pixel_values=x_rgb)
        last_hidden_state = vit_outputs.last_hidden_state 
        
        cls_token = last_hidden_state[:, 0, :]
        patch_tokens = last_hidden_state[:, 1:, :]
        
        patch_std = torch.std(patch_tokens, dim=1) 
        
        chroma_info = x_lab[:, 1:, :, :]
        chroma_feat = self.chroma_head(chroma_info)
        chroma_feat = torch.flatten(chroma_feat, 1)
        chroma_feat = nn.functional.adaptive_avg_pool1d(chroma_feat.unsqueeze(1), 32).squeeze(1)

        combined = torch.cat([cls_token, patch_std, chroma_feat], dim=-1)
        return self.regressor(combined)

def preprocess_image(img_path, target_size=(384, 384)):
    img_pil = Image.open(img_path).convert('RGB').resize(target_size)
    
    transform_rgb = transforms.Compose([
        transforms.ToTensor(),
    ])
    rgb_tensor = transform_rgb(img_pil).unsqueeze(0)
    
    img_np = np.array(img_pil).astype(np.float32) / 255.0
    lab_np = rgb2lab(img_np)
    lab_np[:,:,0] /= 100.0   
    lab_np[:,:,1:] /= 128.0  
    lab_tensor = torch.from_numpy(lab_np).permute(2, 0, 1).float().unsqueeze(0)
    
    return rgb_tensor, lab_tensor

def run_inference(model_path, img_dir, output_json, pretrain_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = ViTColorQualityRegressor(pretrain_path).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    results = {}
    img_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(img_extensions)]
    
    with torch.no_grad():
        for filename in tqdm(img_files):
            img_path = os.path.join(img_dir, filename)
            
            rgb_t, lab_t = preprocess_image(img_path)
            rgb_t, lab_t = rgb_t.to(device), lab_t.to(device)
                
            prediction = model(rgb_t, lab_t)
            score = prediction.item()
            results[filename[:-4]] = score

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    

if __name__ == "__main__":
    CONFIG = {
        "model_weight": "CF.pth", 
        "pretrain_path": "./pretrain/",             
        "input_folder": "./ICQA_2K_test/",          
        "output_file": "ICQA_2K_test_CF_result.json"  
    }
    
    run_inference(
        CONFIG["model_weight"], 
        CONFIG["input_folder"], 
        CONFIG["output_file"],
        CONFIG["pretrain_path"]
    )