import torch
import torch.nn as nn
import json
import os
import glob
import numpy as np
from PIL import Image
from torchvision import models, transforms

class ColorAttentionNet(nn.Module):
    def __init__(self, obj_feat_dim=773, hidden_dim=256):
        super(ColorAttentionNet, self).__init__()
        resnet = models.resnet18(pretrained=False)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.class_embedding = nn.Embedding(num_embeddings=81, embedding_dim=32)
        self.obj_encoder = nn.Sequential(
            nn.Linear(4 + 768 + 32, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 1),
        )
        self.regressor = nn.Sequential(
            nn.Linear(512 + hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )

    def forward(self, img, obj_feats, mask):
        v_img = self.backbone(img).view(img.size(0), -1)
        class_ids = obj_feats[:, :, 0].long()
        coords = obj_feats[:, :, 1:5]
        color_confs = obj_feats[:, :, 5:]
        
        class_emb = self.class_embedding(class_ids)
        combined_obj_features = torch.cat([class_emb, coords, color_confs], dim=-1)
        v_obj = self.obj_encoder(combined_obj_features)
        
        attn_weights = self.attention(v_obj).squeeze(-1)
        attn_weights = attn_weights.masked_fill(mask, -1e9)
        attn_weights = torch.softmax(attn_weights, dim=1).unsqueeze(-1)
        
        v_obj_agg = torch.sum(v_obj * attn_weights, dim=1)
        combined = torch.cat([v_img, v_obj_agg], dim=1)
        return self.regressor(combined)

def process_single_image(img_path, txt_path, color_conf_data, transform, max_objs=20):
    image = Image.open(img_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)

    feat_dim = 1 + 4 + 768
    obj_features = []
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            for line in lines[:max_objs]:
                parts = line.strip().split()
                cls_id_str = parts[0]
                coords = [float(x) for x in parts[1:]]
                
                obj_color_data = color_conf_data.get(cls_id_str, {})
                color_conf_vector = []
                for channel in ['r', 'g', 'b']:
                    channel_data = obj_color_data.get(channel, {})
                    channel_vec = [channel_data.get(str(i), 0.0) for i in range(256)]
                    color_conf_vector.extend(channel_vec)
                
                if not color_conf_vector: color_conf_vector = [0.0] * 768
                feat = [float(cls_id_str)] + coords + color_conf_vector
                obj_features.append(feat)

    num_objs = len(obj_features)
    padded_objs = np.zeros((1, max_objs, feat_dim), dtype=np.float32)
    mask = np.ones((1, max_objs), dtype=bool)
    
    if num_objs > 0:
        padded_objs[0, :num_objs] = np.array(obj_features, dtype=np.float32)
        mask[0, :num_objs] = False
        
    return image_tensor, torch.from_numpy(padded_objs), torch.from_numpy(mask)

def run_inference(img_folder, txt_folder, model_path, color_json_path, output_json):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open(color_json_path, 'r') as f:
        color_conf_data = json.load(f)

    model = ColorAttentionNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    results = {}
    img_list = glob.glob(os.path.join(img_folder, "*.jpg")) + glob.glob(os.path.join(img_folder, "*.png"))

    print(f"Found {len(img_list)} images. Starting inference...")

    with torch.no_grad():
        for img_path in img_list:
            img_name = os.path.basename(img_path)
            base_name = os.path.splitext(img_name)[0]
            txt_path = os.path.join(txt_folder, base_name + ".txt")

            img_tensor, obj_tensor, mask_tensor = process_single_image(
                img_path, txt_path, color_conf_data, transform
            )
            
            img_tensor = img_tensor.to(device)
            obj_tensor = obj_tensor.to(device)
            mask_tensor = mask_tensor.to(device)
            
            output = model(img_tensor, obj_tensor, mask_tensor)
            score = output.item()
            
            results[base_name] = score
            print(f"Processed: {img_name} | Score: {score:.4f}")

    with open(output_json, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    run_inference(
        img_folder='ICQA_2K_train',        
        txt_folder='labels',        
        model_path='SCC.pth',         
        color_json_path='color_.json',   
        output_json='ICQA_2K_train_SCC_result.json' 
    )