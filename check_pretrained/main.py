import sys
sys.path.append("../")
import torch
from pytorch_i3d.pytorch_i3d import InceptionI3d
# from pytorch_i3d.i3d_old import InceptionI3d
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from glob import glob
import umap
import pandas as pd
import random
import numpy as np

def open_images_with_background(l_path, normalize=True):   
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_without_normalize = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    images = torch.stack([transform(Image.open(path).convert("RGB")) for path in l_path]).unsqueeze(0)
    images_without_normalize = torch.stack([transform_without_normalize(Image.open(path).convert("RGB")) for path in l_path]).unsqueeze(0)
    img_background_without_normalize = torch.median(images_without_normalize, dim=1).values.squeeze(0)
    img_background = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img_background_without_normalize)
    # B x T x C x H x W -> B x C x T x H x W
    if normalize:
        return images.permute(0, 2, 1, 3, 4), img_background
    else:
        return images_without_normalize.permute(0, 2, 1, 3, 4), img_background_without_normalize


def open_flows(l_path):    
    flows = torch.stack([
        torch.from_numpy(np.load(path)) for path in l_path + l_path[-1:]
    ]).permute(3, 0, 1, 2)
    flows = F.interpolate(flows, size=(224, 224), mode="bilinear", align_corners=False)
    # T x C x H x W -> B x C x T x H x W
    flows = flows.unsqueeze(0)
    return flows


def path2rgbfeature(idx, normalize=True):    
    model = InceptionI3d(in_channels=3)
    
    model.load_state_dict(torch.load(PATH_TRAINED_MODEL_RGB))
    l_path = sorted(glob(f"/datasets/UCSD_Anomaly_Dataset_v1p2/UCSDped1/Test/Test{idx:03}/*.tif")) 
    images, _ = open_images_with_background(l_path, normalize)
    videos = torch.vstack([images[:, :, seq * 16:seq * 16 + 16] for seq in range(BATCH)])
    model = model.cuda()
    videos = videos.cuda()
    with torch.no_grad():
        model.eval()
        X = model(videos).squeeze(0).squeeze(2).detach().cpu()
    return X



def path2flowfeature(idx):    
    model = InceptionI3d(in_channels=2)
    model.load_state_dict(torch.load(PATH_TRAINED_MODEL_FLOW))    
    
    l_path = sorted(glob(f"../datasets/UCSD_Anomaly_Dataset_v1p2/UCSDped1/Test/Test{idx:03}/flows/*.npy"))
    flows = open_flows(l_path)
    videos = torch.vstack([flows[:, :, seq * 16:seq * 16 + 16] for seq in range(BATCH)])    
    model = model.cuda()
    videos = videos.cuda()
    with torch.no_grad():
        model.eval()
        X = model(videos).squeeze(0).squeeze(2).detach().cpu()
    return X

PATH_TRAINED_MODEL_RGB = "../pytorch_i3d/models/rgb_imagenet.pt"
# PATH_TRAINED_MODEL_RGB = "../pytorch_i3d/models/rgb_i3d_pretrained.pt"
PATH_TRAINED_MODEL_FLOW = "../pytorch_i3d/models/flow_imagenet.pt"
BATCH = 10
NUM = 36

if __name__ == "__main__":
    # rgb
    X = torch.vstack([
        path2rgbfeature(idx)
        for idx in range(1, NUM + 1)
        if idx not in [17]
    ]).numpy()

    # flow
    # X = torch.vstack([
    #     path2flowfeature(idx)
    #     for idx in range(1, NUM + 1)
    #     if idx not in [17]
    # ]).numpy()
    print(X.shape)
    l_path = [
        f"Test{idx:03}/{num:03}"
        for idx in range(1, NUM + 1)
        for num in range(10)
        if idx not in [17]
    ]
    # random color rgb code
    colors = ["#%06x" % random.randint(0, 0xFFFFFF) for num in range(NUM + 1)]
    l_color = [
        colors[idx]
        for idx in range(1, NUM + 1)
        for num in range(10)
        if idx not in [17]
    ]
    print(X.shape)
    print(len(l_path))
    embedding = umap.UMAP().fit_transform(X)    
    # plot embedding
    df = pd.DataFrame(embedding, columns=["x", "y"], index=l_path)
    ax = df.plot.scatter(x="x", y="y", figsize=(10, 10), c=l_color)
    # for k, v in df.iterrows():
    #     ax.annotate(k, v, fontsize=8)
    ax.figure.savefig('scatter_rgb.png')