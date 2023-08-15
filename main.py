from pytorch_i3d.pytorch_i3d import InceptionI3d
import torch.nn as nn
from torchvision.models.feature_extraction import create_feature_extractor
from PIL import Image
from torchvision import transforms
from glob import glob
import torch
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from tqdm import tqdm
import sys
import argparse

MASK_W = 4
MASK_H = 4
BATCH = 10

class Loss(nn.Module):
    # calculate distance between two features
    def __init__(self, margin=0.2, alpha=10.0):
        super().__init__()
        self.margin = margin
        self.alpha = alpha
        
    def forward(self, feature0, feature1, mask):
        reg = torch.exp(-mask.var()) * self.alpha
        return torch.mean(torch.clamp(torch.norm(feature0 - feature1, dim=1) - self.margin, min=0.0)) + reg

class FCL(nn.Module):
    def __init__(self, rgb_input_size, flow_input_size):        
        super().__init__()
        print("rgb_input_size + flow_input_size:", rgb_input_size + flow_input_size)
        # FC Layer
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.layer1 = nn.Linear(rgb_input_size + flow_input_size, 1024)
        self.layer2 = nn.Linear(1024, 256)
        self.layer3 = nn.Linear(256, 1)

        self.dropout1 = nn.Dropout(0.6)
        self.dropout2 = nn.Dropout(0.6)

    def forward(self, feature_rgb, feature_flow):
        x = torch.cat([feature_rgb, feature_flow], dim=1)
        x = self.dropout1(self.activation(self.layer1(x)))
        x = self.dropout2(self.activation(self.layer2(x)))
        return self.sigmoid(self.layer3(x))
    
PATH_TRAINED_MODEL_RGB = "./pytorch_i3d/models/rgb_imagenet.pt"
PATH_TRAINED_MODEL_FLOW = "./pytorch_i3d/models/flow_imagenet.pt"
class MyModel(nn.Module):
    def __init__(self, endpoint="MaxPool3d_2a_3x3"):
        super(MyModel, self).__init__()  
        self.rgb_backbone = InceptionI3d(in_channels=3)
        self.rgb_backbone.load_state_dict(torch.load(PATH_TRAINED_MODEL_RGB))
        self.rgb_backbone._final_endpoint = endpoint
        self.rgb_backbone.build()
        
        self.flow_backbone = InceptionI3d(in_channels=2)
        self.flow_backbone.load_state_dict(torch.load(PATH_TRAINED_MODEL_FLOW))
        self.flow_backbone._final_endpoint = endpoint
        self.flow_backbone.build()
        
        self.feature_backbone = InceptionI3d(in_channels=3)
        self.feature_backbone.load_state_dict(torch.load(PATH_TRAINED_MODEL_RGB))
        
        def __freeze(model):
            for param in model.parameters():
                param.requires_grad = False
        __freeze(self.rgb_backbone)
        __freeze(self.flow_backbone)
        __freeze(self.feature_backbone)

        feat_rgb, feat_flow = self.forward_twostream(torch.randn(BATCH, 3, 16, 224, 224), torch.randn(BATCH, 2, 16, 224, 224))
        self.fcl = FCL(feat_rgb.size(4), feat_flow.size(4))
        
    def _fold_and_flatten(self, x):
        # TODO: これでいいのか？ -> 全体の数が減るぽい(はみ出る分を捨てる)
        # N x C x T x H x W => N x T x 4 x 4 x (C x H/4 x W/4)
        s = x.size(3) // 4
        return x.unfold(3, s, s).unfold(4, s, s).permute(0, 2, 3, 4, 1, 5, 6).flatten(4)
    
    def forward_twostream(self, rgbs, flows):
        # N x T x 4(h) x 4(w) x F
        x_rgb = self._fold_and_flatten(self.rgb_backbone.extract_features(rgbs))
        x_flow = self._fold_and_flatten(self.flow_backbone.extract_features(flows))
        return x_rgb, x_flow
    
    def _masking(self, rgbs, mask, img_background):
        # rgbs = N x C x T x H x W
        # mask = N x T x H x W
        # img_background = C x H x W
        masked_rgb = rgbs * mask.unsqueeze(1)
        masked_background = img_background.unsqueeze(0) * (1 - mask.unsqueeze(2).repeat(1, 1, 3, 1, 1))
        return masked_rgb + masked_background.permute(0, 2, 1, 3, 4)
    
    def _fold_mask(self, mask):
        # mask = H x W x T x 10 x 1
        mask = mask.squeeze(4).permute(3, 2, 0, 1)
        mask = mask.unsqueeze(1)
        mask = F.interpolate(mask, size=(16, 224, 224), mode="trilinear", align_corners=False)
        mask = mask.squeeze(1)
        # ret = N x T x H x W
        return mask
    
    def get_mask(self, rgbs, flows):        
        feat_rgb, feat_flow = self.forward_twostream(rgbs, flows)        
        # N x T x 4(h) x 4(w) x F
        # mask = H x W x T x 10 x 1
        mask = torch.stack([
            torch.stack([
                torch.stack([
                    self.fcl(feat_rgb[:, t, h, w], feat_flow[:, t, h, w]) for t in range(feat_rgb.size(1))
                ])            
                for w in range(feat_rgb.size(3))
            ])
            for h in range(feat_rgb.size(2))
        ])
        return self._fold_mask(mask)
    
    def forward(self, rgbs, flows, img_background):
        mask = self.get_mask(rgbs, flows)
        # rgbs = N x C x T x H x W
        # mask = N x T x H x W
        # masking rgb
        masked0, masked1 = self._masking(rgbs, mask, img_background), self._masking(rgbs, 1.0 - mask, img_background)
        return self.feature_backbone(masked0).squeeze(2), self.feature_backbone(masked1).squeeze(2), mask

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
    
EPOCH = 10000
EVALUTATION_INTERVAL = 100
def train(model, batch_rgbs, batch_flows, img_background, cuda=True):
    model = model.cuda() if cuda else model
    
    batch_rgbs = batch_rgbs.cuda() if cuda else batch_rgbs
    batch_flows = batch_flows.cuda() if cuda else batch_flows
    img_background = img_background.cuda() if cuda else img_background
    
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = Loss()

    loss_sum = 0
    total = 0
        
    with tqdm(total=EPOCH, unit="batch") as pbar:
        for num_step in range(EPOCH):
            if num_step % EVALUTATION_INTERVAL == 0:
                model.eval()
                with torch.no_grad():
                    masks = model.get_mask(batch_rgbs, batch_flows)
                    yield model, masks
            model.train()
            # idx = np.random.randint(0, batch_rgbs.size(0) - 2)
            # feature0, feature1, masks = model(batch_rgbs[idx:idx+2], batch_flows[idx:idx+2], img_background)
            feature0, feature1, masks = model(batch_rgbs, batch_flows, img_background)
                    
            loss = criterion(feature0, feature1, masks)
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * feature0.size(0) 
            total += feature0.size(0)
            running_loss = loss_sum / total

            pbar.set_postfix({
                "exp(-mask_var)": torch.exp(-masks.var()).item(),
                "running_loss": running_loss,
            })
            pbar.update(1)
            
    model.eval()
    with torch.no_grad():
        masks = model.get_mask(batch_rgbs, batch_flows)
        yield model, masks
    
ENDPOINTS = [
        'Conv3d_1a_7x7',
        'MaxPool3d_2a_3x3',
        'Conv3d_2b_1x1',
        'Conv3d_2c_3x3',
        'MaxPool3d_3a_3x3']
if __name__ == "__main__":    
    # endpoint from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", type=str, default="MaxPool3d_2a_3x3")
    args = parser.parse_args()
    endpoint = args.endpoint
    print("endpoint:", endpoint)
    
    l_path = sorted(glob("/datasets/UCSD_Anomaly_Dataset_v1p2/UCSDped1/Test/Test024/*.tif"))
    l_flow_path = sorted(glob("./datasets/UCSD_Anomaly_Dataset_v1p2/UCSDped1/Test/Test024/flows/*.npy"))
    
    images, img_background = open_images_with_background(l_path, True)
    images_without_normalize, img_background_without_normalize = open_images_with_background(l_path, False)
    flows = open_flows(l_flow_path)
    
    batch_rgbs = torch.vstack([images[:, :, seq * 16:seq * 16 + 16] for seq in range(BATCH)])
    batch_images_without_normalize = torch.vstack([images_without_normalize[:, :, seq * 16:seq * 16 + 16] for seq in range(BATCH)])
    batch_flows = torch.vstack([flows[:, :, seq * 16:seq * 16 + 16] for seq in range(BATCH)])
    print("size: ", batch_rgbs.size(), batch_flows.size())
    
    model = MyModel(endpoint=endpoint)
    for model, masks in train(model, batch_rgbs, batch_flows, img_background):
        model.eval()
        with torch.no_grad():
            model = model.cuda()
            masks = masks.cuda()
            batch_images_without_normalize = batch_images_without_normalize.cuda()
            img_background_without_normalize = img_background_without_normalize.cuda()
            masked0 = model._masking(
                batch_images_without_normalize,
                masks,
                img_background_without_normalize
            ).cpu().permute(0, 2, 1, 3, 4).contiguous()[-1:].view(-1, 3, 224, 224).numpy()
            masked1 = model._masking(
                batch_images_without_normalize,
                1 - masks,
                img_background_without_normalize
            ).cpu().permute(0, 2, 1, 3, 4).contiguous()[-1:].view(-1, 3, 224, 224).numpy()
            # save images
            masks = masks[-1:].cpu().contiguous().view(-1, 224, 224).numpy()
            for i, mask in enumerate(masks):
                Image.fromarray((mask * 255).astype(np.uint8)).save(f"./masked/{endpoint}/masked{i:02}-mask.png")
                
            for i, masked in enumerate(masked0):
                Image.fromarray((masked * 255).astype(np.uint8).transpose(1, 2, 0)).save(f"./masked/{endpoint}/masked{i:02}-0.png")
            # save images
            for i, masked in enumerate(masked1):
                Image.fromarray((masked * 255).astype(np.uint8).transpose(1, 2, 0)).save(f"./masked/{endpoint}/masked{i:02}-1.png")
        
        