from pytorch_i3d.pytorch_i3d import InceptionI3d
import torch.nn as nn
from torchvision.models.feature_extraction import create_feature_extractor
from PIL import Image
from torchvision import transforms
from glob import glob
import torch
import torch.nn.functional as F


class FCL(nn.Module):
    def __init__(self, rgb_input_size, flow_input_size, output_size):
        super().__init__()
        # FC Layer
        self.activation = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        self.layer1 = nn.Linear(rgb_input_size + flow_input_size, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, output_size)

        self.dropout1 = nn.Dropout(0.6)
        self.dropout2 = nn.Dropout(0.6)

    def forward(self, rgb_feature, flow_feature):
        x = torch.cat([rgb_feature, flow_feature], dim=1)
        x = self.dropout1(self.activation(self.layer1(x)))
        x = self.dropout2(self.activation(self.layer2(x)))
        return self.sigmoid(self.layer3(x))
    

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        ENDPOINT = "MaxPool3d_3a_3x3"        
        self.rgb_backbone = InceptionI3d(final_endpoint=ENDPOINT)
        self.rgb_backbone.build()
        
        self.flow_backbone = InceptionI3d(final_endpoint=ENDPOINT)
        self.flow_backbone.build()
        
        self.feature_backbone = InceptionI3d()
        
        # 22 x 22 x T
        self.fcl = FCL(484, 484, 484 * 16)
        
    def forward_twostream(self, rgbs, flows):
        def _pooling_and_flatten(x):
            return x.max(dim=1).values.max(dim=1).values.flatten(start_dim=1)
        x_rgb = _pooling_and_flatten(self.rgb_backbone.extract_features(rgbs))
        x_flow = _pooling_and_flatten(self.rgb_backbone.extract_features(flows))
        return x_rgb, x_flow
    
    def forward(self, rgbs, flows=None, img_background=None):
        x_rgb, x_flow = self.forward_twostream(rgbs, flows)
        # N -> W x H 484 -> 22 x 22
        mask = self.fcl(x_rgb, x_flow)
        mask = mask.view(-1, 16, 22, 22)
        # resize mask
        mask = F.interpolate(mask, size=(224, 224), mode="bilinear", align_corners=False).squeeze(1)
        # rgbs = N x C x T x H x W
        # mask = N x T x H x W
        # masking rgb
        x_rgb = rgbs * mask.unsqueeze(1)
        feature = self.feature_backbone(x_rgb).squeeze(2)
        return feature

if __name__ == "__main__":
    model = MyModel()
    l_path = sorted(glob("/datasets/UCSD_Anomaly_Dataset_v1p2/UCSDped1/Test/Test001/*.tif"))
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    images = torch.stack([transform(Image.open(path).convert("RGB")) for path in l_path]).unsqueeze(0)
    # B x T x C x H x W -> B x C x T x H x W
    images = images.permute(0, 2, 1, 3, 4)
    batch = torch.vstack([images[:, :, seq * 16:seq * 16 + 16] for seq in range(10)])
    print(batch.size())
    print(model(batch, batch).size())