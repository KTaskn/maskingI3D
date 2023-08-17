import torch
from main import MyModel, Loss, open_flows

def test_dot():
    def xy(radian: torch.Tensor):
        return torch.stack([torch.cos(radian), torch.sin(radian)], dim=-1)
    # 0.0 to 2pi
    rad = torch.tensor([0.0, 1/6, 2/6, 3/6, 4/6, 5/6, 1.0, 7/6, 8/6, 9/6, 10/6, 11/6]) * torch.pi
    # B x C
    xy = xy(rad)
    l_path = [
        "./datasets/UCSD_Anomaly_Dataset_v1p2/UCSDped1/Test/Test024/flows/001.npy",
        "./datasets/UCSD_Anomaly_Dataset_v1p2/UCSDped1/Test/Test024/flows/002.npy",
        ]
    # B x T x H x W x C
    flow = open_flows(l_path).permute(0, 2, 3, 4, 1)
    print(flow.size())
    print(torch.einsum("bthwc,zc->bthwz", flow, xy).size())

def test_loss():
    criterion = Loss(alpha=0.0)
    N = 5
    mask = torch.zeros(N, 16, 224, 224)
    x1 = torch.randn(N, 100)
    x2 = torch.randn(N, 100)
    with torch.no_grad():
        loss, var = criterion(x1, x1, mask)
    assert loss.item() == 1.0
    
    with torch.no_grad():
        loss, var = criterion(x1, -x1, mask)
    assert loss.item() == 1.0
    
    with torch.no_grad():
        loss, var = criterion(x1, x2, mask)
    assert loss.item() >= 0.0
