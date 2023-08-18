import sys
sys.path.append("../")
import torch
from main import open_flows
import numpy as np
import cv2
import torch.nn.functional as F

def xy(radian: torch.Tensor):
    return torch.stack([torch.cos(radian), torch.sin(radian)], dim=-1)

def flow2rgb(flow, frame):
    hsv = np.zeros((frame.shape[0], frame.shape[1], 3))
    hsv[...,1] = 255    
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

if __name__ == "__main__":
    # 0.0 to 2pi
    rad = torch.tensor([0.0, 1/6, 2/6, 3/6, 4/6, 5/6, 1.0, 7/6, 8/6, 9/6, 10/6, 11/6]) * torch.pi
    # B x C
    xy = xy(rad)
    l_path = [
        "../datasets/UCSD_Anomaly_Dataset_v1p2/UCSDped1/Test/Test024/flows/130.npy",
        "../datasets/UCSD_Anomaly_Dataset_v1p2/UCSDped1/Test/Test024/flows/131.npy",
        "../datasets/UCSD_Anomaly_Dataset_v1p2/UCSDped1/Test/Test024/flows/132.npy",
        ]
    # B x T x H x W x C
    flow = open_flows(l_path).permute(0, 2, 3, 4, 1)
    cv2.imwrite("./flow.png", flow2rgb(flow[0, 0].numpy(), flow[0, 0].numpy()))
    print(flow.size())
    flow = torch.stack([F.avg_pool2d(flow[idx].permute(0, 3, 1, 2), 5, 1, 2).permute(0, 2, 3, 1) for idx in range(flow.size(0))])
    print(flow.size())
    cv2.imwrite("./flow_stride.png", flow2rgb(flow[0, 0].numpy(), flow[0, 0].numpy()))
        
    # 12方向ごとに分ける
    masks = torch.clamp(torch.einsum("bthwc,zc->bthwz", flow, xy), min=0.0)
    for i in range(masks.size(-1)):
        cv2.imwrite("./mask{}.png".format(i), 255 * masks[0, 0, :, :, i].numpy())
        
        
    # switch: N x 1 x 12
    switch = torch.ones(1, masks.size(1), 12)    
    
    # 12方向ごとに分けたものを足し合わせる
    mask = torch.einsum("bthwz,btz->bthw", masks, switch)
    cv2.imwrite("./mask.png", 255 * mask[0, 0].numpy())