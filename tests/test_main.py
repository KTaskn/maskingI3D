import torch
from main import MyModel, Loss


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

def test__fold_mask():
    H, W, T = 224, 224, 16
    model = MyModel()
    mask = torch.randn(H, W, T, 10, 1)
    assert mask.shape == (H, W, T, 10, 1)
    mask_transformed = model._fold_mask(mask)
    assert mask_transformed.shape == (10, 16, 224, 224)
    assert mask_transformed.unsqueeze(0).permute(3, 4, 2, 1, 0).shape == (H, W, T, 10, 1)
    assert torch.allclose(mask, mask_transformed.unsqueeze(0).permute(3, 4, 2, 1, 0))

def test_fold_and_flatten():  
    model = MyModel()
    # N x C x T x H x W => N x T x 4 x 4 x (C x H/4 x W/4)
    N, C, T, H, W = 2, 3, 5, 8, 8
    sh = H // 4
    sw = W // 4
    x = torch.randint(0, 1000, (N, C, T, H, W))
    y = model._fold_and_flatten(x)
    assert y.shape == (N, T, 4, 4, sh * sw * C)
    # same entities, different shapes
    # n = 0, t = 0
    for n in range(N):
        for t in range(T):    
            for h in range(4):                
                for w in range(4):
                    sq0 = x[n, :, t, sh * h:(sh * h) + sh, sw * w:(sw * w) + sw]
                    sq1 = y[n, t, h, w, :]
                    assert torch.allclose(sq0.flatten(), sq1)
    
    N, C, T, H, W = 2, 3, 5, 9, 9
    sh = H // 4
    sw = W // 4
    x = torch.randint(0, 1000, (N, C, T, H, W))
    y = model._fold_and_flatten(x)
    assert y.shape == (N, T, 4, 4, sh * sw * C)
    for n in range(N):
        for t in range(T):       
            for h in range(4):             
                for w in range(4):
                    sw = W // 4
                    sh = H // 4
                    sq0 = x[n, :, t, sh * h:(sh * h) + sh, sw * w:(sw * w) + sw]
                    sq1 = y[n, t, h, w, :]
                    assert torch.allclose(sq0.flatten(), sq1)
    
    
    N, C, T, H, W = 2, 3, 5, 14, 14
    sw = W // 4
    sh = H // 4
    x = torch.randint(0, 1000, (N, C, T, H, W))
    y = model._fold_and_flatten(x)
    assert y.shape == (2, 5, 4, 4, 3 * 3 * 3)