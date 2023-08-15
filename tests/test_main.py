import torch
from main import MyModel

def test_fold_and_flatten():  
    # N x C x T x H x W => N x T x 4 x 4 x (C x H/4 x W/4)
    N, C, T, H, W = 2, 3, 5, 8, 8
    sh = H // 4
    sw = W // 4
    x = torch.randint(0, 1000, (N, C, T, H, W))
    y = MyModel._fold_and_flatten(x)
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
    y = MyModel._fold_and_flatten(x)
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
    y = MyModel._fold_and_flatten(x)
    assert y.shape == (2, 5, 4, 4, 3 * 3 * 3)