import sys
sys.path.append('../../')  # noqa
from ultralytics.nn.modules import Conv
import torch.nn as nn
import torch
from ultralytics.utils.torch_utils import make_divisible
from timm.models.layers import DropPath


def pad(k, p=None, d=1):

    if d > 1:               
        k = d * (k - 1) + 1  
    if p is None:          
        p = k // 2          
    return p


class DWConv(nn.Module):

    def __init__(self, dim=128):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


class SAP(nn.Module):

    def __init__(self, dim, r=8, drop=0.0):
        super().__init__()
        ch = make_divisible(int(dim // r), 8)           
        self.cv1 = nn.Conv2d(dim, ch, 1)                
        self.act = nn.SiLU()
        self.cv2 = nn.Conv2d(ch, dim, 1)                
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity() 

    def forward(self, x):
        return self.drop(self.cv2(self.act((self.cv1(x)))))

class WDS(nn.Module):

    def __init__(self, dim, ks=[3, 3, 3], ds=[1, 1, 1], e=0.5):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, 1)
        self.v = nn.Conv2d(dim, dim, 1)
        self.act = nn.SiLU()

        self.stem = nn.Conv2d(dim, dim, ks[0], 1, padding=pad(ks[0], d=ds[0]), dilation=ds[0])   

        dim2 = make_divisible(int(dim * e), 8)                  
        self.dim2 = dim2
        
        self.len = len(ks)                                      
        
        self.mrs = nn.ModuleList([nn.Conv2d(dim2, dim2, k, 1,
                                            padding=pad(k, d=d), dilation=d, groups=dim2) for k, d in zip(ks[1:], ds[1:])])
        

        self.spatial_mix = nn.Conv2d(2 * len(ks), len(ks), 7, padding=3)     

    def forward(self, x):

        identity = x    
        v = self.v(x)   
        x = self.stem(x)  

        x0, x1 = x.split([x.shape[1] - self.dim2, self.dim2], 1)    
        outs = [x1]

        outs.extend(mr(outs[-1]) for mr in self.mrs)  

        gbs = torch.cat([self._global(x) for x in outs], 1) 
        gbs_attn = self.spatial_mix(gbs).sigmoid()          

        
        ctx = 0 
        for i in range(self.len):
            ctx = ctx + outs[i] * gbs_attn[:, i:i+1]    
        x = v * self.act(torch.cat([x0, ctx], 1))  

        return identity + self.proj(x)  

    def _global(self, x):
        avg_attn = torch.mean(x, dim=1, keepdim=True)   
        max_attn, _ = torch.max(x, dim=1, keepdim=True) 
        return torch.cat([avg_attn, max_attn], dim=1)   


class LayerScale(nn.Module):

    def __init__(self, dim, init_value=1e-2):
        super().__init__()
        self.layer_scale = nn.Parameter(
            init_value * torch.ones((1, dim, 1, 1)), requires_grad=True)    

    def forward(self, x):
        return x * self.layer_scale     

class AFE(nn.Module):

    def __init__(self, dim0, dim, r=0.5, ks=[3, 3, 3], ds=[1, 2, 3], init_value=1e-1, drop_path=0.0):
        super().__init__()
        assert dim0 == dim                                  
        self.norm1 = nn.BatchNorm2d(dim)                    
        self.norm2 = nn.BatchNorm2d(dim)                    
        self.wds = WDS(dim, ks, ds, r)            
        self.sap = SAP(dim)                             
        self.ls1 = LayerScale(dim, init_value)              
        self.ls2 = LayerScale(dim, init_value)              
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()      

    def forward(self, x):
        x = x + self.drop_path(self.ls1(self.wds(self.norm1(x))))     
        x = x + self.drop_path(self.ls2(self.sap(self.norm2(x))))      
        return x


class PoolDown(nn.Module):
   
    def __init__(self, c1, c2, k, s):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)         
        self.cv = Conv(c1, c2, k=1, s=1, g=1)   

    def forward(self, x):
        x = self.cv(self.pool(x))
        return x       



