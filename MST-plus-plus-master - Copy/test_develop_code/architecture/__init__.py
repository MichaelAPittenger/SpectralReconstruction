import torch
from .edsr import EDSR
from .HDNet import HDNet
from .hinet import HINet
from .hrnet import SGN
from .HSCNN_Plus import HSCNN_Plus
from .MIRNet import MIRNet
from .MPRNet import MPRNet
from .MST import MST
from .MST_Plus_Plus import MST_Plus_Plus
from .Restormer import Restormer
from .AWAN import AWAN

def model_generator(method, pretrained_model_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if method == 'mirnet':
        model = MIRNet(n_RRG=3, n_MSRB=1, height=3, width=1).to(device)
    elif method == 'mst_plus_plus':
        model = MST_Plus_Plus().to(device)
    elif method == 'mst':
        model = MST(dim=31, stage=2, num_blocks=[4, 7, 5]).to(device)
    elif method == 'hinet':
        model = HINet(depth=4).to(device)
    elif method == 'mprnet':
        model = MPRNet(num_cab=4).to(device)
    elif method == 'restormer':
        model = Restormer().to(device)
    elif method == 'edsr':
        model = EDSR().to(device)
    elif method == 'hdnet':
        model = HDNet().to(device)
    elif method == 'hrnet':
        model = SGN().to(device)
    elif method == 'hscnn_plus':
        model = HSCNN_Plus().to(device)
    elif method == 'awan':
        model = AWAN().to(device)
    else:
        print(f'Method {method} is not defined !!!!')
    if pretrained_model_path is not None:
        print(f'Loading model from {pretrained_model_path}')
        checkpoint = torch.load(pretrained_model_path, map_location=torch.device('cpu'))  # Force CPU loading
        model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()},
                              strict=True)

    return model
