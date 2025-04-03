import torch
import torch_directml
import numpy as np
import argparse
import os
import torch.backends.cudnn as cudnn
import time
from architecture import *
from utils import AverageMeter, save_matv73, Loss_MRAE, Loss_RMSE, Loss_PSNR, my_summary
from hsi_dataset import TrainDataset, ValidDataset
from torch.utils.data import DataLoader


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

parser = argparse.ArgumentParser(description="Spectral Recovery Toolbox")
parser.add_argument('--data_root', type=str, default='../dataset/')
parser.add_argument('--method', type=str, default='mst_plus_plus') # change for different types
parser.add_argument('--pretrained_model_path', type=str, default='./model_zoo/mst_plus_plus.pth')
parser.add_argument('--outf', type=str, default='./exp/mst_plus_plus/')
opt = parser.parse_args()

if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

# Set device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# DirectML
# device = torch_directml.device()
# print(device)

# Load dataset
val_data = ValidDataset(data_root=opt.data_root, bgr2rgb=True)
val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

# Loss functions
criterion_mrae = Loss_MRAE().to(device)
criterion_rmse = Loss_RMSE().to(device)
criterion_psnr = Loss_PSNR().to(device)

# Validate
with open(f'{opt.data_root}/split_txt/valid_list.txt', 'r') as fin:
    hyper_list = [line.replace('\n', '.mat') for line in fin]
hyper_list.sort()
var_name = 'cube'

def validate(val_loader, model):
    model.eval()
    losses_mrae = AverageMeter()
    losses_rmse = AverageMeter()
    losses_psnr = AverageMeter()
    for i, (input, target) in enumerate(val_loader):
        input = input.to(device)
        target = target.to(device)
        with torch.no_grad():
            output = model(input)
            loss_mrae = criterion_mrae(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])
            loss_rmse = criterion_rmse(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])
            loss_psnr = criterion_psnr(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])
        # Record loss
        losses_mrae.update(loss_mrae.item())
        losses_rmse.update(loss_rmse.item())
        losses_psnr.update(loss_psnr.item())
        
        result = output.cpu().numpy() * 1.0
        result = np.transpose(np.squeeze(result), [1, 2, 0])
        result = np.minimum(result, 1.0)
        result = np.maximum(result, 0)
        mat_name = hyper_list[i]
        mat_dir = os.path.join(opt.outf, mat_name)
        save_matv73(mat_dir, var_name, result)
    return losses_mrae.avg, losses_rmse.avg, losses_psnr.avg

if __name__ == '__main__':
    start_time = time.time()
    cudnn.benchmark = False  # No need for GPU benchmarking
    pretrained_model_path = opt.pretrained_model_path
    method = opt.method
    model = model_generator(method, pretrained_model_path).to(device)
    mrae, rmse, psnr = validate(val_loader, model)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'method:{method}, mrae:{mrae}, rmse:{rmse}, psnr:{psnr}')
    print(f"Total execution time: {elapsed_time:.2f} seconds")