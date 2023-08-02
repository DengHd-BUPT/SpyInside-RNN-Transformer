import R2_qarc as R2
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from tqdm import tqdm
import time
import os, sys, random
from Abstraction_VQPN import QARCModel, QARCModelDP
from utils import get_default_device
import argparse
import numpy as np
import cv2
import argparse

parser = argparse.ArgumentParser(description="QARC verification")
parser.add_argument("--eps", type=float, default=0.007, help="perturbation epsilon.")
parser.add_argument(
    "--bound_method",
    type=str,
    default="lp",
    choices=["lp", "opt"],
    help="bounding method, either lp or opt.",
)
args = parser.parse_args()
dev = get_default_device()

kernel = int(64)
dense_size = int(64)
INPUT_W = 64
INPUT_H = 36
INPUT_D = 3
# long seq
INPUT_SEQ = 25
OUTPUT_DIM = 5

model_name = "Model/vqpn.pt"
devtxt = "cuda" if dev == torch.device("cuda") else "cpu"
stt_dict = torch.load(model_name, map_location=devtxt)
model = QARCModel(kernel, dense_size).to(dev)
model.load_state_dict(torch.load(model_name, map_location=devtxt))
eps = args.eps
bound_method = args.bound_method
r2model = QARCModelDP(kernel, dense_size).to(dev)
r2model.load_state_dict(torch.load(model_name, map_location=devtxt))
r2model.set_bound_method(bound_method)

def diy_get_image():
    _videoindex2 = random.randint(1, 62)
    print("videoindex:" + str(_videoindex2))
    _dirs = os.listdir('img/')
    _video = _dirs[np.random.randint(len(_dirs))]
    print("_video:" + str(_video))
    _index2 = _videoindex2 * 5
    x_buff2 = np.zeros([INPUT_SEQ, INPUT_H, INPUT_W, INPUT_D])

    for pq in range(1, 6):
        x_buff2 = np.roll(x_buff2, -1, axis=1)
        filename = 'img/' + _video + '/' + _video + '_' + str(_index2 + pq) + '.png'
        print("get_image:" + str(filename))
        img = cv2.imread(filename)
        if img is None:
            # print filename
            filename = 'img/' + _video + \
                       '/' + str(_index2 + pq - 1) + '.png'
            img = cv2.imread(filename)
        x_buff2[-1, :, :, :] = img
    return x_buff2

def predict_vmaf_for_diyimage():
    _images2 = diy_get_image()
    _images2 = np.reshape(
        _images2, [1, INPUT_SEQ, INPUT_D, INPUT_H, INPUT_W])
    _images2 = torch.tensor(_images2)
    _test_y2 = model(_images2.to(dev))
    f = open('/home/dhd/DRLVerification/prover/log.txt', 'a')
    print("Actual predict", file=f)
    print("VMAF: " + str(_test_y2[0]), file=f)
    return _test_y2[0]

def predict_vmaf_for_diyimage_dp():
    _images2 = diy_get_image()
    _images2 = np.reshape(
        _images2, [1, INPUT_SEQ, INPUT_D, INPUT_H, INPUT_W])
    _images2 = torch.tensor(_images2)
    f = open('/home/dhd/DRLVerification/prover/log.txt', 'a')
    print("Formal predict", file=f)
    epss = 0.006
    print("eps: " + str(epss), file=f)
    _test_y2 = r2model.certify(_images2.to(dev), epss)
    print("vmaf_dp.lb: " + str(_test_y2.lb), file=f)
    print("vmaf_dp.ub: " + str(_test_y2.ub), file=f)
    print("\n", file=f)
    return _test_y2

# Verification of output
vmaf_pred_for_diyimage = predict_vmaf_for_diyimage()
print("The video quality score of video bitrate {300, 500, 800, 1100, 1400} Kbps:" + str(vmaf_pred_for_diyimage))
vmaf_dp = predict_vmaf_for_diyimage_dp()
print("The video quality score of video bitrate {300, 500, 800, 1100, 1400} Kbps:" + str(vmaf_dp.lb))

