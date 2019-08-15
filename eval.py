from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg
from layers.functions.prior_box import PriorBox
from utils.nms_wrapper import nms
#from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.faceboxes import FaceBoxes
from utils.box_utils import decode
from utils.timer import Timer

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True

def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

#weightfile = '/home/wynmew/workspace/FaceBoxes.PyTorch/weights/FaceBoxes_epoch_60.pth'
weightfile = '/home/wynmew/workspace/FaceBoxes.PyTorch/weights/Final_FaceBoxes.pth'

cpu=False
confidenceTh = 0.05
nmsTh = 0.3
keepTopK=750
top_k = 5000

os.environ['CUDA_VISIBLE_DEVICES']='1'
torch.set_grad_enabled(False)
# net and model
net = FaceBoxes(phase='test', size=None, num_classes=2)    # initialize detector
net = load_model(net, weightfile, cpu)
net.eval()
#print('Finished loading model!')
#print(net)
cudnn.benchmark = True
device = torch.device("cpu" if cpu else "cuda")
net = net.to(device)

#image_path = '/home/wynmew/data/downloads/danbooru2018/original/0795/3036795.jpg'
image_path = '/home/wynmew/data/downloads/danbooru2018/original/0795/1081795.jpg'
imgOrig = cv2.imread(image_path, cv2.IMREAD_COLOR)
img=np.float32(imgOrig)
im_height, im_width, _ = img.shape
scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
img -= (104, 117, 123)
img = img.transpose(2, 0, 1)
img = torch.from_numpy(img).unsqueeze(0)
img = img.to(device)
scale = scale.to(device)

loc, conf = net(img)  # forward pass
priorbox = PriorBox(cfg, image_size=(im_height, im_width))
priors = priorbox.forward()
priors = priors.to(device)
prior_data = priors.data
boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
boxes = boxes * scale
boxes = boxes.cpu().numpy()
scores = conf.data.cpu().numpy()[:, 1]

# ignore low scores
inds = np.where(scores > confidenceTh)[0]
boxes = boxes[inds]
scores = scores[inds]

# keep top-K before NMS
order = scores.argsort()[::-1][:top_k]
boxes = boxes[order]
scores = scores[order]

# do NMS
dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
#keep = py_cpu_nms(dets, args.nms_threshold)
keep = nms(dets, nmsTh,force_cpu=cpu)
dets = dets[keep, :]

# keep top-K faster NMS
dets = dets[:keepTopK, :]

for k in range(dets.shape[0]):
    xmin = dets[k, 0]
    ymin = dets[k, 1]
    xmax = dets[k, 2]
    ymax = dets[k, 3]
    ymin += 0.2 * (ymax - ymin + 1)
    score = dets[k, 4]
    print('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.format(image_path, score, xmin, ymin, xmax, ymax))
    cv2.rectangle(imgOrig, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 15)


cv2.imwrite('out.png', imgOrig)
