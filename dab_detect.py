import os, sys
import torch
import numpy as np

from CodeDependency.dab.models import build_DABDETR, build_dab_deformable_detr
from CodeDependency.dab.util.slconfig import SLConfig
from CodeDependency.dab.datasets import build_dataset
from CodeDependency.dab.util.visualizer import COCOVisualizer
from CodeDependency.dab.util import box_ops
from PIL import Image
import CodeDependency.dab.datasets.transforms as T


model_config_path = "model_zoo/DAB_DETR/R50/config.json" # change the path of the model config file
model_checkpoint_path = "model_zoo/DAB_DETR/R50/checkpoint.pth" # change the path of the model checkpoint
# See our Model Zoo section in README.md for more details about our pretrained models.

args = SLConfig.fromfile(model_config_path)
model, criterion, postprocessors = build_DABDETR(args)
checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['model'])


# 这里需要修改种类
dataset_val = build_dataset(image_set='val', args=args)
cocojs = dataset_val.coco.dataset
id2name = {item['id']: item['name'] for item in cocojs['categories']}

# image, targets = dataset_val[0]
#
# # build gt_dict for vis
# box_label = [id2name[int(item)] for item in targets['labels']]
# gt_dict = {
#     'boxes': targets['boxes'],
#     'image_id': targets['image_id'],
#     'size': targets['size'],
#     'box_label': box_label,
# }
# vslzr = COCOVisualizer()
# vslzr.visualize(image, gt_dict, savedir=None)

# output = model(image[None])
# output = postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]))[0]
# thershold = 0.3 # set a thershold
#
# scores = output['scores']
# labels = output['labels']
# boxes = box_ops.box_xyxy_to_cxcywh(output['boxes'])
# select_mask = scores > thershold
# box_label = [id2name[int(item)] for item in labels[select_mask]]
# pred_dict = {
#     'boxes': boxes[select_mask],
#     'size': targets['size'],
#     'box_label': box_label
# }
# vslzr.visualize(image, pred_dict, savedir=None)

image = Image.open("./figure/idea.jpg").convert("RGB")
# image

# transform images
transform = T.Compose([
    T.RandomResize([800], max_size=1333),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
image, _ = transform(image, None)


# predict images
output = model(image[None])
output = postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]))[0]


# visualize outputs
thershold = 0.3 # set a thershold

scores = output['scores']
labels = output['labels']
boxes = box_ops.box_xyxy_to_cxcywh(output['boxes'])
select_mask = scores > thershold

box_label = [id2name[int(item)] for item in labels[select_mask]]
pred_dict = {
    'boxes': boxes[select_mask],
    'size': torch.Tensor([image.shape[1], image.shape[2]]),
    'box_label': box_label
}
vslzr.visualize(image, pred_dict, savedir=None)



