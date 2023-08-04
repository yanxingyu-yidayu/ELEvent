import os
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import argparse
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image
import numpy as np
import sys

# build_model_deformable = __import__("Deformable-DETR.models")
sys.path.insert(0, "./yolov5")
# sys.path.insert(1, "./detr")
from yolov5.detect import run as run_yolov5
# from detr.models import build_model as build_model_detr

app = Flask(__name__)

CLASSES = ['chick', 'chick', "N/A", "N/A", "N/A", "N/A"]
COLORS = [(0.000, 0.447, 0.741), (0.850, 0.325, 0.098), (0.929, 0.694, 0.125),
          (0.494, 0.184, 0.556), (0.466, 0.674, 0.188), (0.301, 0.745, 0.933)]
transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def detect(im, model, transform):
    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)

    # demo model only support by default images with aspect ratio between 0.5 and 2
    # if you want to use images with an aspect ratio outside this range
    # rescale your image so that the maximum size is at most 1333 for best results
    assert img.shape[-2] <= 1600 and img.shape[
        -1] <= 1600, 'demo model only supports images up to 1600 pixels on each side'

    # propagate through the model
    outputs = model(img)
    # print(f"{'-'*10}outputs{'-'*10}\n{outputs}\n")
    # keep only predictions with 0.7+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    # print(f"{'-' * 10}probas{'-' * 10}\n{probas}\n")
    keep = probas.max(-1).values >= 0.75

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
    return probas[keep], bboxes_scaled


def plot_results(pil_img, prob, boxes, img_name):
    img_w, img_h = pil_img.size
    plt.figure(figsize=((img_w+80)/100, (img_h+80)/100))
    plt.imshow(pil_img)
    ax = plt.gca()
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), COLORS * 100):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=2))
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=6,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.savefig(f"static/input_images/{img_name}", bbox_inches='tight')


@app.route('/')
def index():
    return render_template('index.html')


def get_result(model, model_path, im, result_file_name):
    checkpoint_detr = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint_detr['model'])
    model.eval()
    scores, boxes = detect(im, model, transform)
    plot_results(im, scores, boxes, result_file_name)


@app.route('/', methods=['POST'])
def index_post():
    file = request.files['file']
    filename = secure_filename(file.filename)
    print("upload filename -->>>>", filename)
    file_save_path = os.path.join('static/input_images', filename)
    file.save(file_save_path)
    # im = Image.open(file_save_path)

    # filename = file.filename.split(".")[0].strip()
    detr_file_name_result = None
    deformable_file_name_result = None
    yolov5_file_name_result = None
    # if args.detr_model_path is not None:
    #     model_detr, _, _ = build_model_detr(args)
    #     detr_file_name_result = filename + "_detr_result.png"
    #     get_result(model_detr, args.detr_model_path, im, detr_file_name_result)
    # if args.deformable_model_path is not None:
    #     model_deformable, _, _ = build_model_deformable(args)
    #     deformable_file_name_result = filename + "_deformable_result.png"
    #     get_result(model_deformable, args.deformable_model_path, im, deformable_file_name_result)
    if args.yolov5_path is not None:
        # pass
        yolov5_file_name_result = run_yolov5(weights=args.yolov5_path, source=file_save_path, data=args.yolov5_data_cfg_path, project='static/', exist_ok=True)
        # yolov5_file_name_result = "/".join(yolov5_file_name_result.split("/")[1:])
    print("yolo5 resutl ====>>>  ", filename)
    print("detr_file_name_result ====>>>  ", detr_file_name_result)
    return render_template('index.html', input_file=filename,
                           # detr_file_name_result=detr_file_name_result,
                           # deformable_file_name_result="input_images/"+deformable_file_name_result,
                           yolov5_file_name_result=filename)


@app.route('/display_yolov5/<filename>')
def display_image_yolov5(filename):
    print("diplay_image_yolov5 -->>> ", filename)
    # return redirect(url_for('static', filename='exp/' + filename), code=301)
    return send_from_directory("static/exp", filename, as_attachment=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Transformer object detection demo', add_help=False)

    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    parser.add_argument('--sgd', action='store_true')

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # Running demo server
    parser.add_argument("--port", default=9595)
    parser.add_argument("--detr_model_path", default=None)
    parser.add_argument("--deformable_model_path", default=None)
    parser.add_argument("--yolov5_path", default=None)
    parser.add_argument("--yolov5_data_cfg_path", default=None)


    args = parser.parse_args()
    run_port = args.port
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.exists('static/input_images'):
        os.makedirs('static/input_images')
    app.run('0.0.0.0', port=run_port)

