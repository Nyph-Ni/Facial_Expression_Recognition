from models.efficientnet import efficientnet_b4, preprocess, efficientnet_b2
import torch
from PIL import Image
from utils.utils import load_params
import os

efficientnet = efficientnet_b2
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
# device = torch.device('cpu')

images_dir = r"images"

# read labels
labels = [
    "anger", "disgust", "fear", "happy", "neutral", "sad", "surprised"
]

# pred
model = efficientnet_b4(False, num_classes=len(labels)).to(device)
print(load_params(model, r".\weights\model_epoch39_test0.7173.pth"))
model.eval()
for image_fname in os.listdir(images_dir):
    # read images
    image_path = os.path.join(images_dir, image_fname)
    with Image.open(image_path) as x:
        x = x.convert("L")  # 训练集为黑白图片
        x = preprocess([x], 224).to(device)
    with torch.no_grad():
        pred = torch.softmax(model(x), dim=1)
    values, indices = torch.topk(pred, k=7)

    print("Image Pred: %s" % image_fname)
    print("-------------------------------------")
    for value, idx in zip(values[0], indices[0]):
        value, idx = value.item(), idx.item()
        print("%-75s%.2f%%" % (labels[idx], value * 100))
