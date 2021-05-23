import torch
from models.efficientnet import efficientnet_b4, preprocess, efficientnet_b2
import os
from utils.tools import Tester, get_dataset_from_pickle, AccCounter
from utils.utils import load_params

batch_size = 8

# --------------------------------
efficientnet = efficientnet_b2
dataset_dir = r'../fer2013'
# dataset_dir = r'D:\datasets\Facial Expression Recognition\fer2013'
pkl_folder = 'pkl/'
test_pickle_fname = "images_targets_test.pkl"

labels = [
    "anger", "disgust", "fear", "happy", "neutral", "sad", "surprised"
]


# --------------------------------

def test_transform(image):
    return preprocess([image], 224)[0]


def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = efficientnet(False, image_size=224, num_classes=len(labels))
    print(load_params(model, r".\weights\model_epoch39_test0.7173.pth"))
    test_dataset = get_dataset_from_pickle(os.path.join(dataset_dir, pkl_folder, test_pickle_fname), test_transform)
    acc_counter = AccCounter(labels)
    tester = Tester(model, test_dataset, batch_size, device, acc_counter, 4000)
    print(tester.test())


if __name__ == "__main__":
    main()
