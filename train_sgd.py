import torch
import torch.nn as nn
from models.efficientnet import efficientnet_b4, preprocess, efficientnet_b2
import os
from utils.tools import Trainer, Logger, Tester, Checker, Saver, LRScheduler, get_dataset_from_pickle, AccCounter
from tensorboardX import SummaryWriter
from models.utils import freeze_layers
import torchvision.transforms.functional as transF
import math

efficientnet = efficientnet_b2
weight_decay = 1e-4
batch_size = 32
image_size = 224
dataset_dir = r'../fer2013'
epochs = 100
min_lr, max_lr = 0.001, 0.02
# dataset_dir = r'D:\datasets\Facial Expression Recognition\fer2013'
comment = "-b2,wd_w,bs=32,hflip,224,no_norm,sgd"
# --------------------------------
pkl_folder = 'pkl/'
train_pickle_fname = "images_targets_train.pkl"
test_pickle_fname = "images_targets_test_pub.pkl"
test_pickle_fname2 = "images_targets_test_pri.pkl"

labels = [
    "anger", "disgust", "fear", "happy", "neutral", "sad", "surprised"
]


# --------------------------------
def cosine_annealing_lr(epoch, _T_max, _min_lr, _max_lr):
    return _min_lr + (_max_lr - _min_lr) * (1 + math.cos(epoch / _T_max * math.pi)) / 2


def lr_func(epoch):
    return cosine_annealing_lr(epoch, epochs - 1, min_lr, max_lr)


def train_transform(image):
    if torch.rand(1) < 0.5:
        image = transF.hflip(image)
    return preprocess([image], image_size)[0]


def test_transform(image):
    return preprocess([image], image_size)[0]


def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = efficientnet(True, image_size=image_size, num_classes=len(labels))
    # freeze_layers(model, ["conv_first", "layer1"])
    pg0, pg1, pg2 = [], [], []  # bn_weight, weight, bias
    for k, v in model.named_modules():
        # bias
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases. no decay
        # weight
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay

    optim = torch.optim.SGD(pg0, 0, 0.9)  # bn_weight
    optim.add_param_group({'params': pg1, 'weight_decay': 1e-4})  # add pg1 with weight_decay
    optim.add_param_group({'params': pg2})  # add pg2 (biases)
    del pg0, pg1, pg2

    train_dataset = get_dataset_from_pickle(os.path.join(dataset_dir, pkl_folder, train_pickle_fname), train_transform)
    test_dataset = get_dataset_from_pickle(os.path.join(dataset_dir, pkl_folder, test_pickle_fname), test_transform)
    test_dataset2 = get_dataset_from_pickle(os.path.join(dataset_dir, pkl_folder, test_pickle_fname2), test_transform)
    acc_counter = AccCounter(labels)
    writer = SummaryWriter(comment=comment)
    logger = Logger(50, writer)
    checker = Checker({"PublicTest": Tester(model, test_dataset, batch_size, device, acc_counter, 4000),
                       "PrivateTest": Tester(model, test_dataset2, batch_size, device, acc_counter, 4000)},
                      Saver(model), 1, 0, logger)
    lr_scheduler = LRScheduler(optim, lr_func)
    trainer = Trainer(model, optim, train_dataset, batch_size, device, lr_scheduler, logger, checker)
    print("配置: %s" % comment, flush=True)
    trainer.train((0, epochs))
    writer.close()


if __name__ == "__main__":
    main()
