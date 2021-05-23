from torch.utils.data import DataLoader
import torch
from ..tools.utils import to
import torch.nn.functional as F


class RuntimeErrorHandler:
    def __init__(self, ignore_num):
        self.ignore_num_ori = self.ignore_num = ignore_num

    def error(self, e):
        if self.ignore_num > 0:
            print(e, flush=True)
            self.ignore_num -= 1
        else:
            raise e

    def init(self):
        self.ignore_num = self.ignore_num_ori


def label_smoothing_cross_entropy(pred, target, smoothing: float = 0.1):
    """reference: https://github.com/seominseok0429/label-smoothing-visualization-pytorch

    :param pred: shape(N, In). 未过softmax
    :param target: shape(N,)
    :param smoothing: float
    :return: shape()
    """
    pred = F.log_softmax(pred, dim=-1)
    ce_loss = F.nll_loss(pred, target)
    smooth_loss = -torch.mean(pred)
    return (1 - smoothing) * ce_loss + smoothing * smooth_loss


class Trainer:
    def __init__(self, model, optim, train_dataset, batch_size, device,
                 lr_scheduler=None, logger=None, checker=None, runtime_error_handler=None):
        self.model = model.to(device)
        self.optim = optim
        # self.train_loader = DataLoader(train_dataset, batch_size, True, pin_memory=True)
        self.train_loader = DataLoader(train_dataset, batch_size, True, num_workers=8, pin_memory=True)
        self.device = device
        self.lr_scheduler = lr_scheduler
        self.logger = logger
        assert checker
        self.checker = checker
        self.runtime_error_handler = runtime_error_handler or RuntimeErrorHandler(ignore_num=2)

    def train(self, epoch_range):
        for epoch in range(*epoch_range):
            self.model.train()
            if self.lr_scheduler:
                self.lr_scheduler.step(epoch)
            lr = self.optim.param_groups[0]['lr']
            self.logger.new_epoch(epoch, len(self.train_loader), lr)
            for i, (x, target) in enumerate(self.train_loader):
                try:
                    x, target = to(x, target, self.device)
                    pred = self.model(x)
                    # loss = F.cross_entropy(pred, target)
                    loss = label_smoothing_cross_entropy(pred, target, 0.01)
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()
                    self.logger.step(loss.item())
                    self.runtime_error_handler.init()
                except RuntimeError as e:
                    x, y, loss = None, None, None
                    torch.cuda.empty_cache()
                    try:
                        self.runtime_error_handler.error(e)
                    except RuntimeError as e:
                        self.checker.saver.save("tmp_epoch%d_step%d" % (epoch, i + 1))
                        raise e

            if self.checker:
                self.checker.step(epoch, last=(epoch == epoch_range[1] - 1))
