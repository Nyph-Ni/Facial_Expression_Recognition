class LRScheduler:
    def __init__(self, optim, lr_func):
        self.optim = optim
        self.lr_func = lr_func

    def step(self, epoch):
        lr = self.lr_func(epoch)
        for pg in self.optim.param_groups:
            pg['lr'] = lr
        return lr
