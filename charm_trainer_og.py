from autocommand import autocommand
#from torch.utils.tensorboard import SummaryWriter
import rn_model
import datetime
import deep_gambler as dg
import numpy as np
import os
import read_IQ as riq
import signal
import torch
import torch.nn as nn
import torch.optim as optim


def print_stats(acc_mat, name, epoch, tensorboard):
    classes = acc_mat.shape[0]
    ones = np.ones((classes, 1)).squeeze(-1)

    corrects = np.diag(acc_mat)
    acc = corrects.sum()/acc_mat.sum()
    recall = (corrects/acc_mat.dot(ones)).round(4)
    precision = (corrects/ones.dot(acc_mat)).round(4)
    f1 = (2*recall*precision/(recall+precision)).round(4)

    print(f"Epoch {epoch} on {name} dataset")
    print(f"Accuracy: {acc}")
    if tensorboard:
        tensorboard.add_scalar(f"accuracy/{name}", acc, epoch)
    print(f"\t\tRecall\tPrecision\tF1")
    for c in range(classes):
        print(f"Class {c}\t\t{recall[c]}\t{precision[c]}\t\t{f1[c]}")
        #if tensorboard:
        #    tensorboard.add_scalar(f"recall_{c}/{name}", recall[c], epoch)
        #    tensorboard.add_scalar(f"precision_{c}/{name}", precision[c], epoch)
        #    tensorboard.add_scalar(f"f1_{c}/{name}", f1[c], epoch)
        #    tensorboard.flush()


def tensorboard_parse(tensorboard):
    '''
    tensorboard: a string with comma separated <key>=<value> substrings, each of
    them mapping to a tensorboard.SummaryWriter constructor parameter.
    E.g.,
    log_dir='./runs',comment='',purge_step=None,max_queue=10,flush_secs=120,filename_suffix=''
    '''
    writer = None
    if tensorboard:
        conf = {}
        for tok in tensorboard.split(','):
            kv = tok.split('=')
            if len(kv) == 2:
                if kv[1] == 'None':
                    kv[1] = None
                conf[kv[0]] = kv[1]
        writer = SummaryWriter(**conf)
    return writer


class EarlyExitException(Exception):
    def __str__(self):
        return "Received termination signal"


class CharmTrainer(object):
    def __init__(self, id_gpu="0", data_folder=".", batch_size=64, chunk_size=200000, sample_stride=0, loaders=8, dg_coverage=0.999, tensorboard=None):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = id_gpu
        self.device = (torch.device('cuda') if torch.cuda.is_available()
                      else torch.device('cpu'))
        print(f"Training on {self.device}")
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

        self.chunk_size = chunk_size
        self.loss_fn = nn.CrossEntropyLoss()  #dg.GamblerLoss(3)
        self.dg_coverage = dg_coverage

        self.train_data = riq.IQDataset(data_folder=data_folder, chunk_size=chunk_size, stride=sample_stride)
        self.train_data.normalize(torch.tensor([-2.7671e-06, -7.3102e-07]), torch.tensor([0.0002, 0.0002]))
        self.train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=batch_size, shuffle=True, num_workers=loaders, pin_memory=True)

        self.val_data = riq.IQDataset(data_folder=data_folder, chunk_size=chunk_size, stride=sample_stride, subset='validation')
        self.val_data.normalize(torch.tensor([-2.7671e-06, -7.3102e-07]), torch.tensor([0.0002, 0.0002]))
        self.val_loader = torch.utils.data.DataLoader(self.val_data, batch_size=batch_size, shuffle=False, num_workers=loaders, pin_memory=True)

        self.running = False
        self.best_val_accuracy = 0.0
        self.tensorboard = tensorboard_parse(tensorboard)


    def init(self):
        self.model = brain.CharmBrain(self.chunk_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters())
        self.best_val_accuracy = 0.0


    def training_loop(self, n_epochs):
        for self.loss_fn.o in [1.7]:
            self.init()
            self.model.train()
            for epoch in range(n_epochs):
                loss_train = 0.0
                for chunks, labels in self.train_loader:
                    if not self.running:
                        raise EarlyExitException
                    chunks = chunks.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)

                    output = self.model(chunks)
                    loss = self.loss_fn(output, labels)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    loss_train += loss.item()
                if self.tensorboard:
                    self.tensorboard.add_scalar("Loss/train", loss_train/len(self.train_loader), epoch)
                if True:
                    print(f"{datetime.datetime.now()} Epoch {epoch}, loss {loss_train/len(self.train_loader)}")
                    print(f"Coverage: {self.dg_coverage}, o-parameter {self.loss_fn.o}")
                    self.validate(epoch, train=True)
                    self.model.train()

    def validate(self, epoch, train=True):
        loaders = [('val', self.val_loader)]
        if train:
            loaders.append(('train', self.train_loader))

        self.model.eval()
        for name, loader in loaders:
            correct = 0
            total = 0
            acc_mat = np.zeros((len(self.train_data.label), len(self.train_data.label)))

            with torch.no_grad():
                for chunks, labels in loader:
                    if not self.running:
                        raise EarlyExitException
                    chunks = chunks.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                    output = self.model(chunks)
                    #predicted = dg.output2class(output, self.dg_coverage, 3)
                    _, predicted = torch.max(output, dim=1)
                    total += labels.shape[0]
                    correct += int((predicted == labels).sum())
                    for i in range(labels.shape[0]):
                        acc_mat[labels[i]][predicted[i]] += 1

            accuracy = correct/total
            print(f"{name} accuracy: {accuracy}")
            if name == 'val' and accuracy>self.best_val_accuracy:
                self.save_model(f"charm_{self.dg_coverage}_{self.loss_fn.o}_{round(accuracy, 2)}.pt")
                self.best_val_accuracy = accuracy

            print_stats(acc_mat, name, epoch, self.tensorboard)

    def save_model(self, filename='charm.pt'):
        '''
        load your model with:
        >>> model = brain.CharmBrain()
        >>> model.load_state_dict(torch.load(filename))
        '''
        torch.save(self.model.state_dict(), filename)

    def execute(self, n_epochs):
        self.running = True
        try:
            self.training_loop(n_epochs)
            self.validate(n_epochs-1, train=True)
        except EarlyExitException:
            pass
        if self.tensorboard:
            self.tensorboard.close()
        print("[Done]")

    def exit_gracefully(self, signum, frame):
        self.running = False


@autocommand(__name__)
def charm_trainer(id_gpu="0", data_folder=".", n_epochs=25, batch_size=512, chunk_size=20000, sample_stride=0, loaders=6, dg_coverage=0.75, tensorboard=None):
    ct = CharmTrainer(id_gpu=id_gpu, data_folder=data_folder, batch_size=batch_size, chunk_size=chunk_size, sample_stride=sample_stride,
                      loaders=loaders, dg_coverage=dg_coverage, tensorboard=tensorboard)
    ct.execute(n_epochs=n_epochs)