from autocommand import autocommand
#from torch.utils.tensorboard import SummaryWriter
import rn_model, datetime, os, signal, torch
#import deep_gambler as dg
import numpy as np
import readCharmDataset as riq
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pandas as pd

def compute_metrics(labels, acc_mat, avg_loss, best_val_accuracy):
    classes = acc_mat.shape[0]
    ones = np.ones((classes, 1)).squeeze(-1)

    corrects = np.diag(acc_mat)
    acc = corrects.sum()/acc_mat.sum()
    recall = (corrects/acc_mat.dot(ones)).round(4)
    precision = (corrects/ones.dot(acc_mat)).round(4)
    f1 = (2*recall*precision/(recall+precision)).round(4)

    print(f"Accuracy: {acc}")

    #if tensorboard:
    #    tensorboard.add_scalar(f"accuracy/{name}", acc, epoch)
    print(f"\t\tRecall\tPrecision\tF1")
    
    results = {"acc": acc, "avg_loss": avg_loss, "best_val_accuracy": best_val_accuracy}

    #for c, label in enumerate(labels):
    #    print(f"Class {label}\t\t{recall[c]}\t{precision[c]}\t\t{f1[c]}")

    #    results.update({"recall_%s"%(label): recall[c], "precision_%s"%(label): precision[c],
    #        "f1_%s"%(label): f1[c]})
    for c in range(classes):
        print(f"Class {c}\t\t{recall[c]}\t{precision[c]}\t\t{f1[c]}")
        #results.update({"recall_%s"%(labels[c]): recall[c], "precision_%s"%(labels[c]): precision[c],
        #    "f1_%s"%(labels[c]): f1[c]})

    #print(results)
        #if tensorboard:
        #    tensorboard.add_scalar(f"recall_{c}/{name}", recall[c], epoch)
        #    tensorboard.add_scalar(f"precision_{c}/{name}", precision[c], epoch)
        #    tensorboard.add_scalar(f"f1_{c}/{name}", f1[c], epoch)
        #    tensorboard.flush()
    return results



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
    def __init__(self, model_name="rn", id_gpu="0", data_folder=".", modelPath=".", resultPath=".", batch_size=64, chunk_size=200000, 
        sample_stride=0, loaders=8, dg_coverage=0.999, tensorboard=None):
        
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = id_gpu
        self.device = (torch.device('cuda') if torch.cuda.is_available()
                      else torch.device('cpu'))
        
        self.model_name = model_name
        self.history_path = os.path.join(resultPath, "history_%s_og.csv"%(self.model_name))
        self.modelSavePath = os.path.join(modelPath, "%s_model_og.pt"%(self.model_name))

        self.metricsEvaluationPath = os.path.join(resultPath, "dnn_metrics_performance_test_set.csv")

        self.labels = ['Clear', 'LTE', 'WiFi']

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

        self.test_data = riq.IQDataset(data_folder=data_folder, chunk_size=chunk_size, stride=sample_stride, subset='test')
        self.test_data.normalize(torch.tensor([-2.7671e-06, -7.3102e-07]), torch.tensor([0.0002, 0.0002]))
        self.test_loader = torch.utils.data.DataLoader(self.val_data, batch_size=batch_size, shuffle=False, num_workers=loaders, pin_memory=True)


        self.running = False
        self.best_val_accuracy = 0.0
        self.tensorboard = tensorboard_parse(tensorboard)

        print("Init OK")


    def save_history(self, metrics, epoch, subset):
        metrics.update({"epoch": epoch, "subset": subset})
        df = pd.DataFrame([metrics])
        df.to_csv(self.history_path, mode='a', header=not os.path.exists(self.history_path))

    def save_metrics_performance_test(self, metrics):
        df = pd.DataFrame([metrics])
        df.to_csv(self.metricsEvaluationPath, mode='a', header=not os.path.exists(self.metricsEvaluationPath))

    def save_model(self, metrics):
        '''
        load your model with:
        >>> model = brain.CharmBrain()
        >>> model.load_state_dict(torch.load(filename))
        '''
        save_dict  = {} 
        save_dict.update(metrics)
        save_dict.update({"best_val_accuracy": self.best_val_accuracy})
        save_dict.update({"model_state_dict": self.model.state_dict()})
        torch.save(save_dict, self.modelSavePath)


    def init(self):
        self.model = rn_model.CharmBrain(self.chunk_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters())
        self.best_val_accuracy = 0.0


    def training_loop(self, n_epochs):
        for self.loss_fn.o in [1.7]:
            self.init()
            self.model.train()
            for epoch in range(n_epochs):
                loss_train = 0.0
                for chunks, labels in tqdm(self.train_loader):
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

                #if self.tensorboard:
                #    self.tensorboard.add_scalar("Loss/train", loss_train/len(self.train_loader), epoch)
                
                #if True:
                print(f"{datetime.datetime.now()} Epoch {epoch}, loss {loss_train/len(self.train_loader)}")
                #print(f"Coverage: {self.dg_coverage}, o-parameter {self.loss_fn.o}")
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
            loss_total = 0
            acc_mat = np.zeros((len(self.train_data.label), len(self.train_data.label)))
            #acc_mat = np.zeros((len(loader.label), len(loader.label)))

            with torch.no_grad():
                for chunks, labels in tqdm(loader):
                    if not self.running:
                        raise EarlyExitException
                    chunks = chunks.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                    output = self.model(chunks)
                    loss = self.loss_fn(output, labels)
                    #predicted = dg.output2class(output, self.dg_coverage, 3)
                    loss_total += loss.item()
                    _, predicted = torch.max(output, dim=1)
                    total += labels.shape[0]
                    correct += int((predicted == labels).sum())
                    for i in range(labels.shape[0]):
                        acc_mat[labels[i]][predicted[i]] += 1


            accuracy = correct/total
            avg_loss = loss_total/len(loader)

            print(f"Epoch {epoch} on {name} dataset")
            print(f"{name} accuracy: {accuracy}")

            metrics = compute_metrics(self.labels, acc_mat, avg_loss, self.best_val_accuracy)
            
            self.save_history(metrics, epoch, subset=name)

            if name == 'val' and accuracy>self.best_val_accuracy:
                self.best_val_accuracy = accuracy
                self.save_model(metrics)


    def test(self):
        
        self.model.eval()
        correct = 0
        total = 0
        loss_total = 0
        acc_mat = np.zeros((len(self.train_data.label), len(self.train_data.label)))

        with torch.no_grad():
            for chunks, labels in tqdm(self.test_loader):
                if not self.running:
                    raise EarlyExitException
                    
                chunks = chunks.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                output = self.model(chunks)
                loss = self.loss_fn(output, labels)
                #predicted = dg.output2class(output, self.dg_coverage, 3)
                loss_total += loss.item()
                _, predicted = torch.max(output, dim=1)
                total += labels.shape[0]
                correct += int((predicted == labels).sum())
                for i in range(labels.shape[0]):
                    acc_mat[labels[i]][predicted[i]] += 1

        accuracy = correct/total
        avg_loss = loss_total/len(self.test_loader)

        print(f"Test Accuracy: {accuracy}")

        metrics = compute_metrics(self.labels, acc_mat, avg_loss, self.best_val_accuracy)

        self.save_metrics_performance_test(metrics)


    def execute(self, n_epochs):
        self.running = True
        try:
            self.training_loop(n_epochs)
            #self.validate(n_epochs-1, train=True)
            self.test()

        except EarlyExitException:
            pass
        if self.tensorboard:
            self.tensorboard.close()
        print("[Done]")

    def exit_gracefully(self, signum, frame):
        self.running = False

@autocommand(__name__)
def charm_trainer(model_name="rn", id_gpu="0", data_folder=".", 
    modelPath="./models", resultPath="./results", n_epochs=25, batch_size=512, 
    chunk_size=20000, sample_stride=0, loaders=6, dg_coverage=0.75, tensorboard=None):
    
    ct = CharmTrainer(id_gpu=id_gpu, data_folder=data_folder, modelPath=modelPath, resultPath=resultPath, 
        batch_size=batch_size, chunk_size=chunk_size, sample_stride=sample_stride,
        loaders=loaders, dg_coverage=dg_coverage, tensorboard=tensorboard)
    
    ct.execute(n_epochs=n_epochs)