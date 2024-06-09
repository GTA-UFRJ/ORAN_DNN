from autocommand import autocommand
import rn_model, datetime, os, signal, torch, cnn_model, lstm, conv_lstm, sys, conv_rn, ee_dnns
import numpy as np
import readCharmDataset as riq
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pandas as pd



def compute_conf_matrix(labels, acc_mat):

    conf_mat = {"Classes": labels}


    for label in labels:
        conf_mat.update({label: []})

    for c in range(len(labels)):  
        for j in range(len(labels)):
            conf_mat[labels[c]].append(acc_mat[c, j])

    return conf_mat


def compute_performance_metrics_branches(labels, acc_mat, avg_loss, best_val_accuracy, n_branches):

    results_dict = {}

    for i in range(n_branches):


        classes = acc_mat[i].shape[0]
        ones = np.ones((classes, 1)).squeeze(-1)

        correct_branch = np.diag(acc_mat[i])
        acc_branch = correct_branch.sum()/acc_mat[i].sum()
        recall_branch = (correct_branch/acc_mat[i].dot(ones)).round(4)
        precision_branch = (correct_branch/ones.dot(acc_mat[i])).round(4)
        f1_branch = (2*recall_branch*precision_branch/(recall_branch+precision_branch)).round(4)

        #print(f"Accuracy Branch: %s: %s"%(i+1, acc_branch))

        #print(f"\t\tRecall\tPrecision\tF1")
        
        results = {"acc_branch_%s"%(i+1): acc_branch, "avg_loss_branch_%s"%(i+1): avg_loss[i], "best_val_accuracy": best_val_accuracy,
        "precision_branch_%s"%(i+1): np.mean(precision_branch[:-1]), "recall_branch_%s"%(i+1): np.mean(recall_branch[:-1]), 
        "f1_branch_%s"%(i+1): np.mean(f1_branch[:-1])}

        results_dict.update(results)

        conf_mat = compute_conf_matrix(labels, acc_mat[i])

        #for c in range(classes-1):
        #    print(c)
        #    print(f"Class {c}\t\t{recall_branch[c]}\t{precision_branch[c]}\t\t{f1_branch[c]}")
        ##    results.update({"recall_%s"%(labels[c]): recall_branch[c], "precision_%s"%(labels[c]): precision_branch[c],
        #        "f1_%s"%(labels[c]): f1_branch[c]})

    return results_dict, conf_mat



class CharmEETrainer(object):
    def __init__(self, model, model_name, loss_weights, id_gpu, data_folder, modelPath, resultPath, batch_size, chunk_size, 
        sample_stride, loaders, dg_coverage, tensorboard, loss_weight_type):
        
        self.device = (torch.device('cuda') if torch.cuda.is_available()
                      else torch.device('cpu'))
        
        self.model = model
        self.model_name = model_name
        self.history_path = os.path.join(resultPath, "history_%s_%s.csv"%(self.model_name, loss_weight_type))
        self.modelSavePath = os.path.join(modelPath, "%s_model_%s.pt"%(self.model_name, loss_weight_type))
        self.metricsEvaluationPath = os.path.join(resultPath, "dnn_metrics_performance_%s.csv"%(loss_weight_type))
        self.confMatrixPath = os.path.join(resultPath, "%s_confusion_matrix_%s.csv"%(self.model_name, loss_weight_type))

        self.labels = ['Clear', 'LTE', 'WiFi']

        self.chunk_size = chunk_size
        self.loss_fn = nn.CrossEntropyLoss()  #dg.GamblerLoss(3)
        self.dg_coverage = dg_coverage
        self.loss_weights = loss_weights

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


        lr = [0.01, 0.01]
        #self.optimizer = optim.SGD([{'params': self.model.stages.parameters(), 'lr': lr[0]}, 
        # {'params': self.model.exits.parameters(), 'lr': lr[0]},
        #  {'params': self.model.classifier.parameters(), 'lr': lr[0]}], momentum=0.9, weight_decay=2e-05)

        self.optimizer = optim.Adam(self.model.parameters())


        #self.tensorboard = tensorboard_parse(tensorboard)
        print("Init OK")

    def compute_metrics(self, output_list, conf_list, class_list, target):
        model_loss = 0
        ee_loss, acc_branches = [], []

        for i, (output, inf_class, weight) in enumerate(zip(output_list, class_list, self.loss_weights), 1):
            loss_branch = self.loss_fn(output, target)
            model_loss += weight*loss_branch

            acc_branch = 100*inf_class.eq(target.view_as(inf_class)).sum().item()/target.size(0)
            ee_loss.append(loss_branch.item()), acc_branches.append(acc_branch)
            acc_model = np.mean(np.array(acc_branches))

        return model_loss, ee_loss, acc_model, acc_branches

    def training_loop(self, n_epochs):
        model_loss_list, ee_loss_list, model_acc_list, ee_acc_list = [], [], [], []
        for self.loss_fn.o in [1.7]:
            self.model.train()
            for epoch in range(n_epochs):
                loss_train = 0.0
                for chunks, labels in tqdm(self.train_loader):

                    chunks = chunks.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)


                    output_list, conf_list, class_list = self.model.forwardTraining(chunks)
                    model_loss, ee_loss, model_acc, ee_acc = self.compute_metrics(output_list, conf_list, class_list, labels)

                    self.optimizer.zero_grad()
                    model_loss.backward()
                    self.optimizer.step()
                    torch.cuda.empty_cache()

                    model_loss_list.append(float(model_loss.item())), ee_loss_list.append(ee_loss)
                    model_acc_list.append(model_acc), ee_acc_list.append(ee_acc)

                    # clear variables
                    del chunks, labels, output_list, conf_list, class_list
                    torch.cuda.empty_cache()

                avg_loss, avg_ee_loss = round(np.mean(model_loss_list), 4), np.mean(ee_loss_list, axis=0)
                avg_acc, avg_ee_acc = round(np.mean(model_acc_list), 2), np.mean(ee_acc_list, axis=0)

                #print(f"{datetime.datetime.now()} Epoch {epoch}, loss {loss_train/len(self.train_loader)}")
                print("Epoch: %s, Train Model Loss: %s, Train Model Acc: %s"%(epoch, avg_loss, avg_acc))

                for i in range(self.model.n_branches+1):
                    print("Branch %s: Train Acc: %s, Train Loss: %s"%(i+1, avg_ee_acc[i], avg_ee_loss[i])) 

                #self.validate(epoch)
                self.model.train()


    def validate(self, epoch):

        self.model.eval()
        correct_branch = np.zeros(self.model.n_branches)
        #total, loss_total = 0, 0
        acc_mat = [np.zeros((len(self.train_data.label), len(self.train_data.label))) for i in range(self.model.n_branches)]
        metrics_branches_dict = {}
        model_loss_list, ee_loss_list, model_acc_list, ee_acc_list = [], [], [], []

        with torch.no_grad():
            for chunks, labels in tqdm(self.val_loader):
                chunks = chunks.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)


                output_list, conf_list, class_list = self.model.forwardTraining(chunks)
                model_loss, ee_loss, model_acc, ee_acc = self.compute_metrics(output_list, conf_list, class_list, labels)

                model_loss_list.append(float(model_loss.item())), ee_loss_list.append(ee_loss), model_acc_list.append(model_acc), ee_acc_list.append(ee_acc)

                for j in range(self.model.n_branches): 
                    _, predicted = torch.max(output_list[j], dim=1)
                    correct_branch[j] += int((predicted == labels).sum())
                    for i in range(labels.shape[0]):
                        acc_mat[j][labels[i]][predicted[i]] += 1

        avg_loss, avg_ee_loss = round(np.mean(model_loss_list), 4), np.mean(ee_loss_list, axis=0)
        avg_acc, avg_ee_acc = round(np.mean(model_acc_list), 2), np.mean(ee_acc_list, axis=0)

        print("Epoch: %s, Val Model Loss: %s, Val Model Acc: %s"%(epoch, avg_loss, avg_acc))
        for j in range(self.model.n_branches+1):
            print("Branch %s: Val Acc: %s, Val Loss: %s"%(j+1, avg_ee_acc[j], avg_ee_loss[j])) 

        metrics_branches, _ = compute_performance_metrics_branches(self.labels, acc_mat, avg_ee_loss, self.best_val_accuracy, self.model.n_branches)
            
        self.save_history(metrics_branches_dict, epoch)

        if (avg_acc > self.best_val_accuracy):
            self.best_val_accuracy = avg_acc
            self.save_model(metrics_branches)

    def save_history(self, metrics, epoch):
        metrics.update({"epoch": epoch})
        df = pd.DataFrame([metrics])
        df.to_csv(self.history_path, mode='a', header=not os.path.exists(self.history_path))

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


    def execute(self, n_epochs):

      self.training_loop(n_epochs)
      #self.validate(n_epochs)
      #self.test()



@autocommand(__name__)
def charm_trainer(model_name="cnn", id_gpu="0", data_folder="./oran_dataset", 
    modelPath="./models", resultPath="./results", n_epochs=100, batch_size=512, 
    chunk_size=20000, sample_stride=0, loaders=6, dg_coverage=0.75, tensorboard=None,
    exit_type="bnpool", n_branches=2, n_classes=3, loss_weights_type="decrescent"):

    device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    exit_positions = [5, 11]
    loss_weights_dict = {"crescent": np.linspace(0.3, 1, n_branches+1), 
    "decrescent": np.linspace(1, 0.3, n_branches+1), 
    "equal": np.ones(n_branches+1)}

    loss_weights = loss_weights_dict[loss_weights_type]


    ee_model = ee_dnns.Early_Exit_DNN(model_name, n_classes, n_branches, exit_type, device, exit_positions=exit_positions)    
    
    
    ct = CharmEETrainer(ee_model, model_name, loss_weights, id_gpu, data_folder, modelPath, 
                        resultPath, batch_size, chunk_size, sample_stride,loaders, dg_coverage, tensorboard, loss_weights_type)
    ct.execute(n_epochs)
