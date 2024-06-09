from autocommand import autocommand
import rn_model, datetime, os, signal, torch, cnn_model, lstm, conv_lstm, sys, conv_rn
import numpy as np
import readCharmDataset as riq
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pandas as pd


class CharmEETrainer(object):
    def __init__(self, model, model_name, loss_weights, id_gpu, data_folder, modelPath, resultPath, batch_size, chunk_size, 
        sample_stride, loaders, dg_coverage, tensorboard):
        
        self.device = (torch.device('cuda') if torch.cuda.is_available()
                      else torch.device('cpu'))
        
        self.model = model
        self.model_name = model_name
        self.history_path = os.path.join(resultPath, "history_%s.csv"%(self.model_name))
        self.modelSavePath = os.path.join(modelPath, "%s_model.pt"%(self.model_name))
        self.metricsEvaluationPath = os.path.join(resultPath, "dnn_metrics_performance_test_set.csv")
        self.confMatrixPath = os.path.join(resultPath, "%s_confusion_matrix.csv"%(self.model_name))

        self.labels = ['Clear', 'LTE', 'WiFi']

        self.chunk_size = chunk_size
        self.loss_fn = nn.CrossEntropyLoss()  #dg.GamblerLoss(3)
        self.dg_coverage = dg_coverage
        self.loss_weights = loss_weights

        self.train_data = IQDataset(data_folder=data_folder, chunk_size=chunk_size, stride=sample_stride)
        self.train_data.normalize(torch.tensor([-2.7671e-06, -7.3102e-07]), torch.tensor([0.0002, 0.0002]))
        self.train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=batch_size, shuffle=True, num_workers=loaders, pin_memory=True)

        self.val_data = IQDataset(data_folder=data_folder, chunk_size=chunk_size, stride=sample_stride, subset='validation')
        self.val_data.normalize(torch.tensor([-2.7671e-06, -7.3102e-07]), torch.tensor([0.0002, 0.0002]))
        self.val_loader = torch.utils.data.DataLoader(self.val_data, batch_size=batch_size, shuffle=False, num_workers=loaders, pin_memory=True)

        self.test_data = IQDataset(data_folder=data_folder, chunk_size=chunk_size, stride=sample_stride, subset='test')
        self.test_data.normalize(torch.tensor([-2.7671e-06, -7.3102e-07]), torch.tensor([0.0002, 0.0002]))
        self.test_loader = torch.utils.data.DataLoader(self.val_data, batch_size=batch_size, shuffle=False, num_workers=loaders, pin_memory=True)


        self.running = False
        self.best_val_accuracy = 0.0


        lr = [0.01, 0.01]
        self.optimizer = optim.SGD([{'params': ee_model.stages.parameters(), 'lr': lr[0]}, 
         {'params': ee_model.exits.parameters(), 'lr': lr[0]},
          {'params': ee_model.classifier.parameters(), 'lr': lr[0]}], momentum=0.9, weight_decay=2e-05)



        #self.tensorboard = tensorboard_parse(tensorboard)
        print("Init OK")

    def compute_metrics(self, output_list, conf_list, class_list, target):
        model_loss = 0
        ee_loss, acc_branches = [], []

        for i, (output, inf_class, weight) in enumerate(zip(output_list, class_list, loss_weights), 1):
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


                    output_list, conf_list, class_list = self.model(chunks)
                    model_loss, ee_loss, model_acc, ee_acc = self.compute_metrics(output_list, conf_list, class_list, labels)

                    self.optimizer.zero_grad()
                    model_loss.backward()
                    self.optimizer.step
                    torch.cuda.empty_cache()

                    model_loss_list.append(float(model_loss.item())), ee_loss_list.append(ee_loss)
                    model_acc_list.append(model_acc), ee_acc_list.append(ee_acc)

                    # clear variables
                    del chunks, labels, output_list, conf_list, class_list
                    torch.cuda.empty_cache()

                avg_loss, avg_ee_loss = round(np.mean(model_loss_list), 4), np.mean(ee_loss_list, axis=0)
                avg_acc, avg_ee_acc = round(np.mean(model_acc_list), 2), np.mean(ee_acc_list, axis=0)

                print(f"{datetime.datetime.now()} Epoch {epoch}, loss {loss_train/len(self.train_loader)}")
                print("Epoch: %s, Train Model Loss: %s, Train Model Acc: %s"%(epoch, avg_loss, avg_acc))

                #self.validate(epoch, train=True)
                #self.model.train()

    def execute(self, n_epochs):

      self.training_loop(n_epochs)
      #self.validate(n_epochs-1, train=True)
      self.test()



@autocommand(__name__)
def charm_trainer(model_name="cnn", id_gpu="0", data_folder="./", 
    modelPath="./models", resultPath="./results", n_epochs=25, batch_size=512, 
    chunk_size=20000, sample_stride=0, loaders=6, dg_coverage=0.75, tensorboard=None,
    exit_type="bnpool", n_branches=2, n_classes=3, loss_weights_type="decrescent"):
    



    device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    exit_positions = [5, 11]
    loss_weights_dict = {"crescent": np.linspace(0.3, 1, n_branches+1), 
    "decrescent": np.linspace(1, 0.3, n_branches+1), 
    "equal": np.ones(n_branches+1)}

    loss_weights = loss_weights_dict[loss_weights_type]


    ee_model = Early_Exit_DNN(model_name, n_classes, n_branches, exit_type, device, exit_positions=exit_positions)    
    
    
    ct = CharmEETrainer(ee_model, model_name, loss_weights, id_gpu, data_folder, modelPath, 
                        resultPath, batch_size, chunk_size, sample_stride,loaders, dg_coverage, tensorboard)
    ct.execute(n_epochs)
