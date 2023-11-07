from autocommand import autocommand
from torch.utils.tensorboard import SummaryWriter
import datetime, os, signal, torch
import rn_model, cnn_model
#import deep_gambler as dg
import numpy as np
import read_IQ as riq
import torch.nn as nn
import torch.optim as optim

def compute_metrics(labels, acc_mat, avg_loss, best_val_accuracy):
    classes = acc_mat.shape[0]
    ones = np.ones((classes, 1)).squeeze(-1)

    corrects = np.diag(acc_mat)
    acc = corrects.sum()/acc_mat.sum()
    recall = (corrects/acc_mat.dot(ones)).round(4)
    precision = (corrects/ones.dot(acc_mat)).round(4)
    f1 = (2*recall*precision/(recall+precision)).round(4)

    print(f"Accuracy: {acc}")

    print(f"\t\tRecall\tPrecision\tF1")
    
    results = {"acc": acc, "avg_loss": avg_loss, "best_val_accuracy": best_val_accuracy}

    for c in range(classes):
        print(f"Class {c}\t\t{recall[c]}\t{precision[c]}\t\t{f1[c]}")
        results.update({"recall_%s"%(labels[c]): recall[c], "precision_%s"%(labels[c]): precision[c],
            "f1_%s"%(labels[c]): f1[c]})

    return results


class EarlyExitException(Exception):
    def __str__(self):
        return "Received termination signal"


class CharmTrainer(object):
    def __init__(self, model_name, id_gpu="0", data_folder=".", batch_size=64, chunk_size=200000, 
        sample_stride=0, loaders=8):

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = id_gpu
        self.device = torch.device('cuda') if torch.cuda.is_available()
                      else torch.device('cpu')
        
        self.model_name = model_name
        self.history_path = os.path.join(resultPath, "history_%s_own.csv"%(self.model_name))
        self.modelSavePath = os.path.join(modelPath, "%s_model_own.pt"%(self.model_name))
        self.metricsEvaluationPath = os.path.join(resultPath, "dnn_metrics_performance_test_set_own.csv")

        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

        self.chunk_size = chunk_size
        self.loss_fn = nn.CrossEntropyLoss()

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

        print("Training %s on %s"%(self.model_name, self.device) )


    def initialize_model(self):

        if(self.model_name == "rn"):
            self.model = rn_model.CharmBrain(self.chunk_size).to(self.device)

        elif(self.model_name == "cnn"):
            self.model = cnn_model.ConvModel().to(self.device)

        else:
            raise Exception("This DNN model has not implemented yet.")



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

    def training_loop(self, epoch):

        loss_train = 0.0
        correct, total = 0, 0
        acc_mat = np.zeros((len(self.train_loader.label), len(self.train_loader.label)))
        self.model.train()

        for chunks, labels in self.train_loader:
            if not self.running:
                raise EarlyExitException
            
            chunks = chunks.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            output = self.model(chunks)
            _, predicted = torch.max(output, dim=1)

            loss = self.loss_fn(output, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_train += loss.item()
            total += labels.shape[0]
            correct += int((predicted == labels).sum())
                    
            for i in range(labels.shape[0]):
                acc_mat[labels[i]][predicted[i]] += 1

            # clear variables
            del chunks, labels, output, predicted
            torch.cuda.empty_cache()

        metrics = compute_metrics(correct, total, total_loss, acc_mat, epoch)

        self.save_history(metrics, subset="train")

        print("Epoch: %s, Train Loss: %s, Train Accuracy: %s"%(epoch, metrics['avg_loss'], metrics['acc']))

    def validation_loop(self, epoch):
        
        self.model.eval()
        
        correct, total, total_loss = 0, 0, 0
        acc_mat = np.zeros((len(self.val_loader.label), len(self.val_loader.label)))

        with torch.no_grad():
            for chunks, labels in self.val_loader:
                if not self.running:
                    raise EarlyExitException
                
                chunks = chunks.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                output = self.model(chunks)
                loss = self.loss_fn(output, labels)

                _, predicted = torch.max(output, dim=1)
                total += labels.shape[0]
                correct += int((predicted == labels).sum())
                total_loss += loss.item()

                for i in range(labels.shape[0]):
                    acc_mat[labels[i]][predicted[i]] += 1

                # clear variables
                del chunks, labels, output, predicted
                torch.cuda.empty_cache()

        metrics = compute_metrics(correct, total, total_loss, acc_mat, epoch)

        self.save_history(metrics, subset="val")

        print("Epoch: %s, Val Loss: %s, Val Accuracy: %s"%(epoch, metrics['avg_loss'], metrics['acc']))

        if (metrics['acc'] > self.best_val_accuracy):
            self.save_model(metrics)
            self.best_val_accuracy = metrics['acc']


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

    def run(self, n_epochs):
        
        self.initialize_model()
        self.optimizer = optim.Adam(self.model.parameters())
        self.best_val_accuracy = 0.0

        for epoch in range(n_epochs):
            self.training_loop(epoch)
            self.validation_loop(epoch)

        self.test()

    def save_model(self, metrics):

        save_dict  = {} 
        save_dict.update(metrics)
        save_dict.update("best_val_accuracy": self.best_val_accuracy)        
        save_dict.update({"model_state_dict": self.model.state_dict()})
        torch.save(save_dict, self.modelSavePath)

    def exit_gracefully(self, signum, frame):
        self.running = False


def main(args):

    ct = CharmTrainer(model_name=args.model_name, id_gpu=config.id_gpu, data_folder=config.datasetPath, 
        batch_size=args.batch_size, chunk_size=args.chunk_size, 
        sample_stride=config.sample_stride,loaders=config.loaders, 
        dg_coverage=config.dg_coverage)
    
    ct.run(args.n_epochs)

if (__name__ == "__main__"):

    # Input Arguments to configure the early-exit model .
    parser = argparse.ArgumentParser(description="Training Early-exit DNN. These are the hyperparameters")

    #We here insert the argument model_name
    parser.add_argument('--model_name', type=str, default=config.model_name, 
        choices=["rn", "cnn"], help='DNN model name (default: %s)'%(config.model_name))

    #parser.add_argument('--max_patience', type=int, default=20, help='Max Patience.')

    parser.add_argument('--model_id', type=int, default=1, help='Model_id.')

    parser.add_argument('--n_epochs', type=int, default=config.n_epochs, help='Number of epochs.')

    parser.add_argument('--chunk_size', type=int, default=config.chunk_size, help='Chunk Size.')

    parser.add_argument('--batch_size', type=int, default=config.batch_size, help='Chunk Size.')

    args = parser.parse_args()

    main(args)