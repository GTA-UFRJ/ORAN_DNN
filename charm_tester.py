from autocommand import autocommand
#from torch.utils.tensorboard import SummaryWriter
import datetime, os, signal, torch
import numpy as np
import readCharmDataset as riq
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
import charm_trainer

@autocommand(__name__)
def charm_tester(model_name="cnn", id_gpu="0", data_folder="./", 
    modelPath="./models", resultPath="./results", n_epochs=25, batch_size=512, 
    chunk_size=20000, sample_stride=0, loaders=6, dg_coverage=0.75, tensorboard=None):
    
    ct = charm_trainer.CharmTrainer(model_name=model_name, id_gpu=id_gpu, data_folder=data_folder, modelPath=modelPath, resultPath=resultPath, 
        batch_size=batch_size, chunk_size=chunk_size, sample_stride=sample_stride,
        loaders=loaders, dg_coverage=dg_coverage, tensorboard=tensorboard)
    
    ct.load_model()
    ct.test()