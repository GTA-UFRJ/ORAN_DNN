import torchvision
import os, sys, time, math
from PIL import Image
import torch, functools
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import torchvision.models as models
from pthflops import count_ops
from torch import Tensor
from torchvision.prototype import models as PM
import cnn_model

class EarlyExitBlock(nn.Module):
  """
  This EarlyExitBlock allows the model to terminate early when it is confident for classification.
  """
  def __init__(self, input_shape, pool_size, n_classes, exit_type, device):
    super(EarlyExitBlock, self).__init__()
    self.input_shape = input_shape

    _, width, height = input_shape
    self.expansion = width * height if exit_type == 'plain' else 1

    self.layers = nn.ModuleList()

    if (exit_type == 'bnpool'):
      #self.layers.append(nn.BatchNorm2d(pool_size))
      self.layers.append(nn.AdaptiveAvgPool2d(pool_size))

    if (exit_type == 'layers'):
      self.layers.append(nn.MaxPool1d(kernel_size=5, stride=5, padding=0, dilation=1, ceil_mode=False))
      self.layers.append(nn.Conv1d(7, 7, kernel_size=(1,), stride=(1,)))
      self.layers.append(nn.ReLU())     
        
    #This line defines the data shape that fully-connected layer receives.
    current_width, current_height = self.get_current_data_shape()

    self.layers = self.layers.to(device)

    #This line builds the fully-connected layer
    self.classifier = nn.Sequential(nn.Linear(current_width*current_height, current_width*current_height//5),
                                    nn.Tanh(), nn.Linear(current_width*current_height//5, n_classes)).to(device)

  def get_current_data_shape(self):
    _, width, height = self.input_shape
    temp_layers = nn.Sequential(*self.layers)

    input_tensor = torch.rand(1, width, height)
    #print(input_tensor.shape)
    #print(temp_layers)
    _, output_width, output_height = temp_layers(input_tensor).shape
    return output_width, output_height
        
  def forward(self, x):
    for layer in self.layers:
      x = layer(x)
    x = x.view(x.size(0), -1)
    output = self.classifier(x)
    return output




class Early_Exit_DNN(nn.Module):
  def __init__(self, model_name: str, n_classes: int, n_branches: int, 
               exit_type: str, device, threshold, distribution="predefined", exit_positions=None):
    super(Early_Exit_DNN, self).__init__()

    self.model_name = model_name
    self.n_classes = n_classes
    self.n_branches = n_branches
    self.exit_type = exit_type
    self.distribution = distribution
    self.device = device
    self.exit_positions = exit_positions
    self.threshold = threshold
    #input_ dim = torch.Size([batch_size, 2, 20000])

    build_early_exit_dnn = self.select_dnn_model()
    build_early_exit_dnn()


  def select_dnn_model(self):
    """
    This method selects the backbone to insert the early exits.
    """

    architecture_dnn_model_dict = {"cnn": self.early_exit_cnn}
    return architecture_dnn_model_dict.get(self.model_name, self.invalid_model)

  def invalid_model(self):
    raise Exception("This DNN model has not implemented yet.")


  def countFlops(self, model):
    input_data = torch.rand(1, 3, self.input_dim, self.input_dim).to(self.device)
    flops, all_data = count_ops(model, input_data, print_readable=False, verbose=False)
    return flops

  def add_exit_block(self):

    self.stages.append(nn.Sequential(*self.layers))
    x = torch.rand(1, 2, 20000).to(self.device)
    feature_shape = nn.Sequential(*self.stages)(x).shape
    #print(self.stages)
    #print(feature_shape)
    #sys.exit()
    self.exits.append(EarlyExitBlock(feature_shape, 5, self.n_classes, self.exit_type, self.device))#.to(self.device))
    self.layers = nn.ModuleList()
    self.stage_id += 1    

  def early_exit_cnn(self):
    """
    This method inserts early exits into a Mobilenet V2 model
    """

    self.stages = nn.ModuleList()
    self.exits = nn.ModuleList()
    self.layers = nn.ModuleList()
    self.cost = []
    self.stage_id = 0
    
    backbone_model = cnn_model.ConvModel().to(self.device)

    for n_layer, layer in enumerate(backbone_model.conv_layers.children()):
      self.layers.append(layer)
      if (n_layer in self.exit_positions):
        self.add_exit_block()

    self.stages.append(nn.Sequential(*self.layers))
    
    self.classifier = nn.Sequential(*backbone_model.line_layers)

    self.softmax = nn.Softmax(dim=1)


  def forward(self, x):
    """
    This method runs the DNN model during the training phase.
    x (tensor): input image
    """

    output_list, conf_list, class_list  = [], [], []

    for i, exitBlock in enumerate(self.exits):

      #This line process a DNN backbone until the (i+1)-th side branch (early-exit)
      x = self.stages[i](x)

      #This runs the early-exit classifications (prediction)
      output_branch = exitBlock(x)
			
      #This obtains the classification and confidence value in each side branch
      #Confidence is the maximum probability of belongs one of the predefined classes
      #The prediction , a.k.a inference_class,  is the argmax output. 
      conf_branch, prediction = torch.max(self.softmax(output_branch), 1)

      threshold_met = conf_branch > self.threshold

      if(threshold_met.item()):
      	return output_branch, conf_branch, prediction

      #This apprends the gathered confidences and classifications into a list
      output_list.append(output_branch), conf_list.append(conf_branch), class_list.append(prediction)

    #This executes the last piece of DNN backbone
    x = self.stages[-1](x)

    x = torch.flatten(x, 1)

    #This generates the last-layer classification
    output = self.classifier(x)
    infered_conf, infered_class = torch.max(self.softmax(output), 1)

    output_list.append(output), conf_list.append(infered_conf), class_list.append(infered_class)

    return output, infered_conf, infered_class



  def forwardTraining(self, x):
    """
    This method runs the DNN model during the training phase.
    x (tensor): input image
    """

    output_list, conf_list, class_list  = [], [], []

    for i, exitBlock in enumerate(self.exits):

      #This line process a DNN backbone until the (i+1)-th side branch (early-exit)
      x = self.stages[i](x)

      #This runs the early-exit classifications (prediction)
      output_branch = exitBlock(x)
			
      #This obtains the classification and confidence value in each side branch
      #Confidence is the maximum probability of belongs one of the predefined classes
      #The prediction , a.k.a inference_class,  is the argmax output. 
      conf_branch, prediction = torch.max(self.softmax(output_branch), 1)

      #This apprends the gathered confidences and classifications into a list
      output_list.append(output_branch), conf_list.append(conf_branch), class_list.append(prediction)

    #This executes the last piece of DNN backbone
    x = self.stages[-1](x)

    x = torch.flatten(x, 1)

    #This generates the last-layer classification
    output = self.classifier(x)
    infered_conf, infered_class = torch.max(self.softmax(output), 1)

    output_list.append(output), conf_list.append(infered_conf), class_list.append(infered_class)

    return output_list, conf_list, class_list





  def forwardEval(self, x):
    """
    This method runs the DNN model during the training phase.
    x (tensor): input image
    """

    output_list, conf_list, class_list, inf_time_list  = [], [], [], []
    flops_branch_list, total_flops_list = [], []
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    cumulative_inf_time = 0.0

    x_input = x

    ee_branch = nn.ModuleList()

    for i, exitBlock in enumerate(self.exits):

      ee_branch.append(self.stages[i])
      ee_branch.append(exitBlock)

      flops_branch, _ = count_ops(nn.Sequential(*ee_branch), x_input, print_readable=False, verbose=False)

      flops_backbone, _ = count_ops(self.stages[i], x, print_readable=False, verbose=False)

      x_exit = self.stages[i](x)

      flops_exit, _ = count_ops(exitBlock, x_exit, print_readable=False, verbose=False)

      flops_branch_list.append(flops_branch), total_flops_list.append(flops_backbone+flops_exit)

      del ee_branch[-1]

      #This lines starts a timer to measure processing time
      starter.record()

      #This line process a DNN backbone until the (i+1)-th side branch (early-exit)
      x = self.stages[i](x)

      #This runs the early-exit classifications (prediction)
      output_branch = exitBlock(x)

      #This obtains the classification and confidence value in each side branch
      #Confidence is the maximum probability of belongs one of the predefined classes
      #The prediction , a.k.a inference_class,  is the argmax output. 
      conf_branch, prediction = torch.max(self.softmax(output_branch), 1)

      #This line terminates the timer started previously.
      ender.record()
      torch.cuda.synchronize()
      curr_time = starter.elapsed_time(ender)

      #This apprends the gathered confidences and classifications into a list
      output_list.append(output_branch), conf_list.append(conf_branch.item()), class_list.append(prediction), inf_time_list.append(curr_time)


    ee_branch.append(self.stages[-1])
    ee_branch.append(nn.Flatten())
    ee_branch.append(self.classifier)

    flops_branch, _ = count_ops(nn.Sequential(*ee_branch), x_input, print_readable=False, verbose=False)
    flops_backbone, _ = count_ops(self.stages[-1], x, print_readable=False, verbose=False)

    x_exit = self.stages[-1](x)

    x_exit = torch.flatten(x_exit, 1)

    flops_exit, _ = count_ops(self.classifier, x_exit, print_readable=False, verbose=False)

    flops_branch_list.append(flops_branch), total_flops_list.append(flops_backbone+flops_exit)


    #This measures the processing time for the last piece of DNN backbone
    starter.record()

    #This executes the last piece of DNN backbone
    x = self.stages[-1](x)

    x = torch.flatten(x, 1)

    #This generates the last-layer classification
    output = self.classifier(x)
    infered_conf, infered_class = torch.max(self.softmax(output), 1)


    #This ends the timer
    ender.record()
    torch.cuda.synchronize()
    curr_time = starter.elapsed_time(ender)

    output_list.append(output), conf_list.append(infered_conf.item()), class_list.append(infered_class), inf_time_list.append(curr_time)

    cumulative_inf_time_list = np.cumsum(inf_time_list)

    cumulative_total_flops_list = np.cumsum(total_flops_list)

    return conf_list, class_list, inf_time_list, cumulative_inf_time_list, flops_branch_list, cumulative_total_flops_list




  def forwardFlops(self, x):
    """
    This method runs the DNN model during the training phase.
    x (tensor): input image
    """

    output_list, conf_list, class_list, inf_time_list  = [], [], [], []
    flops = 0
    flops_list = []

    for i, exitBlock in enumerate(self.exits):

      #This line process a DNN backbone until the (i+1)-th side branch (early-exit)

      flops += count_ops(self.stages[i], x, print_readable=False, verbose=False)

      x = self.stages[i](x)


      #This runs the early-exit classifications (prediction)
      output_branch = exitBlock(x)
      flops += count_ops(exitBlock, x, print_readable=False, verbose=False)

      flops_list.append(flops)

      #This obtains the classification and confidence value in each side branch
      #Confidence is the maximum probability of belongs one of the predefined classes
      #The prediction , a.k.a inference_class,  is the argmax output. 
      conf_branch, prediction = torch.max(self.softmax(output_branch), 1)

      #This apprends the gathered confidences and classifications into a list
      output_list.append(output_branch), conf_list.append(conf_branch.item()), class_list.append(prediction)


    flops += count_ops(self.stages[-1], x, print_readable=False, verbose=False)


    #This executes the last piece of DNN backbone
    x = self.stages[-1](x)
    



    x = torch.flatten(x, 1)

    #This generates the last-layer classification
    output = self.classifier(x)
    flops += count_ops(self.classifier, x, print_readable=False, verbose=False)

    flops_list.append(flops)

    infered_conf, infered_class = torch.max(self.softmax(output), 1)

    output_list.append(output), conf_list.append(infered_conf.item()), class_list.append(infered_class), inf_time_list.append(curr_time)

    return flops_list
