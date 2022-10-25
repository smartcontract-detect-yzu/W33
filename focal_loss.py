####################################################
##### This is focal loss class for multi class #####
####################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# I refered https://github.com/c0nn3r/RetinaNet/blob/master/focal_loss.py

class FocalLoss2d(nn.modules.loss._WeightedLoss):

    def __init__(self, gamma=2, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean', balance_param=0.25, class_size=2):
        super(FocalLoss2d, self).__init__(weight, size_average, reduce, reduction)
        self.gamma = gamma
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.balance_param = balance_param
        
        if weight is None:
            self.weight = Variable(torch.ones(class_size))
        elif isinstance(weight,Variable):
            self.weight = weight
        else:
            self.weight = Variable(weight)

    def forward(self, inputs, targets):
        
        # inputs and targets are assumed to be BatchxClasses
        assert len(inputs.shape) == len(targets.shape)
        assert inputs.size(0) == targets.size(0)
        assert inputs.size(1) == targets.size(1)
           
        # compute the negative likelyhood
        logpt = - F.binary_cross_entropy_with_logits(inputs, targets, pos_weight= self.weight, reduction=self.reduction)
        pt = torch.exp(logpt)

        # compute the loss
        focal_loss = -( (1-pt)**self.gamma ) * logpt
        balanced_focal_loss = self.balance_param * focal_loss
        return balanced_focal_loss