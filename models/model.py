from __future__ import division
import torch
from torch import nn
from models import resnext,shufflenetv2hs,shufflevit,ghostnet
import pdb

# def generate_model():
#     model = ghostnet.get_model(
#         num_classes=101,
#         width=1.0)
#     model = model.cuda()
#     model = nn.DataParallel(model)
#     print(model)
#     return model

# def generate_model():
#     model = shufflevit.get_model(
#         num_classes=101,
#         sample_size=224)

#     model = model.cuda()
#     model = nn.DataParallel(model)
#     #print(model)
#     return model

def generate_model():
    model = shufflenetv2hs.get_model(
        num_classes=101,
        sample_size=224,
        width_mult=1.)
    model = model.cuda()
    model = nn.DataParallel(model)
    print(model)
    return model

if __name__ == '__main__':
    mm = generate_model()


# def generate_model():

#     from models.resnext import get_fine_tuning_parameters
#     model = resnext.resnet101(
#             num_classes=51,
#             shortcut_type='B',
#             cardinality=32,
#             sample_size=112,
#             sample_duration=64,
#             input_channels=3)
    

#     model = model.cuda()
#     model = nn.DataParallel(model)
#     return model

