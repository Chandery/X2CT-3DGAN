# ------------------------------------------------------------------------------
# Copyright (c) Tencent
# Licensed under the GPLv3 License.
# Created by Kai Ma (makai0324@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
from torch.utils.data import Subset

def collate_gan(batch):
  '''
  :param batch: [imgs, boxes, labels] dtype = np.ndarray
  imgs:
    shape = (C H W)
  :return:
  '''
  ct = [x[0] for x in batch]
  xray = [x[1] for x in batch]
  file_path = [x[2] for x in batch]

  return torch.stack(ct), torch.stack(xray), file_path

def collate_gan_views(batch):
  '''
  :param batch: [imgs, boxes, labels] dtype = np.ndarray
  imgs:
    shape = (C H W)
  :return:
  '''
  # ct = [x[0] for x in batch]
  # xray1 = [x[1] for x in batch]
  # xray2 = [x[2] for x in batch]
  # file_path = [x[3] for x in batch]

  ct = [x['image'] for x in batch]
  xray1 = [x['cond1'] for x in batch]
  xray2 = [x['cond2'] for x in batch]
  file_path = [x['file_path'] for x in batch]

  ct = torch.stack(ct)
  xray1 = torch.stack(xray1)
  xray2 = torch.stack(xray2)

  if(ct.dim() == 5):
    ct = ct.squeeze(1)
  # if(xray1.dim() == 4):
    # xray1 = xray1.squeeze(1)
  # if(xray2.dim() == 4):
    # xray2 = xray2.squeeze(1)

  xray = [xray1, xray2]
  # print("ct.shape=", ct.shape)
  # print("xray[0]、[1].shape=",xray[0].shape, xray[1].shape)
  # // return torch.stack(ct), [torch.stack(xray1), torch.stack(xray2)]
  return ct, xray, file_path