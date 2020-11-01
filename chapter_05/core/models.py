import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias, 1e-2)

class LossFn:
    def __init__(self, cls_factor=1, box_factor=1, landmark_factor=1):
        # loss function
        self.cls_factor = cls_factor
        self.box_factor = box_factor
        self.land_factor = landmark_factor
        self.loss_cls = nn.BCELoss() # binary cross entropy
        self.loss_box = nn.MSELoss(reduce=True, reduction='mean') # mean square error
        self.loss_landmark = nn.MSELoss()


    def cls_loss(self,gt_label,pred_label):
        pred_label = torch.squeeze(pred_label)
        gt_label = torch.squeeze(gt_label)
        # get the mask element which >= 0, only 0 and 1 can effect the detection loss
        mask = torch.ge(gt_label,0.0)
        valid_gt_label = torch.masked_select(gt_label,mask)
        valid_pred_label = torch.masked_select(pred_label,mask)
        return self.loss_cls(valid_pred_label,valid_gt_label)*self.cls_factor

    def focal_Loss(self,gt_label,pred_label):#focal_Loss
        # print('focal_Loss ~ ')
        w = 1e-6
        pred_label = torch.squeeze(pred_label)
        gt_label = torch.squeeze(gt_label)
        # get the mask element which >= 0, only 0 and 1 can effect the detection loss
        mask_pos = torch.eq(gt_label,1.)# 正样本  gt_label == 1
        mask_neg = torch.eq(gt_label,0.)# 负样本  gt_label == 0

        pos_pre = torch.masked_select(pred_label,mask_pos)

        neg_pre = torch.masked_select(pred_label,mask_neg)

        pos_loss = -torch.log(pos_pre+w) * torch.pow(1. - pos_pre, 2.) # pos
        neg_loss = -torch.log(1. - neg_pre+w) * torch.pow(neg_pre, 2.) # neg

        alfa = 0.25
        loss = alfa * torch.mean(pos_loss)+(1.-alfa)*torch.mean(neg_loss)

        return loss*self.cls_factor


    # def box_loss(self,gt_label,gt_offset,pred_offset):
    #     pred_offset = torch.squeeze(pred_offset)
    #     gt_offset = torch.squeeze(gt_offset)
    #     gt_label = torch.squeeze(gt_label)
    #
    #     #get the mask element which != 0
    #     mask = torch.ne(gt_label,0.0)
    #     #convert mask to dim index
    #     chose_index = torch.nonzero(mask.data)
    #     chose_index = torch.squeeze(chose_index)
    #     #only valid element can effect the loss
    #     valid_gt_offset = gt_offset[chose_index,:]
    #     valid_pred_offset = pred_offset[chose_index,:]
    #     return self.loss_box(valid_pred_offset,valid_gt_offset)*self.box_factor

    def box_loss(self,gt_label,gt_offset,pred_offset):
        pred_offset = torch.squeeze(pred_offset)
        gt_offset = torch.squeeze(gt_offset)
        gt_label = torch.squeeze(gt_label)

        #get the mask element which != 0
        unmask = torch.eq(gt_label,0)
        mask = torch.eq(unmask,0)
        #convert mask to dim index
        chose_index = torch.nonzero(mask.data)
        chose_index = torch.squeeze(chose_index)
        #only valid element can effect the loss
        valid_gt_offset = gt_offset[chose_index,:]
        valid_pred_offset = pred_offset[chose_index,:]

        loss = torch.abs(valid_pred_offset-valid_gt_offset)
    
        loss = torch.sum(loss,dim = 1)
      
        loss = torch.mean(loss)*self.box_factor
        return loss


    def landmark_loss(self,gt_label,gt_landmark,pred_landmark):
        pred_landmark = torch.squeeze(pred_landmark)
        gt_landmark = torch.squeeze(gt_landmark)
        gt_label = torch.squeeze(gt_label)
        mask = torch.eq(gt_label,-2)

        chose_index = torch.nonzero(mask.data)
        chose_index = torch.squeeze(chose_index)

        valid_gt_landmark = gt_landmark[chose_index, :]
        valid_pred_landmark = pred_landmark[chose_index, :]
        return self.loss_landmark(valid_pred_landmark,valid_gt_landmark)*self.land_factor





class PNet(nn.Module):
    ''' PNet '''

    def __init__(self):
        super(PNet, self).__init__()

        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=3, stride=1),  # conv1
            nn.BatchNorm2d(12,affine=True),
            nn.PReLU(num_parameters = 12),  # PReLU1
            nn.MaxPool2d(kernel_size=2, stride=2),  # pool1
            nn.Conv2d(12, 24, kernel_size=3, stride=1),  # conv2
            nn.BatchNorm2d(24,affine=True),
            nn.PReLU(num_parameters = 24),  # PReLU2
            nn.Conv2d(24, 48, kernel_size=3, stride=1),  # conv3
            nn.BatchNorm2d(48,affine=True),
            nn.PReLU(num_parameters = 48)  # PReLU3
        )

        # detection
        self.conv4_1 = nn.Conv2d(48, 1, kernel_size=1, stride=1)
        # bounding box regresion
        self.conv4_2 = nn.Conv2d(48, 4, kernel_size=1, stride=1)
        # weight initiation with xavier
        self.apply(weights_init)

    def forward(self, x):
        x = (x - 127.5) / 128.
        x = self.pre_layer(x)
        label = torch.sigmoid(self.conv4_1(x))
        offset = self.conv4_2(x)
        return label, offset



class RNet(nn.Module):
    ''' RNet '''

    def __init__(self,is_train=False, use_cuda=True):
        super(RNet, self).__init__()
        self.is_train = is_train
        self.use_cuda = use_cuda
        # backend
        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, stride=1),  # conv1
            nn.BatchNorm2d(24,affine=True),
            nn.PReLU(num_parameters = 24),  # prelu1
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool1
            nn.Conv2d(24, 48, kernel_size=3, stride=1),  # conv2
            nn.BatchNorm2d(48,affine=True),
            nn.PReLU(num_parameters = 48),  # prelu2
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool2
            nn.Conv2d(48, 64, kernel_size=2, stride=1),  # conv3
            nn.BatchNorm2d(64,affine=True),
            nn.PReLU(num_parameters = 64)  # prelu3

        )
        self.conv4 = nn.Linear(64*2*2, 128)  # conv4
        self.prelu4 = nn.PReLU()  # prelu4
        self.conv5_1 = nn.Linear(128, 1)
        self.conv5_2 = nn.Linear(128, 4)
        self.apply(weights_init)

    def forward(self, x):
        # backend
        x = (x - 127.5) / 128.
        x = self.pre_layer(x)
        x = x.view(x.size(0), -1)
        x = self.conv4(x)
        x = self.prelu4(x)
        # detection
        label = torch.sigmoid(self.conv5_1(x))
        offset = self.conv5_2(x)

        return label, offset




class ONet(nn.Module):
    ''' ONet '''

    def __init__(self,is_train=False, use_cuda=True):
        super(ONet, self).__init__()
        self.is_train = is_train
        self.use_cuda = use_cuda
        # backend
        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1),  # conv1
            nn.BatchNorm2d(32,affine=True),
            nn.PReLU(num_parameters = 32),  # prelu1
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool1
            nn.Conv2d(32, 64, kernel_size=3, stride=1),  # conv2
            nn.BatchNorm2d(64,affine=True),
            nn.PReLU(num_parameters = 64),  # prelu2
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool2
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # conv3
            nn.BatchNorm2d(64,affine=True),
            nn.PReLU(num_parameters = 64), # prelu3
            nn.MaxPool2d(kernel_size=2,stride=2), # pool3
            nn.Conv2d(64,128,kernel_size=2,stride=1), # conv4
            nn.BatchNorm2d(128,affine=True),
            nn.PReLU(num_parameters = 128) # prelu4
        )
        self.conv5 = nn.Linear(128*2*2, 256)  # conv5
        self.prelu5 = nn.PReLU(num_parameters = 256)  # prelu5
        # detection
        self.conv6_1 = nn.Linear(256, 1)
        # bounding box regression
        self.conv6_2 = nn.Linear(256, 4)
        # lanbmark localization
        # self.conv6_3 = nn.Linear(256, 10)
        # weight initiation weih xavier
        self.apply(weights_init)

    def forward(self, x):
        # backend
        x = (x - 127.5) / 128.
        x = self.pre_layer(x)
        x = x.view(x.size(0), -1)
        x = self.conv5(x)
        x = self.prelu5(x)
        # detection
        label = torch.sigmoid(self.conv6_1(x))
        offset = self.conv6_2(x)

        return label,offset
