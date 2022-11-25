#encoding:utf-8
#
#created by xiongzihua 2017.12.26
#
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def detection_loss_4_yolo(output, target, device, prune):
    from utilities.utils import one_hot

    # hyper parameter

    lambda_coord = 5
    lambda_noobj = 0.5

    # check batch size
    b, _, _, _ = target.shape
    _, _, _, n = output.shape

    # output tensor slice
    # output tensor shape is [batch, 7, 7, 5 + classes]
    objness1_output = output[:, :, :, 0]
    x_offset1_output = output[:, :, :, 1]
    y_offset1_output = output[:, :, :, 2]
    width_ratio1_output = output[:, :, :, 3]
    height_ratio1_output = output[:, :, :, 4]
    class_output = output[:, :, :, 5:]

    num_cls = class_output.shape[-1]

    # label tensor slice
    objness_label = target[:, :, :, 0]
    x_offset_label = target[:, :, :, 1]
    y_offset_label = target[:, :, :, 2]
    width_ratio_label = target[:, :, :, 3]
    height_ratio_label = target[:, :, :, 4]
    #print("5:",target[:,:,:,5])
    if prune == False:
        class_label = one_hot(class_output, target[:, :, :, 5], device)
    else:
        class_label = target[: ,: ,: ,5:]

    noobjness_label = torch.neg(torch.add(objness_label, -1))

    obj_coord1_loss = lambda_coord * \
                      torch.sum(objness_label *
                        (torch.pow(x_offset1_output - x_offset_label, 2) +
                                    torch.pow(y_offset1_output - y_offset_label, 2)))
    #print("CHECKING!!!!\n",width_ratio1_output,width_ratio_label,height_ratio1_output,height_ratio_label)
    #print("CHECKING!!!!\n",height_ratio_label,torch.sqrt(height_ratio_label))
    if prune == False:
        obj_size1_loss = lambda_coord * \
                         torch.sum(objness_label *
                                   (torch.pow(width_ratio1_output - torch.sqrt(width_ratio_label), 2) +
                                    torch.pow(height_ratio1_output - torch.sqrt(height_ratio_label), 2)))
    else:
        obj_size1_loss = lambda_coord * \
                         torch.sum(objness_label *
                                   (torch.pow(width_ratio1_output - width_ratio_label, 2) +
                                    torch.pow(height_ratio1_output - height_ratio_label, 2)))

    objectness_cls_map = objness_label.unsqueeze(-1)

    for i in range(num_cls - 1):
        objectness_cls_map = torch.cat((objectness_cls_map, objness_label.unsqueeze(-1)), 3)

    obj_class_loss = torch.sum(objectness_cls_map * torch.pow(class_output - class_label, 2))

    noobjness1_loss = lambda_noobj * torch.sum(noobjness_label * torch.pow(objness1_output - objness_label, 2))
    objness1_loss = torch.sum(objness_label * torch.pow(objness1_output - objness_label, 2))

    total_loss = (obj_coord1_loss + obj_size1_loss + noobjness1_loss + objness1_loss + obj_class_loss)
    total_loss = total_loss / b

    return total_loss, obj_coord1_loss / b, obj_size1_loss / b, obj_class_loss / b, noobjness1_loss / b, objness1_loss / b

class yoloLoss(nn.Module):
    def __init__(self,S,B,l_coord,l_noobj):
        super(yoloLoss,self).__init__()
        self.S = S
        self.B = B
        self.l_coord = l_coord
        self.l_noobj = l_noobj

    def compute_iou(self, box1, box2):
        '''Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
        Args:
          box1: (tensor) bounding boxes, sized [N,4].
          box2: (tensor) bounding boxes, sized [M,4].
        Return:
          (tensor) iou, sized [N,M].
        '''
        N = box1.size(0)
        M = box2.size(0)

        lt = torch.max(
            box1[:,:2].unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:,:2].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        rb = torch.min(
            box1[:,2:].unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:,2:].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        wh = rb - lt  # [N,M,2]
        wh[wh<0] = 0  # clip at 0
        inter = wh[:,:,0] * wh[:,:,1]  # [N,M]

        area1 = (box1[:,2]-box1[:,0]) * (box1[:,3]-box1[:,1])  # [N,]
        area2 = (box2[:,2]-box2[:,0]) * (box2[:,3]-box2[:,1])  # [M,]
        area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
        area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

        iou = inter / (area1 + area2 - inter)
        return iou
    def forward(self,pred_tensor,target_tensor):
        '''
        pred_tensor: (tensor) size(batchsize,S,S,Bx5+20=30) [x,y,w,h,c]
        target_tensor: (tensor) size(batchsize,S,S,30)
        '''
        N = pred_tensor.size()[0]
        coo_mask = target_tensor[:,:,:,4] > 0
        noo_mask = target_tensor[:,:,:,4] == 0
        coo_mask = coo_mask.unsqueeze(-1).expand_as(target_tensor)
        noo_mask = noo_mask.unsqueeze(-1).expand_as(target_tensor)

        coo_pred = pred_tensor[coo_mask].view(-1,30)
        box_pred = coo_pred[:,:10].contiguous().view(-1,5) #box[x1,y1,w1,h1,c1]
        class_pred = coo_pred[:,10:]                       #[x2,y2,w2,h2,c2]
        
        coo_target = target_tensor[coo_mask].view(-1,30)
        box_target = coo_target[:,:10].contiguous().view(-1,5)
        class_target = coo_target[:,10:]

        # compute not contain obj loss
        noo_pred = pred_tensor[noo_mask].view(-1,30)
        noo_target = target_tensor[noo_mask].view(-1,30)
        noo_pred_mask = torch.cuda.BoolTensor(noo_pred.size())
        noo_pred_mask.zero_()
        noo_pred_mask[:,4]=1
        noo_pred_mask[:,9]=1
        noo_pred_c = noo_pred[noo_pred_mask] #noo pred只需要计算 c 的损失 size[-1,2]
        noo_target_c = noo_target[noo_pred_mask]
        nooobj_loss = F.mse_loss(noo_pred_c,noo_target_c,size_average=False)

        #compute contain obj loss
        coo_response_mask = torch.cuda.BoolTensor(box_target.size())
        coo_response_mask.zero_()
        coo_not_response_mask = torch.cuda.BoolTensor(box_target.size())
        coo_not_response_mask.zero_()
        box_target_iou = torch.zeros(box_target.size()).cuda()
        from tqdm import tqdm
        for i in tqdm(range(0,box_target.size()[0],2)): #choose the best iou box
            box1 = box_pred[i:i+2]
            box1_xyxy = Variable(torch.FloatTensor(box1.size()))
            box1_xyxy[:,:2] = box1[:,:2]/14. -0.5*box1[:,2:4]
            box1_xyxy[:,2:4] = box1[:,:2]/14. +0.5*box1[:,2:4]
            box2 = box_target[i].view(-1,5)
            box2_xyxy = Variable(torch.FloatTensor(box2.size()))
            box2_xyxy[:,:2] = box2[:,:2]/14. -0.5*box2[:,2:4]
            box2_xyxy[:,2:4] = box2[:,:2]/14. +0.5*box2[:,2:4]
            iou = self.compute_iou(box1_xyxy[:,:4],box2_xyxy[:,:4]) #[2,1]
            max_iou,max_index = iou.max(0)
            max_index = max_index.data.cuda()
            
            coo_response_mask[i+max_index]=1
            coo_not_response_mask[i+1-max_index]=1

            #####
            # we want the confidence score to equal the
            # intersection over union (IOU) between the predicted box
            # and the ground truth
            #####
            box_target_iou[i+max_index,torch.LongTensor([4]).cuda()] = (max_iou).data.cuda()
        box_target_iou = Variable(box_target_iou).cuda()
        #1.response loss
        box_pred_response = box_pred[coo_response_mask].view(-1,5)
        box_target_response_iou = box_target_iou[coo_response_mask].view(-1,5)
        box_target_response = box_target[coo_response_mask].view(-1,5)
        contain_loss = F.mse_loss(box_pred_response[:,4],box_target_response_iou[:,4],size_average=False)
        loc_loss = F.mse_loss(box_pred_response[:,:2],box_target_response[:,:2],size_average=False) + F.mse_loss(torch.sqrt(box_pred_response[:,2:4]),torch.sqrt(box_target_response[:,2:4]),size_average=False)
        #2.not response loss
        box_pred_not_response = box_pred[coo_not_response_mask].view(-1,5)
        box_target_not_response = box_target[coo_not_response_mask].view(-1,5)
        box_target_not_response[:,4]= 0
        #not_contain_loss = F.mse_loss(box_pred_response[:,4],box_target_response[:,4],size_average=False)
        
        #I believe this bug is simply a typo
        not_contain_loss = F.mse_loss(box_pred_not_response[:,4], box_target_not_response[:,4],size_average=False)

        #3.class loss
        class_loss = F.mse_loss(class_pred,class_target,size_average=False)

        return (self.l_coord*loc_loss + 2*contain_loss + not_contain_loss + self.l_noobj*nooobj_loss + class_loss)/N




