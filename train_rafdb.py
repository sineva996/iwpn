import warnings
warnings.filterwarnings("ignore")
import os
import argparse
from PIL import Image
import numpy as np
import pandas as pd
import torch
import copy
from sklearn.metrics import balanced_accuracy_score
from sklearn import metrics
from net.Network import ResNet18_IWPN as IWPN
import dataset
import Freeze
import Loss
import Fintune as FT
import utils_transform as utils
import Mask_prune as MP
#import plotCM as pt

parser = argparse.ArgumentParser()

parser.add_argument('--raf_path', type=str, default='datasets/RAF-DB/', help='Raf-DB dataset path.')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate for sgd.')
parser.add_argument('--inc_lr', type=float, default=1e-6, help='Initial learning rate for Incre.')
parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers.')
parser.add_argument('--epochs', type=int, default=70, help='Total training epochs.')
parser.add_argument('--inc_epochs', type=int, default=24, help='Total training epochs.')
parser.add_argument("--proc", default=[2, 2, 3], type=list, help="stage-wise numbers of classes")
parser.add_argument("--order", default=[[0,3],[4,6],[1,2,5]], type=list, help="Classes at stage-wise of learning")
parser.add_argument('--frac', default=[1, 1, 1], type=list, help='frac at stage-wise of data.')
parser.add_argument('--Lrate', type=float, default=0.8, help='Initial rate of Aff in Loss.')
parser.add_argument('--prun', default=[0.3, 0.3], type=list, help='frac at stage-wise of prun.')
parser.add_argument('--fix', default=[0.3, 0.3], type=list, help='frac at stage-wise of prun.')
parser.add_argument("--confusion", action="store_true", default=True, help="show confusion matrix")

args = parser.parse_args()

def training():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
    else:
        device = torch.device("cpu")

    model = IWPN()
    model.to(device)

    train_data_transforms = utils.data_transforms()
    test_data_transforms = utils.val_data_transforms()

    train_dataset = dataset.RafDataSet(args.raf_path, phase = 'train', transform = train_data_transforms)    
    test_dataset = dataset.RafDataSet(args.raf_path, phase = 'test', transform = test_data_transforms)   
    
    print('The RAF-DB train set size:', train_dataset.__len__())
    print('The RAF-DB test set size:', test_dataset.__len__())
    
    """ train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size = args.batch_size,
                                               num_workers = args.workers,
                                               shuffle = True,  
                                               pin_memory = True) """
    val_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size = args.batch_size,
                                               num_workers = args.workers,
                                               shuffle = False,  
                                               pin_memory = True)
    
    cro_loss = Loss.cross_loss()
    c_weight = Loss.c_weight()
    cro_loss = torch.nn.CrossEntropyLoss(weight=c_weight).to(device)
    """ c_counts1=torch.from_numpy( np.array([1290,0,0,4772,0,0,0]).astype(np.float32))
    c_counts2=torch.from_numpy( np.array([0,0,0,0,1982,0,2524]).astype(np.float32))
    c_counts3=torch.from_numpy( np.array([0,281,717,0,0,705,0]).astype(np.float32))
    c_mask1=torch.from_numpy( np.array([1,0,0,1,0,0,0]).astype(np.float32))
    c_mask2=torch.from_numpy( np.array([0,0,0,0,1,0,1]).astype(np.float32))
    c_mask3=torch.from_numpy( np.array([0,1,1,0,0,1,0]).astype(np.float32))
    c_weight1 = ((torch.sum(c_counts1)-c_counts1)*c_mask1)/torch.sum(c_counts1)
    c_weight2 = ((torch.sum(c_counts2)-c_counts2)*c_mask2)/torch.sum(c_counts2)
    c_weight3 = ((torch.sum(c_counts3)-c_counts3)*c_mask3)/torch.sum(c_counts3) """
    Affnity =Loss.Affnity(device)
    
    """ params = list(model.parameters()) + list(Affnity.parameters())
    optimizer = torch.optim.SGD(params,lr=args.lr, weight_decay = 1e-4, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1) """

    model=FT.fintune(model,args.epochs,train_dataset,test_dataset,args.lr,args.batch_size,args.workers)
    model.to(device)
    aff_params = list(Affnity.parameters())
    
    #model=Freeze.BN_freeze(model,True,True)

    #best_acc = 0
    for itera in range(len(args.proc)):
        best_acc = 0.0
        print ("\n")
        print("Stage {} of {} stages".format((itera+1), len(args.proc)))
                  
        inc_train_dataset = dataset.Raf_increData(args.raf_path, args.order, itera, args.frac, phase = 'train', transform = train_data_transforms)
        inc_test_dataset = dataset.Raf_increData(args.raf_path,  args.order, itera, args.frac, phase = 'test', transform = test_data_transforms)
        print("The RAF-DB train set size of {} stage: {}".format((itera+1), inc_train_dataset.__len__())) 
        print("The RAF-DB test set size of {} stage: {}".format((itera+1), inc_test_dataset.__len__()))

        inc_train_loader = torch.utils.data.DataLoader(inc_train_dataset,
                                               batch_size = args.batch_size,
                                               num_workers = args.workers,
                                               shuffle = True,  
                                               pin_memory = True)
        """ inc_val_loader = torch.utils.data.DataLoader(inc_test_dataset,
                                               batch_size = args.batch_size,
                                               num_workers = args.workers,
                                               shuffle = False,  
                                               pin_memory = True) """
        #CrossEntropy_weight
        """ if itera==0:
            cro_loss = torch.nn.CrossEntropyLoss(weight=c_weight1).to(device)
        elif itera == 1:
            cro_loss = torch.nn.CrossEntropyLoss(weight=c_weight2).to(device)
        elif itera ==2:
            cro_loss = torch.nn.CrossEntropyLoss(weight=c_weight3).to(device) """
        
        if itera==0:
            feat_params= Freeze.get_feat_params(model,itera)
            fc_params  = Freeze.get_fc_params(model)
            #optimizer = torch.optim.SGD(params,lr=(0.1)*args.inc_lr, weight_decay = 1e-4, momentum=0.9)
            optimizer = torch.optim.SGD([{'params':feat_params,'lr':(args.inc_lr)},{'params':fc_params,'lr':(args.inc_lr)},{'params':aff_params, 'lr':(args.lr)}],lr=args.inc_lr,weight_decay = 1e-4, momentum=0.9)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)
        elif itera >0:
            Differ_freeze  = Freeze.Differ_freeze(model,itera)#冻结
            model = Differ_freeze.getmodel()
            feat_params= Freeze.get_feat_params(model,itera)
            fc_params  = Freeze.get_fc_params(model)
            #optimizer = torch.optim.SGD([{'params':feat_params,'lr':(((0.1)/(itera))*args.inc_lr)},{'params':aff_params},{'params':fc_params, 'lr':(((0.1)*(itera))*args.inc_lr)}],lr=args.inc_lr,weight_decay = 1e-4, momentum=0.9)
            #optimizer = torch.optim.SGD([{'params':feat_params,'lr':(((0.1)/(itera))*args.inc_lr)},{'params':fc_params, 'lr':(((0.1)*(itera))*args.inc_lr)}],lr=args.inc_lr,weight_decay = 1e-4, momentum=0.9)
            optimizer = torch.optim.SGD([{'params':feat_params,'lr':(10*itera*args.inc_lr)},{'params':fc_params,'lr':(4*itera*args.inc_lr)},{'params':aff_params, 'lr':(itera*args.lr)}],lr=(10*itera*args.inc_lr),weight_decay = 1e-4, momentum=0.9)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        init_mask = MP.get_initmask(model) 
        init_mask={key:init_mask[key].cuda() for key in init_mask}
        if itera==0:
            pre_mask = init_mask 

        pre_mask=MP.frozen_mask(pre_mask,model,itera)

        for epoch in range(1, (args.inc_epochs+1)): 
            running_loss = 0.0  
            correct_sum = 0     
            iter_cnt = 0        
            acc_break = 0.0     
            
            model.train()
            model=Freeze.BN_freeze(model,True,True)

            for (imgs, targets) in inc_train_loader:
                #images_train=imgs
                #labels_train=targets
                iter_cnt += 1 
                optimizer.zero_grad()

                imgs = imgs.to(device) 
                #targets = torch.from_numpy(np.asarray(targets))
                targets = targets.to(device)
                out,feat = model(imgs)
                
                loss = args.Lrate * Affnity(feat,targets) +cro_loss(out,targets)
                loss.backward()
                optimizer.step()
                running_loss += loss
                _, predicts = torch.max(out, 1)
                correct_num = torch.eq(predicts, targets).sum()
                correct_sum += correct_num
            
            acc = correct_sum.float() / float(inc_train_dataset.__len__())
            acc_break = acc
            running_loss = running_loss/iter_cnt
            print('Epoch %d : Training accuracy: %.4f. Loss: %.3f. LearningRate %.6f' % (epoch, acc, running_loss,optimizer.param_groups[0]['lr']))
            
            pre_labels = []
            tar_labels = []
            with torch.no_grad():
                running_loss = 0.0
                iter_cnt = 0
                bingo_cnt = 0
                sample_cnt = 0
                baccs = []

                model.eval()
                for (imgs, targets) in val_loader:
                #for (imgs, targets) in inc_train_loader:
                    imgs = imgs.to(device)
                    targets = targets.to(device)
                
                    out,feat = model(imgs)
                    #loss = torch.Tensor([0.0]).float().cuda()
                    loss = args.Lrate * Affnity(feat,targets) +cro_loss(out,targets)
                    #loss = cro_loss(out,targets)

                    running_loss += loss
                    iter_cnt+=1
                    _, predicts = torch.max(out, 1)
                    correct_num  = torch.eq(predicts,targets)
                    bingo_cnt += correct_num.sum().cpu()
                    sample_cnt += out.size(0)
                    pre_labels += predicts.cpu().tolist()
                    tar_labels += targets.cpu().tolist()
                
                    baccs.append(balanced_accuracy_score(targets.cpu().numpy(),predicts.cpu().numpy()))
                
                running_loss = running_loss/iter_cnt   
                scheduler.step()

                acc = bingo_cnt.float()/float(sample_cnt)
                acc = np.around(acc.numpy(),4)
                best_acc = max(acc,best_acc)
                bacc = np.around(np.mean(baccs),4)
                all_acc=Loss.get_allacc(tar_labels,pre_labels)
                print("Epoch %d : Validation accuracy:%.4f. bacc:%.4f. Loss:%.3f" % (epoch, acc, bacc, running_loss))
                print(all_acc)
                print("best_acc:" + str(best_acc))
            
            if itera<2 and (epoch==(args.epochs/2)) :
                aprun=MP.get_all_prunable(model,pre_mask)
                cutf=MP.get_cutoff(aprun,args.prun[itera])
                fixable=MP.get_all_fixable(model,pre_mask)
                fixed=MP.get_fixed(fixable,args.fix[itera])
                model,pre_mask=MP.mask_prune(model,pre_mask,cutf,fixed)

            model=MP.zero_fixed_gradient(model,pre_mask)
            model=MP.zero_prun_gradient(model,pre_mask)
            if itera==0:
                if (acc_break > 0.90) and (epoch>8):
                    break
            if itera==1:
                if (acc_break > 0.90) and (epoch>8):
                    break
    """The Key of the mask:
        Conv1Weight
        Layer1.0.Conv1Weight
        Layer1.0.Conv2Weight
        Layer1.1.Conv1Weight
        Layer1.1.Conv2Weight
        Layer2.0.Conv1Weight
        Layer2.0.Conv2Weight
        Layer2.0.Downsample.0Weight
        Layer2.1.Conv1Weight
        Layer2.1.Conv2Weight
        Layer3.0.Conv1Weight
        Layer3.0.Conv2Weight
        Layer3.0.Downsample.0Weight
        Layer3.1.Conv1Weight
        Layer3.1.Conv2Weight
        Layer4.0.Conv1Weight
        Layer4.0.Conv2Weight
        Layer4.0.Downsample.0Weight
        Layer4.1.Conv1Weight
        Layer4.1.Conv2Weight
        Fcweight"""       

    pre_labels = []
    tar_labels = []
    with torch.no_grad():
        iter_cnt = 0
        bingo_cnt = 0
        sample_cnt = 0
        baccs = []
        model.eval()
        for (imgs, targets) in val_loader:
            #imgs.shape : torch.Size([epoch, 3, 224, 224])
            #targets.shape : torch.Size([epoch])
            imgs = imgs.to(device)
            targets = targets.to(device)
                
            out,feat = model(imgs)
            #loss = torch.Tensor([0.0]).float().cuda()
            loss = cro_loss(out,targets) + args.Lrate * Affnity(feat,targets)
            #loss = cro_loss(out,targets)

            iter_cnt+=1
            _, predicts = torch.max(out, 1)
            correct_num  = torch.eq(predicts,targets)
            bingo_cnt += correct_num.sum().cpu()
            sample_cnt += out.size(0)

            pre_labels += predicts.cpu().tolist()
            tar_labels += targets.cpu().tolist()
            baccs.append(balanced_accuracy_score(targets.cpu().numpy(),predicts.cpu().numpy()))
            

        acc = bingo_cnt.float()/float(sample_cnt)
        acc = np.around(acc.numpy(),4)
        all_acc=Loss.get_allacc(tar_labels,pre_labels)
        bacc = np.around(np.mean(baccs),4)
        print("The last Validation accuracy:%.4f. bacc:%.4f." % (acc, bacc))
        print(all_acc)

        if args.confusion:
            print('Confusion matrix:')
            print(metrics.confusion_matrix(tar_labels,pre_labels))  
    
    torch.save({'model_state_dict': model.state_dict(),},
        os.path.join('checkpoints', "rafdb"+".pth"))
    print('Model saved.')

if __name__ == "__main__":        
    training()