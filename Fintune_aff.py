import torch
import Loss
import numpy as np

def fintune(model,epoch,train_dataset,test_dataset,lr1,batch_size,workers):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
    else:
        device = torch.device("cpu")
    cro_loss = Loss.cross_loss().to(device)
    train1_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size = batch_size,
                                               num_workers = workers,
                                               shuffle = True,  
                                               pin_memory = True)
    val1_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size = batch_size,
                                               num_workers = workers,
                                               shuffle = False,  
                                               pin_memory = True)
    params = list(model.parameters())
    #optimizer = torch.optim.SGD(params,lr=lr1, weight_decay = 1e-4, momentum=0.9)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.8)
    optimizer = torch.optim.Adam(params,lr=lr1, weight_decay = 0)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)
    
    for epoch in range(1,epoch+ 1): 
        running_loss = 0.0  
        correct_sum = 0     
        iter_cnt = 0        

        model.train()
        for (imgs, targets) in train1_loader:
            iter_cnt += 1
            optimizer.zero_grad()

            imgs = imgs.to(device)
            targets = targets.to(device)
            out,feat = model(imgs)
            loss = cro_loss(out,targets)
            loss.backward()
            optimizer.step()
        
            running_loss += loss
            _, predicts = torch.max(out, 1)
            correct_num = torch.eq(predicts, targets).sum()
            correct_sum += correct_num
        acc = correct_sum.float() / float(train_dataset.__len__())
        running_loss = running_loss/iter_cnt
        print('FT_Epoch %d : Train_acc: %.4f Loss: %.3f LR %.6f' % (epoch, acc, running_loss,optimizer.param_groups[0]['lr']))
        
        pre_labels = []
        tar_labels = []
        with torch.no_grad():
            iter_cnt = 0
            bingo_cnt = 0
            model.eval()
            for (imgs, targets) in val1_loader:
                imgs = imgs.to(device)
                targets = targets.to(device)
                    
                out,feat = model(imgs)
                loss = cro_loss(out,targets)

                iter_cnt+=1
                _, predicts = torch.max(out, 1)
                correct_num  = torch.eq(predicts,targets).sum()
                bingo_cnt += correct_num
                pre_labels += predicts.cpu().tolist()
                tar_labels += targets.cpu().tolist()

            acc = bingo_cnt.float()/float(test_dataset.__len__())
            print('FT_Epoch %d : Test_acc: %.4f' % (epoch, acc))
        
        scheduler.step()
    
    return model
def print_parameter_grad_info(net):
    print('-------parameters requires grad info--------')
    for name, p in net.named_parameters():
        print(f'{name}:\t{p.requires_grad}')

def print_net_state_dict(net):
    for key, v in net.state_dict().items():
        print(f'{key}')