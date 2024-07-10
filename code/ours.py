from utils.utils import setup_seed
from dataset.av_dataset import AVDataset_CD
import copy
from torch.utils.data import DataLoader
from models.models import AVClassifier
from sklearn import metrics
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from numpy import *
from tqdm import tqdm
from sklearn.metrics import f1_score
import argparse
import os
import pickle
from sklearn.manifold import TSNE
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from operator import mod
import warnings
warnings.filterwarnings('ignore')



def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, type=str,
                        help='KineticSound, CREMAD, K400, VGGSound, Audioset,VGGPart,UCF101')
    parser.add_argument('--model', default='model', type=str)
    parser.add_argument('--n_classes', default=6, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--optimizer', default='sgd',
                        type=str, choices=['sgd', 'adam'])
    parser.add_argument('--learning_rate', default=0.002, type=float, help='initial learning rate')
    parser.add_argument('--lr_decay_step', default=30, type=int, help='where learning rate decays')
    parser.add_argument('--lr_decay_ratio', default=0.1,
                        type=float, help='decay coefficient')
    parser.add_argument('--ckpt_path', default='log_cd',
                        type=str, help='path to save trained models')
    parser.add_argument('--train', action='store_true',
                        help='turn on train mode')
    parser.add_argument('--clip_grad', action='store_true',
                        help='turn on train mode')
    parser.add_argument('--tensorboard_path', default='log_cd',
                        type=str, help='path to save tensorboard logs')
    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--gpu_ids', default='0,1,2,3',
                        type=str, help='GPU ids')
    parser.add_argument('--move_lambda', default=3,
                        type=float, help='move lambda')
    parser.add_argument('--reinit_epoch', default=20, type=int)
    parser.add_argument('--reinit_num', default=3, type=int)


    return parser.parse_args()



def purity_score(y_true, y_pred):
    """Purity score
        Args:
            y_true(np.ndarray): n*1 matrix Ground truth labels
            y_pred(np.ndarray): n*1 matrix Predicted clusters

        Returns:
            float: Purity score
    """
    # matrix which will hold the majority-voted labels
    y_voted_labels = np.zeros(y_true.shape)
    # Ordering labels
    ## Labels might be missing e.g with set like 0,2 where 1 is missing
    ## First find the unique labels, then map the labels to an ordered set
    ## 0,2 should become 0,1
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true==labels[k]] = ordered_labels[k]
    # Update unique labels
    labels = np.unique(y_true)
    # We set the number of bins to be n_classes+2 so that 
    # we count the actual occurence of classes between two consecutive bins
    # the bigger being excluded [bin_i, bin_i+1[
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred==cluster], bins=bins)
        # Find the most present label in the cluster
        winner = np.argmax(hist)
        y_voted_labels[y_pred==cluster] = winner

    return accuracy_score(y_true, y_voted_labels)



def reinit_score(args, train_audio,train_visual,train_label,val_audio,val_visual,val_label):
    all_feature=[train_audio,val_audio,train_visual,val_visual]
    stages=['train_audio','val_audio','train_visual','val_visual']
    all_purity=[]

    for idx,fea in enumerate(all_feature):
        print('Computing t-SNE embedding')
        result = fea
        # 归一化处理
        scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
        result = scaler.fit_transform(result)
        y_pred = KMeans(n_clusters=args.n_classes, random_state=0).fit_predict(result)

        if(stages[idx][:5]=='train'):
            purity=purity_score(np.array(train_label),y_pred)
        else:
            purity=purity_score(np.array(val_label),y_pred)
        all_purity.append(purity)


        print('%s purity= %.4f' % (stages[idx],purity))
        print('%%%%%%%%%%%%%%%%%%%%%%%%') 
    
    purity_gap_audio=np.abs(all_purity[0]-all_purity[1])
    purity_gap_visual=np.abs(all_purity[2]-all_purity[3])


    weight_audio=torch.tanh(torch.tensor(args.move_lambda*purity_gap_audio))
    weight_visual=torch.tanh(torch.tensor(args.move_lambda*purity_gap_visual))

    print('weight audio')
    print(weight_audio)
    print('weight visual')
    print(weight_visual)


    return weight_audio,weight_visual
 


def reinit(args, model,checkpoint,weight_audio,weight_visual):


    print("Start reinit ... ")


    record_names_audio = []
    record_names_visual = []
    for name, param in model.named_parameters():
        if 'audio_net' in name:
            if('conv' in name):
                record_names_audio.append((name, param))
        elif 'visual_net' in name:
            if('conv' in name):
                record_names_visual.append((name, param))


    for name, param in model.named_parameters():
        if 'audio_net' in name:
            init_weight=checkpoint[name]
            current_weight=param.data
            new_weight=weight_audio*init_weight+(1-weight_audio).cuda()*current_weight
            param.data=new_weight
        elif 'visual_net' in name:
            init_weight=checkpoint[name]
            current_weight=param.data
            new_weight=weight_visual*init_weight+(1-weight_visual).cuda()*current_weight
            param.data=new_weight

    
    return model


def train_epoch(args, epoch, model, device, dataloader, optimizer):
    criterion = nn.CrossEntropyLoss()


    model.train()
    print("Start training ... ")

    _loss = 0

    # for name, param in model.named_parameters():
    #     print(name)
    #     print(param.data)
    #     print("requires_grad:", param.requires_grad)
    #     print("-----------------------------------")



    for step, (spec, images, label) in tqdm(enumerate(dataloader)):


        optimizer.zero_grad()
        images = images.to(device)
        spec = spec.to(device)
        label = label.to(device)
        out,_,_,_,_ = model(spec.float(), images.float())

        loss_mm = criterion(out, label)


        loss=loss_mm


        loss.backward()


        optimizer.step()

        _loss += loss.item()



    return _loss / len(dataloader)




def valid(args, model, device, dataloader):

    if args.dataset == 'VGGSound':
        n_classes = 309
    elif args.dataset == 'KineticSound':
        n_classes = 31
    elif args.dataset == 'VGGPart':
        n_classes = 100
    elif args.dataset == 'CREMAD':
        n_classes = 6
    elif args.dataset == 'K400':
        n_classes = 400
    elif args.dataset == 'Audioset':
        n_classes = 527
    elif args.dataset == 'UCF101':
        n_classes = 101

    cri = nn.CrossEntropyLoss()
    _loss = 0

    prob_all = []
    label_all = []

    with torch.no_grad():
        model.eval()
        num = [0.0 for _ in range(n_classes)]
        acc = [0.0 for _ in range(n_classes)]
        all_label = []
        all_out = []

        for step, (spec, images, label) in tqdm(enumerate(dataloader)):


            spec = spec.to(device)
            images = images.to(device)
            label = label.to(device)

            prediction_all = model(spec.float(), images.float())


            _, prob = torch.max(prediction_all[0], 1)
            prob_all.extend(prob.cpu().numpy()) #求每一行的最大值索引
            label_all.extend(label.cpu().numpy())



            prediction=F.softmax(prediction_all[0])

            loss = cri(prediction, label)
            _loss += loss.item()

            for i, item in enumerate(label):

                ma = prediction[i].cpu().data.numpy()
                index_ma = np.argmax(ma)
                # print(index_ma, label_index)
                num[label[i]] += 1.0
                if index_ma == label[i]:
                    acc[label[i]] += 1.0

                
                all_out.append(prediction[i].cpu().data.numpy())
                if args.dataset == 'KineticSound':
                    ss = torch.zeros(31)
                elif args.dataset == 'CREMAD':
                    ss = torch.zeros(6)
                ss[label[i]] = 1
                all_label.append(ss.numpy())


    acc = sum(acc) / sum(num)
    print("F1-Score weighted:{:.4f}".format(f1_score(label_all,prob_all,average='weighted')))
    print("F1-Score macro:{:.4f}".format(f1_score(label_all,prob_all,average='macro')))
    print("Acc:{:.4f}".format(metrics.accuracy_score(label_all,prob_all)))

    print("Acc: {:.4f}".format(acc))


    return acc



def get_feature(args, epoch, model, device, dataloader):
    model.eval()
    all_audio=[]
    all_visual=[]
    all_label=[]


    with torch.no_grad():
        for step, (spec, images, label) in tqdm(enumerate(dataloader)):

            images = images.to(device)
            spec = spec.to(device)
            label = label.to(device)
            _,_,_,a,v = model(spec.float(), images.float())
            all_audio.append(a.data.cpu())
            all_visual.append(v.data.cpu())
            all_label.append(label.data.cpu())
    
    all_audio=torch.cat(all_audio)
    all_visual=torch.cat(all_visual)
    all_label=torch.cat(all_label)



    return all_audio,all_visual,all_label




def main():

    args = get_arguments()
    print(args)

    setup_seed(args.random_seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    gpu_ids = list(range(torch.cuda.device_count()))

    device = torch.device('cuda:0')
    model = AVClassifier(args)
    model.to(device)
    model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    model.cuda()

    torch.save(model.state_dict(), 'init_para.pkl')
    

    PATH='init_para.pkl'
    checkpoint = torch.load(PATH)
    print('get init weight')


    if args.dataset == 'CREMAD':
        train_dataset = AVDataset_CD(mode='my_train')
        test_dataset = AVDataset_CD(mode='test')
        val_dataset=AVDataset_CD(mode='val')

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=16, pin_memory=False)
    
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size,
                                  shuffle=False, num_workers=16, pin_memory=False)

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=16)

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(
            model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08)


    if args.train:
        best_acc = -1
        flag_reinit=0

        for epoch in range(args.epochs):

            print('Epoch: {}: '.format(epoch))


            batch_loss = train_epoch(
                    args, epoch, model, device, train_dataloader, optimizer)
                
            acc = valid(args, model, device, test_dataloader)


            if acc > best_acc:
                best_acc = float(acc)

                if not os.path.exists(args.ckpt_path):
                    os.mkdir(args.ckpt_path)

                model_name = 'best_model_{}_of_{}_{}_epoch{}_batch{}_lr{}_k{}.pth'.format(
                    args.model, args.optimizer,  args.dataset, args.epochs, args.batch_size, args.learning_rate,args.move_lambda)

                saved_dict = {'saved_epoch': epoch,
                                'acc': acc,
                                'model': model.state_dict(),
                                'optimizer': optimizer.state_dict()}

                save_dir = os.path.join(args.ckpt_path, model_name)

                torch.save(saved_dict, save_dir)
                
                print('The best model has been saved at {}.'.format(save_dir))
                print("Loss: {:.4f}, Acc: {:.4f}".format(
                    batch_loss, acc))
            else:
                print("Loss: {:.4f}, Acc: {:.4f},Best Acc: {:.4f}".format(
                    batch_loss, acc,best_acc))

                
            ### reinit
                
    
            if((epoch % args.reinit_epoch == 0)&(epoch>0)):
                flag_reinit+=1
                if(flag_reinit<=args.reinit_num):
                    print('reinit %d' % flag_reinit)
                    print("Start getting training feature ... ")
                    train_audio,train_visual,train_label=get_feature(args, epoch, model, device, train_dataloader)
                    print("Start getting evluating feature ... ")
                    val_audio,val_visual,val_label=get_feature(args, epoch, model, device, val_dataloader)
                    weight_audio,weight_visual= reinit_score(args, train_audio,train_visual,train_label,val_audio,val_visual,val_label)
                    model=reinit(args, model,checkpoint,weight_audio,weight_visual)



if __name__ == "__main__":
    main()
