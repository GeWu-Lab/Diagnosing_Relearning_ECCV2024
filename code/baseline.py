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
from tqdm import tqdm
import argparse
import os
from sklearn.metrics import f1_score
import pickle
from operator import mod



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


    return parser.parse_args()



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


    if args.dataset == 'CREMAD':
        train_dataset = AVDataset_CD(mode='train')
        test_dataset = AVDataset_CD(mode='test')

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=16, pin_memory=False)

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=16)

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(
            model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08)


    if args.train:
        best_acc = -1

        for epoch in range(args.epochs):

            print('Epoch: {}: '.format(epoch))


            batch_loss = train_epoch(
                    args, epoch, model, device, train_dataloader, optimizer)
                
            acc = valid(args, model, device, test_dataloader)

            if acc > best_acc:
                best_acc = float(acc)

                if not os.path.exists(args.ckpt_path):
                    os.mkdir(args.ckpt_path)

                model_name = 'best_model_{}_of_{}_{}_epoch{}_batch{}_lr{}.pth'.format(
                    args.model, args.optimizer,  args.dataset, args.epochs, args.batch_size, args.learning_rate)

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




if __name__ == "__main__":
    main()
