import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import resnet18


class ConcatFusion(nn.Module):
    def __init__(self, input_dim=1024+512, output_dim=100):
        super(ConcatFusion, self).__init__()
        self.fc_out = nn.Linear(input_dim, output_dim)

    def forward(self, out):
        # output = torch.cat((x, y), dim=1)
        output = self.fc_out(out)
        return output



class RGBClassifier(nn.Module):
    def __init__(self, args):
        super(RGBClassifier, self).__init__()

        n_classes = 101

        self.visual_net = resnet18(modality='visual')
        self.visual_net.load_state_dict(torch.load('/home/yake_wei/models/resnet18.pth'), strict=False)
        self.fc = nn.Linear(512, n_classes)

    def forward(self, visual):
        B = visual.size()[0]
        v = self.visual_net(visual)

        (_, C, H, W) = v.size()
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)

        v = F.adaptive_avg_pool3d(v, 1)

        v = torch.flatten(v, 1)

        out = self.fc(v)

        return out



class AVClassifier(nn.Module):
    def __init__(self, args):
        super(AVClassifier, self).__init__()

        if args.dataset == 'VGGSound':
            n_classes = 309
        elif args.dataset == 'KineticSound':
            n_classes = 31
        elif args.dataset == 'CREMAD':
            n_classes = 6
        elif args.dataset == 'AVE':
            n_classes = 28
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))


        self.dataset = args.dataset

        self.audio_net = resnet18(modality='audio')
        self.visual_net = resnet18(modality='visual')

        self.head = nn.Linear(1024, n_classes)
        self.head_audio = nn.Linear(512, n_classes)
        self.head_video = nn.Linear(512, n_classes)



    def forward(self, audio, visual):
        if self.dataset != 'CREMAD':
            visual = visual.permute(0, 2, 1, 3, 4).contiguous()
        a = self.audio_net(audio)
        v = self.visual_net(visual)

        (_, C, H, W) = v.size()
        B = a.size()[0]
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)

        a = F.adaptive_avg_pool2d(a, 1)
        v = F.adaptive_avg_pool3d(v, 1)

        a = torch.flatten(a, 1)
        v = torch.flatten(v, 1)


        out = torch.cat((a,v),1)
        out = self.head(out)


        return out,_,_,a,v

    

        
    




