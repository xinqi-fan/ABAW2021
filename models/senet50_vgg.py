import torch
import torch.nn as nn

from .senet50_ft_dag import Senet50_ft_dag


class Senet50Vgg(nn.Module):
    def __init__(self, output_dim, ckpt=None):
        super(Senet50Vgg, self).__init__()
        self.output_dim = output_dim

        self.cnn = Senet50_ft_dag(output_dim=self.output_dim)
        if ckpt:
            self._load_pretrain_senet50vgg(ckpt)

    def forward(self, x):
        # cnn
        x = self.cnn(x)

        return x

    def _load_pretrain_senet50vgg(self, ckpt):
        checkpoint = torch.load(ckpt)

        classifier_name = 'classifier'
        del checkpoint[classifier_name + '.weight']
        del checkpoint[classifier_name + '.bias']
        print('checkpoint head is discarded.')

        self.cnn.load_state_dict(checkpoint, strict=False)
        print(f'CNN pretrained weights is loaded')

        # freeze weights
        # for p in self.cnn.parameters():
        #     p.requires_grad = False
        # # unfreeze classifier
        # self.cnn.classifier.weight.requires_grad = True
        # self.cnn.classifier.bias.requires_grad = True
        # print('Partial weights freezed')








