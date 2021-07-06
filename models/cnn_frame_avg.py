import torch
import torch.nn as nn

from .resnet50_ft_dag import Resnet50_ft_dag
from .vision_transformer_temporal import TemporalVisionTransformer


class CnnFrameAvg(nn.Module):
    def __init__(self, num_patch, embed_dim, output_dim, cnn_ckpt=None):
        super(CnnFrameAvg, self).__init__()

        self.embed_dim = embed_dim
        self.num_patch = num_patch

        self.cnn = Resnet50_ft_dag(output_dim=self.embed_dim)
        if cnn_ckpt:
            self._load_pretrain_resnet50vgg(cnn_ckpt)

        self.norm = nn.LayerNorm(self.embed_dim)
        self.avgpool = nn.AvgPool1d(self.num_patch)

        self.fc = nn.Linear(self.embed_dim, output_dim)

    def forward(self, x):
        # cnn
        bs, num_patch, num_c, H, W = x.shape
        assert num_patch == self.num_patch

        x = x.reshape(bs*num_patch, num_c, H, W).contiguous()
        x = self.cnn(x)
        x = self.norm(x)
        # reshape for transformer input
        x = x.reshape(bs, num_patch, self.embed_dim).contiguous()
        # permute for AvgPool1D
        x = x.permute(0, 2, 1)
        # pool
        x = torch.squeeze(self.avgpool(x), 2)
        # classifier
        x = self.fc(x)

        return x

    def _load_pretrain_resnet50vgg(self, ckpt):
        # model_dict = self.cnn.state_dict()
        checkpoint = torch.load(ckpt)
        # checkpoint = torch.load(ckpt)['model']
        classifier_name = 'classifier'
        # model_head_dim = model_dict[classifier_name + '.weight'].shape[0]
        # checkpoint_head_dim = checkpoint[classifier_name + '.weight'].shape[0]
        # print(f'model head dim {model_head_dim} | checkpoint_head_dim {checkpoint_head_dim}')
        #
        # if model_head_dim != checkpoint_head_dim:
        print('model head and checkpoint head is different. checkpoint head is discarded.')
        del checkpoint[classifier_name + '.weight']
        del checkpoint[classifier_name + '.bias']

        self.cnn.load_state_dict(checkpoint, strict=False)

        print(f'CNN pretrained weights is loaded')

        # freeze weights
        # for p in self.cnn.parameters():
        #     p.requires_grad = False
        # # unfreeze classifier
        # self.cnn.classifier.weight.requires_grad = True
        # self.cnn.classifier.bias.requires_grad = True
        # print('Partial weights freezed')





