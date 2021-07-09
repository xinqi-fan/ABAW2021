import torch
import torch.nn as nn

from .resnet50_ft_dag import Resnet50_ft_dag
from .transformer_temporal import TemporalTransformer


class CnnTransformer(nn.Module):
    def __init__(self, num_patch, embed_dim, output_dim, num_heads, dropout, cnn_ckpt=None):
        super(CnnTransformer, self).__init__()

        self.embed_dim = embed_dim
        self.num_patch = num_patch
        self.num_heads = num_heads

        self.cnn = Resnet50_ft_dag(output_dim=self.embed_dim)
        if cnn_ckpt:
            self._load_pretrain_resnet50vgg(cnn_ckpt)
        self.norm_cnn = nn.LayerNorm(self.embed_dim)

        last_layer_name, last_module = list(self.cnn.named_modules())[-1]
        try:
            in_channels, out_channels = last_module.in_features, last_module.out_features
        except:
            in_channels, out_channels = last_module.in_channels, last_module.out_channels
        setattr(self.cnn, '{}'.format(last_layer_name), Identity()) # the second last layer has 512 dimensions
        setattr(self.cnn, 'output_feature_dim', in_channels)

        self.proj = nn.Linear(self.cnn.output_feature_dim, self.embed_dim)

        # transformer
        self.trans = TemporalTransformer(num_patches=self.num_patch, num_classes=output_dim, drop_rate=0.5)

        # self.norm_sum = nn.BatchNorm1d(self.embed_dim)

        self.fc = nn.Linear(self.embed_dim, output_dim)


    def forward(self, x):
        # cnn
        bs, num_patch, num_c, H, W = x.shape
        assert num_patch == self.num_patch
        # print(x.shape)

        x = x.reshape(bs*self.num_patch, num_c, H, W).contiguous()

        with torch.no_grad():
            x = self.cnn(x)
        # print(x.shape)
        x = x.detach()
        # x = x.squeeze(-1).squeeze(-1)
        x = self.proj(x)
        # print(x.shape)

        # reshape for transformer input
        x = x.reshape(bs, self.num_patch, self.embed_dim).contiguous()

        # only torch 1.9 above support batch_first
        x = self.trans(x)

        # reshape for output
        # x = x.permute(1, 0, 2)
        # x = x.reshape(bs, self.num_patch*self.embed_dim)
        # x = torch.mean(x, dim=1)
        # classifier
        x = self.fc(x)

        return x

    def _load_pretrain_resnet50vgg(self, ckpt):
        model_dict = self.cnn.state_dict()
        try:
            checkpoint_org = torch.load(ckpt)['model']
        except:
            checkpoint_org = torch.load(ckpt)

        checkpoint = {}
        for i, k in enumerate(checkpoint_org.keys()):
            new_k = k[11:] # remove module.cnn.
            checkpoint[new_k] = checkpoint_org[k]

        classifier_name = 'classifier'
        del checkpoint[classifier_name + '.weight']
        del checkpoint[classifier_name + '.bias']
        print('checkpoint head is discarded.')

        self.cnn.load_state_dict(checkpoint, strict=False)
        print(f'CNN pretrained weights is loaded')

        # freeze weights
        for p in self.cnn.parameters():
            p.requires_grad = False
        # unfreeze classifier
        self.cnn.classifier.weight.requires_grad = True
        self.cnn.classifier.bias.requires_grad = True
        print('Partial weights freezed')



class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x



