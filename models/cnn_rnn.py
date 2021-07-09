import torch
import torch.nn as nn

from .resnet50_ft_dag import Resnet50_ft_dag
# from .resnet50_ferplus_dag import Resnet50_ferplus_dag


class CnnRnn(nn.Module):
    def __init__(self, num_patch, embed_dim, output_dim, num_heads, dropout, cnn_ckpt=None):
        super(CnnRnn, self).__init__()

        self.embed_dim = embed_dim
        self.num_patch = num_patch
        self.num_heads = num_heads

        # self.cnn = Resnet50_ferplus_dag()
        self.cnn = Resnet50_ft_dag(output_dim=self.embed_dim)
        # if cnn_ckpt:
        #     self._load_pretrain_cnn(cnn_ckpt)

        last_layer_name, last_module = list(self.cnn.named_modules())[-1]
        try:
            in_channels, out_channels = last_module.in_features, last_module.out_features
        except:
            in_channels, out_channels = last_module.in_channels, last_module.out_channels
        setattr(self.cnn, '{}'.format(last_layer_name), Identity()) # the second last layer has 512 dimensions
        setattr(self.cnn, 'output_feature_dim', in_channels)

        self.mlp = ProjectHead(self.cnn.output_feature_dim, self.embed_dim)

        self.gru = GruHead(self.embed_dim, self.embed_dim//2, output_dim)

    def forward(self, x):
        # cnn
        bs, num_patch, num_c, H, W = x.shape
        assert num_patch == self.num_patch

        x = x.reshape(bs*self.num_patch, num_c, H, W).contiguous()
        # print(f'x {x.shape}')
        with torch.no_grad():
            x = self.cnn(x)
        x = x.detach()
        # print(f'x {x.shape}')

        x = x.squeeze(-1).squeeze(-1)
        # print(f'x {x.shape}')
        x = self.mlp(x)
        # print(f'x {x.shape}')

        x = x.reshape(bs, self.num_patch, self.embed_dim).contiguous()
        # print(f'x {x.shape}')
        # many-to-many
        x = self.gru(x)[:, 4, :]
        # many-to-one: the last
        # x = self.gru(x)[:, -1, :]
        # print(f'x {x.shape}')

        return x

    def _load_pretrain_cnn(self, ckpt):
        model_dict = self.cnn.state_dict()
        try:
            checkpoint_org = torch.load(ckpt)['model']
        except:
            checkpoint_org = torch.load(ckpt)

        checkpoint = {}
        for i, k in enumerate(checkpoint_org.keys()):
            new_k = k.replace('module.cnn.', '')
            checkpoint[new_k] = checkpoint_org[k]

        # classifier_name = 'classifier'
        # del checkpoint[classifier_name + '.weight']
        # del checkpoint[classifier_name + '.bias']
        # print('checkpoint head is discarded.')

        self.cnn.load_state_dict(checkpoint, strict=False)
        print(f'CNN pretrained weights is loaded')

        # freeze weights
        for p in self.cnn.parameters():
            p.requires_grad = False
        # unfreeze classifier
        self.cnn.classifier.weight.requires_grad = True
        self.cnn.classifier.bias.requires_grad = True
        print('Partial weights freezed')


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x


class ProjectHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=0):
        super(ProjectHead, self).__init__()
        self._name = 'ProjectHead'
        # self.bn0 = nn.BatchNorm1d(input_dim)
        self.fc_0 = nn.Linear(input_dim, hidden_dim)
        # self.bn1 = nn.BatchNorm1d(hidden_dim)
        # self.fc_1 = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        # x = self.bn0(x)
        # f0 = self.bn1(torch.relu(self.fc_0(x)))
        f0 = torch.relu(self.fc_0(x))
        # output = self.fc_1(f0)
        return f0


class GruHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_class = 8):
        super(GruHead, self).__init__()
        self._name = 'GruHead'
        self.GRU_layer = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc_1 = nn.Linear(hidden_dim*2, n_class)
    def forward(self, x):
        B, N, C = x.size()
        self.GRU_layer.flatten_parameters()
        f0 = torch.relu(self.GRU_layer(x)[0])
        output = self.fc_1(f0)
        return output



