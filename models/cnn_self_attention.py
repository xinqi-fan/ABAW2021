import torch
import torch.nn as nn

from .resnet50_ft_dag import Resnet50_ft_dag


class CnnSelfAtt(nn.Module):
    def __init__(self, num_patch, embed_dim, output_dim, num_heads, dropout, cnn_ckpt=None):
        super(CnnSelfAtt, self).__init__()

        self.embed_dim = embed_dim
        self.num_patch = num_patch
        self.num_heads = num_heads

        self.cnn = Resnet50_ft_dag(output_dim=self.embed_dim)
        if cnn_ckpt:
            self._load_pretrain_resnet50vgg(cnn_ckpt)
        self.norm_cnn = nn.LayerNorm(self.embed_dim)

        self.self_att1 = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads)
        self.norm_att1 = nn.LayerNorm(self.embed_dim)
        self.self_att2 = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads)
        self.norm_att2 = nn.LayerNorm(self.embed_dim)

        self.fc = nn.Linear(self.num_patch*self.embed_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # cnn
        bs, num_patch, num_c, H, W = x.shape
        assert num_patch == self.num_patch

        x = x.reshape(bs*self.num_patch, num_c, H, W).contiguous()
        x = self.cnn(x)
        # reshape for transformer input
        x = x.reshape(bs, self.num_patch, self.embed_dim).contiguous()
        x = self.norm_cnn(x)

        # only torch 1.9 above support batch_first
        x = x.permute(1, 0, 2)
        x, _ = self.self_att1(x, x, x)  # output: attn_output, attn_output_weights
        x = self.dropout(self.norm_att1(x))
        x, _ = self.self_att2(x, x, x)  # output: attn_output, attn_output_weights
        x = self.dropout(self.norm_att2(x))

        # reshape for output
        x = x.permute(1, 0, 2)
        x = x.reshape(bs, self.num_patch*self.embed_dim).contiguous()
        # classifier
        x = self.fc(x)

        return x

    def _load_pretrain_resnet50vgg(self, ckpt):
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








