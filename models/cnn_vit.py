import torch
import torch.nn as nn

from .resnet50_ft_dag import Resnet50_ft_dag
from .vision_transformer_temporal import TemporalVisionTransformer


class CNN_ViT(nn.Module):
    def __init__(self, num_patch, embed_dim, output_dim, cnn_ckpt=None, vit_ckpt=None):
        super(CNN_ViT, self).__init__()

        self.embed_dim = embed_dim
        self.num_patch = num_patch

        self.cnn = Resnet50_ft_dag(mode='single-task', output_dim=embed_dim)
        if cnn_ckpt:
            self._load_pretrain_resnet50vgg(cnn_ckpt)

        self.vit = TemporalVisionTransformer(num_patches=self.num_patch, num_classes=output_dim)
        if vit_ckpt:
            self._load_pretrain_ViT(vit_ckpt)

        self.norm = nn.LayerNorm([self.num_patch, self.embed_dim])

    def forward(self, x):
        # cnn
        bs, num_patch, num_c, H, W = x.shape
        assert num_patch == self.num_patch

        x = x.reshape(bs*num_patch, num_c, H, W).contiguous()
        x = self.cnn(x)
        # reshape for transformer input
        z = x.reshape(bs, num_patch, self.embed_dim).contiguous()
        # layer norm
        z = self.norm(z)
        # vit
        z = self.vit(z)

        return z

    def _load_pretrain_resnet50vgg(self, ckpt):
        # model_dict = self.cnn.state_dict()
        checkpoint = torch.load(ckpt)['model']
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


    def _load_pretrain_ViT(self, ckpt):
        model_dict = self.vit.state_dict()
        checkpoint = torch.load(ckpt)
        model_head_dim = model_dict['head.weight'].shape[0]
        checkpoint_head_dim = checkpoint['head.weight'].shape[0]
        print(f'model head dim {model_head_dim} | checkpoint_head_dim {checkpoint_head_dim}')

        if model_head_dim != checkpoint_head_dim:
            print('model head and checkpoint head is different. checkpoint head is discarded.')
            del checkpoint['head.weight']
            del checkpoint['head.bias']

        del checkpoint['pos_embed']
        del checkpoint['patch_embed.proj.bias']
        del checkpoint['patch_embed.proj.weight']

        # since we may remove some layers of Transformer
        # filter out unnecessary keys
        new_state_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
        # overwrite entries in the existing state dict
        model_dict.update(new_state_dict)

        self.vit.load_state_dict(checkpoint, strict=False)

        print(f'Transformer pretrained weights is loaded')






