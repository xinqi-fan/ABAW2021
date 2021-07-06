from __future__ import print_function

import os
import sys
import argparse
import time
import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms

from data import AffWild2EXPRDataset
from models import Resnet50Vgg, CnnVit, CnnFrameAvg, CnnEmbedAvg, CnnSelfAtt, CnnSelfAttSum
from utils import AverageMeter, accuracy, EXPR_metric
from utils import Logger

# try:
#     import apex
#     from apex import amp, optimizers
# except ImportError:
#     pass


def parse_arguments():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--optimizer', type=str, default='sgd',
                        help='optimizer')
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,75,90',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--scheduler', action='store_true',
                        help='using learning rate scheduler')

    # model dataset
    parser.add_argument('--model', type=str, default='Resnet50Vgg',
                        choices=['Resnet50Vgg', 'CnnVit', 'CnnFrameAvg', 'CnnEmbedAvg', 'CnnSelfAtt', 'CnnSelfAttSum'])
    parser.add_argument('--dataset', type=str, default='Aff-Wild2',
                        choices=['Aff-Wild2', 'RAF-DB'], help='dataset')
    parser.add_argument('--data_folder', type=str, default='/home/xinqifan2/Data/Facial_Expression/Aff-Wild2/ABAW-2021', help='path to custom dataset')
    parser.add_argument('--img_relative_folder', type=str, default='Cropped_aligned_image/cropped_aligned', help='path to relative image folder')
    parser.add_argument('--label_relative_folder', type=str, default='Annotation/annotations', help='path to relative label folder')
    parser.add_argument('--task', type=str, default='EXPR')
    parser.add_argument('--data_mode', type=str, default='static',
                        choices=['static', 'sequence_naive', 'sequence_video_middle', 'sequence_video_middle_repeat', 'sequence_video_non_middle'])

    # other setting
    parser.add_argument('--cnn_ckpt', type=str, default='weights/resnet50_ft_dag.pth',
                        help='path to pre-trained model')
    parser.add_argument('--vit_ckpt', type=str, default='weights/jx_vit_base_p16_224-80ecf9dd.pth',
                        help='path to pre-trained model')
    parser.add_argument('--embed_dim', type=int, default=768,
                        help='CNN output dimension')
    parser.add_argument('--num_patch', type=int, default=9,
                        help='number of frames used at each time stamp (sequence length = number of batches)')
    parser.add_argument('--save_model', action='store_true',
                        help='save model')
    parser.add_argument('--hpc', action='store_true', help='whether train on hpc')

    args = parser.parse_args()

    args.model_path = './save/{}_models'.format(args.dataset)
    args.model_name = '{}_{}_{}_{}_lr_{}_bsz_{}'.\
        format(args.dataset, args.data_mode, args.task, args.model, args.learning_rate, args.batch_size)

    args.save_folder = os.path.join(args.model_path, args.model_name)
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)

    if not args.hpc:
        log_file = os.path.join(args.save_folder, 'output_log.txt')
        sys.stdout = Logger(filename=log_file)
        sys.stderr = sys.stdout

    print(f'model name: {args.model_name}')
    print(f'args: {args}')

    if args.dataset == 'Aff-Wild2':
        args.n_cls = 7
    else:
        raise ValueError('dataset not supported: {}'.format(args.dataset))

    return args


def set_loader(args):
    # construct data loader
    if args.dataset == 'Aff-Wild2':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    else:
        raise ValueError('dataset not supported: {}'.format(args.dataset))

    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    if args.dataset == 'Aff-Wild2':
        train_dataset = AffWild2EXPRDataset(args.data_folder, args.img_relative_folder, args.label_relative_folder, data_mode=args.data_mode, phase='train', transform=train_transform, sequence_len=args.num_patch)
        val_dataset = AffWild2EXPRDataset(args.data_folder, args.img_relative_folder, args.label_relative_folder, data_mode=args.data_mode, phase='validation', transform=val_transform, sequence_len=args.num_patch)
        print('Train set size:', train_dataset.__len__())
        print('Validation set size:', val_dataset.__len__())
        # train_sampler = weighted_sampler_generator(data_txt_dir, args.dataset)
        train_sampler = None
    else:
        raise ValueError(args.dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    return train_loader, val_loader


def set_model(args):

    if args.model == 'Resnet50Vgg':
        model = Resnet50Vgg(output_dim=args.n_cls, ckpt=args.cnn_ckpt)
    elif args.model == 'CnnVit':
        model = CnnVit(num_patch=args.num_patch, embed_dim=args.embed_dim, output_dim=args.n_cls, cnn_ckpt=args.cnn_ckpt, vit_ckpt=args.vit_ckpt)
    elif args.model == 'CnnFrameAvg':
        model = CnnFrameAvg(num_patch=args.num_patch, embed_dim=args.embed_dim, output_dim=args.n_cls, cnn_ckpt=args.cnn_ckpt)
    elif args.model == 'CnnEmbedAvg':
        model = CnnEmbedAvg(num_patch=args.num_patch, embed_dim=args.embed_dim, output_dim=args.n_cls, cnn_ckpt=args.cnn_ckpt)
    elif args.model == 'CnnSelfAtt':
        model = CnnSelfAtt(num_patch=args.num_patch, embed_dim=args.embed_dim, output_dim=args.n_cls, num_heads=4, dropout=0.2, cnn_ckpt=args.cnn_ckpt)
    elif args.model == 'CnnSelfAttSum':
        model = CnnSelfAttSum(num_patch=args.num_patch, embed_dim=args.embed_dim, output_dim=args.n_cls, num_heads=4, dropout=0.2, cnn_ckpt=args.cnn_ckpt)
    else:
        raise ValueError('model not supported: {}'.format(args.model))

    print(model)
    # check requires grad
    # for name, param in model.named_parameters():
    #     print(f'{name}, {param.requires_grad}')

    criterion = torch.nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def set_optimizer(args, model):

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                              lr=args.learning_rate,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                               lr=args.learning_rate,
                               weight_decay=args.weight_decay)
    else:
        raise ValueError('optimizer not supported: {}'.format(args.optimizer))
    return optimizer


def save_model(model, optimizer, args, epoch, save_file):
    print('==> Saving...')
    state = {
        'args': args,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state


def train(train_loader, model, criterion, optimizer, epoch, args):
    """one epoch training"""

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    label = {'gt': [], 'pred': []}

    end = time.time()
    for idx, (images, targets) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda()
        targets = targets.cuda()
        bsz = targets.shape[0]

        # model
        output = model(images)

        loss = criterion(output, targets)

        # update metric
        losses.update(loss, bsz)
        acc_batch = accuracy(output, targets)
        acc.update(acc_batch[0], bsz)
        label['gt'].append(targets.cpu().detach().numpy())
        label['pred'].append(output.cpu().detach().numpy())

        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (idx + 1) % args.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@ {acc.val:.3f} ({acc.avg:.3f})'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, acc=acc))
            sys.stdout.flush()

    label_gt = np.concatenate(label['gt'], axis=0)
    label_pred = np.concatenate(label['pred'], axis=0)
    f1, acc, total_acc = EXPR_metric(label_pred, label_gt)

    return losses.avg, f1, acc, total_acc


def validate(val_loader, model, criterion, args):
    """validation"""
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    label = {'gt': [], 'pred': []}

    with torch.no_grad():
        end = time.time()
        for idx, (images, targets) in enumerate(val_loader):
            images = images.cuda()
            targets = targets.cuda()
            bsz = targets.shape[0]

            # model
            output = model(images)
            loss = criterion(output, targets)

            # update metric
            losses.update(loss.item(), bsz)
            acc_batch = accuracy(output, targets)
            acc.update(acc_batch[0], bsz)
            label['gt'].append(targets.cpu().detach().numpy())
            label['pred'].append(output.cpu().detach().numpy())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       loss=losses, acc=acc))

    label_gt = np.concatenate(label['gt'], axis=0)
    label_pred = np.concatenate(label['pred'], axis=0)
    f1, acc, total_acc = EXPR_metric(label_pred, label_gt)

    return losses.avg, f1, acc, total_acc


def main():
    best_total_acc = 0
    args = parse_arguments()

    # build data loader
    train_loader, val_loader = set_loader(args)

    # build model and criterion
    model, criterion = set_model(args)

    # build optimizer
    optimizer = set_optimizer(args, model)

    if args.scheduler:
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 4, 8], gamma=0.1)

    # training routine
    for epoch in range(1, args.epochs + 1):

        # train for one epoch
        time1 = time.time()
        loss, train_f1, train_acc, train_total_acc = train(train_loader, model, criterion, optimizer, epoch, args)
        time2 = time.time()
        print('Train epoch {}, total time {:.2f}, F1:{:.4f}, accuracy:{:.4f}, total accuracy:{:.4f}'.format(
            epoch, time2 - time1, train_f1, train_acc, train_total_acc))

        # eval for one epoch
        time1 = time.time()
        loss, val_f1, val_acc, val_total_acc = validate(val_loader, model, criterion, args)
        time2 = time.time()
        print('Validation epoch {}, total time {:.4f}, F1:{:.4f}, accuracy:{:.4f}, total accuracy:{:.4f}'.format(
            epoch, time2 - time1, val_f1, val_acc, val_total_acc))
        if val_total_acc > best_total_acc:
            best_total_acc = val_total_acc

        if args.scheduler:
            scheduler.step()

        # save the last model
        if args.save_model:
            # if epoch % args.save_freq == 0:
            if val_total_acc > 0.46:
                save_file = os.path.join(
                    args.save_folder, 'ckpt_epoch_{epoch}_{total_acc:.4f}.pth'.format(epoch=epoch, total_acc=val_total_acc))
                save_model(model, optimizer, args, epoch, save_file)

    print('best accuracy: {:.4f}'.format(best_total_acc))


if __name__ == '__main__':
    main()
