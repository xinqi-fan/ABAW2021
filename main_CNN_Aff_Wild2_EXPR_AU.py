from __future__ import print_function

import os
import sys
import argparse
import time
import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms

from data import AffWild2ExprAuDataset
from models.resnet50_ft_dag import Resnet50_ft_dag
from utils import AverageMeter, accuracy, AU_metric, EXPR_metric

# try:
#     import apex
#     from apex import amp, optimizers
# except ImportError:
#     pass


def parse_arguments():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=2,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=48,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=6,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--optimizer', type=str, default='sgd',
                        help='optimizer')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--scheduler', action='store_true',
                        help='using learning rate scheduler')


    # model dataset
    parser.add_argument('--model', type=str, default='ResNet50VGG')
    parser.add_argument('--dataset', type=str, default='Aff-Wild2',
                        choices=['Aff-Wild2', 'RAF-DB'], help='dataset')
    parser.add_argument('--data_folder', type=str, default='/home/xinqifan2/Data/Facial_Expression/Aff-Wild2/ABAW-2021', help='path to custom dataset')
    parser.add_argument('--task', type=str, default='EXPR_AU')

    # other setting
    parser.add_argument('--ckpt', type=str, default='weights/resnet50_ft_dag.pth',
                        help='path to pre-trained model')
    parser.add_argument('--save_model', action='store_true',
                        help='save model')

    args = parser.parse_args()

    args.model_path = './save/{}_models'.format(args.dataset)
    args.model_name = '{}_{}_{}_lr_{}_bsz_{}'.\
        format(args.dataset, args.task, args.model, args.learning_rate, args.batch_size)
    print(f'model name: {args.model_name}')

    args.save_folder = os.path.join(args.model_path, args.model_name)
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)

    if args.dataset == 'Aff-Wild2':
        args.n_cls = [7, 12]
    else:
        raise ValueError('dataset not supported: {}'.format(args.dataset))

    return args


def set_loader(args):

    # construct data loader
    if args.dataset == 'Aff-Wild2':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
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
        train_dataset = AffWild2ExprAuDataset(args.data_folder, phase='train', transform=train_transform)
        val_dataset = AffWild2ExprAuDataset(args.data_folder, phase='validation', transform=val_transform)
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


def load_pretrain_resnet50vgg(args, model):
    checkpoint = torch.load(args.ckpt)

    classifier_name = 'classifier'
    del checkpoint[classifier_name + '.weight']
    del checkpoint[classifier_name + '.bias']
    print('checkpoint head is discarded.')

    model.load_state_dict(checkpoint, strict=False)
    print(f'pretrained weights is loaded')


def set_model(args):

    model = Resnet50_ft_dag(mode='multi-task', output_dim=args.n_cls)

    load_pretrain_resnet50vgg(args, model)

    criterion_EXPR = torch.nn.CrossEntropyLoss()
    criterion_AU = torch.nn.BCEWithLogitsLoss()

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.cuda()
        criterion_EXPR = criterion_EXPR.cuda()
        criterion_AU = criterion_AU.cuda()
        cudnn.benchmark = True

    return model, [criterion_EXPR, criterion_AU]


def set_optimizer(opt, model):

    if opt.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                              lr=opt.learning_rate,
                              momentum=opt.momentum,
                              weight_decay=opt.weight_decay)
    elif opt.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                               lr=opt.learning_rate,
                               weight_decay=opt.weight_decay)
    else:
        raise ValueError('optimizer not supported: {}'.format(opt.optimizer))

    return optimizer


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
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
    acc_AU = AverageMeter()
    acc_EXPR = AverageMeter()
    label_EXPR = {'gt': [], 'pred': []}
    label_AU = {'gt': [], 'pred': []}

    end = time.time()
    for idx, (images, targets) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda()
        targets[0] = targets[0].cuda()
        targets[1] = targets[1].cuda()
        bsz = targets[0].shape[0]

        # model
        output = model(images)
        loss_EXPR = criterion[0](output[0], targets[0])
        loss_AU = criterion[1](output[1], targets[1])
        loss = 0.5 * loss_EXPR + 0.5 * loss_AU

        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update metric
        losses.update(loss, bsz)
        acc_EXPR_batch = accuracy(output[0], targets[0])
        acc_EXPR.update(acc_EXPR_batch[0], bsz)
        label_EXPR['gt'].append(targets[0].cpu().detach().numpy())
        label_EXPR['pred'].append(output[0].cpu().detach().numpy())
        predict_AU = torch.sigmoid(output[1])
        predict_AU = torch.round(predict_AU)
        correct_sum = sum(predict_AU == targets[1]).sum()
        acc_AU_batch = correct_sum.float()/(bsz*args.n_cls[1])
        acc_AU.update(acc_AU_batch, bsz)
        label_AU['gt'].append(targets[1].cpu().detach().numpy())
        label_AU['pred'].append(predict_AU.cpu().detach().numpy())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (idx + 1) % args.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'AccEXPR {acc_EXPR.val:.3f} ({acc_EXPR.avg:.3f})\t'
                  'AccAU {acc_AU.val:.3f} ({acc_AU.avg:.3f})\t'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, acc_EXPR=acc_EXPR, acc_AU=acc_AU))
            sys.stdout.flush()

    label_gt = np.concatenate(label_AU['gt'], axis=0)
    label_pred = np.concatenate(label_AU['pred'], axis=0)
    f1, acc, total_acc = EXPR_metric(label_pred, label_gt)
    EXPR_accs = [f1, acc, total_acc]
    label_gt = np.concatenate(label_AU['gt'], axis=0)
    label_pred = np.concatenate(label_AU['pred'], axis=0)
    f1, acc, total_acc = AU_metric(label_pred, label_gt)
    AU_accs = [f1, acc, total_acc]

    return losses.avg, EXPR_accs, AU_accs


def validate(val_loader, model, criterion, args):
    """validation"""
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    acc_AU = AverageMeter()
    acc_EXPR = AverageMeter()
    label_EXPR = {'gt': [], 'pred': []}
    label_AU = {'gt': [], 'pred': []}

    with torch.no_grad():
        end = time.time()
        for idx, (images, targets) in enumerate(val_loader):
            images = images.cuda()
            targets[0] = targets[0].cuda()
            targets[1] = targets[1].cuda()
            bsz = targets[0].shape[0]

            # model
            output = model(images)
            loss_EXPR = criterion[0](output[0], targets[0])
            loss_AU = criterion[1](output[1], targets[1])
            loss = 0.5 * loss_EXPR + 0.5 * loss_AU

            # update metric
            losses.update(loss, bsz)
            acc_EXPR_batch = accuracy(output[0], targets[0])
            acc_EXPR.update(acc_EXPR_batch[0], bsz)
            label_EXPR['gt'].append(targets[0].cpu().detach().numpy())
            label_EXPR['pred'].append(output[0].cpu().detach().numpy())
            predict_AU = torch.sigmoid(output[1])
            predict_AU = torch.round(predict_AU)
            correct_sum = sum(predict_AU == targets[1]).sum()
            acc_AU_batch = correct_sum.float() / (bsz * args.n_cls[1])
            acc_AU.update(acc_AU_batch, bsz)
            label_AU['gt'].append(targets[1].cpu().detach().numpy())
            label_AU['pred'].append(predict_AU.cpu().detach().numpy())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'AccEXPR {acc_EXPR.val:.3f} ({acc_EXPR.avg:.3f})\t'
                      'AccAU {acc_AU.val:.3f} ({acc_AU.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       loss=losses, acc_EXPR=acc_EXPR, acc_AU=acc_AU))
                sys.stdout.flush()

    label_gt = np.concatenate(label_AU['gt'], axis=0)
    label_pred = np.concatenate(label_AU['pred'], axis=0)
    f1, acc, total_acc = EXPR_metric(label_pred, label_gt)
    EXPR_accs = [f1, acc, total_acc]
    label_gt = np.concatenate(label_AU['gt'], axis=0)
    label_pred = np.concatenate(label_AU['pred'], axis=0)
    f1, acc, total_acc = AU_metric(label_pred, label_gt)
    AU_accs = [f1, acc, total_acc]

    return losses.avg, EXPR_accs, AU_accs


def main():
    args = parse_arguments()

    # build data loader
    train_loader, val_loader = set_loader(args)

    # build model and criterion
    model, criterion = set_model(args)

    # build optimizer
    optimizer = set_optimizer(args, model)

    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 4, 8], gamma=0.1)

    # training routine
    for epoch in range(1, args.epochs + 1):

        # train for one epoch
        time1 = time.time()
        loss, EXPR_accs, AU_accs = train(train_loader, model, criterion, optimizer, epoch, args)
        time2 = time.time()
        print('Train epoch {}, total time {:.2f}, EXPR F1:{:.4f}, accuracy:{:.4f}, total accuracy:{:.4f}, AU F1:{:.4f}, accuracy:{:.4f}, total accuracy:{:.4f}'.format(
            epoch, time2 - time1, EXPR_accs[0], EXPR_accs[1], EXPR_accs[2], AU_accs[0], AU_accs[1], AU_accs[2]))

        # eval for one epoch
        time1 = time.time()
        loss, EXPR_accs, AU_accs = validate(val_loader, model, criterion, args)
        time2 = time.time()
        print('Validation epoch {}, total time {:.2f}, EXPR F1:{:.4f}, accuracy:{:.4f}, total accuracy:{:.4f}, AU F1:{:.4f}, accuracy:{:.4f}, total accuracy:{:.4f}'.format(
            epoch, time2 - time1, EXPR_accs[0], EXPR_accs[1], EXPR_accs[2], AU_accs[0], AU_accs[1], AU_accs[2]))

        if args.scheduler:
            scheduler.step()

        if args.save_model:
            # if epoch % args.save_freq == 0:
                save_file = os.path.join(
                    args.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
                save_model(model, optimizer, args, epoch, save_file)


if __name__ == '__main__':
    main()
