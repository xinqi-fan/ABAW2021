import torch


def frames_collate(batch):

    targets = []
    images = []
    image_names = []
    for sample in batch:
        imgs, labels, img_names = sample
        images.append(torch.stack(imgs, 0))
        targets.append(labels)
        image_names.extend(img_names)

    return torch.stack(images, 0), torch.stack(targets, 0), image_names