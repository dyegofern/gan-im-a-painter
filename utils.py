import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import os
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, root, transform=None, mode='train'):
        self.transform = transform
        self.files = sorted([os.path.join(root, f) for f in os.listdir(root) if f.endswith(('.jpg', '.png'))])

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.files)


class ReplayBuffer:
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if np.random.uniform(0, 1) > 0.5:
                    i = np.random.randint(0, self.max_size)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)


class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


def get_transforms(img_size=256):
    transforms_train = transforms.Compose([
        transforms.Resize(int(img_size * 1.12), Image.BICUBIC),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transforms_test = transforms.Compose([
        transforms.Resize((img_size, img_size), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    return transforms_train, transforms_test


def denormalize(tensor):
    return (tensor * 0.5 + 0.5).clamp(0, 1)


def save_image_grid(images, path, nrow=4):
    images = denormalize(images)
    grid = torch.cat([img.unsqueeze(0) for img in images], dim=0)

    fig, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 4))
    if len(images) == 1:
        axes = [axes]

    for ax, img in zip(axes, images):
        img_np = img.permute(1, 2, 0).cpu().numpy()
        ax.imshow(img_np)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()


def tensor_to_pil(tensor):
    img = denormalize(tensor).cpu().squeeze().permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)
