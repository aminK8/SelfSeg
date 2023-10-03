import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from utils import read_url_files
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, root_dirs, sequence_length=8, transform=None):
        self.root_dirs = root_dirs
        self.sequence_length = sequence_length
        self.transform = transform

        self.folder_paths =  []
        for root_dir in self.root_dirs:
            self.folder_paths.extend([os.path.join(root_dir, folder) 
                                      for folder in os.listdir(root_dir)])
        self.image_paths = []

        for folder_path in self.folder_paths:
            images = [os.path.join(folder_path, img) for img in os.listdir(folder_path) if not img.endswith('.json')]

            images.sort()
            self.image_paths.extend(images)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        sequence = []
        for i in range(self.sequence_length):
            img_path = self.image_paths[idx + i]
            img = Image.open(img_path)

            if self.transform:
                img = self.transform(img)

            sequence.append(img)

        return torch.stack(sequence)


def get_data_loader(args):
    if args.in_channel == 1:
        transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, 
                                   hue=0.2),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2,
                                   hue=0.2),
            transforms.ToTensor(),
        ])
    train_dataset = ImageDataset(read_url_files(args.data_type), args.sequence_length, 
                                 transform=transform)
    
    if args.distributed:
        train_sampler = DistributedSampler(train_dataset)
    else: 
        train_sampler = None
        
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              sampler=train_sampler, shuffle=(train_sampler is None))
    return train_loader
