import argparse
import os
import sys
import myDataset
import networks
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from model import MrcNet

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  


# Training settings
parser = argparse.ArgumentParser(
    description='Screen Recapture Detection'
)
parser.add_argument(
    '--batch-size', type=int, default=8, metavar='N',
    help='input batch size for training (default: 8)'
)
parser.add_argument(
    '--test-batch-size', type=int, default=8, metavar='N',
    help='input batch size for testing (default: 8)'
)
parser.add_argument(
    '--patch-size', type=int, default=96, metavar='N',
    help='input the patch size of the network during training and testing '
    '(default: 96)'
)
parser.add_argument(
    '--log-dir', default='./log',
    help='folder to output model checkpoints'
)
parser.add_argument(
    '--epochs', type=int, default=30, metavar='N',
    help='number of epochs to train (default: 30)'
)
parser.add_argument(
    '--lr', type=float, default=0.0005, metavar='LR',
    help='learning rate (default: 0.0005)'
)
parser.add_argument(
    '--wd', default=1e-4, type=float,
    metavar='W', help='weight decay (default: 1e-4)'
)
parser.add_argument(
    '--optimizer', default='adam', type=str,
    metavar='OPT', help='The optimizer to use (default: adam)'
)
parser.add_argument(
    '--lr-decay', default=0.95, type=float,
    metavar='LRD', help='learning rate decay for adagrad (default: 0.95)'
)
parser.add_argument(
    '--seed', type=int, default=1, metavar='S',
    help='random seed (default: 1)'
)
parser.add_argument(
    '--early-stopping-patience', type=int, default=5,
    help='early stopping patience (default: 5)'
)
parser.add_argument(
    '--grad-clip', type=float, default=1.0,
    help='gradient clipping value (default: 1.0, 0 to disable)'
)
parser.add_argument(
    '--data-root', default='./dataset/',
    help='path to dataset root directory (default: ./dataset/)'
)

args = parser.parse_args()

np.random.seed(args.seed)
if torch.cuda.is_available():
    cudnn.benchmark = True
    kwargs = {'num_workers': 4, 'pin_memory': True}
else:
    kwargs = {'num_workers': 0}

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

train_dir = os.path.join(args.data_root, 'train')
val_dir = os.path.join(args.data_root, 'val')

if not os.path.exists(train_dir):
    print(f"Error: Training directory not found: {train_dir}", file=sys.stderr)
    sys.exit(1)

if not os.path.exists(val_dir):
    print(f"Error: Validation directory not found: {val_dir}", file=sys.stderr)
    sys.exit(1)

normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


class GaussianNoise:
    """Add Gaussian noise to image."""
    def __init__(self, mean=0.0, std=0.01):
        self.mean = mean
        self.std = std
    
    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise


train_loader = myDataset.DataLoaderHalf(
    myDataset.MyDataset(train_dir,
                   transforms.Compose([
                       transforms.RandomResizedCrop(args.patch_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                       transforms.RandomHorizontalFlip(p=0.5),
                       transforms.RandomRotation(degrees=5),
                       transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
                       transforms.ColorJitter(brightness=0.2, contrast=0.2),
                       transforms.ToTensor(),
                       GaussianNoise(mean=0.0, std=0.005),
                       normalize
                   ])),
    batch_size=args.batch_size, shuffle=False, half_constraint=True,
    sampler_type='RandomBalancedSampler', **kwargs)

val_loader = torch.utils.data.DataLoader(
    myDataset.MyDataset(val_dir,
                    transforms.Compose([
                        transforms.CenterCrop(args.patch_size),
                        transforms.ToTensor(),
                        normalize
                    ])),
    batch_size=args.test_batch_size, shuffle=False, **kwargs)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose=True, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 10
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: True
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                           Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.val_loss_min = val_loss
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.val_loss_min = val_loss
            self.counter = 0


def main():
    model = MrcNet()
    networks.init_weights(model, init_type='normal')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
    optimizer = create_optimizer(model, args.lr)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=args.early_stopping_patience, verbose=True)
    
    best_val_acc = 0.0

    for epoch in range(1, args.epochs+1):
        adjust_learning_rate(optimizer, epoch)

        train_acc, train_loss = train(train_loader, model, optimizer, criterion, epoch)
        val_acc, val_loss = val(val_loader, model, criterion, epoch)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'val_acc': val_acc},
                       '{}/best_model.pth'.format(args.log_dir))
            print(f'New best validation accuracy: {val_acc:.2f}%')
        
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

def train(train_loader, model, optimizer, criterion, epoch):
    model.train()
    running_loss = 0
    running_corrects = 0
 
    for batch_idx, (data, target) in enumerate(train_loader):
                
        data, target = data.to(device), target.to(device)

        prediction = model(data)
        _, preds = torch.max(prediction.data, 1)
        
        loss = criterion(prediction, target)

        running_loss += loss.item()   
        running_corrects += torch.sum(preds == target.data).cpu().item()

        optimizer.zero_grad()
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if (batch_idx+1) % 5 == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]   Loss: {:.6f}'.format(
                    epoch, (batch_idx+1) * len(data), len(train_loader.dataset),
                    100. * (batch_idx+1) / len(train_loader),
                    loss.item()))
        


    running_loss = running_loss / (len(train_loader.dataset) // args.batch_size)
    ave_corrects = 100. * running_corrects / len(train_loader.dataset)
    print('Train Epoch {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        epoch, running_loss, running_corrects, len(train_loader.dataset), ave_corrects))
    return ave_corrects, running_loss


def val(val_loader, model, criterion, epoch):
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

    test_loss = test_loss / len(val_loader)
    ave_correct = 100. * correct / len(val_loader.dataset)
    print('Val Epoch {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        epoch, test_loss, correct, len(val_loader.dataset), ave_correct))
    return ave_correct, test_loss


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 15 epochs"""
    lr = args.lr * (0.1 ** (epoch // 15))   
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def create_optimizer(model, new_lr):
    # setup optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=new_lr,
                              momentum=0.9, weight_decay=args.wd)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=new_lr,
                               weight_decay=args.wd)
    elif args.optimizer == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=new_lr,
                                  lr_decay=args.lr_decay, weight_decay=args.wd)
    return optimizer


if __name__ == '__main__':
    main()


