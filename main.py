import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataset import Flowers
from models.model import Model
from models.vgg import VGG16
import os
from pdb import set_trace
from tensorboardX import SummaryWriter
from tqdm import tqdm


writer = SummaryWriter()

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # loss = nn.CrossEntropyLoss(output, target)
        # set_trace()
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
    
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            writer.add_scalar('train_loss', loss.item(), epoch * len(train_loader.dataset) + batch_idx * len(data))
            writer.add_scalar('train_acc', 100. * correct/len(target), epoch * len(train_loader.dataset) + batch_idx * len(data))
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    print('\nTaining set: Accuracy: {}/{} ({:.0f}%)\n'.format(
            correct, len(train_loader.dataset), 100. * correct / len(train_loader.dataset)))

def test(args, model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        # for data, target in test_loader:
        for batch_idx, (data, target) in enumerate(tqdm(test_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)
            # test_loss += nn.CrossEntropyLoss(output, target, reduction='sum').item() # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            if batch_idx % args.log_interval == 0:
                writer.add_scalar('test_loss', test_loss, epoch * len(test_loader.dataset) + batch_idx * len(data))
                writer.add_scalar('test_acc', 100. * correct/len(target), epoch * len(test_loader.dataset) + batch_idx * len(data))
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch VGG16 Classification')
    parser.add_argument('--batch-size', type=int, default=30, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=30, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--vgg', default=True, metavar='TV', type=lambda x: (str(x).lower() == 'true'),
                        help='training vgg from original images')
    parser.add_argument('--MSRN', default=True, metavar='TMSRN', type=lambda x: (str(x).lower() == 'true'),
                        help='testing if is from MSRN --false is from bicubic')
    parser.add_argument("--vggw", default="Weights/vgg/29.pth", type=str, help="path to VGG16 model")
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    print('args.vgg: {}'.format(args.vgg == True))
    print('args.MSRN: {}'.format(args.MSRN == True))
    print('Weight loaded from: {}'.format(args.vggw))
    if args.vgg:
        #Original Images
        print("Training VGG -- Creating train_loader and test loader for VGG16 model")
        train_loader = torch.utils.data.DataLoader(Flowers(is_train=True, downsample=False, upsample=False), batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(Flowers(is_train=False, downsample=False, upsample=False), batch_size=args.test_batch_size, shuffle=False, **kwargs)
        model = VGG16(clf_pth=args.vggw).to(device)
        weight_folder = 'vgg/'

        # model.load_state_dict(torch.load(args.vggw))
    else:
        if args.MSRN:
            #MSRN x4
            print("Test MSRN x4 -- Creating test_loader for Model including MSRN and VGG")
            # train_loader = torch.utils.data.DataLoader(Flowers(is_train=True, downsample=True, upsample=False), batch_size=args.batch_size, shuffle=True, **kwargs)
            test_loader = torch.utils.data.DataLoader(Flowers(is_train=False, downsample=True, upsample=False), batch_size=args.test_batch_size, shuffle=False, **kwargs)
            model = Model(clf_pth=args.vggw).to(device)
            weight_folder = 'msrn/'
        else:
            #Interpolation x4
            print("Test interpolation x4 -- Creating test_loader for VGG model")
            # train_loader = torch.utils.data.DataLoader(Flowers(is_train=True, downsample=True, upsample=True), batch_size=args.batch_size, shuffle=True, **kwargs)
            test_loader = torch.utils.data.DataLoader(Flowers(is_train=False, downsample=True, upsample=True), batch_size=args.test_batch_size, shuffle=False, **kwargs)
            model = VGG16(clf_pth=args.vggw).to(device)
            weight_folder = 'bicubic/'
    
    
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.classifier.parameters()), lr=args.lr)
    if args.vgg:
        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch)
            test(args, model, device, test_loader, epoch)

            if epoch > args.epochs * 0.5:
            # if epoch > 3:
                model_folder = "Weights/" + weight_folder
                model_out_path = model_folder + "{}.pth".format(epoch)
                if not os.path.exists(model_folder):
                    os.makedirs(model_folder)
            # torch.save(model, model_out_path)

                torch.save(model.state_dict(), model_out_path)

    else:
        test(args, model, device, test_loader, 1)


if __name__ == '__main__':
    main()
