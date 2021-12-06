import torch.nn.functional as F
from torchvision import datasets, transforms
import torch
import matplotlib.pyplot as plt
import pylab
from torch.utils.data import DataLoader
import argparse
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn

class Net(nn.Module):
    def __init__(self) :
        super(Net, self).__init__()
        self.conv1=nn.Conv2d(3,32,3,1)
        self.conv2=nn.Conv2d(32,64,3,1)
        self.conv3=nn.Conv2d(64,128,3,1)
        self.conv4=nn.Conv2d(128,256,2,1)
        self.conv5=nn.Conv2d(256,512,3,1)
        self.dropout1=nn.Dropout(0.25)
        self.dropout2=nn.Dropout(0.5)
        self.fc1=nn.Linear(4608,512)
        self.fc2=nn.Linear(512,128)
        self.fc3=nn.Linear(128,10)
        #self.nom1=nn.BatchNorm1d(512)

    def forward(self, x):
        x=self.conv1(x)
        x=F.relu(x) 
        #x=F.max_pool2d(x,2)
        x=self.conv2(x)
        x=F.relu(x)
        x=F.max_pool2d(x,2)
        x=self.dropout1(x)
        x=self.conv3(x)
        x=F.relu(x) 
        x=F.max_pool2d(x,2)
        #x=self.conv4(x)
        #x=F.relu(x) 
        #x=F.max_pool2d(x,2)
        #x=self.conv5(x)
        #x=F.max_pool2d(x,2)
        x=self.dropout2(x)
        x=torch.flatten(x,1)
        x=self.fc1(x)
        x=F.relu(x)
        #x=self.nom1(x)
        x=self.fc2(x)
        x=F.relu(x)
        #x=self.nom1(x)
        x=self.fc3(x)
        output=F.log_softmax(x,dim=1)
        return output

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target=data.to(device), target.to(device)
        optimizer.zero_grad()
        output=model(data)
        loss=F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        #print(model.conv1.weight.grad)
        #optimizer.zero_grad()
        #print(model.conv1.weight.grad)

        if batch_idx % args.log_interval == 0:
            print('Train Epoch:{}[{}/{}({:.0f}%)]\tLoss:{:.6f}'.format(
                epoch, batch_idx*len(data), len(train_loader.dataset),
                100.*batch_idx/len(train_loader), loss.item()
            ))
            if args.dry_run:
                break

    

def test(model, device, test_loader):
    model.eval()
    test_loss=0
    correct=0
    with torch.no_grad():
        for data,target in test_loader:
            data, target = data.to(device), target.to(device)
            output=model(data)
            test_loss+=F.nll_loss(output, target, reduction='sum').item()
            pred=output.argmax(dim=1, keepdim=True)
            correct+=pred.eq(target.view_as(pred)).sum().item()
    test_loss/=len(test_loader.dataset)

    print('\nTest set:Average loss:{:.4f}, Accuracy:{}/{}({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100.*correct/len(test_loader.dataset) ))


def main():
    parser = argparse.ArgumentParser(description='pytoch CIFAR-10 classifier')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training(default=64)')
    parser.add_argument('--test_batch_size', type=int, default=100, metavar='N',
                        help='input batch size for testing(default=100)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train(default=14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate(default=1.0)')
    parser.add_argument('--gamma', type=float, default='0.7', metavar='M',
                        help='learning rate step gamma(default=0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args=parser.parse_args()

    use_cuda=not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    device=torch.device('cuda' if use_cuda else 'cpu')

    train_kwargs={'batch_size':args.batch_size}
    test_kwargs={'batch_size':args.test_batch_size}

    if use_cuda:
        cuda_kwargs={'num_workers':0,
        'pin_memory':True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)




    transform= transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                            (0.2470, 0.2435, 0.2616))
    ])



    data_path = './data-universioned/p1ch7/'
    transform_tensor_cifar10 = datasets.CIFAR10(data_path, train=True, download=True,
                                transform=transform)
    transform_tensor_cifar10_val = datasets.CIFAR10(data_path, train=False, download=True,
                                transform=transform)

    train_loader=DataLoader(
        dataset=transform_tensor_cifar10,
        shuffle= True,
        **train_kwargs
    )
    test_loader=DataLoader(
        dataset=transform_tensor_cifar10_val,
        shuffle=True,
        **test_kwargs
    )


    # main
    model=Net().to(device)
    optimizer=optim.Adam(model.parameters(),lr=args.lr)
    #scheduler=StepLR(optimizer, step_size=1,gamma=args.gamma)

    for epoch in range(1,args.epochs+1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        #scheduler.step()
    
    if args.save_model:
        torch.save(model.state_dict(),'CIFAR-10_cnn.pth')

if __name__=='__main__':
    main()


    '''img, label = transform_tensor_cifar10[99]
    print((img, label))
    print(img.shape)
    print(img.min(), img.max())
    plt.imshow(img.permute(1, 2, 0)) 
    #plt.imshow(img)
    pylab.show()
'''
'''imgs = torch.stack([img for img, _ in transform_tensor_cifar10], dim=3)
print(imgs.shape)
print(imgs)
mean=imgs.view(3,-1).mean(dim=1)
std=imgs.view(3,-1).std(dim=1)
'''



