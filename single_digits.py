import torch
import torchvision.transforms as transforms
import torch.utils.data as data_utils

def prepare_data(args):
    # Define transformations
    transform_CM1 = transforms.Compose([
        transforms.StandardScaler(),
        transforms.CPA() ])
    
    transform_PC1 = transforms.Compose([
        transforms.StandardScaler(),
        transforms.CPA() ])
    
    # Define datasets
    CM1_trainset = data_utils.DigitsDataset( data_path="../data/CM1",  num_partitions=args.num_partitions，transform=transform_CM1）
    CM1_testset = data_utils.DigitsDataset( data_path="../data/CM1",num_partitions=args.num_partitions， transform=transform_CM1  ）
    
    PC1_trainset = data_utils.DigitsDataset(data_path="../data/PC1", num_partitions=args.num_partitions，transform=transform_PC1  )
    PC1_testset = data_utils.DigitsDataset(data_path="../data/PC1", num_partitions=args.num_partitions，transform=transform_PC1 )
    
    # Define data loaders based on selected dataset
    if args.data.lower() == 'cm1':
        train_loader = torch.utils.data.DataLoader(CM1_trainset, batch_size=args.batch, shuffle=True)
        test_loader = torch.utils.data.DataLoader(CM1_testset,  batch_size=args.batch,  shuffle=False )
    elif args.data.lower() == 'pc1':
        train_loader = torch.utils.data.DataLoader( PC1_trainset, batch_size=args.batch, shuffle=True)
        test_loader = torch.utils.data.DataLoader(PC1_testset, batch_size=args.batch, shuffle=False)
    else:
        raise ValueError('Unknown dataset')
    
    return train_loader, test_loader

def train(data_loader, optimizer, loss_fun, device):
    loss_all = 0
    total = 0
    correct = 0
    for data, target in data_loader:
        optimizer.zero_grad()

        data = data.to(device).float()
        target = target.to(device).long()
        output = model(data)
        loss = loss_fun(output, target)
        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

        loss.backward()
        optimizer.step()

    return loss_all / len(data_loader), correct/total

def test(data_loader,site, loss_fun, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device).float()
            target = target.to(device).long()
            output = model(data)
            test_loss += loss_fun(output, target).item()
            pred = output.data.max(1)[1]
            correct += pred.eq(target.view(-1)).sum().item()
            total += target.size(0)

    test_loss /= len(data_loader)
    correct /= total
    print(' {} | Test loss: {:.4f} | Test acc: {:.4f}'.format(site, test_loss, correct))

    if log:
        logfile.write(' {} | Test loss: {:.4f} | Test acc: {:.4f}\n'.format(site, test_loss, correct))


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_partitions', type = float, default= 4, help ='num_partitions of dataset to train')
    parser.add_argument('--log', action='store_true', help='whether to log')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--batch', type = int, default= 32, help ='batch size')
    parser.add_argument('--data', type = str, default= 'CM1', help='[CM1| PC1]')
    parser.add_argument('--save_path', type = str, default='../checkpoint/digits', help='path to save the checkpoint')
    args = parser.parse_args()

    exp_folder = 'singleset_digits'

    args.save_path = os.path.join(args.save_path, exp_folder)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    SAVE_PATH = os.path.join(args.save_path,'SingleSet_{}'.format(args.data))

    log = args.log
    if log:
        log_path = os.path.join('../logs/digits', exp_folder)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        logfile = open(os.path.join(log_path,'SingleSet_{}.log'.format(args.data)), 'a')
        logfile.write('==={}===\n'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        logfile.write('===Setting===\n')
        logfile.write('    lr: {}\n'.format(args.lr))
        logfile.write('    batch: {}\n'.format(args.batch))
        logfile.write('    dataset: {}\n'.format(args.data))
        logfile.write('    epochs: {}\n'.format(args.epochs))

    model = DigitModel().to(device)
    loss_fun = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    train_loader, test_loader = prepare_data()

    for epoch in range(args.epochs):
        print(f"Epoch: {epoch}" )
        loss,acc = train(train_loader, optimizer, loss_fun, device)
        print(' {} | Train loss: {:.4f} | Train acc : {:.4f}'.format(args.data, loss,acc))

        if log:
            logfile.write('Epoch Number {}\n'.format(epoch))
            logfile.write('Train loss: {:.4f} and accuracy : {:.4f}\n'.format(loss, acc))
            logfile.flush()

        test(test_loader, args.data, loss_fun, device)

    print(' Saving the best checkpoint to {}...'.format(SAVE_PATH))
    torch.save({
        'model': model.state_dict(),
        'epoch': epoch
    }, SAVE_PATH)

    if log:
        logfile.flush()
        logfile.close()


