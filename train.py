# Created by Baole Fang at 4/2/24
import argparse
import os.path

import wandb
import yaml
from utils import prepare
import torch.nn as nn
import torch
from tqdm import tqdm
import shutil


def train(train_loader, model, optimizer, scheduler, criterion, epoch):
    model.train()
    total_loss = 0
    phar = tqdm(train_loader, desc=f'Epoch {epoch}')
    correct = 0
    for data, target in phar:
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        pred = model(data)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        curr_correct = (pred.argmax(1) == target).sum().item()
        correct += curr_correct
        phar.set_postfix(loss=loss.item(), auc=curr_correct / len(data))
    scheduler.step()
    return total_loss / len(train_loader), correct / len(train_loader.dataset)


def evaluate(test_loader, model, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            pred = model(data)
            loss = criterion(pred, target)
            total_loss += loss.item()
            correct += (pred.argmax(1) == target).sum().item()
    return total_loss / len(test_loader), correct / len(test_loader.dataset)


def main(config, gpus):
    # init wandb
    run = wandb.init(config=config, project=config['experiment'])

    train_loader, test_loader, model, optimizer, scheduler = prepare(config, gpus)
    criterion = nn.CrossEntropyLoss()
    # train your model
    for epoch in range(config['epochs']):
        # train your model
        train_loss, train_acc = train(train_loader, model, optimizer, scheduler, criterion, epoch)
        # validate your model
        test_loss, test_acc = evaluate(test_loader, model, criterion)
        # log your results
        with open(os.path.join(config['output_dir'], config['experiment'], 'log.txt'), 'a') as f:
            f.write(f'Epoch {epoch}: train_loss: {train_loss}, train_acc: {train_acc}, test_loss: {test_loss}, test_acc: {test_acc}\n')
        wandb.log({'epoch': epoch, 'train_loss': train_loss, train_acc: {train_acc}, 'test_loss': test_loss, 'test_acc': test_acc})
        # save your model
        if epoch % config['save_interval'] == 0:
            output_path = os.path.join(config['output_dir'], config['experiment'], 'checkpoints', f'epoch_{epoch}.pth')
            if isinstance(model, nn.DataParallel):
                model_state = model.module.state_dict()
            else:
                model_state = model.state_dict()
            torch.save({
                'model': model_state,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, output_path)
            wandb.save(output_path)
    run.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/cifar100.yaml', help='path to the configuration file')
    parser.add_argument('--gpus', type=str, default='0', help='gpus to use')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    output_root = os.path.join(config['output_dir'], config['experiment'], 'checkpoints')
    if os.path.exists(output_root):
        if not config['experiment'].endswith('_'):
            print(f'Experiment {config["experiment"]} already exists. Enter y to overwrite.')
            choice = input()
            if choice != 'y':
                exit()
        shutil.rmtree(os.path.join(config['output_dir'], config['experiment']))
    os.makedirs(output_root, exist_ok=True)
    main(config, list(map(int, args.gpus.split(','))))
