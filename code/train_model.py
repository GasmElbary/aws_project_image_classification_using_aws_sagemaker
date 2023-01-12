import json
import logging
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import argparse

import smdebug.pytorch as smd

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

logger.info("Torchvision Version is: " + str(torchvision.__version__))

def test(model, test_loader, criterion, device, hook):
    model.eval()
    hook.set_mode(smd.modes.EVAL)
    running_loss = 0
    running_corrects = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs=inputs.to(device)
            labels=labels.to(device)
            outputs=model(inputs)
            loss=criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()

    avg_loss = running_loss / len(test_loader.dataset)
    acc = running_corrects/ len(test_loader.dataset)
    logger.info("    Testing - loss: {:.4f}, Accuracy: {:.2f}% \n".format(
        avg_loss, 
        100.0 * acc
    ))

def train(model, train_loader, criterion, optimizer, device, hook):    
    model.train()
    hook.set_mode(smd.modes.TRAIN)
    running_loss = 0
    running_corrects = 0
    running_samples = 0
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()
        running_samples+=len(inputs)
        if running_samples % 4000  == 0:
            logger.info(f"    Running_samples: {running_samples}/14,000" )
        
    epoch_loss = running_loss / running_samples
    epoch_acc = running_corrects / running_samples
    logger.info("    Training - loss: {:.4f}, Accuracy: {:.2f}% \n".format(
        epoch_loss, 
        100.0 * epoch_acc
    ))
    return model
    
def net():
    model = models.efficientnet_b7(pretrained = True) #using a pretrained efficientnetV2 model, which is accurate and fast.
    
    for param in model.parameters():
        param.requires_grad = False

    num_features=model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(nn.Linear(num_features, 256),
                             nn.ReLU(),
                             nn.Linear(256, 256),
                             nn.ReLU(),
                             nn.Linear(256, 128),
                             nn.ReLU(),
                             nn.Linear(128, 6),
                             nn.LogSoftmax(dim=1))
    
    return model

def create_data_loaders(data_dir, batch_size):
    logger.info("Getting train/test data loaders")
    train_data_dir = os.path.join(data_dir, "seg_train")
    test_data_dir = os.path.join(data_dir, "seg_test")
    
    training_transform = transforms.Compose([
                                    transforms.RandomHorizontalFlip(p=0.5), 
                                    transforms.Resize((128, 128)),
                                    transforms.ToTensor(),                  
                                    ])
    
    testing_transform = transforms.Compose([
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.Resize((128, 128)),
                                    transforms.ToTensor(),                  
                                    ])

    train_data_loader = torch.utils.data.DataLoader(
                                torchvision.datasets.ImageFolder(
                                                    root=train_data_dir, 
                                                    transform=training_transform
                                ), 
                                batch_size=batch_size, 
                                shuffle=True)
    
    test_data_loader = torch.utils.data.DataLoader(
                                torchvision.datasets.ImageFolder(
                                                    root=test_data_dir, 
                                                    transform=testing_transform
                                ), 
                                batch_size=batch_size, 
                                shuffle=True)
    
    return train_data_loader, test_data_loader 

def main(args):
    logger.info("Initializing the model\n")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("\n Hyperparameters:")
    logger.info(f"\n     Epochs: {args.epochs}")
    logger.info(f"\n     Batch Size: {args.batch_size}")
    logger.info(f"\n     LR: {args.lr}")
    logger.info(f"\n     Eps: {args.eps}")
    
    model=net()
    model=model.to(device)
    
    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)

    loss_criterion = nn.NLLLoss()
    hook.register_loss(loss_criterion)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=args.eps)
    
    train_loader, test_loader = create_data_loaders(args.data_dir, args.batch_size )
    
    logger.info("Training/testing Started")
    for epoch in range(1, args.epochs + 1):
        logger.info(f"\n  Epoch: {epoch}" )
        model = train(model, train_loader, loss_criterion, optimizer, device, hook)
        test(model, test_loader, loss_criterion, device, hook)
        
    logger.info("Saving the model")
    model_path = os.path.join(args.model_dir, "model.pth")
    torch.save(model.state_dict(), model_path)
    logger.info("Model Saved")

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size", type=int, default=128, metavar="N", help="input batch size for training (default: 128)",
    )

    parser.add_argument(
        "--epochs", type=int, default=5, metavar="N", help="number of epochs to train (default: 5)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.05, metavar="LR", help="learning rate (default: 0.05)"
    )
    parser.add_argument(
        "--eps", type=float, default=0.0000001, metavar="EPS", help="epsilon (default: 0.0000001)"
    )

    # Container environment
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data_dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])

    args=parser.parse_args()
    
    main(args)
