# Import libraries
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from torch import nn, optim
from collections import OrderedDict
from PIL import Image

def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_transforms = transforms.Compose([transforms.RandomRotation([-30,30]), transforms.RandomResizedCrop(224),
                                            transforms.RandomHorizontalFlip(), transforms.ToTensor(), 
                                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    test_transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), 
                                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform = test_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    
    class_to_idx = train_data.class_to_idx
    class_labels = train_data.classes
    
    return trainloader, validloader, testloader, class_to_idx, class_labels

# validation
def validation(device, model, dataloader, criterion): 
    device, model = device, model
    dataloader = dataloader
    criterion = criterion
    correct, total, loss, accuracy = 0, 0, 0, 0
    for data in dataloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        loss += criterion(outputs, labels).item()
    accuracy = (correct / total) * 100
    
    return loss, accuracy

# training
def train_model(device, model, criterion, optimizer, trainloader, validloader, epochs, print_every):
    steps = 0
    # change to device
    model.to(device)

    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1        
            inputs, labels = inputs.to(device), labels.to(device)        
            optimizer.zero_grad()        
            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()        
            running_loss += loss.item()        
            if steps % print_every == 0:
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Loss: {:.4f}".format(running_loss/print_every))            
                running_loss = 0                
        # eval mode
        model.eval()
        
        # validation
        with torch.no_grad():            
            train_loss, train_accuracy = validation(device, model, trainloader, criterion)
            valid_loss, valid_accuracy = validation(device, model, validloader, criterion)
            print("Epoch: {}/{}... ".format(e+1, epochs), 
                  "Training Loss: {:.3f}, ".format(train_loss), "Training Accuracy: {:.3f}, ".format(train_accuracy),
                  "Validation Loss: {:.3f}, ".format(valid_loss), "Validation Accuracy: {:.3f}".format(valid_accuracy))
        # training mode
        model.train()

def trainer(device, arch, hidden_units, trainloader, validloader, output_size, learning_rate, epochs):
    if arch!= 'alexnet':
        arch = 'densenet121'
    
    # load a pre-trained model : DensetNet
    if arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        # Freeze parameters of pre-trained model
        for param in model.parameters():
            param.requires_grad = False
        # define feed-forward classifier
        classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(1024, hidden_units)),
                          ('relu1', nn.ReLU()),
                          ('drop1', nn.Dropout(p=0.2)),
                          ('fc2', nn.Linear(hidden_units, 256)),
                          ('relu2', nn.ReLU()),
                          ('drop2', nn.Dropout(p=0.1)),
                          ('fc3', nn.Linear(256, output_size)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))    
        model.classifier = classifier
    else:
        # load a pre-trained model : AlexNet
        model = models.alexnet(pretrained=True)
        # Freeze parameters of pre-trained model
        for param in model.parameters():
            param.requires_grad = False
        # define feed-forward classifier
        classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(9216, hidden_units)),
                          ('relu1', nn.ReLU()),
                          ('drop1', nn.Dropout(p=0.2)),
                          ('fc2', nn.Linear(hidden_units, output_size)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))    
        model.classifier = classifier      
        
    # loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
        
    # train for epochs
    train_model(device, model, criterion, optimizer, trainloader, validloader, epochs=epochs, print_every=40)
    print('------------------ {} Model Trained Successfully ------------------'.format(arch))
    
    return model, criterion, optimizer
        
def tester(device, model, testloader, criterion):
    # eval mode
    model.eval()
      
    with torch.no_grad():            
        test_loss, test_accuracy = validation(device, model, testloader, criterion)

    # training mode
    model.train()

    print('Accuracy of the network on the test images: %d %%' % test_accuracy)

def load_checkpoint(device, checkpoint_file):    
    checkpoint = torch.load(checkpoint_file)
    arch = checkpoint['model_arch']
    class_labels = checkpoint['class_labels']
    hidden_units = checkpoint['hidden_units']
    output_size = len(class_labels)
    
    # load a pre-trained model : DensetNet
    if arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        # Freeze parameters of pre-trained model
        for param in model.parameters():
            param.requires_grad = False
        # define feed-forward classifier
        classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(1024, hidden_units)),
                          ('relu1', nn.ReLU()),
                          ('drop1', nn.Dropout(p=0.2)),
                          ('fc2', nn.Linear(hidden_units, 256)),
                          ('relu2', nn.ReLU()),
                          ('drop2', nn.Dropout(p=0.1)),
                          ('fc3', nn.Linear(256, output_size)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))    
        model.classifier = classifier
    else:
        # load a pre-trained model : AlexNet
        model = models.alexnet(pretrained=True)
        # Freeze parameters of pre-trained model
        for param in model.parameters():
            param.requires_grad = False
        # define feed-forward classifier
        classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(9216, hidden_units)),
                          ('relu1', nn.ReLU()),
                          ('drop1', nn.Dropout(p=0.2)),
                          ('fc2', nn.Linear(hidden_units, output_size)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))    
        model.classifier = classifier  
    
    model.to(device)   
    
    model.load_state_dict(checkpoint['model_state'])
    
    # loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=checkpoint['lr'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model, criterion, optimizer, class_labels

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    
    test_transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), 
                                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    
    pil_image = Image.open(image)
    pil_image = test_transforms(pil_image)
    
    np_array = np.array(pil_image.float())
    
    return np_array

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax
