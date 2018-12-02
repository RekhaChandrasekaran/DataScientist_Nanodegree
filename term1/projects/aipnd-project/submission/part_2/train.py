# Import libraries
import argparse
import helper
import os
import torch

# Argument Parser
parser = argparse.ArgumentParser(description='Image Classifier Application')
parser.add_argument('data_dir', action = 'store', help='Provide directory that contains /train, /test, /valid.')
parser.add_argument('--save_dir', action = 'store', dest='save_dir', default='./checkpoints', 
                    help='Provide directory to save checkpoints.')
parser.add_argument('--arch', action = 'store', dest='arch', default='densenet121', 
                    help='Pre-trained model architecture : densenet121 or alexnet')
parser.add_argument('--learning_rate', action = 'store', dest='learning_rate', type=float, default=0.001,
                    help='Learning rate')
parser.add_argument('--hidden_units', action = 'store', dest='hidden_units', type=int, default=512,
                    help='Number of hidden nodes')
parser.add_argument('--epochs', action = 'store', dest='epochs', type=int, default=5,
                    help='Number of Epochs')
parser.add_argument('--gpu', action = 'store_true', dest='gpu', default=False,
                    help='Set to use gpu for training')

# Parse arguments
args = parser.parse_args()

data_dir = str(args.data_dir)
save_dir = str(args.save_dir)
arch = str(args.arch)
learning_rate = float(args.learning_rate)
hidden_units = int(args.hidden_units)
epochs = int(args.epochs)
gpu = args.gpu

print('------- Selected Options -------')
print('data_dir       = {}'.format(data_dir))
print('save_dir       = {}'.format(save_dir))
print('arch           = {}'.format(arch))
print('learning_rate  = {}'.format(learning_rate))
print('hidden_units   = {}'.format(hidden_units))
print('epochs         = {}'.format(epochs))
print('gpu            = {}'.format(gpu))
print('--------------------------------')

# Load data
trainloader, validloader, testloader, class_to_idx, class_labels = helper.load_data(data_dir)
output_size = len(class_labels)

# Train model
device = 'cuda' if gpu else 'cpu'
model, criterion, optimizer = helper.trainer(device, arch, hidden_units, trainloader, validloader, output_size, learning_rate, epochs)

# Test model
print('------------ Testing ------------')
helper.tester(device, model, testloader, criterion)

# Save Checkpoint
checkpoint = {'model_arch': arch,
              'hidden_units': hidden_units,
              'model_state': model.state_dict(),
              'optimizer_state': optimizer.state_dict(),
              'lr': learning_rate,
              'epochs': epochs,
              'class_to_idx': class_to_idx,
              'class_labels': class_labels}

directory = str(save_dir)
if not os.path.exists(directory):
    os.makedirs(directory)

checkpoint_path = save_dir + '/IC_checkpoint_{}_{}_{}.pth'.format(arch, hidden_units, epochs)
torch.save(checkpoint, checkpoint_path)
print('--- Model saved at {} ---'.format(checkpoint_path))


