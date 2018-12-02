# Import libraries
import argparse
import helper
import os
import torch
import numpy as np
import json

''' Predict the class (or classes) of an image using a trained deep learning model.
'''

# Argument Parser
parser = argparse.ArgumentParser(description='Image Classifier Application')
parser.add_argument('image_path', action = 'store', help='Test Image Path')
parser.add_argument('checkpoint_path', action = 'store', help='Checkpoint File Path')
parser.add_argument('--top_k', action = 'store', dest = 'top_k', type = int, default = 1, 
                    help='Top K most likely classes')
parser.add_argument('--category_names', action = 'store', dest = 'category_names', default = 'cat_to_name.json', 
                    help='Mapping of categories to real names')
parser.add_argument('--gpu', action = 'store_true', dest='gpu', default=False,
                    help='Set to use gpu for inference')

# Parse arguments
args = parser.parse_args()

image_path = str(args.image_path)
checkpoint_path = str(args.checkpoint_path)
top_k = int(args.top_k)
category_names = str(args.category_names)
gpu = args.gpu

print('------- Selected Options -------')
print('img_path        = {}'.format(image_path))
print('checkpoint_path = {}'.format(checkpoint_path))
print('top_k           = {}'.format(top_k))
print('category_names  = {}'.format(category_names))
print('gpu             = {}'.format(gpu))
print('--------------------------------')

# Load Categories
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
# Load checkpoint
device = 'cuda' if gpu else 'cpu'
model, criterion, optimizer, class_labels = helper.load_checkpoint(device, checkpoint_path)

# Process image
image = helper.process_image(image_path)
image_tensor = torch.from_numpy(image).type(torch.FloatTensor)
image_tensor.resize_([1, 3, 224, 224])

model.eval()
model.to(device)
image_tensor = image_tensor.to(device)

result = model(image_tensor)
result = torch.exp(result)
    
probs, idx = result.topk(top_k)
probs.detach_()
probs.resize_([top_k])
probs = probs.tolist()
    
idx.detach_()
idx.resize_([top_k])
idx = idx.tolist()
    
classes = [class_labels[i] for i in idx]
class_names = [cat_to_name[i] for i in classes]

print('----------- Predicted Class(es) -----------')
for i in range(len(probs)):
    print("Probability of the test image to be \'{}\' is {:.2f}.".format(class_names[i], probs[i]))

