import argparse
parser = argparse.ArgumentParser(description ='HW02 Task2')
parser.add_argument("--imagenet_root", type =str , required= True)
parser.add_argument("--class_list", nargs ="*",type =str , required = True)
args,args_other = parser.parse_known_args()
my_imageroot=args.imagenet_root
my_classlist=args.class_list

my_imageroot=args.imagenet_root
my_classlist=args.class_list

#my_imageroot='/content/drive/MyDrive/Deep_learning/HW2/'
#my_classlist=['cat', 'dog']
import os
import glob
from PIL import Image
import numpy as np

#data_path = os.path.join(input_dir,'*gif')
#files = glob.glob(data_path)
#files.sort()
from torchvision import datasets
from torchvision import transforms as tvt
from torch.utils.data import DataLoader
transform = tvt.Compose([ tvt.ToTensor () , tvt. Normalize((0.5,0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset = datasets.ImageFolder(my_imageroot + '/Train', transform=transform)
test_dataset = datasets.ImageFolder(my_imageroot + '/Val', transform=transform)
train_data_loader = DataLoader(dataset = train_dataset,batch_size =10,shuffle =True,num_workers=4)
val_data_loader = DataLoader(dataset = train_dataset,batch_size =10,shuffle =True,num_workers=4)
import torch
import random
dtype = torch.float64
device = torch.device ("cuda:0" if torch.cuda.is_available () 
                                  else "cpu")
seed = 0
random.seed(seed)
torch.manual_seed(seed)
epochs = 50 # feel free to adjust this parameter
D_in , H1 , H2 , D_out = 3 * 64 * 64 , 1000 , 256 , 2
w1 = torch . randn ( D_in , H1 , device = device , dtype = dtype )
w2 = torch . randn ( H1 , H2 , device = device , dtype = dtype )
w3 = torch . randn ( H2 , D_out , device = device , dtype = dtype )
os.chdir(my_imageroot)
learning_rate = 1e-9
for t in range ( epochs ):
    epoch_loss = []
    for i, data in enumerate (train_data_loader):
        inputs , labels = data
        inputs = inputs.to(device).double()
        labels = labels.to(device)
        labels=torch.nn.functional.one_hot(labels)
        y=labels
        x = inputs.view(inputs.size ( 0 ), -1)
        h1 = x.mm(w1)
        h1_relu = h1.clamp(min =0)
        h2 = h1_relu.mm(w2)
        h2_relu = h2.clamp (min =0)
        y_pred = h2_relu.mm(w3)
        loss = (y_pred-y).pow(2).sum().item()
        y_error = y_pred-y
        grad_w3 = h2_relu.t().mm(2 * y_error)
        h2_error = 2.0 * y_error .mm(w3.t())
        h2_error[h2 < 0] = 0
        grad_w2 = h1_relu.t().mm(2*h2_error)
        h1_error = 2.0 * h2_error.mm(w2.t())
        h1_error[h1 < 0] = 0
        grad_w1 = x.t().mm(2*h1_error )
        w1 -= learning_rate * grad_w1
        w2 -= learning_rate * grad_w2
        w3 -= learning_rate * grad_w3
        epoch_loss.append(loss)
    epoch_loss = np.mean(epoch_loss)    
    if t==0:
       with open('output.txt', 'w') as f:
	          print( 'Epoch %d:\t %0.4f'%(t , epoch_loss) , file = f)
    else:
	   	  with open('output.txt', 'a') as f:
	           print( 'Epoch %d:\t %0.4f'%(t , epoch_loss) , file = f)
torch.save ({'w1':w1 ,'w2':w2 ,'w3':w3},'./wts.pkl')
###validation
os.chdir(my_imageroot)
weights = torch.load(my_imageroot+'wts.pkl')
w1 = weights['w1']
w2 = weights['w2']
w3 = weights['w3']

for i , data in enumerate (val_data_loader):
    inputs , labels = data
    inputs = inputs.to(device).double()
    labels = labels.to(device)
    labels=torch.nn.functional.one_hot(labels)
    y=labels
    x = inputs.view(inputs.size(0),-1)
    h1 = x.mm(w1)
    h1_relu = h1.clamp(min =0)
    h2 = h1_relu.mm(w2)
    h2_relu = h2.clamp (min =0)
    y_pred = h2_relu.mm(w3)
    loss = (y_pred-y).pow(2).sum().item()
acc=0
for i in range(len(y)):
		if torch.argmax(y_pred[i]) == torch.argmax(y[i]):
			acc += 1
val_accuracy_value =acc/len(y)*100
val_loss=loss/len(y)
with open('output.txt', 'a') as f:
			f.write('\n')
			print( 'Val Loss:\t %0.2f'%(val_loss) , file = f)
			print( 'Val Accuracy:\t %0.2f'%(val_accuracy_value)+'%', file = f) 