# Deep_learning_using_ImageNet_dataset_for_classification
This repository contains two ".py" files. The first file, "ImageNet_downloader.py" downloads the images from ImageNet database and the second file "cat_dog_classification.py" classifies images using a feed-forward neural network. The names of the images are stored in a JSON file named "imagenet_class_info.json". "ImageNet_downloader.py" reads the file "imagenet_class_info.json" and then downloads the images corresponding to the URLs of the cats or dogs. Then it downsizes all the images to the size 3*64*64 and stores files in val/train directories for training and validation. 
The "cat_dog_classification.py" used PyTorch DataLoader for loading the data and classifying the images using the one-hot coding. 


The file "cat_dog_classification.py" utilisez argparse and after execution asks for the input root.
