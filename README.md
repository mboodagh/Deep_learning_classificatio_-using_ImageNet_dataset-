# Deep_learning_classification_using_ImageNet_dataset
Classifying images of the ImageNet
This repositoray contains two files. The first file "ImageNet_downloader.py", downloades the images from ImageNet. The name of the imgase are stored in a json file named "imagenet_class_info.json". "ImageNet_downloader.py" reads the file "imagenet_class_info.json" and then downloads the images correspnding to the urls of the cats or dogs. Then it downsizes all the imgaes to the size 3*64*64 and stores files in val/train directories for training and validation. "ImageNet_downloader.py"
The "cat_dog_classification.py" used pytorch DataLoader for loading the data and classifies the images using the one-hot coding.  
