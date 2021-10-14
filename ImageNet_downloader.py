####Parser section
import argparse
parser = argparse.ArgumentParser ( description ='HW02 Task1')

import json
import os
import requests
from requests.exceptions import ConnectionError, ReadTimeout, TooManyRedirects, MissingSchema, InvalidURL
from PIL import Image


im_per_subclass=200
sub_cls_list=['working dog','police dog']# choose the groups of dogs or cats
d_root='F:\Deep_learning\Val'
path_to_jason='F:\Deep_learning\imagenet_class_info.json'
class_name = 'dog'
#class_name = 'cat' #choose either cat or dog


x = open(path_to_jason)
y = json.load(x)
ns=[]
for k in y.keys():
    for ss in range(len(sub_cls_list)):
        if y[k]['class_name']==sub_cls_list[ss]:
           ns.append(k)
ur='http://www.image-net.org/api/text/imagenet.synset.geturls?wnid='
class_folder=os.path.join(d_root,class_name)
if not os.path.isdir(class_folder):
   os.makedirs(class_folder)
k=0
for n in ns:
    the_list_url=ur+n
    resp = requests.get(the_list_url)
    urls = [url.decode("utf-8") for url in resp.content.splitlines()]
    urls.remove('')
    i=0
    j=0
    for url in urls:
        img_url=url
        if len ( img_url ) <= 1:
           continue
        try:
           img_resp = requests.get( img_url , timeout = 1)
        except ConnectionError :
              #print('There is a ConnectionError')
               i+=1
               continue
        except ReadTimeout:
              #print('There is a ReadTimeout error')
               i+=1
               continue
        except TooManyRedirects:
              #print('There is a TooManyRedirects error')
               i+=1
               continue
        except MissingSchema:
              #print('There is a MissingSchema error')
               i+=1
               continue
        except InvalidURL:
              #print('There is an InvalidURL error')
               i+=1
               continue
        if not 'content-type' in img_resp.headers:
           i+=1
           continue
        if not 'image' in img_resp.headers ['content-type']:
           i+=1 
           continue
        if (len(img_resp.content)< 1000):
           i+=1
           continue
        img_name = img_url.split ('/')[-1]
        img_name = img_name.split ("?")[0]
        if (len ( img_name ) <= 1):
           i+=1
           continue
        if not 'flickr' in img_url:
           i+=1
           continue
        img_file_path = os.path.join(class_folder,img_name) 
        if os.path.exists(img_file_path):
           continue
        #print(class_folder)
        if j==im_per_subclass:
           break
        j+=1
        k+=1
        with open(img_file_path ,'wb') as img_f:
            #try: 
             #print(img_file_path)
             img_f.write(img_resp.content)
        im = Image.open(img_file_path)
            #except UnidentifiedImageError:
             #      continue
        if im.mode != " RGB ":
           im = im.convert(mode ="RGB")
        im_resized = im.resize((64 , 64), Image.BOX)
        im_resized.save(img_file_path)

print(k)  

