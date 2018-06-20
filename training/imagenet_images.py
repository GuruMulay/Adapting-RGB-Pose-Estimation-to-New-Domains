import os
import numpy as np
import random


imagenet_folders_without_human = ['train_set_1/n01443537/', 'train_set_1/n01580077/', 'train_set_2/n02226429/', 'train_set_5/n13044778/', 'train_set_2/n02268443/', 'train_set_1/n01775062/', 'train_set_1/n01608432/']

imagenet_images_with_rare_human = ['train_set_1/n01530575/', 'train_set_3/n03530642/', 'train_set_4/n03717622/', 'train_set_4/n03777754/', 'train_set_5/n07715103/']


imagenet_folders_all = imagenet_folders_without_human  + imagenet_images_with_rare_human
random.shuffle(imagenet_folders_all)

imagenet_train = imagenet_folders_all[:int(0.8*len(imagenet_folders_all))]
imagenet_val = imagenet_folders_all[int(0.8*len(imagenet_folders_all)):]

print("train and val", imagenet_train, imagenet_val)


class ImagenetImages:

    def __init__(self, imagenet_base_dir='/s/red/a/nobackup/imagenet/images/train/'):

        self.imagenet_base_dir = imagenet_base_dir
        
        self.partition_dict = {}
            
    
    def get_n_images(self, mode='train', n_images=1):
        partition = []
        
        if mode == 'train':
            imagenet_folders = imagenet_train
        else:
            imagenet_folders = imagenet_val
        
        for f in imagenet_folders:
            for img in os.listdir(os.path.join(self.imagenet_base_dir, f)):
                if img.endswith(".JPEG"):
                    partition.append(f + img)
        
        print("len(partition)", len(partition))
        print("partition example", partition[1], partition[-1])
    
        random.shuffle(partition)
        
        return partition[0:n_images]
                    
    
    
def test_imagenet_image_reader():
    im = ImagenetImages('/s/red/a/nobackup/imagenet/images/train/')
    train = im.get_n_images("train", 100)
    val = im.get_n_images("val", 100)
    
    # print("train and val", train, val)
    

if __name__ == "__main__":
    test_imagenet_image_reader()