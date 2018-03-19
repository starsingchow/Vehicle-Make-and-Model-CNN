import tensorflow as tf
import cv2
import os

def fancy_pca(dir):
    filenames = os.listdir(dir)
    filenames.remove('.DS_Store')
    
    for file in filenames:
        file_dir = os.path.join(dir, file)
        image = cv2.imread(file_dir)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        
