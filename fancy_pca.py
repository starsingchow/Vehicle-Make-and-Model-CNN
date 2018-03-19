import cv2
import numpy as np
import os

dir = './car_data/save/train'
save = './car_data/m'

def fancy_pca(dir,save):
    filenames = os.listdir(dir)
    filenames.remove('.DS_Store')
    
    for file in filenames:
        file_dir = os.path.join(dir, file)
        image = cv2.imread(file_dir)
        image = image.astype(np.float64)
        height = image.shape[0]
        width = image.shape[1]   

        image_T = image.reshape(height*width,3)

        image_T -= np.mean(image_T, axis=0)
        cov = np.dot(image_T.T, image_T)/(image_T.shape[0] - 1)
        val, vect = np.linalg.eig(cov)

        alpha = np.random.normal(loc=0.0, scale =1.0, size=(3,))
        weights = alpha * val
        add = np.dot(vect, weights.T)
        for i in range(height):
            for j in range(width):
                image[i,j,0] += add[0]
                image[i,j,1] += add[1]
                image[i,j,2] += add[2]
        
        image.astype(np.uint8)
        cv2.imwrite(os.path.join(save, file), image)

if __name__=='__main__':
    fancy_pca(dir, save)
