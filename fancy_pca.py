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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float64)

        image -= np.mean(image, axis = 0)
        cov = np.dot(image.T, image)/(image.shape[0]-1)
        print(cov.shape)
        val, vect = np.linalg.eig(cov)

        alpha = np.random.normal(loc=0.0, scale =1.0, size=(3,))
        weights = alpha * val
        add = np.dot(vect, weights.reshape((-1, 1)))
        r_num = add[0][0]
        g_num = add[1][0]
        b_num = add[2][0]

        heigh = image.shape[0]
        width = image.shape[1]

        for i in range(heigh):
            for j in range(width):
                pixel = image[i,j,:]
                pixel[0] += r_num
                pixel[0] = int(pixel[0])
                pixel[1] += g_num
                pixel[1] = int(pixel[1])
                pixel[2] += b_num
                pixel[2] = int(pixel[2])
        
        cv2.imwrite(os.path.join(save, file),cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
def 

if __name__=='__main__':
    fancy_pca(dir, save)
