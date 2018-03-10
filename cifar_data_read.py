import tensorflow as tf
import numpy as np

import os

class cifar():
    def __init__(self,dir):
        '''
        dir: the file dir
        '''
        self.train_filenames = [os.path.join(dir, 'data_batch_%d'%i) for i in range(1,6)]
        self.file_select= 0
        self.test_filename = os.path.join(dir,'test_batch') 
        self.train_start = 0
        self.test_start = 0
        self.train_data = None
        self.tarin_label = None
        self.test_data = None
        self.test_label = None

    def setData(self,file_select,type='train'):
        '''
        file_select: the file which is selected
        '''
        import pickle
        with tf.gfile.Open(file_select, 'rb') as fo:
                dict=pickle.load(fo, encoding='bytes')
        if type == 'train':
            train_data = dict[b'data'].reshape(10000,3,32,32)
            self.train_data = np.transpose(np.array(train_data),(0,2,3,1))
            self.train_label = np.array(dict[b'labels']).reshape(10000,1)
        elif type == 'test':
            test_data = dict[b'data'].reshape(10000,3,32,32)
            self.test_data = np.transpose(np.array(test_data),(0,2,3,1))
            self.test_label = np.array(dict[b'labels']).reshape(10000,1)
        else:
            raise Exception("type error")
 
    def batch(self, BATCH_SIZE,type='train'):
        select_type={
            'train':self.train_batch(BATCH_SIZE),
            'test':self.test_batch(BATCH_SIZE)
        }
        return select_type.get(type)
        
    def train_batch(self,BATCH_SIZE):
        if (self.train_start >= 10000 or self.train_start == 0):
            try:
                filename = self.train_filenames[self.file_select]
            except IndexError:
                self.file_select = 0
                filename = self.train_filenames[self.file_select]
            finally:
                self.file_select += 1

            self.setData(filename)
            if(self.train_start >= 10000):
                self.train_start = 0
        
        train_start = self.train_start

        data = self.train_data[train_start:(train_start+BATCH_SIZE),:,:,:]
        label = self.train_label[train_start:(train_start+BATCH_SIZE),:]

            
        self.train_start += BATCH_SIZE
        return data,label
    
    def test_batch(self,BATCH_SIZE):
        self.setData(self.test_filename,'test')
        if self.test_start >= 10000:
            return None,None
        test_start = self.test_start

        data = self.test_data[test_start:(test_start+BATCH_SIZE),:,:,:]
        label = self.test_label[test_start:(test_start+BATCH_SIZE),:]


        
        self.test_start += BATCH_SIZE
        return data,label