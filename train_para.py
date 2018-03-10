# -*- coding: utf-8 -*-

class train_para():
    def __init__(self, image_size, lr, lr_decay, 
                train_steps, reg_rate = None, 
                moving_average_decay = None,
                skip = None, train_type = 'full train',
                train_list = None
                ):
        '''
        image_size: int, input data size
        lr: float, learning rate
        lr_decay: float, learning rate decay 
        train_steps: int, training steps
        reg_rate: float, regularaztion rate
        moving_average_decay: float moving average decay
        skip: list skip layer
        '''
        self.image_size = image_size
        self.lr = lr
        self.lr_decay = lr_decay
        self.train_steps = train_steps
        self.reg_rate = reg_rate
        self.moving_average_decay = moving_average_decay
        self.skip = skip
        self.train_list = train_list
        if self.skip == None:
                self.skip = []
        
        if train_type == 'full train':
                self.train_type = 0
        elif train_type == 'part fune':
                self.trian_type = 1
        elif train_type == 'fine tune':
                self.train_type = 2
        else:
                raise ValueError('please input right train type lilk: full trian, part fune fine tune')


    
