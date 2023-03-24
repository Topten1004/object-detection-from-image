from re import I
from turtle import color, width
from matplotlib import pyplot as plt, testing
import numpy as np
from regex import P
import tensorflow as tf
import os
from torch import Size
from tqdm import tqdm
import cv2
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt



def standardize_dataset(x, axis=None):
    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.sqrt(((x - mean)**2).mean(axis=axis, keepdims=True))
    return (x - mean) / std

def add_gaussian_noise(X, mean=0, std=1):
    """Returns a copy of X with Gaussian noise."""
    return X.copy() + std * np.random.standard_normal(X.shape) + mean

class DataManager:
    def __init__(self):                

        self.test_input_img = []
        self.training_set_size = None
        self.load_data()                

    def load_data(self):        
        LR_SIZE = 128
        SIZE = 512
                
                
        # path = './dataset/DIV2K_train_LR_bicubic/train/hr_res'
        # path_npy_LR = 'dataset/DIV2K_train_LR_bicubic/train/npy_files/LR_res/'

        path = './dataset/train/low_res'
        path_npy_LR = 'dataset/train/npy_files/LR_res/'
        files = os.listdir(path_npy_LR)
        origin_files = os.listdir( path )
        
        if len(files) == 0:
            for i in tqdm(origin_files):
                img = cv2.imread(path+'/'+i, 1)        
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)        
                # img = cv2.resize(img, (LR_SIZE, LR_SIZE))
                img = cv2.resize(img, (SIZE, SIZE))
                img = img.astype('float32') / 255.0
                img_npy = np.array(img_to_array(img))
                np.save(path_npy_LR+i, img_npy)            
        
        # low resolution image load for Y^ output
        HR_SIZE = 512
        
        # path = './dataset//DIV2K_train_LR_bicubic/train/hr_res'
        # path_npy_HR = 'dataset/DIV2K_train_LR_bicubic/train/npy_files/HR_res/'

        path = './dataset/train/high_res'
        path_npy_HR = 'dataset/train/npy_files/HR_res/'
        files = os.listdir(path_npy_HR)
        origin_files = os.listdir( path )
        if len(files) == 0:
            for i in tqdm(origin_files):
                img = cv2.imread(path+'/'+i, 1)             
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   
                img = cv2.resize(img, (HR_SIZE, HR_SIZE))
                img = img.astype('float32') / 255.0
                img_npy = np.array(img_to_array(img))
                np.save(path_npy_HR+i, img_npy)
            
        files = os.listdir(path_npy_HR)
        self.training_set_size = len(files)
        print('training files number is', self.training_set_size)

        

    def load_test_data(self):        
        SIZE = 256
        path = './celeba'        
        files = os.listdir(path)
        
        for i in tqdm(files):
            img = cv2.imread(path+'/'+i, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (SIZE, SIZE))
            img = img.astype('float32') / 255.0
            self.test_input_img.append(img_to_array(img))        

    def get_batch(self, batch_size, batch_num, use_noise=False):
        path_npy_LR = './dataset/DIV2K_train_LR_bicubic/train/npy_files/LR_res/'
        # path_npy_LR = 'dataset/train/npy_files/LR_res/'

        files = os.listdir(path_npy_LR)
        X = []        
        for file in files[batch_num*batch_size:(batch_num*batch_size+batch_size)]:
            X.append(np.load(path_npy_LR + file))

        path_npy_LR = './dataset/DIV2K_train_LR_bicubic/train/npy_files/HR_res/'
        # path_npy_LR = 'dataset/train/npy_files/HR_res/'
        files = os.listdir(path_npy_LR)
        Y = []        
        for file in files[batch_num*batch_size:(batch_num*batch_size+batch_size)]:
            Y.append(np.load(path_npy_LR + file))

        return np.array(X), np.array(Y)        
    
    def get_batch_test(self, batch_size, use_noise=False):
        SIZE = 512
        # SIZE = 256
        # LR_SIZE = 128
        path = './celeba'        
        # path = './dataset/DIV2K_train_LR_bicubic/train/hr_res'
        # path = './dataset/train/low_res'
        files = os.listdir(path)
        indexes = np.random.randint(len(files), size=batch_size)
        test_input_img = []
        test_input_imgName = []
        width = []
        height = []

        for index in indexes:
            print(files[index])
            img = cv2.imread(path+'/'+files[index], 1)
            height.append(img.shape[0])
            width.append(img.shape[1])   
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   
            # img = cv2.resize(img, (LR_SIZE, LR_SIZE))         
            img = cv2.resize(img, (SIZE, SIZE))
            img = img.astype('float32') / 255.0
            test_input_img.append(img_to_array(img))        
            test_input_imgName.append(files[index])

        test_img = np.array(test_input_img)        
        return test_img, test_input_imgName, width, height

    def data_visualize(self ):
        for i in range(4):            
            a = np.random.randint(0, 700)            
            plt.figure( figsize=(10, 10))
            plt.subplot(1,2,1)
            plt.title( 'High Resolution Image', color='green', fontsize=20)            
            plt.imshow(self.X[a])
            plt.axis('off')
            plt.show()

            plt.subplot(1,2,2)
            plt.title('Low Resolution Image', color='black', fontsize=20)
            plt.imshow(self.Y[a])
            plt.axis('off')
            plt.show()
    
    def save_512(self):
        SIZE = 512     
        LR_SIZE = 128           
        path = './dataset/DIV2K_train_LR_bicubic/train/hr_res'
        path_512 = './dataset/DIV2K_train_LR_bicubic/train/from256to(512_512)'
        files = os.listdir(path)        

        for index in tqdm(files):            
            img = cv2.imread(path+'/'+index, 1)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    
            img = cv2.resize(img, (LR_SIZE, LR_SIZE))        
            img = cv2.resize(img, (SIZE, SIZE))

            cv2.imwrite(path_512+'/'+index, img)
          