import cv2
import os, random
import numpy as np
from parameter import letters
import json

# # Input data generator
def labels_to_text(labels):     # letters index -> text (string)
    return ''.join(list(map(lambda x: letters[int(x)], labels)))

def text_to_labels(text):      # text -> letters 
    return list(map(lambda x: letters.index(x), text))

##load json file labels
def load_json(file_path):
    with open(file_path) as f:
        data = json.load(f) 
    return data


class TextImageGenerator:
    def __init__(self, json_path, img_w, img_h,
                 batch_size, status, downsample_factor, max_text_len=80):
        self.img_h = img_h
        self.img_w = img_w
        self.batch_size = batch_size
        self.max_text_len = max_text_len
        self.downsample_factor = downsample_factor
        self.json_path = json_path
        self.n = len(load_json(self.json_path).keys())
        #self.indexes = list(range(self.n))
        #self.cur_index = 0
        #self.imgs = np.zeros((self.n, self.img_h, self.img_w))
        self.texts = []
        self.status = status

    ## load data
    def load_data(self):
        print("Json Loading start.....")
        data = load_json(self.json_path)
        img_path = []
        labels = []
        print("Found " + str(self.status) + ' ' + str(len(data.keys())))
        for key in data:
            img_path.append(key)
            labels.append(data[key])
        return img_path, labels
    '''
    def build_data(self):
        
        
        print(self.n, " Image Loading start...")

        for i in range(len(data)):
            img = cv2.imread('.'+data[str(i+values_lec)][0]['path'], cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (self.img_w, self.img_h))
            img = img.astype(np.float32)
            img = (img / 255.0) * 2.0 - 1.0

            self.imgs[i, :, :] = img
            self.texts.append(data[str(i+values_lec)][0]['class'])
        print(len(self.texts) == self.n)
        print(self.texts)
        print(self.n, " Image Loading finish...")

    def next_sample(self):      ## index max -> 0 
        self.cur_index += 1
        if self.cur_index >= self.n:
            self.cur_index = 0
            random.shuffle(self.indexes)
        return self.imgs[self.indexes[self.cur_index]], self.texts[self.indexes[self.cur_index]]
    '''
    def load_img(self,img_path):

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.img_w, self.img_h))
        img = img.astype(np.float32)
        img = (img / 255.0) * 2.0 - 1.0
        return img

    def next_batch(self):       ## batch size


        img_path, labels = self.load_data()
        img_path = np.array(img_path)
        labels = np.array(labels)
        order = np.arange(len(img_path))
        while True:

            np.random.shuffle(order)
            x = img_path[order]
            y = labels[order]

            X_data = np.ones([self.batch_size, self.img_w, self.img_h, 1])     # (bs, 128, 64, 1)
            Y_data = np.ones([self.batch_size, self.max_text_len])             # (bs, 9)
            input_length = np.ones((self.batch_size, 1)) * (self.img_w // self.downsample_factor - 2)  # (bs, 1)
            label_length = np.zeros((self.batch_size, 1))           # (bs, 1)

            for i in range(self.batch_size):

                img = self.load_img(x[i])

                text = y[i]
                #img, text = self.next_sample()
                img = img.T
                img = np.expand_dims(img, -1)
                X_data[i] = img
                ##padding
                text_padding = text_to_labels(text)
                
                len_padding = self.max_text_len - len(text_padding)
                text_padding.extend(np.zeros((len_padding,), dtype=int))

                Y_data[i] = text_padding
                label_length[i] = len(text)

            # dict 
            inputs = {
                'the_input': X_data,  # (bs, 128, 64, 1)
                'the_labels': Y_data,  # (bs, 8)
                'input_length': input_length,  # (bs, 1) ->  value = 30
                'label_length': label_length  # (bs, 1) -> value = 8
            }
            outputs = {'ctc': np.zeros([self.batch_size])}   # (bs, 1) ->  0
            
            yield (inputs, outputs)