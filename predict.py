import cv2
import numpy as np
import itertools, os, time
from Model import get_Model
from parameter import *
import json
from keras import backend as K
K.set_learning_phase(0)

##load json file labels
def load_json(file_path):
    with open(file_path) as f:
        data = json.load(f) 
    return data

class predict():
    def __init__(self, json_path, model_path):
        self.epoch = 0
        self.output_path = 0
        self.model_name = 0
        self.json_path = json_path
        self.model_path = model_path
        self.model = get_Model(training=False)

    def load_model(self):
        print('load')
        self.model.load_weights(self.model_path)
        print("...Previous weight data...")

    def decode_label(self,out):
    
        out_best = list(np.argmax(out[0, 2:], axis=1))  # get max index -> len = 32
        out_best = [k for k, g in itertools.groupby(out_best)]  # remove overlap value
        outstr = ''
        for i in out_best:
            if i < len(letters):
                outstr += letters[i]
        return outstr

    def load_data(self):
        print("Json Loading start.....")
        data = load_json(self.json_path)
        img_path = []
        labels = []
        print("Data Test Found " + str(len(data.keys())))
        for key in data:
            img_path.append(key)
            labels.append(data[key])
        return img_path, labels

    def TrainCheck(self):
        self.load_model()
        img_path, labels = self.load_data()
        total = 0
        acc = 0
        letter_total = 0
        letter_acc = 0
        start = time.time()
        for index in range(len(img_path)):
            img = cv2.imread('./KANA_data100-1/'+img_path[index], cv2.IMREAD_GRAYSCALE)

            img_pred = img.astype(np.float32)
            img_pred = cv2.resize(img_pred, (img_w, img_h))
            img_pred = (img_pred / 255.0) * 2.0 - 1.0
            img_pred = img_pred.T
            img_pred = np.expand_dims(img_pred, axis=-1)
            img_pred = np.expand_dims(img_pred, axis=0)

            net_out_value = self.model.predict(img_pred)

            pred_texts = self.decode_label(net_out_value)

            true_texts = labels[index]
            match_char = 0
            for i in range(min(len(pred_texts), len(true_texts))):
                if pred_texts[i] == true_texts[i]:
                    letter_acc += 1
                    match_char += 1
            letter_total += max(len(pred_texts), len(true_texts))
            letter_length = max(len(pred_texts), len(true_texts))
            if pred_texts == true_texts:
                acc += 1
            total += 1
            print('Predicted: %s  /  True: %s  /Acc: %s' % (pred_texts, true_texts, (match_char / letter_length)*100 ))
            
            # cv2.rectangle(img, (0,0), (150, 30), (0,0,0), -1)
            # cv2.putText(img, pred_texts, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255),2)

            #cv2.imshow("q", img)
            #if cv2.waitKey(0) == 27:
            #   break
            #cv2.destroyAllWindows()

        end = time.time()
        total_time = (end - start)
        print("Time : ",total_time / total)
        print("ACC : ", (acc / total)*100)
        print("letter ACC : ", (letter_acc / letter_total)*100)

a = predict('/home/hogwarts/Documents/OCR_Japan/KANA_data100-1/val.json','/home/hogwarts/Downloads/LSTM+BN5--01--24.476.hdf5')
a.TrainCheck()