from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from Image_Generator import TextImageGenerator
from Model import get_Model
import time
from callbacks import TrainCheck,myCallback
from parameter import *
K.set_learning_phase(0)

# # Model description and training

model = get_Model(training=True)

try:
    model.load_weights('./model_OCR/LSTM+BN5--08--1.671.hdf5')
    print("...Previous weight data...")
except:
    print("...New weight data...")
    pass
print(model.summary())

json_train_path = './train_all.json'
tiger_train = TextImageGenerator(json_train_path, img_w, img_h, batch_size,'train', downsample_factor, max_text_len = max_text_len)
#tiger_train.build_data()

json_val_path = './val_all.json'
tiger_val = TextImageGenerator(json_val_path, img_w, img_h, val_batch_size,'val', downsample_factor, max_text_len = max_text_len)
#tiger_val.build_data()
print('>>>>>>>>>>>>>>>>>',val_batch_size)
print(tiger_train.n)
print(tiger_val.n)

ada = Adam()


early_stop = EarlyStopping(monitor='loss', min_delta=0.001, patience=4, mode='min', verbose=1)
checkpoint = ModelCheckpoint(filepath='./model_OCR/' +'LSTM+BN5--{epoch:02d}--{val_loss:.3f}.hdf5', monitor='loss', verbose=1, mode='min', period=1)
tensorboard = TensorBoard(log_dir="logs_OCR/LSTM+BN5{}".format(time.time()),
                              batch_size=batch_size, write_images=True)
save_batch = myCallback()
#train_check = TrainCheck()
# the loss calc occurs elsewhere, so use a dummy lambda func for the loss
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=ada)

# captures output of softmax so we can decode the output during visualization
model.fit_generator(generator=tiger_train.next_batch(),
                    steps_per_epoch=int(tiger_train.n / batch_size),
                    epochs=300,
                    callbacks=[checkpoint,tensorboard,save_batch],
                    validation_data=tiger_val.next_batch(),
                    validation_steps=int(tiger_val.n / val_batch_size) )
