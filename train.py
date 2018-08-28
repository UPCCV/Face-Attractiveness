import os, keras,cv2,math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.applications import ResNet50
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.models import Sequential

img_width, img_height, channels = 80, 80, 3
sample_dir='Images'

class DataGenerator(keras.utils.Sequence):
    def __init__(self,labels,batch_size=8,shuffle=True):
        self.labels=labels
        self.indexes = np.arange(len(self.labels))
        self.batch_size=batch_size
        self.shuffle = shuffle
        self.dir=sample_dir
        self.files=os.listdir(self.dir)

    def __len__(self):
        return math.ceil(len(self.labels) / float(self.batch_size))

    def __getitem__(self, index):
        batch_indexs = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.data_generation(batch_indexs)
        return X, y

    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            #print("Epoch end")

    def data_generation(self, batch_indexs):
        x = []
        y = []
        for bi in batch_indexs:
            file=self.files[bi]
            imgpath=self.dir+"/"+file
            #print(bi,imgpath)
            img = load_img(imgpath, target_size=(img_height, img_width))
            img = img_to_array(img).reshape(img_height, img_width, channels)
            img=img.astype('float32') / 255.
            x.append(img)
            l=self.labels[file]
            y.append(l)
        return np.array(x), np.array(y)

def convert_to_gt_txt(gtfile="All_Ratings.xlsx"):
    ratings = pd.read_excel(gtfile, sheet_name=None)["ALL"]
    filenames = ratings['Filename']
    scores = ratings['Rating']
    labels = {}
    files = set(filenames)
    for f in files:
        labels[f] = []
    for f, s in zip(filenames, scores):
        labels[f].append(s)
    with open("gt.txt", "w")as fgt:
        for f in files:
            sum = 0
            for s in labels[f]:
                sum += s
            s = sum / len(labels[f])
            print(f, s)
            fgt.write(f + " " + str(s) + "\n")
    return labels

def load_gt_file(gtfile="gt.txt"):
    with open(gtfile) as f:
        labels = {}
        lines = f.readlines()
        for line in lines:
            items = line.split()
            filename = items[0]
            label = (float)(items[1])
            labels[filename] = label
    return labels

def get_model():
    input_shape = (img_width, img_height, channels)
    #resnet = ResNet50(include_top=False, pooling='avg', input_shape=input_shape)
    #model = Sequential()
    #model.add(resnet)
    #model.add(Dense(1))
    #model.layers[0].trainable = False

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.summary()
    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

def plot_histrory(history):
    plt.rcParams['figure.figsize'] = (6, 6)

    loss = history.history['loss']
    epochs = range(1, len(loss) + 1)

    plt.figure()
    plt.title('Training loss')
    plt.plot(epochs, loss, 'red', label='Training loss')
    plt.legend()
    plt.show()


def train():
    labels = load_gt_file()
    model = get_model()
    earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto')
    filepath = "{epoch:02d}-{val_loss:.2f}.h5"
    checkpoints = keras.callbacks.ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True, mode='min')
    tensorboard = keras.callbacks.TensorBoard(log_dir='logs', histogram_freq=0, write_graph=True, write_images=False,embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    callbacks = [earlystop, checkpoints, tensorboard]
    model.layers[0].trainable = True
    model.compile(loss='mse', optimizer='adam')
    train_generator=DataGenerator(labels)
    val_generator=DataGenerator(labels)
    #max_queue_size=1,workers=1,verbose=1,
    history = model.fit_generator(train_generator,validation_data=val_generator,validation_steps=5000/8,callbacks=callbacks,steps_per_epoch=None,epochs=10)
    plot_histrory(history)
    model.save("model.h5")

def evaluate():
    model = load_model("model.h5")
    labels = load_gt_file()
    val_generator = DataGenerator(labels)
    scores = model.evaluate_generator(val_generator)
    print(scores)
    #plt.scatter(y_train, model.predict(x_train))
    #plt.plot(y_train, y_train, 'ro')
    #plt.show()

def test(imgpath="Images/AF1.jpg"):
    model = load_model("model.h5")
    img = load_img(imgpath)
    x = img_to_array(img).reshape(img_height, img_width, channels)
    x = x.astype('float32') / 255.
    x=np.expand_dims(x,axis=0)
    l = model.predict(x)
    print(l[0])

def test_one_image(imgpath="Images/AF1.jpg"):
    model = load_model("model.h5")
    img=cv2.imread(imgpath)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img=cv2.resize(img,(img_height,img_width))
    img=img/255.0
    img = np.expand_dims(img, axis=0)
    l=model.predict(img)
    print(l[0])

def train_all():
    labels = load_gt_file()
    nb_samples = len(os.listdir(sample_dir))
    x_total = np.empty((nb_samples, img_width, img_height, channels), dtype=np.float32)
    y_total = np.empty((nb_samples, 1), dtype=np.float32)
    for i, fn in enumerate(os.listdir(sample_dir)):
        img = load_img('%s/%s' % (sample_dir, fn),target_size=(img_width,img_height))
        x = img_to_array(img).reshape(img_height, img_width, channels)
        x = x.astype('float32') / 255.
        y = labels[fn]
        x_total[i] = x
        y_total[i] = y
    seed = 42
    x_train_all, x_test, y_train_all, y_test = train_test_split(x_total, y_total, test_size=0.2, random_state=seed)
    x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=seed)
    model = get_model()
    filepath = "{epoch:02d}-{val_loss:.2f}.h5"
    checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    reduce_learning_rate = keras.callbacks.ReduceLROnPlateau(monitor='loss',
           factor=0.1,patience=2,cooldown=2,min_lr=0.00001,verbose=1)
    callback_list = [checkpoint, reduce_learning_rate]
    model.layers[0].trainable = True
    model.compile(loss='mse', optimizer='adam')
    history = model.fit(x=x_train,y=y_train,batch_size=8,epochs=10,validation_data=(x_val, y_val),callbacks=callback_list)
    #plot_histrory(history)
    model.save("model.h5")


def plot_scatter():
    model=load_model("model.h5")
    labels = load_gt_file()
    nb_samples = len(os.listdir(sample_dir))
    x_total = np.empty((nb_samples, img_width, img_height, channels), dtype=np.float32)
    y_total = np.empty((nb_samples, 1), dtype=np.float32)
    for i, fn in enumerate(os.listdir(sample_dir)):
        img = load_img('%s/%s' % (sample_dir, fn), target_size=(img_width, img_height))
        x = img_to_array(img).reshape(img_height, img_width, channels)
        x = x.astype('float32') / 255.
        y = labels[fn]
        x_total[i] = x
        y_total[i] = y
    plt.scatter(y_total, model.predict(x_total))
    plt.plot(y_total, y_total, 'ro')
    plt.show()

if __name__ == "__main__":
    # convert_to_gt_txt()
    #train_all()
    #train()
    #evaluate()
    plot_scatter()
    #test_one_image()
    #test()