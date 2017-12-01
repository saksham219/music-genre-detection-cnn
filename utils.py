import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pickle

def song_data(song_path,save_path):
    y,sr=librosa.core.load(song_path)
    dur=librosa.get_duration(y=y,sr=sr)

#Creating spectrograms
    S=librosa.feature.melspectrogram(y=y,sr=sr)
    plt.figure(figsize=(dur*(2/3), 1.71))
    librosa.display.specshow(librosa.power_to_db(S),fmax=8000)
#plt.colorbar(format='%+2.0f dB')
    plt.axis("off")
    plt.savefig(fname=save_path,bbox_inches="tight",pad_inches=0)

#Slicing spectrograms into samples of 128 by 128 pixels
    img =Image.open(save_path)
    width,height=img.size
    nb_samples=int(width/128)

    data = []
    for i in range(nb_samples):
        startPixel = i*128
        imgTmp = img.crop((startPixel, 1, startPixel + 128, 129))
        imgTmp= imgTmp.convert('L')
        imgData = np.asarray(imgTmp, dtype=np.uint8).reshape(128*128)
        imgData = imgData/255
        data.append(imgData)

    return data

def get_train_test():
    f1 = open("train_set.pickle","rb")
    train_data = pickle.loads(f1.read())
    f1.close()

    f2 = open("test_set.pickle","rb")
    test_data = pickle.loads(f2.read())
    f2.close()

    def one_hot(y):
        b = np.zeros((len(y),4))
        b[np.arange(len(y)),y.astype(int)] = 1
        return b

    X_train = train_data[:, 0: train_data.shape[1] - 1]
    Y_train =  one_hot(train_data[:,train_data.shape[1]-1])

    x_test =   test_data[:, 0:test_data.shape[1] - 1]
    y_test =  one_hot(test_data[:,test_data.shape[1]-1])

    X_train = X_train.reshape([-1,128,128,1])
    x_test = x_test.reshape([-1,128,128,1])

    return(X_train,Y_train,x_test,y_test)
