import os
from utils import song_data
from model import get_model
import numpy as np
import pickle

def predict(x_test):
    song_dir_path="C:/Users/user/Music/predictions/"
    song_list=os.listdir(song_dir_path)

    img_path = "C:/Users/user/pictures/img.png"
    model = get_model()
    print('loading model')
    model.load('temp_model.tflearn')

    genre_dict = {0:'Country',1:'Pop',2:'RnB',3:'Rock'}

    i = 0
    for song in song_list:
        print('listening to  ' + song)
        data = song_data(song_dir_path+song,img_path)
        data = np.asarray(data)
        data = data.reshape([-1,128,128,1])

        #print(x_test[i:i+90].shape)
        predictions = model.predict(x_test[i:i+70])

        slice_genre_list = np.argmax(predictions,axis = 1)

        unique, counts = np.unique(slice_genre_list, return_counts=True)
        genre_counts = dict(zip(unique, counts))
        #print(genre_counts)
        max_genre = max(genre_counts,key = genre_counts.get)
        #print(genre_counts[max_genre]/slice_genre_list.shape[0])
        if(0.20 < genre_counts[max_genre]/slice_genre_list.shape[0] <= 0.5):
            print('This sounds like %s \n' %(genre_dict[max_genre]))
        elif(1 >= genre_counts[max_genre]/slice_genre_list.shape[0] >=0.5):
            print('This is definitely %s \n' %(genre_dict[max_genre]))
        else:
            print('Hmmm. I am not really sure. Sorry. \n')

        i = i + 90


