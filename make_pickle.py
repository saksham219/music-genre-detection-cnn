import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pickle
import os

print('making pickles for POP songs\n')
#Initialising basic path
song_path="/Users/shrutijha/Documents/music/"
list1=os.listdir(song_path)
del list1[0]

#Initialising main list
data=[]

#Parameter calculation
for a in list1:
    y,sr=librosa.core.load(song_path+a)
    dur=librosa.get_duration(y=y,sr=sr)
#print(dur)

#Creating spectrograms
    S=librosa.feature.melspectrogram(y=y,sr=sr)
    plt.figure(figsize=(dur*(2/3), 1.71))
    librosa.display.specshow(librosa.power_to_db(S),fmax=8000)
#plt.colorbar(format='%+2.0f dB')
    plt.axis("off")
    plt.savefig(fname="/Users/shrutijha/Documents/images/img.png",bbox_inches="tight",pad_inches=0)

#Slicing spectrograms into samples of 128 by 128 pixels
    img=Image.open("/Users/shrutijha/Documents/images/img.png")
    width,height=img.size
    nb_samples=int(width/128)

    for i in range(nb_samples):
        startPixel = i*128
        imgTmp = img.crop((startPixel, 1, startPixel + 128, 129))
        imgTmp= imgTmp.convert('L')
        imgData = np.asarray(imgTmp, dtype=np.uint8).reshape(128*128)
        imgData = imgData/255
        imgData=np.append(imgData,1) #Appending 1 for pop
        data.append(imgData)
    
#creating pickle file
pickle_out=open("list.pickle","wb")
pickle.dump(data,pickle_out)
pickle_out.close()

