import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

def get_model():
    #defining dimensions of each layer

    #input layer
    convnet = input_data(shape = [None,128,128,1],name='input')

    #1st conv layer          no. of filters=64    filtersize =2
    convnet = conv_2d(convnet,64 ,2,activation = 'elu', weights_init = 'Xavier')
    convnet = max_pool_2d(convnet,2)

    #2nd conv layer     no. of filters = 128 , filtersize = 2
    convnet = conv_2d(convnet,128,2,activation = 'elu', weights_init = 'Xavier')
    convnet = max_pool_2d(convnet,2)

    convnet = conv_2d(convnet,256,2,activation = 'elu', weights_init = 'Xavier')
    convnet = max_pool_2d(convnet,2)

    convnet = conv_2d(convnet,512,2,activation = 'elu', weights_init = 'Xavier')
    convnet = max_pool_2d(convnet,2)


    #no of nodes in FC layer
    convnet = fully_connected(convnet,1024, activation = 'elu')
    convnet = dropout(convnet,0.5)

    #no of classes = 4
    convnet = fully_connected(convnet,4,activation = 'softmax')

    convnet = regression(convnet,optimizer = 'rmsprop',learning_rate = 0.0001,
                         loss='categorical_crossentropy', name = 'targets')

    model = tflearn.DNN(convnet, tensorboard_verbose =3)

    return model

def train(X_train,Y_train,x_test,y_test,model):
#start traning the model
    model.fit({'input' : X_train},{'targets':Y_train}, n_epoch =2,
           validation_set = ({'input' : x_test},{'targets':y_test}),
          snapshot_step = 50000, show_metric = True, run_id = 'music')

#saves the weights after model is trained
    model.save('temp_model.tflearn')


#train(X_train,Y_train,x_test,y_test)

'''model = get_model()
#print(x_test[1].shape)
#load a saved model
#run a prediction
print(x_test[1:62].shape)
predictions = model.predict(x_test[1:62])
print(predictions)

print(np.argmax(predictions,axis = 1))

model.load('finalcnn.model')
predictions = model.predict(x_test[1:62])
print(predictions)

print(np.argmax(predictions,axis = 1))
'''

'''
model = get_model()
model.load('finalcnn.model')
f = open('list_rock.pickle','rb')
data = pickle.loads(f.read())
data = np.asarray(data)
data = data[:,0:data.shape[1]-1]
print(data.shape)
data = data.reshape([-1,128,128,1])

#print(x_test.shape)
predictions = model.predict(data[0:100])
print(np.argmax(predictions,axis = 1))
'''


