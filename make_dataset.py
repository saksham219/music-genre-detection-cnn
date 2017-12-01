import pickle
import numpy as np
import os

pickle_files = ['25_country.pickle', '25_pop.pickle', '25_rock.pickle', '25_RnB.pickle']

for file in pickle_files:

    with open(file,'rb') as f:
        print('opening' + file)
        data = f.read()
        slices = pickle.loads(data)
        slices = np.asarray(slices)

        #seperating into test and train files- 75 percent of data into train , 25 percent in test
        train_index = int(len(slices) * 0.75)
        test_index = train_index +1

        train_examples = slices[0:train_index]
        test_examples = slices[test_index:len(slices)]

        #writing into train set pickle
        if os.path.exists('train_set.pickle'):
            with open('train_set.pickle','rb') as train_file:        #appending if file exists already
                train_slices = pickle.loads(train_file.read())
                train_slices = np.asarray(train_slices)
                train_data = np.concatenate([train_slices,train_examples])
                np.random.shuffle(train_data)                          #randomly shuffling rows

        else:
            train_data = train_examples

        with open('train_set.pickle','wb') as train_file:
            pickle.dump(train_data,train_file)

        #writing into test set pickle
        if os.path.exists('test_set.pickle'):
            with open('test_set.pickle','rb') as test_file:      #appending if file exists already
                test_slices = pickle.loads(test_file.read())
                test_slices = np.asarray(test_slices)
                test_data = np.concatenate([test_slices,test_examples])
                np.random.shuffle(test_data)                    #randomly shuffling rows
        else:
            test_data = test_examples

        with open('test_set.pickle','wb') as test_file:
            pickle.dump(test_data,test_file)



print('making train pickle')
print('making_test pickle')
