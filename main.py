from model import train, get_model
from utils import get_train_test
from predict import predict

print('Welcome to Music Genre Detection\n')

exit = 0

X_train,Y_train,x_test,y_test = get_train_test()
print(x_test.shape)

while(1):

    choice = int(input('What would you like to do?\n 1.Train the network\n 2.Run Predictions on new songs\n 3.Exit\n'))

    if(choice ==1):

        print('Training the model\n')
        model = get_model()
        train(X_train,Y_train,x_test,y_test,model)

    elif(choice==2):
        print('making predictions\n')
        predict(x_test)

    elif(choice==3):
        break


