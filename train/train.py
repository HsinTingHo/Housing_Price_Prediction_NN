import os
import argparse
import tensorflow as tf
import pandas as pd
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential

#from model import DNN

def keras_model_fn(hyperparameters):
    """
    Args:
        hyperparameters = {'input_dim':19, 'hidden_dim':[19,19,19]}
        
    Returns: 
        A compiled Keras model
    """
    model = Sequential()
    model.add(Dense(input_dim,activation='relu'))
    for each_layer in hyperparameters['hidden_dim']: 
            model.add(Dense(each_layer,activation='relu'))
    model.add(Dense(1,activation='relu'))
    
    
    optimizer = Adam()
    model.compile(loss='mse',
                  optimizer=opt,
                  metrics=['mse'])
    print(model.summary())
    return model


def serving_input_fn(hyperparameters):
    """
    Define place holder to the model during serving
    For more information: https://github.com/aws/sagemaker-python-sdk#creating-a-serving_input_fn
    Args:
        hyperparameters = {'input_dim':19, 'hidden_dim':[19,19,19], 'batch_size':128}
        
    Returns: 
        ServingInputReceiver or fn that returns a ServingInputReceiver
    """
    tensor = tf.placeholder(tf.float32, shape=[None, hyperparameters['input_dim']])
    # the features key PREDICT_INPUTS matches the Keras Input Layer name
    features = {PREDICT_INPUTS: tensor}
    return build_raw_serving_input_receiver_fn(features)


def train_input_fn(training_dir, hyperparameters):
    return _input(tf.estimator.ModeKeys.TRAIN,
                  data_dir=training_dir)


def eval_input_fn(training_dir, hyperparameters):
    return _input(tf.estimator.ModeKeys.EVAL,
                  data_dir=training_dir)

def _input(mode, data_dir):
    """
    Args:
        mode: Standard names for model modes (tf.estimators.ModeKeys).
        batch_size: The number of samples per batch of input requested.
    """
    dataset = pd.read_csv(file_name(mode, data_dir))
    y = dataset['price'].values
    X = dataset.drop('price',axis =1).values
    return {PREDICT_INPUTS: X}, y


def file_name(mode, data_dir):   
    if mode == tf.estimator.ModeKeys.EVAL:
        print('test')
        name = data_dir+'/test.csv'
        print(name)
    elif mode == tf.estimator.ModeKeys.TRAIN:
        print('train')
        name = data_dir+'/train.csv'
    
    return name
                                               
# ********************************************************************
# Gets training data in batches from the train.csv file
# def _get_train_data_loader(batch_size, training_dir):
#     print("Get train data loader.")
    
#     train_ds = tf.data.experimental.make_csv_dataset(
#         file_pattern=training_dir, batch_size=batch_size, 
#         column_names=data.columns, label_name = data.columns[0],
#         header=False, num_epochs=1, shuffle=False
#     )

#     return train_ds

# def train(model, data_path, batch_size, epochs, optimizer):
#     print("Fit model on training data")  
#     data = pd.read_csv(data_path)
#     y_train = data['price'].values
#     x_train = data.drop('price',axis =1).values
#     x_val = x_train[-10000:]
#     y_val = y_train[-10000:]
#     model.compile(optimizer=optimizer ,loss='mse', metrics=['mse'])
#     history = model.fit(
#         x_train,
#         y_train,
#         batch_size=batch_size,
#         epochs=epochs,
#         # monitoring validation loss and metrics at the end of each epoch       
#         validation_data=(x_val, y_val)
#     )
#     model.summary()
    
#     #plot loss
#     loss_df = pd.DataFrame(model.history.history)
#     loss_df.plot(figsize=(12,8))


# if __name__ == '__main__':
    
#     INPUTS = 18
#     #set sagemaker parameters
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
#     parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
#     parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
#     #training parameters
#     parser.add_argument('--batch-size', type=int, default=10, metavar='N',
#                         help='input batch size for training (default: 10)')
#     parser.add_argument('--epochs', type=int, default=10, metavar='N',
#                         help='number of epochs to train (default: 10)')
#     parser.add_argument('--seed', type=int, default=1, metavar='S',
#                         help='random seed (default: 1)')
    
#     #model parameters
#     parser.add_argument('--hidden_units', type=int, default=[INPUTS,INPUTS], metavar='N',
#                         help='number of epochs to train (default: [10,10,10])')
#     args = parser.parse_args()
    
#     #tf.keras models will transparently run on a single GPU with no code changes required
    
#     tf.random.set_seed(args.seed)
    
#     # Load the training data.
#     #train_loader = _get_train_data_loader(args.batch_size, args.data_dir)
#     print('Loading training data')
    
#     model = DNN(input_dim=INPUTS,hidden_dim=args.hidden_units,output_dim=1).model
#     optimizer = Adam()
    
#     train(model, args.data_dir,args.batch_size, args.epochs, optimizer)
    
#     #save model parameter
#     model.save('trained_model')