from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam

class DNN():
    def __init__(self, input_dim, hidden_dim , output_dim):
        self.model = Sequential()
        self.model.add(Dense(input_dim,activation='relu'))
        for each_layer in hidden_dim: 
            self.model.add(Dense(each_layer,activation='relu'))
        self.model.add(Dense(output_dim,activation='relu'))
#  ******** keras sequential model ********
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Activation
# from tensorflow.keras.optimizers import Adam
# import tensorflow as tf
# import torch.utils.data
# import torch
# import os



# INPUT_SIZE = 19
# INPUT_TENSOR_NAME = "inputs_input" 
# BATCH_SIZE = 128

# def keras_model_fn(hyperparameters):
#     #create a fully connected network
#     model = Sequential()
#     model.add(Dense(INPUT_SIZE,activation='relu'))
#     model.add(Dense(INPUT_SIZE,activation='relu'))
#     model.add(Dense(INPUT_SIZE,activation='relu'))
#     model.add(Dense(INPUT_SIZE,activation='relu'))
#     model.add(Dense(1))
#     #opt = Adam(learning_rate=hyperparameters['learning_rate'])
#     #model.compile(optimizer=opt ,loss='mse', metrics=['mse'])
#     return model


# # def serving_input_fn(hyperparameters):
# #     tensor = tf.placeholder(tf.float32, shape=[None, INPUT_SIZE])
# #     inputs = {INPUT_TENSOR_NAME: tensor}
# #     return tf.estimator.export.ServingInputReceiver(inputs, inputs)


# # def train_input_fn(training_dir, hyperparameters):
# #     return _input(tf.estimator.ModeKeys.TRAIN, batch_size=BATCH_SIZE, data_dir=training_dir)


# # def eval_input_fn(training_dir, hyperparameters):
# #     pass
# #     #return _input(tf.estimator.ModeKeys.EVAL, batch_size=BATCH_SIZE, data_dir=training_dir)


# # def _input(mode, batch_size, data_dir):
# #     assert os.path.exists(data_dir), ("Unable to find datasets for input")
# #     #how to deal with batch
# #     if mode == tf.estimator.ModeKeys.TRAIN:
# #     return {INPUT_TENSOR_NAME: features}, labels



# # Gets training data in batches from the train.csv file
# def _get_train_data_loader(batch_size, training_dir):
#     print("Get train data loader.")

#     train_data = pd.read_csv(os.path.join(training_dir, "train.csv"), header=None, names=None)

#     train_y = torch.from_numpy(train_data[[0]].values).float().squeeze()
#     train_x = torch.from_numpy(train_data.drop([0], axis=1).values).float()

#     train_ds = torch.utils.data.TensorDataset(train_x, train_y)

#     return torch.utils.data.DataLoader(train_ds, batch_size=batch_size)


# def train(model, train_loader, epochs, criterion, optimizer, device):
#     """
#     This is the training method that is called by the PyTorch training script. The parameters
#     passed are as follows:
#     model        - The model that we wish to train.
#     train_loader - The PyTorch DataLoader that should be used during training.
#     epochs       - The total number of epochs to train for.
#     criterion    - The loss function used for training. 
#     optimizer    - The optimizer to use during training.
#     device       - Where the model and data should be loaded (gpu or cpu).
#     """
    
#     # training loop is provided
#     for epoch in range(1, epochs + 1):
#         model.train() # Make sure that the model is in training mode.

#         total_loss = 0

#         for batch in train_loader:
#             # get data
#             batch_x, batch_y = batch

#             batch_x = batch_x.to(device)
#             batch_y = batch_y.to(device)

#             #optimizer.zero_grad()

#             # get predictions from model
#             y_pred = model(batch_x)
            
#             # perform backprop
#             loss = criterion(y_pred, batch_y)
#             loss.backward()
#             #optimizer.step()
            
#             total_loss += loss.data.item()

#         print("Epoch: {}, Loss: {}".format(epoch, total_loss / len(train_loader)))


# if __name__ == '__main__':