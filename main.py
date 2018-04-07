import torch
import random
import math
import pandas as pd
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.autograd import Variable
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


class PassengerDataset(Dataset):
    def __init__(self, df, test):
        self.test = test
        self.passergers_id = df['PassengerId'].tolist()
        self.passengers_frame = self.__prep_data__(df)

    def __len__(self):
        size, _ = self.passengers_frame.shape
        return size

    def __getitem__(self, index):
        passenger = self.passengers_frame.iloc[index].as_matrix().tolist()
        if self.test:
            return torch.FloatTensor(passenger), self.passergers_id[index]
        else:
            label = [passenger[0]]
            del passenger[0]
            return torch.FloatTensor(passenger), torch.FloatTensor(label)

    @staticmethod
    def __prep_data__(df):
        passengers_frame = df
        # drop unwanted feature
        passengers_frame = passengers_frame.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)
        # Fill missing data: Age and Fare with the mean, Embarked with most frequent value
        passengers_frame[['Age']] = passengers_frame[['Age']].fillna(value=passengers_frame[['Age']].mean())
        passengers_frame[['Fare']] = passengers_frame[['Fare']].fillna(value=passengers_frame[['Fare']].mean())
        passengers_frame[['Embarked']] = passengers_frame[['Embarked']].fillna(value=passengers_frame['Embarked'].value_counts().idxmax())
        # Convert categorical  features into numeric
        passengers_frame['Sex'] = passengers_frame['Sex'].map({'female': 1, 'male': 0}).astype(int)
        # Convert Embarked to one-hot
        enbarked_one_hot = pd.get_dummies(passengers_frame['Embarked'], prefix='Embarked')
        passengers_frame = passengers_frame.drop('Embarked', axis=1)
        passengers_frame = passengers_frame.join(enbarked_one_hot)
        # normalize to 0-1
        x = passengers_frame.values  # returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        passengers_frame = pd.DataFrame(x_scaled)
        return passengers_frame


class SinDataset(Dataset):
    def __init__(self, data_size):
        self.data_size = data_size
        self.data = [random.uniform(0,1) for _ in range(data_size)]
        self.labels = [math.sin(self.data[i]) for i in range(data_size)]

    def __len__(self):
        return self.data_size

    def __getitem__(self, index):
        return torch.FloatTensor([self.data[index]]), torch.FloatTensor([self.labels[index]])


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(9, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return F.sigmoid(x)


def load_train_data():
    df = pd.read_csv('data/train.csv')
    train, validation = train_test_split(df, test_size=0.2)
    return train, validation


def generate_kaggle_data(model):
    test_df = pd.read_csv('data/test.csv')
    test_data_set = PassengerDataset(test_df, test=True)
    test_data_loader = DataLoader(test_data_set, batch_size=1, shuffle=False, num_workers=4)
    survived_col = []
    id_col = []
    for index, (test_input, test_id) in enumerate(test_data_loader):
        test_input = Variable(test_input)
        test_id = test_id.numpy()[0]
        test_output = model(test_input).data.numpy()[0][0]
        predicted = 1 if test_output > 0.5 else 0
        survived_col.append(predicted)
        id_col.append(test_id)
    df_data = {'PassengerId':id_col, 'Survived':survived_col}
    df = pd.DataFrame(data=df_data)
    df.to_csv('data/test_output.csv', index=False)


def main():
    # setup data
    train_set, validation_set = load_train_data()
    train_data_set = PassengerDataset(train_set, test=False)
    train_data_loader = DataLoader(train_data_set, batch_size=32, shuffle=True, num_workers=4)
    validation_data_set = PassengerDataset(validation_set, test=False)
    validation_data_loader = DataLoader(validation_data_set, batch_size=1, shuffle=True, num_workers=4)
    # setup network
    model = NeuralNetwork()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.MSELoss()
    # train
    epochs = 100
    for epoch in range(epochs):
        model.train()
        for index, (inputs, targets) in enumerate(train_data_loader):
            inputs, targets = Variable(inputs), Variable(targets)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        # get accuracy
        correct_num = 0
        for index, (inputs, targets) in enumerate(validation_data_loader):
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = model(inputs).data.numpy()
            predicted = np.where(outputs > 0.5, 1, 0)
            answer = targets.data.numpy()
            correct_num += (predicted == answer).sum()
        print('Epoch [%d/%d], Loss:%.4f, Accuracy:%.4f' % (epoch+1, epochs, loss.data[0], correct_num/len(validation_set)))
    # generate_kaggle_data(model)


if __name__ == '__main__':
    main()

