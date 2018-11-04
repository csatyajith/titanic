# from keras.utils.np_utils import to_categorical
import os

import numpy as np
import pandas
from keras.layers import Dense
from keras.models import Sequential

import root_dir

with open(os.path.join(root_dir.get(), "data", "titanic_train.csv")) as train_file:
    train_data = pandas.read_csv(train_file)

with open(os.path.join(root_dir.get(), "data", "titanic_test.csv")) as test_file:
    test_data = pandas.read_csv(test_file)

train_data = train_data.fillna(-1)
test_data = test_data.fillna(-1)


def transform_csv_to_data(data, type):
    survival = list()
    passenger_ids = list(data["PassengerId"])
    if type != "test":
        survival = list(data["Survived"])
    p_class = list(data["Pclass"])
    names = list(data["Name"])
    sex = list(data["Sex"])
    age = normalize_data(list(data["Age"]))
    sibsp = list(data["SibSp"])
    parch = list(data["Parch"])
    ticket = list(data["Ticket"])
    fare = normalize_data(list(data["Fare"]))
    cabin = list(data["Cabin"])
    embarked = list(data["Embarked"])
    embarked_set = list(set(embarked))
    data_y = list()
    data_x = list()
    for i in range(len(passenger_ids)):
        if type != "test":
            data_y.append(survival[i])
        input_properties = list()
        passenger_sex = 0 if sex[i] == "male" else 1
        input_properties.append(passenger_sex)
        input_properties.append(age[i])
        input_properties.append(fare[i])
        input_properties.append(p_class[i])
        input_properties.append(embarked_set.index(embarked[i]) if embarked[i] != -1 else -1)
        input_properties.append(parch[i])
        input_properties.append(sibsp[i])
        data_x.append(input_properties)

    data_x = np.array(data_x)
    data_y = np.array(data_y)
    return data_x, data_y


def normalize_data(data):
    for i in range(len(data)):
        data[i] = (data[i] - min(data)) / (max(data) - min(data))
    return data


data_x, data_y = transform_csv_to_data(train_data, "train")
test_data_x, test_data_y = transform_csv_to_data(test_data, "test")

model = Sequential()
model.add(Dense(units=20, input_dim=data_x.shape[1], activation='tanh'))
model.add(Dense(units=10, input_dim=data_x.shape[1], activation='tanh'))
model.add(Dense(units=1, input_dim=data_x.shape[1], activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(data_x, data_y, epochs=300)

# model.train_on_batch(data_x, data_y)
# print(data_x.shape, data_y.shape)
# print(data_y)
# loss_and_metrics = model.evaluate(data_x, data_y, batch_size=train_batch_size)
classes = model.predict(test_data_x)
# print(classes)
survival = list()
for i in range(len(list(test_data["PassengerId"]))):
    survival.append(1 if classes[i][0] > 0.5 else 0)
output_df = pandas.DataFrame({"PassengerId": list(test_data["PassengerId"]),
                              "Survived": survival})
output_df.to_csv(os.path.join(root_dir.get(), "titanic1.csv"), index=False)
# print(loss_and_metrics)
# for i in range(len(data_x)):
#     print("{}: {}" .format(np.sum(data_x[i]), classes[i]))
