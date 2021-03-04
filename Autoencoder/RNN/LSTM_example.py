import numpy
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import  pandas as pd
import  os
from keras.models import Sequential, load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np

dataframe = pd.read_csv('./ads.csv', usecols=[1], engine='python', skipfooter=0)
dataset = dataframe.values
# 将整型变为float
dataset = dataset.astype('float32')
#归一化 加快lose的下降速度
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# 65%作为训练集，35%作为测试集
train_size = int(len(dataset) * 0.65)
trainlist = dataset[:train_size]
testlist = dataset[train_size:]

def create_dataset(dataset, look_back):
#这里的look_back与timestep相同，即前面的多少步
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return numpy.array(dataX),numpy.array(dataY)
# 训练数据太少 look_back并不能过大,即利用前多少个预测下一个数值
look_back = 10
trainX,trainY  = create_dataset(trainlist,look_back)
testX,testY = create_dataset(testlist,look_back)

# trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
# testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1] ,1 ))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(1, input_shape=(None,1),return_sequences=False))
# model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
model.save(os.path.join("Test" + ".h5"))
# make predictions
# model = load_model(os.path.join("DATA","Test" + ".h5"))
trainPredict = []
testPredict = []
trainX1 = trainX[0]
for i in range(len(trainX)-5):
    trainPredict.append(model.predict(np.array([trainX1]))[0])
    trainX1 = trainX1[1:]
    trainX1 = np.insert(trainX1,[look_back-1],trainPredict[-1],axis=0)
trainPredict = np.array(trainPredict)
trainPredictOnePoint = model.predict(trainX)
testPredict = model.predict(testX)

# 反归一化
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform(trainY)
trainPredictOnePoint = scaler.inverse_transform(trainPredictOnePoint)
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform(testY)

plt.plot(trainY,color = 'green') # 绿色是真实数据
plt.plot(trainPredictOnePoint,color = 'orange')
plt.plot(trainPredict[1:],color = 'red') # 红色是预测数据
plt.title('train_set')
plt.show()
plt.savefig('训练.png')
plt.plot(testY,color = 'green')
plt.plot(testPredict[1:],color = 'red')
plt.title('test_set')
plt.savefig('测试.png')
plt.show()


