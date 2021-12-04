from random import randint

from numpy import array
from numpy import argmax
from numpy import array_equal
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense


# 随机产生在(1,n_features)区间的整数序列，序列长度为n_steps_in
def generate_sequence(length, n_unique):
    return [randint(1, n_unique - 1) for _ in range(length)]


# 构造LSTM模型输入需要的训练数据
def get_dataset(n_in, n_out, cardinality, n_samples):
    X1, X2, y = list(), list(), list()
    for _ in range(n_samples):
        # 生成输入序列
        source = generate_sequence(n_in, cardinality)
        # 定义目标序列，这里就是输入序列的前三个数据
        target = source[:n_out]
        target.reverse()
        # 向前偏移一个时间步目标序列
        target_in = [0] + target[:-1]
        # 直接使用to_categorical函数进行on_hot编码
        src_encoded = to_categorical(source, num_classes=cardinality)
        tar_encoded = to_categorical(target, num_classes=cardinality)
        tar2_encoded = to_categorical(target_in, num_classes=cardinality)

        X1.append(src_encoded)
        X2.append(tar2_encoded)
        y.append(tar_encoded)
    return array(X1), array(X2), array(y)


# 构造Seq2Seq训练模型model, 以及进行新序列预测时需要的的Encoder模型:encoder_model 与Decoder模型:decoder_model
def define_models(n_input, n_output, n_units):
    # 训练模型中的encoder
    encoder_inputs = Input(shape=(None, n_input))
    encoder = LSTM(n_units, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]  # 仅保留编码状态向量
    # 训练模型中的decoder
    decoder_inputs = Input(shape=(None, n_output))
    decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(n_output, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    # 新序列预测时需要的encoder
    encoder_model = Model(encoder_inputs, encoder_states)
    # 新序列预测时需要的decoder
    decoder_state_input_h = Input(shape=(n_units,))
    decoder_state_input_c = Input(shape=(n_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    # 返回需要的三个模型
    return model, encoder_model, decoder_model


def predict_sequence(infenc, infdec, source, n_steps, cardinality):
    # 输入序列编码得到编码状态向量
    state = infenc.predict(source)
    # 初始目标序列输入：通过开始字符计算目标序列第一个字符，这里是0
    target_seq = array([0.0 for _ in range(cardinality)]).reshape(1, 1, cardinality)
    # 输出序列列表
    output = list()
    for t in range(n_steps):
        # predict next char
        yhat, h, c = infdec.predict([target_seq] + state)
        # 截取输出序列，取后三个
        output.append(yhat[0, 0, :])
        # 更新状态
        state = [h, c]
        # 更新目标序列(用于下一个词预测的输入)
        target_seq = yhat
    return array(output)


# one_hot解码
def one_hot_decode(encoded_seq):
    return [argmax(vector) for vector in encoded_seq]


# 参数设置
# 表示随机产生(1,n_features)范围内的数据
n_features = 50 + 1
n_steps_in = 6
n_steps_out = 3
# 定义模型
# 128代表LSTM中神经元个数
train, infenc, infdec = define_models(n_features, n_features, 128)
train.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
# 生成训练数据
# 10代表10组数据
X1, X2, y = get_dataset(n_steps_in, n_steps_out, n_features, 100000)
print(X1.shape, X2.shape, y.shape)
# 训练模型
train.fit([X1, X2], y, epochs=1)
# 评估模型效果
total, correct = 100, 0
for _ in range(total):
    X1, X2, y = get_dataset(n_steps_in, n_steps_out, n_features, 1)
    target = predict_sequence(infenc, infdec, X1, n_steps_out, n_features)
    if array_equal(one_hot_decode(y[0]), one_hot_decode(target)):
        correct += 1
print('Accuracy: %.2f%%' % (float(correct) / float(total) * 100.0))
# 查看预测结果
for _ in range(10):
    X1, X2, y = get_dataset(n_steps_in, n_steps_out, n_features, 1)
    target = predict_sequence(infenc, infdec, X1, n_steps_out, n_features)
    print('X=%s y=%s, yhat=%s' % (one_hot_decode(X1[0]), one_hot_decode(y[0]), one_hot_decode(target)))
