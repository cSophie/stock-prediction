import tushare as ts
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler


# 贵州茅台 600519.SH
# 平安银行 000001.SZ
# 云南白药 000538.SZ
# 中信证券 600030.SH
# 张家界 000430.SZ
num_dict = {0: '600519.SH',
            1: '000001.SZ',
            2: '000538.SZ',
            3: '600030.SH',
            4: '000430.SZ'}
feature_dict = {0: 'open',
                1: 'high',
                2: 'low',
                3: 'close'}
name_dict = {
    '600519.SH': 'maotai',
    '000001.SZ': 'pingan',
    '000538.SZ': 'yunnan',
    '600030.SH': 'zhongxin',
    '000430.SZ': 'zhangjiajie'
}
pro = ts.pro_api('e9b31113ccd628c7933a0af4e9c45f38aee75b5d9a4fb89fde3c460a')
end_dt = '20230320'
timesteps = 60
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
input_size = 9
hidden_size = 64
num_layers = 2
output_size = 7


def lstm_train(ts_cd, index):
    print('lstm_train')
    df = pro.daily(ts_code=ts_cd, start_date='20190101', end_date=end_dt)
    df = df.sort_index(ascending=True)
    df = df.set_index('trade_date')
    df.index = pd.to_datetime(df.index)
    df = df.drop(columns=['ts_code'])
    df = df.fillna(method='ffill')
    scaler1 = MinMaxScaler(feature_range=(-1, 1))
    scaler2 = MinMaxScaler(feature_range=(-1, 1))
    df.iloc[:, index] = scaler1.fit_transform(df.iloc[:, index].values.reshape(-1, 1))
    for i in range(9):
        df.iloc[:,i] = scaler2.fit_transform(df.iloc[:,i].values.reshape(-1, 1))
    x_train, y_train, x_test, y_test = load_data(df, timesteps, index)
    x_train = torch.from_numpy(x_train).type(torch.Tensor)
    x_test = torch.from_numpy(x_test).type(torch.Tensor)
    y_train = torch.from_numpy(y_train).type(torch.Tensor)
    y_test = torch.from_numpy(y_test).type(torch.Tensor)
    model = StockLSTM(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, output_size = output_size)
    loss_fn = torch.nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr = 0.01)
    model=model.to(device)
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)
    #
    y_train = y_train.squeeze()
    y_test = y_test.squeeze()
    #
    num_epochs = 100
    for epoch in range(num_epochs):
        y_train_pred = model(x_train)
        loss = loss_fn(y_train_pred, y_train)
        if epoch % 10 == 0 and epoch != 0:
            print('Epoch ', epoch, 'MSE: ', loss.item())
        opt.zero_grad()
        loss.backward()
        opt.step()
    state = {'model': model.state_dict(), 'optimizer': opt.state_dict(), 'epoch': num_epochs}
    torch.save(state, './model_para_stock/{}_{}_LSTM.pth'.format(name_dict[ts_cd], feature_dict[index]))


def gru_train(ts_cd, index):
    print('gru_train')
    df = pro.daily(ts_code=ts_cd, start_date='20190101', end_date=end_dt)
    df = df.sort_index(ascending=True)
    df = df.set_index('trade_date')
    df.index = pd.to_datetime(df.index)
    df = df.drop(columns=['ts_code'])
    df = df.fillna(method='ffill')
    scaler1 = MinMaxScaler(feature_range=(-1, 1))
    scaler2 = MinMaxScaler(feature_range=(-1, 1))
    df.iloc[:, index] = scaler1.fit_transform(df.iloc[:, index].values.reshape(-1, 1))
    for i in range(9):
        df.iloc[:,i] = scaler2.fit_transform(df.iloc[:,i].values.reshape(-1, 1))
    x_train, y_train, x_test, y_test = load_data(df, timesteps, index)
    x_train = torch.from_numpy(x_train).type(torch.Tensor)
    x_test = torch.from_numpy(x_test).type(torch.Tensor)
    y_train = torch.from_numpy(y_train).type(torch.Tensor)
    y_test = torch.from_numpy(y_test).type(torch.Tensor)
    model = StockGRU(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, output_size = output_size)
    loss_fn = torch.nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr = 0.01)
    model=model.to(device)
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)
    #
    y_train = y_train.squeeze()
    y_test = y_test.squeeze()
    #
    num_epochs = 100
    for epoch in range(num_epochs):
        y_train_pred = model(x_train)
        loss = loss_fn(y_train_pred, y_train)
        if epoch % 10 == 0 and epoch != 0:
            print('Epoch ', epoch, 'MSE: ', loss.item())
        opt.zero_grad()
        loss.backward()
        opt.step()
    state = {'model': model.state_dict(), 'optimizer': opt.state_dict(), 'epoch': num_epochs}
    torch.save(state, './model_para_stock/{}_{}_GRU.pth'.format(name_dict[ts_cd], feature_dict[index]))


def dnn_train(ts_cd, index):
    print('dnn_train')
    df = pro.daily(ts_code=ts_cd, start_date='20190101', end_date=end_dt)
    df = df.sort_index(ascending=True)
    df = df.set_index('trade_date')
    df.index = pd.to_datetime(df.index)
    df = df.drop(columns=['ts_code'])
    df = df.fillna(method='ffill')
    scaler1 = MinMaxScaler(feature_range=(-1, 1))
    scaler2 = MinMaxScaler(feature_range=(-1, 1))
    df.iloc[:, index] = scaler1.fit_transform(df.iloc[:, index].values.reshape(-1, 1))
    for i in range(9):
        df.iloc[:,i] = scaler2.fit_transform(df.iloc[:,i].values.reshape(-1, 1))
    x_train, y_train, x_test, y_test = load_data(df, timesteps, index)
    x_train = torch.from_numpy(x_train).type(torch.Tensor)
    x_test = torch.from_numpy(x_test).type(torch.Tensor)
    y_train = torch.from_numpy(y_train).type(torch.Tensor)
    y_test = torch.from_numpy(y_test).type(torch.Tensor)
    model = StockDNN()
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr = 0.01)
    model=model.to(device)
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)
    #
    y_train = y_train.squeeze()
    y_test = y_test.squeeze()
    #
    num_epochs = 100
    for epoch in range(num_epochs):
        y_train_pred = model(x_train)
        loss = loss_fn(y_train_pred, y_train)
        if epoch % 10 == 0 and epoch != 0:
            print('Epoch ', epoch, 'MSE: ', loss.item())
        opt.zero_grad()
        loss.backward()
        opt.step()
    state = {'model': model.state_dict(), 'optimizer': opt.state_dict(), 'epoch': num_epochs}
    torch.save(state, '/kaggle/working/{}_{}_DNN.pth'.format(name_dict[ts_cd], feature_dict[index]))


def lstm_pred(ts_cd, index_config):
    """
    指定待预测的股票代码和数据索引，使用LSTM预测明天的相应值。
    :param ts_cd: 股票代码，包括：
    600519.SH 贵州茅台
    000001.SZ 平安银行
    000538.SZ 云南白药
    000430.SZ 张家界
    600030.SH 中信证券
    :param index_config: 想预测的数据的索引，对应关系如下：
    0：open
    1：high
    2：low
    3：close
    :return: 预测值
    """
    print('lstm_pred')
    df = pro.daily(ts_code=ts_cd, start_date='20220101', end_date='')
    # 数据处理开始
    df = df.sort_index(ascending=True)
    df = df.set_index('trade_date')
    df.index = pd.to_datetime(df.index)
    df = df.drop(columns=['ts_code'])
    df = df.fillna(method='ffill')
    scaler1 = MinMaxScaler(feature_range=(-1, 1))
    scaler2 = MinMaxScaler(feature_range=(-1, 1))
    df.iloc[:, index_config] = scaler1.fit_transform(df.iloc[:, index_config].values.reshape(-1, 1))
    for i in range(9):
        df.iloc[:, i] = scaler2.fit_transform(df.iloc[:, i].values.reshape(-1, 1))
    # 取最近timesteps天的数据，预测明天的open
    df = df.iloc[0: timesteps, ]
    x = test_data(df, timesteps)
    x = torch.from_numpy(x).type(torch.Tensor)
    x = x.to(device)
    # 数据处理结束

    # 模型结构
    model = StockLSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size)
    model=model.to(device)
    # 加载模型参数
    checkpoint = torch.load('./model_para_stock/{}_{}_LSTM.pth'.format(name_dict[ts_cd], feature_dict[index_config]))
    model.load_state_dict(checkpoint['model'])
    # 预测
    y_pred = model(x)
    y_pred = scaler1.inverse_transform(y_pred.detach().cpu().numpy())
    return y_pred


def gru_pred(ts_cd, index_config):
    """
    指定待预测的股票代码和数据索引，使用GRU预测明天的相应值。
    :param ts_cd: 股票代码，包括：
    600519.SH 贵州茅台
    000001.SZ 平安银行
    000538.SZ 云南白药
    000430.SZ 张家界
    600030.SH 中信证券
    :param index_config: 想预测的数据的索引，对应关系如下：
    0：open
    1：high
    2：low
    3：close
    :return: 预测值
    """
    print('gru_pred')
    df = pro.daily(ts_code=ts_cd, start_date='20220101', end_date='')
    # 数据处理开始
    df = df.sort_index(ascending=True)
    df = df.set_index('trade_date')
    df.index = pd.to_datetime(df.index)
    df = df.drop(columns=['ts_code'])
    df = df.fillna(method='ffill')
    scaler1 = MinMaxScaler(feature_range=(-1, 1))
    scaler2 = MinMaxScaler(feature_range=(-1, 1))
    df.iloc[:, index_config] = scaler1.fit_transform(df.iloc[:, index_config].values.reshape(-1, 1))
    for i in range(9):
        df.iloc[:, i] = scaler2.fit_transform(df.iloc[:, i].values.reshape(-1, 1))
    # 取最近timesteps天的数据，预测明天的open
    df = df.iloc[0: timesteps, ]
    x = test_data(df, timesteps)
    x = torch.from_numpy(x).type(torch.Tensor)
    x = x.to(device)
    # 数据处理结束

    # 模型结构
    model = StockGRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size)
    model=model.to(device)
    # 加载模型参数
    checkpoint = torch.load('./model_para_stock/{}_{}_GRU.pth'.format(name_dict[ts_cd], feature_dict[index_config]))
    model.load_state_dict(checkpoint['model'])
    # 预测
    y_pred = model(x)
    y_pred = scaler1.inverse_transform(y_pred.detach().cpu().numpy())
    return y_pred


def dnn_pred(ts_cd, index_config):
    print('dnn_pred')
    """
    指定待预测的股票代码和数据索引，使用GRU预测明天的相应值。
    :param ts_cd: 股票代码，包括：
    600519.SH 贵州茅台
    000001.SZ 平安银行
    000538.SZ 云南白药
    000430.SZ 张家界
    600030.SH 中信证券
    :param index_config: 想预测的数据的索引，对应关系如下：
    0：open
    1：high
    2：low
    3：close
    :return: 预测值
    """
    print('gru_pred')
    df = pro.daily(ts_code=ts_cd, start_date='20220101', end_date='')
    # 数据处理开始
    df = df.sort_index(ascending=True)
    df = df.set_index('trade_date')
    df.index = pd.to_datetime(df.index)
    df = df.drop(columns=['ts_code'])
    df = df.fillna(method='ffill')
    scaler1 = MinMaxScaler(feature_range=(-1, 1))
    scaler2 = MinMaxScaler(feature_range=(-1, 1))
    df.iloc[:, index_config] = scaler1.fit_transform(df.iloc[:, index_config].values.reshape(-1, 1))
    for i in range(9):
        df.iloc[:, i] = scaler2.fit_transform(df.iloc[:, i].values.reshape(-1, 1))
    # 取最近timesteps天的数据，预测明天的open
    df = df.iloc[0: timesteps - 7, ]
    x = test_data(df, timesteps - 7)
    x = torch.from_numpy(x).type(torch.Tensor)
    x = x.to(device)
    # 数据处理结束

    # 模型结构
    model = StockDNN()
    model=model.to(device)
    # 加载模型参数
    checkpoint = torch.load('/kaggle/working/{}_{}_DNN.pth'.format(name_dict[ts_cd], feature_dict[index_config]))
    model.load_state_dict(checkpoint['model'])
    # 预测
    y_pred = model(x)
    y_pred = scaler1.inverse_transform(y_pred.detach().cpu().numpy())
    return y_pred


def load_data(dataframe, timesteps, index_config):
    data_raw = dataframe.values
    data = []
    for index in range(len(data_raw) - timesteps):
        data.append(data_raw[index: index + timesteps])
    data = np.array(data)
    test_size = int(np.round(0.3 * data.shape[0]))
    train_size = data.shape[0] - test_size
    x_train = data[:train_size, :-7, :]     # data[批, timesteps， 特征]
    y_train = data[:train_size, -7:, index_config:index_config+1]      # 0:1 open   1:2 high   2:3 low   3:4 close
    x_test = data[train_size: , :-7, :]
    y_test = data[train_size: , -7:, index_config:index_config+1]
    return [x_train, y_train, x_test, y_test]


def test_data(dataframe, timesteps):
    data_raw = dataframe.values
    data = []
    data.append(data_raw[0: timesteps])
    data = np.array(data)
    x_test = data[:, :, :]  # data[批, timesteps， 特征]
    return x_test


class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(StockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_().to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_().to(device)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.linear(out[:, -1, :])
        return out


class StockGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(StockGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_().to(device)
        #         c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_().to(device)
        out, (hn) = self.gru(x, (h0.detach()))
        out = self.linear(out[:, -1, :])
        return out


class StockDNN(nn.Module):
    def __init__(self):
        super(StockDNN, self).__init__()
        self.linear1 = nn.Linear(53 * 9, 340)
        self.linear2 = nn.Linear(340, 240)
        self.linear3 = nn.Linear(240, 120)
        self.linear4 = nn.Linear(120, 7)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = x.view(-1, 53 * 9)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        return self.linear4(x)


def train_and_pred(ts_cd=''):
    if ts_cd == '':
        # LSTM
        # 5支股票的4个标签训练&测试
        lstm_result = []
        for stock in range(5):
            lstm_result.append([])
            for i in range(4):
                lstm_train(num_dict[stock], i)
                lstm_result[stock].append(lstm_pred(num_dict[stock], i))
        # GRU
        gru_result = []
        for stock in range(5):
            gru_result.append([])
            for i in range(4):
                gru_train(num_dict[stock], i)
                gru_result[stock].append(gru_pred(num_dict[stock], i))
        # DNN
        dnn_result = []
        for stock in range(5):
            dnn_result.append([])
            for i in range(4):
                dnn_train(num_dict[stock], i)
                dnn_result[stock].append(dnn_pred(num_dict[stock], i))
    else:
        # LSTM
        lstm_result = []
        for i in range(4):
            lstm_train(ts_cd, i)
            lstm_result.append(lstm_pred(ts_cd, i))
        # GRU
        gru_result = []
        for i in range(4):
            gru_train(ts_cd, i)
            gru_result.append(gru_pred(ts_cd, i))
        # DNN
        dnn_result = []
        for i in range(4):
            dnn_train(ts_cd, i)
            dnn_result.append(dnn_pred(ts_cd, i))
    return [np.array(lstm_result).squeeze(), np.array(gru_result).squeeze(), np.array(dnn_result).squeeze()]

if __name__ == '__main__':
    lstm, gru, dnn = train_and_pred()
    print(lstm)
    print(gru)
    print(dnn)
