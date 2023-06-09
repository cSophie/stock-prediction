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

# 预训练股票代码
pre_train_stock = ['600519.SH', '000001.SZ', '000538.SZ', '600030.SH', '000430.SZ']

# 在文件读写中标记open/high/low/close
feature_dict = {0: 'open',
                1: 'high',
                2: 'low',
                3: 'close'}

pro = ts.pro_api('e9b31113ccd628c7933a0af4e9c45f38aee75b5d9a4fb89fde3c460a')
start_dt = '20190101'
end_dt = ''
timesteps = 60
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
input_size = 9
hidden_size = 64
num_layers = 2
output_size = 7


def lstm_train(ts_cd, index, dataframe):
    print('lstm_train')
    df = dataframe
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
        opt.zero_grad()
        loss.backward()
        opt.step()
    state = {'model': model.state_dict(), 'optimizer': opt.state_dict(), 'epoch': num_epochs}
    torch.save(state, './model_para_stock/{}_{}_LSTM.pth'.format(ts_cd, feature_dict[index]))


def gru_train(ts_cd, index, dataframe):
    print('gru_train')
    df = dataframe
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
        opt.zero_grad()
        loss.backward()
        opt.step()
    state = {'model': model.state_dict(), 'optimizer': opt.state_dict(), 'epoch': num_epochs}
    torch.save(state, './model_para_stock/{}_{}_GRU.pth'.format(ts_cd, feature_dict[index]))


def dnn_train(ts_cd, index, dataframe):
    print('dnn_train')
    df = dataframe
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
#         if epoch % 10 == 0 and epoch != 0:
#             print('Epoch ', epoch, 'MSE: ', loss.item())
        opt.zero_grad()
        loss.backward()
        opt.step()
    state = {'model': model.state_dict(), 'optimizer': opt.state_dict(), 'epoch': num_epochs}
    torch.save(state, './model_para_stock/{}_{}_DNN.pth'.format(ts_cd, feature_dict[index]))


def lstm_pred(ts_cd, index, dataframe):
    """
    指定待预测的股票代码和数据索引，使用LSTM预测下周的相应值。
    :param ts_cd: 股票代码，包括：
    600519.SH 贵州茅台
    000001.SZ 平安银行
    000538.SZ 云南白药
    000430.SZ 张家界
    600030.SH 中信证券
    :param index: 想预测的数据的索引，对应关系如下：
    0：open
    1：high
    2：low
    3：close
    :return: 预测值
    """
    print('lstm_pred')
    df = dataframe
    # 数据处理开始
    scaler1 = MinMaxScaler(feature_range=(-1, 1))
    scaler2 = MinMaxScaler(feature_range=(-1, 1))
    df.iloc[:, index] = scaler1.fit_transform(df.iloc[:, index].values.reshape(-1, 1))
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
    checkpoint = torch.load('./model_para_stock/{}_{}_LSTM.pth'.format(ts_cd, feature_dict[index]))
    model.load_state_dict(checkpoint['model'])
    # 预测
    y_pred = model(x)
    y_pred = scaler1.inverse_transform(y_pred.detach().cpu().numpy())
    return y_pred


def gru_pred(ts_cd, index, dataframe):
    """
    指定待预测的股票代码和数据索引，使用GRU预测下周的相应值。
    :param ts_cd: 股票代码，包括：
    600519.SH 贵州茅台
    000001.SZ 平安银行
    000538.SZ 云南白药
    000430.SZ 张家界
    600030.SH 中信证券
    :param index: 想预测的数据的索引，对应关系如下：
    0：open
    1：high
    2：low
    3：close
    :return: 预测值
    """
    print('gru_pred')
    df = dataframe
    # 数据处理开始
    scaler1 = MinMaxScaler(feature_range=(-1, 1))
    scaler2 = MinMaxScaler(feature_range=(-1, 1))
    df.iloc[:, index] = scaler1.fit_transform(df.iloc[:, index].values.reshape(-1, 1))
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
    checkpoint = torch.load('./model_para_stock/{}_{}_GRU.pth'.format(ts_cd, feature_dict[index]))
    model.load_state_dict(checkpoint['model'])
    # 预测
    y_pred = model(x)
    y_pred = scaler1.inverse_transform(y_pred.detach().cpu().numpy())
    return y_pred


def dnn_pred(ts_cd, index, dataframe):
    """
    指定待预测的股票代码和数据索引，使用DNN预测下周的相应值。
    :param ts_cd: 股票代码，包括：
    600519.SH 贵州茅台
    000001.SZ 平安银行
    000538.SZ 云南白药
    000430.SZ 张家界
    600030.SH 中信证券
    :param index: 想预测的数据的索引，对应关系如下：
    0：open
    1：high
    2：low
    3：close
    :return: 预测值
    """
    print('dnn_pred')
    df = dataframe
    # 数据处理开始
    scaler1 = MinMaxScaler(feature_range=(-1, 1))
    scaler2 = MinMaxScaler(feature_range=(-1, 1))
    df.iloc[:, index] = scaler1.fit_transform(df.iloc[:, index].values.reshape(-1, 1))
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
    checkpoint = torch.load('./model_para_stock/{}_{}_DNN.pth'.format(ts_cd, feature_dict[index]))
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


def data_process_train(index, dataframe):
    df = dataframe
    df = df.sort_index(ascending=True)
    df = df.set_index('trade_date')
    df.index = pd.to_datetime(df.index)
    df = df.drop(columns=['ts_code'])
    df = df.fillna(method='ffill')
    scaler1 = MinMaxScaler(feature_range=(-1, 1))
    scaler2 = MinMaxScaler(feature_range=(-1, 1))
    df.iloc[:, index] = scaler1.fit_transform(df.iloc[:, index].values.reshape(-1, 1))
    for i in range(9):
        df.iloc[:, i] = scaler2.fit_transform(df.iloc[:, i].values.reshape(-1, 1))
    return df


def data_process_pred(index, dataframe):
    df = dataframe
    df = df.sort_index(ascending=True)
    df = df.set_index('trade_date')
    df.index = pd.to_datetime(df.index)
    df = df.drop(columns=['ts_code'])
    df = df.fillna(method='ffill')
    return df


def train(ts_cd):
    df = pro.daily(ts_code=ts_cd, start_date=start_dt, end_date=end_dt)
    for i in range(4):
        lstm_train(ts_cd, i, data_process_train(i, df))
        gru_train(ts_cd, i, data_process_train(i, df))
        dnn_train(ts_cd, i, data_process_train(i, df))


def pred(ts_cd):
    lstm_res = []
    gru_res = []
    dnn_res = []
    df = pro.daily(ts_code=ts_cd, start_date=start_dt, end_date=end_dt)
    for i in range(4):
        lstm_res.append(lstm_pred(ts_cd, i, data_process_pred(i, df)))
        gru_res.append(gru_pred(ts_cd, i, data_process_pred(i, df)))
        dnn_res.append(dnn_pred(ts_cd, i, data_process_pred(i, df)))
    # dnn数据处理
    for stock in range(5):
        for day in range(7):
            high = dnn_res[stock][1][day]
            low = dnn_res[stock][2][day]
            for i in range(4):
                if high < dnn_res[stock][i][day]:
                    high = dnn_res[stock][i][day]
                if low > dnn_res[stock][i][day]:
                    low = dnn_res[stock][i][day]
            dnn_res[stock][1][day] = high
            dnn_res[stock][2][day] = low
    return [np.array(lstm_res).squeeze(), np.array(gru_res).squeeze(), np.array(dnn_res).squeeze()]


def pre_train_and_pred():
    """
    预训练5支股票，如需要增减、改变预训练的股票，在pre_train_stock中更改
    """
    lstm = []
    gru = []
    dnn = []
    for stock in pre_train_stock:
        print(stock)
        train(stock)
        temp1, temp2, temp3 = pred(stock)
        lstm.append(temp1)
        gru.append(temp2)
        dnn.append(temp3)
    return [np.array(lstm), np.array(gru), np.array(dnn)]


def select_train_and_pred(ts_cd):
    train(ts_cd)
    return pred(ts_cd)


def pct_chg(today, next_week):
    res = []
    temp = (next_week[0] - today) / today
    res.append(temp)
    for i in range(6):
        temp = (next_week[i + 1] - next_week[i]) / next_week[i] * 100
        res.append(temp)
    return np.array(res)


def get_pct_chg(ts_cd, next_week):
    df = pro.daily(ts_code=ts_cd, start_date=start_dt, end_date=end_dt)
    return pct_chg(df.iloc[0, 5], next_week)        # sector: df.iloc[0, 2]


if __name__ == '__main__':
    lstm, gru, dnn = pre_train_and_pred()
    print(lstm.shape)           # (5, 4, 7)   [股票种类, 标签, 天数]
    print(lstm)
    print(gru)
    print(dnn)
    # 涨跌幅
    for i in range(5):
        stock = pre_train_stock[i]
        temp = get_pct_chg(stock, lstm[i, 3])
        print(temp)
    # 自选
    lstm, gru, dnn = select_train_and_pred('300999.SZ')
    print(lstm.shape)           # (4, 7)      [标签, 天数]
    print(lstm)
    print(gru)
    print(dnn)
    # 涨跌幅
    print(get_pct_chg('300999.SZ', lstm[3]))


