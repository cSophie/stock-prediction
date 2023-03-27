import tushare as ts
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler


# 885431.TI 新能源汽车
# 885338.TI 融资融券
# 885728.TI 人工智能
# 885757.TI 区块链
# 885362.TI 云计算

# 预训练板块代码
pre_train_sector = [
    '885431.TI',
    '885338.TI',
    '885728.TI',
    '885757.TI',
    '885362.TI']
# 在文件读写中标记close/open/high/low
feature_dict = {0: 'close',
                1: 'open',
                2: 'high',
                3: 'low'}


pro = ts.pro_api('e9b31113ccd628c7933a0af4e9c45f38aee75b5d9a4fb89fde3c460a')
start_dt = '20190101'
end_dt = ''
timesteps = 60
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
input_size = 10
hidden_size = 64
num_layers = 2
output_size = 7


def gru_train(ts_cd, index, dataframe):
    print('gru_train')
    df = dataframe
    df = df.sort_index(ascending=True)
    df = df.set_index('trade_date')
    df.index = pd.to_datetime(df.index)
    df = df.drop(columns=['ts_code'])
    df = df.fillna(method='ffill')
    scaler1 = MinMaxScaler(feature_range=(-1, 1))
    scaler2 = MinMaxScaler(feature_range=(-1, 1))
    df.iloc[:, index] = scaler1.fit_transform(df.iloc[:, index].values.reshape(-1, 1))
    for i in range(10):                                                          ###
        df.iloc[:,i] = scaler2.fit_transform(df.iloc[:,i].values.reshape(-1, 1))
    x_train, y_train, x_test, y_test = load_data(df, timesteps, index)
    x_train = torch.from_numpy(x_train).type(torch.Tensor)
    x_test = torch.from_numpy(x_test).type(torch.Tensor)
    y_train = torch.from_numpy(y_train).type(torch.Tensor)
    y_test = torch.from_numpy(y_test).type(torch.Tensor)
    model = StockGRU(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, output_size = output_size)
    loss_fn = torch.nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr = 0.01)
    model = model.to(device)
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
    torch.save(state, './model_para_sector/{}_{}_GRU.pth'.format(ts_cd, feature_dict[index]))


def gru_pred(ts_cd, index_config, dataframe):
    """
    指定待预测的板块代码和数据索引，使用GRU预测下周的相应值。
    :param ts_cd: 板块代码，包括：
    885431.TI 新能源汽车
    885338.TI 融资融券
    885728.TI 人工智能
    885757.TI 区块链
    885362.TI 云计算
    ...
    :param index_config: 想预测的数据的索引，对应关系如下：
    0：close
    1：open
    2：high
    3：low
    :return: 预测值
    """
    print('gru_pred')
    df = dataframe
    # 数据处理开始
    df = df.sort_index(ascending=True)
    df = df.set_index('trade_date')
    df.index = pd.to_datetime(df.index)
    df = df.drop(columns=['ts_code'])
    df = df.fillna(method='ffill')
    scaler1 = MinMaxScaler(feature_range=(-1, 1))
    scaler2 = MinMaxScaler(feature_range=(-1, 1))
    df.iloc[:, index_config] = scaler1.fit_transform(df.iloc[:, index_config].values.reshape(-1, 1))
    for i in range(10):        ###
        df.iloc[:, i] = scaler2.fit_transform(df.iloc[:, i].values.reshape(-1, 1))
    # 取最近timesteps天的数据，预测明天的open
    df = df.iloc[0: timesteps, ]
    x = test_data(df, timesteps)
    x = torch.from_numpy(x).type(torch.Tensor)
    x = x.to(device)
    # 数据处理结束

    # 模型结构
    model = StockGRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size)
    model = model.to(device)
    # 加载模型参数
    checkpoint = torch.load('./model_para_sector/{}_{}_GRU.pth'.format(ts_cd, feature_dict[index_config]))
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


def train(ts_cd):
    df = pro.ths_daily(ts_code=ts_cd, start_date=start_dt, end_date=end_dt)  ###
    for i in range(4):
        gru_train(ts_cd, i, df)


def pred(ts_cd):
    df = pro.ths_daily(ts_code=ts_cd, start_date=start_dt, end_date=end_dt)  ###
    gru_res = []
    for i in range(4):
        gru_res.append(gru_pred(ts_cd, i, df))
    return np.array(gru_res).squeeze()


def pre_train_and_pred():
    """
    预训练5个板块，如需要增减、改变预训练的板块，在pre_train_sector中更改
    """
    gru = []
    for stock in pre_train_sector:
        print(stock)
        train(stock)
        temp = pred(stock)
        gru.append(temp)
    return np.array(gru)


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
    return pct_chg(df.iloc[0, 2], next_week)        # sector: df.iloc[0, 2]


if __name__ == '__main__':
    gru = pre_train_and_pred()
    print(gru.shape)     # (5, 4, 7)   [板块种类, 标签, 天数]
    print(gru)
    """
    gru.shape == (5, 4, 7)      [板块种类, 标签, 天数]
    gru[ , , 0]: 
                        close   open   high   low
    885431.TI 新能源汽车
    885338.TI 融资融券
    885728.TI 人工智能
    885757.TI 区块链
    885362.TI 云计算
    """
    # 自选
    gru = select_train_and_pred('885976.TI')
    print(gru.shape)     # (4, 7)      [标签, 天数]
    print(gru)
