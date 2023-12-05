import pandas as pd
import numpy as np
import math
import csv
import statistics

from FinMind.data import DataLoader
import talib

from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mpl_dates
import matplotlib.pyplot as plt

from datetime import datetime
from io import StringIO
import pprint as pp

api_token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJkYXRlIjoiMjAyMS0xMi0yNyAxNDo1OTowOSIsInVzZXJfaWQiOiJkdXJhbnQ3MTA5MTYiLCJpcCI6IjE0MC4xMjAuMTMuMjMwIn0.8-KIC3-OA4D6JcOtQ_fJBOVkyugx60t1Gy82c57TLz4"

api = DataLoader()
api.login_by_token(api_token = api_token)

stock_list = ['2330', '2603', '2002','1301', '2801']
#input_type = "train"
input_type = "test"

if input_type == "train":
    start_date = '2000-11-17'
    end_date = '2021-05-31'
elif input_type == "test":
    start_date = '2020-11-14'
    end_date = '2021-12-31'

    
for stock_id in stock_list:
    print(stock_id)
    TaiwanStockPriceDay = api.taiwan_stock_daily(
    stock_id = stock_id,
    start_date = start_date,
    end_date = end_date
    )
    
    
    TaiwanStockPriceDay['EMA5'] = talib.EMA(TaiwanStockPriceDay['close'], timeperiod=5)
    TaiwanStockPriceDay['EMA20'] = talib.EMA(TaiwanStockPriceDay['close'], timeperiod=20)
    TaiwanStockPriceDay['K'], TaiwanStockPriceDay['D'] = talib.STOCH(TaiwanStockPriceDay['max'], TaiwanStockPriceDay['min'], TaiwanStockPriceDay['close'], fastk_period=14, slowk_period=2, slowk_matype=0, slowd_period=3, slowd_matype=0)
    TaiwanStockPriceDay[34:]



    ## Strategy 1: 移動平均線交叉
    print("SMA")
    flag = 0

    trading_info = []
    buy_sell_tuple = []
    buy_list = []
    sell_list = []
    return_list = []
    temp = []
    trading_dic = {}

    for i in range(len(TaiwanStockPriceDay[34:])):
        if TaiwanStockPriceDay['EMA5'][i+32] < TaiwanStockPriceDay['EMA20'][i+32] and TaiwanStockPriceDay['EMA5'][i+33] > TaiwanStockPriceDay['EMA20'][i+33]:
            # 黃金交叉，買點
            if flag == 0:
                flag = 1
                trading_info.append([i, 'buy', TaiwanStockPriceDay['close'][i+34]])
                trading_dic[i] = ['buy', TaiwanStockPriceDay['close'][i+34]]
                temp.append([i, TaiwanStockPriceDay['close'][i+34]])
            else:
                trading_info.append([i, 'hold', TaiwanStockPriceDay['close'][i+34]])
                trading_dic[i] = ['hold', TaiwanStockPriceDay['close'][i+34]]
        elif TaiwanStockPriceDay['EMA5'][i+32] > TaiwanStockPriceDay['EMA20'][i+32] and TaiwanStockPriceDay['EMA5'][i+33] < TaiwanStockPriceDay['EMA20'][i+33]:
            # 死亡交叉，賣點
            if flag == 1:
                flag = 0
                trading_info.append([i, 'sell', TaiwanStockPriceDay['close'][i+34]])
                trading_dic[i] = ['sell', TaiwanStockPriceDay['close'][i+34]]
                temp.append([i, TaiwanStockPriceDay['close'][i+34]])
                temp.append((temp[1][1] - temp[0][1])/temp[0][1])
                return_list.append((temp[1][1] - temp[0][1])/temp[0][1])
                buy_sell_tuple.append(temp)
                temp = []
            else:
                trading_info.append([i, 'hold', TaiwanStockPriceDay['close'][i+34]])
                trading_dic[i] = ['hold', TaiwanStockPriceDay['close'][i+34]]
        else:
            trading_info.append([i, 'hold', TaiwanStockPriceDay['close'][i+34]])
            trading_dic[i] = ['hold', TaiwanStockPriceDay['close'][i+34]]

    if flag == 1:
        trading_info[-1][1] = 'sell'
        trading_dic[len(TaiwanStockPriceDay[34:])-1][0] = 'sell'
        temp.append([i, TaiwanStockPriceDay['close'][i+34]])
        temp.append((temp[1][1] - temp[0][1])/temp[0][1])
        return_list.append((temp[1][1] - temp[0][1])/temp[0][1])
        buy_sell_tuple.append(temp)
        temp = []

    for trade in buy_sell_tuple:
        buy_list.append(trade[0])
        sell_list.append(trade[1])

    #plot_stock(TaiwanStockPriceDay[34:], buy_list, sell_list)
    
    
    TaiwanStockPriceDay = api.taiwan_stock_daily(
        stock_id = stock_id,
        start_date = start_date,
        end_date = end_date
    )
    TaiwanStockPriceDay['EMA5'] = talib.EMA(TaiwanStockPriceDay['close'], timeperiod=5)
    TaiwanStockPriceDay['EMA20'] = talib.EMA(TaiwanStockPriceDay['close'], timeperiod=20)
    TaiwanStockPriceDay['K'], TaiwanStockPriceDay['D'] = talib.STOCH(TaiwanStockPriceDay['max'], TaiwanStockPriceDay['min'], TaiwanStockPriceDay['close'], fastk_period=14, slowk_period=2, slowk_matype=0, slowd_period=3, slowd_matype=0)
    
    
    if input_type == 'train':
        with open("./data/Trajectory/Train/" + stock_id + "_" + "SMA_trajectory_all_train2.csv",  'w', encoding='utf8', newline='') as csvFile:
            writer = csv.writer(csvFile)
            for trade in trading_info:
                writer.writerow(trade)
                
    elif input_type == 'test':
        with open("./data/Trajectory/Test/" + stock_id + "_" + "SMA_trajectory_all_test.csv",  'w', encoding='utf8', newline='') as csvFile:
            writer = csv.writer(csvFile)
            for trade in trading_info:
                writer.writerow(trade)

    total_num = len(buy_sell_tuple)
    total_sum = 0
    selected_num = 0
    selected_sum = 0

    profit_num = 0

    median = statistics.median(return_list)

    buy_list = []
    sell_list = []

    for trade in buy_sell_tuple:
        total_sum += trade[2]

        if trade[2] > 0:
            profit_num += 1

        if trade[2] >= median:
            selected_num += 1
            selected_sum += trade[2]
            buy_list.append(trade[0])
            sell_list.append(trade[1])
        else:
            trading_dic[trade[0][0]][0] = 'hold'
            trading_dic[trade[1][0]][0] = 'hold'

    print(median)

    print("【未篩選前】")
    print('賺錢的交易比例：', profit_num/total_num)
    print('平均報酬率：', total_sum/total_num)
    print('交易次數：',  total_num)
    print()


    print("【篩選後：保留一半】")
    print('平均報酬率：', selected_sum/selected_num)
    print('交易次數：',  selected_num)
    print()

    #plot_stock(TaiwanStockPriceDay[34:], buy_list, sell_list)

    #pp.pprint(trading_dic)
    
    if input_type == 'train':
        with open("./data/Trajectory/Train/" + stock_id + "_" + "SMA_trajectory_50_train2.csv",  'w', encoding='utf8', newline='') as csvFile:
            writer = csv.writer(csvFile)
            for key in trading_dic.keys():
                #print(key)
                #print(trading_dic[key])
                writer.writerow([key, trading_dic[key][0], trading_dic[key][1]])
    elif input_type == 'test':
        with open("./data/Trajectory/Test/" + stock_id + "_" + "SMA_trajectory_50_test.csv",  'w', encoding='utf8', newline='') as csvFile:
            writer = csv.writer(csvFile)
            for key in trading_dic.keys():
                #print(key)
                #print(trading_dic[key])
                writer.writerow([key, trading_dic[key][0], trading_dic[key][1]])




    ## Strategy 2: KD
    print("KD")
                
                
    TaiwanStockPriceDay = api.taiwan_stock_daily(
        stock_id = stock_id,
        start_date = start_date,
        end_date = end_date
    )

    TaiwanStockPriceDay['K'], TaiwanStockPriceDay['D'] = talib.STOCH(TaiwanStockPriceDay['max'], TaiwanStockPriceDay['min'], TaiwanStockPriceDay['close'], fastk_period=14, slowk_period=2, slowk_matype=0, slowd_period=3, slowd_matype=0)


    flag = 0

    trading_info = []
    buy_sell_tuple = []
    buy_list = []
    sell_list = []
    return_list = []
    temp = []

    for i in range(len(TaiwanStockPriceDay[34:])):
        if TaiwanStockPriceDay['K'][i+32] < TaiwanStockPriceDay['D'][i+32] and TaiwanStockPriceDay['K'][i+33] > TaiwanStockPriceDay['D'][i+33]:
            # 黃金交叉，買點
            if flag == 0:
                flag = 1
                trading_info.append([i, 'buy', TaiwanStockPriceDay['close'][i+34]])
                trading_dic[i] = ['buy', TaiwanStockPriceDay['close'][i+34]]
                temp.append([i, TaiwanStockPriceDay['close'][i+34]])
            else:
                trading_info.append([i, 'hold', TaiwanStockPriceDay['close'][i+34]])
                trading_dic[i] = ['hold', TaiwanStockPriceDay['close'][i+34]]
        elif TaiwanStockPriceDay['K'][i+32] > TaiwanStockPriceDay['D'][i+32] and TaiwanStockPriceDay['K'][i+33] < TaiwanStockPriceDay['D'][i+33]:
            # 死亡交叉，賣點
            if flag == 1:
                flag = 0
                trading_info.append([i, 'sell', TaiwanStockPriceDay['close'][i+34]])
                trading_dic[i] = ['sell', TaiwanStockPriceDay['close'][i+34]]
                temp.append([i, TaiwanStockPriceDay['close'][i+34]])
                temp.append((temp[1][1] - temp[0][1])/temp[0][1])
                return_list.append((temp[1][1] - temp[0][1])/temp[0][1])
                buy_sell_tuple.append(temp)
                temp = []
            else:
                trading_info.append([i, 'hold', TaiwanStockPriceDay['close'][i+34]])
                trading_dic[i] = ['hold', TaiwanStockPriceDay['close'][i+34]]
        else:
            trading_info.append([i, 'hold', TaiwanStockPriceDay['close'][i+34]])
            trading_dic[i] = ['hold', TaiwanStockPriceDay['close'][i+34]]

    if flag == 1:
        trading_info[-1][1] = 'sell'
        trading_dic[len(TaiwanStockPriceDay[34:])-1][0] = 'sell'
        temp.append([i, TaiwanStockPriceDay['close'][i+34]])
        temp.append((temp[1][1] - temp[0][1])/temp[0][1])
        return_list.append((temp[1][1] - temp[0][1])/temp[0][1])
        buy_sell_tuple.append(temp)
        temp = []



    for trade in buy_sell_tuple:
        buy_list.append(trade[0])
        sell_list.append(trade[1])


    #plot_stock(TaiwanStockPriceDay[34:], buy_list, sell_list)
    TaiwanStockPriceDay = api.taiwan_stock_daily(
        stock_id = stock_id,
        start_date = start_date,
        end_date = end_date
    )
    TaiwanStockPriceDay['K'], TaiwanStockPriceDay['D'] = talib.STOCH(TaiwanStockPriceDay['max'], TaiwanStockPriceDay['min'], TaiwanStockPriceDay['close'], fastk_period=14, slowk_period=2, slowk_matype=0, slowd_period=3, slowd_matype=0)

    
    if input_type == 'train':
        with open("./data/Trajectory/Train/" + stock_id + "_" + "KD_trajectory_all_train2.csv",  'w', encoding='utf8', newline='') as csvFile:
            writer = csv.writer(csvFile)
            for trade in trading_info:
                writer.writerow(trade)
    elif input_type == 'test':
        with open("./data/Trajectory/Test/" + stock_id + "_" + "KD_trajectory_all_test.csv",  'w', encoding='utf8', newline='') as csvFile:
            writer = csv.writer(csvFile)
            for trade in trading_info:
                writer.writerow(trade)
                

    total_num = len(buy_sell_tuple)
    total_sum = 0
    selected_num = 0
    selected_sum = 0

    profit_num = 0

    median = statistics.median(return_list)

    buy_list = []
    sell_list = []

    for trade in buy_sell_tuple:
        total_sum += trade[2]

        if trade[2] > 0:
            profit_num += 1

        if trade[2] >= median:
            selected_num += 1
            selected_sum += trade[2]
            buy_list.append(trade[0])
            sell_list.append(trade[1])
        else:
            trading_dic[trade[0][0]][0] = 'hold'
            trading_dic[trade[1][0]][0] = 'hold'

    print(median)

    print("【未篩選前】")
    print('賺錢的交易比例：', profit_num/total_num)
    print('平均報酬率：', total_sum/total_num)
    print('交易次數：',  total_num)
    print()


    print("【篩選後：保留一半】")
    print('平均報酬率：', selected_sum/selected_num)
    print('交易次數：',  selected_num)
    print()

    #plot_stock(TaiwanStockPriceDay[34:], buy_list, sell_list)

    #pp.pprint(trading_dic)

    if input_type == 'train':
        with open("./data/Trajectory/Train/" + stock_id + "_" + "KD_trajectory_50_train2.csv",  'w', encoding='utf8', newline='') as csvFile:
            writer = csv.writer(csvFile)
            for key in trading_dic.keys():
                #print(key)
                #print(trading_dic[key])
                writer.writerow([key, trading_dic[key][0], trading_dic[key][1]])
    elif input_type == 'test':
        with open("./data/Trajectory/Test/" + stock_id + "_" + "KD_trajectory_50_test.csv",  'w', encoding='utf8', newline='') as csvFile:
            writer = csv.writer(csvFile)
            for key in trading_dic.keys():
                #print(key)
                #print(trading_dic[key])
                writer.writerow([key, trading_dic[key][0], trading_dic[key][1]])


    #Strategy 3: 威廉指標

    print("WilliamR")

    TaiwanStockPriceDay = api.taiwan_stock_daily(
        stock_id = stock_id,
        start_date = start_date,
        end_date = end_date
    )


    TaiwanStockPriceDay['willr'] = talib.WILLR(TaiwanStockPriceDay['max'], TaiwanStockPriceDay['min'], TaiwanStockPriceDay['close'], timeperiod=14)
    #TaiwanStockPriceDay['Bias'] = (TaiwanStockPriceDay['close'] - TaiwanStockPriceDay['close'].rolling(24, min_periods=1).mean())/ TaiwanStockPriceDay['close'].rolling(24, min_periods=1).mean()*100


    flag = 0

    trading_info = []
    buy_sell_tuple = []
    buy_list = []
    sell_list = []
    return_list = []
    temp = []

    for i in range(len(TaiwanStockPriceDay[34:])):
        if TaiwanStockPriceDay['willr'][i+33] < -0.8:
            # 黃金交叉，買點
            if flag == 0:
                flag = 1
                trading_info.append([i, 'buy', TaiwanStockPriceDay['close'][i+34]])
                trading_dic[i] = ['buy', TaiwanStockPriceDay['close'][i+34]]
                temp.append([i, TaiwanStockPriceDay['close'][i+34]])
            else:
                trading_info.append([i, 'hold', TaiwanStockPriceDay['close'][i+34]])
                trading_dic[i] = ['hold', TaiwanStockPriceDay['close'][i+34]]
        elif TaiwanStockPriceDay['willr'][i+33] > -0.2:
            # 死亡交叉，賣點
            if flag == 1:
                flag = 0
                trading_info.append([i, 'sell', TaiwanStockPriceDay['close'][i+34]])
                trading_dic[i] = ['sell', TaiwanStockPriceDay['close'][i+34]]
                temp.append([i, TaiwanStockPriceDay['close'][i+34]])
                temp.append((temp[1][1] - temp[0][1])/temp[0][1])
                return_list.append((temp[1][1] - temp[0][1])/temp[0][1])
                buy_sell_tuple.append(temp)
                temp = []
            else:
                trading_info.append([i, 'hold', TaiwanStockPriceDay['close'][i+34]])
                trading_dic[i] = ['hold', TaiwanStockPriceDay['close'][i+34]]
        else:
            trading_info.append([i, 'hold', TaiwanStockPriceDay['close'][i+34]])
            trading_dic[i] = ['hold', TaiwanStockPriceDay['close'][i+34]]

    if flag == 1:
        trading_info[-1][1] = 'sell'
        trading_dic[len(TaiwanStockPriceDay[34:])-1][0] = 'sell'
        temp.append([i, TaiwanStockPriceDay['close'][i+34]])
        temp.append((temp[1][1] - temp[0][1])/temp[0][1])
        return_list.append((temp[1][1] - temp[0][1])/temp[0][1])
        buy_sell_tuple.append(temp)
        temp = []

    #pp.pprint(buy_sell_tuple)

    for trade in buy_sell_tuple:
        buy_list.append(trade[0])
        sell_list.append(trade[1])


    #plot_stock(TaiwanStockPriceDay[34:], buy_list, sell_list)
    TaiwanStockPriceDay = api.taiwan_stock_daily(
        stock_id = stock_id,
        start_date = start_date,
        end_date = end_date
    )
    TaiwanStockPriceDay['willr'] = talib.WILLR(TaiwanStockPriceDay['max'], TaiwanStockPriceDay['min'], TaiwanStockPriceDay['close'], timeperiod=14)

    if input_type == 'train':
        with open("./data/Trajectory/Train/" + stock_id + "_" + "William_trajectory_all_train2.csv",  'w', encoding='utf8', newline='') as csvFile:
            writer = csv.writer(csvFile)
            for trade in trading_info:
                writer.writerow(trade)
    elif input_type == 'test':
        with open("./data/Trajectory/Test/" + stock_id + "_" + "William_trajectory_all_test.csv",  'w', encoding='utf8', newline='') as csvFile:
            writer = csv.writer(csvFile)
            for trade in trading_info:
                writer.writerow(trade)


    total_num = len(buy_sell_tuple)
    total_sum = 0
    selected_num = 0
    selected_sum = 0

    profit_num = 0

    median = statistics.median(return_list)

    buy_list = []
    sell_list = []

    for trade in buy_sell_tuple:
        total_sum += trade[2]

        if trade[2] > 0:
            profit_num += 1

        if trade[2] >= median:
            selected_num += 1
            selected_sum += trade[2]
            buy_list.append(trade[0])
            sell_list.append(trade[1])
        else:
            trading_dic[trade[0][0]][0] = 'hold'
            trading_dic[trade[1][0]][0] = 'hold'

    print(median)

    print("【未篩選前】")
    print('賺錢的交易比例：', profit_num/total_num)
    print('平均報酬率：', total_sum/total_num)
    print('交易次數：',  total_num)
    print()


    print("【篩選後：保留一半】")
    print('平均報酬率：', selected_sum/selected_num)
    print('交易次數：',  selected_num)
    print()

    #plot_stock(TaiwanStockPriceDay[34:], buy_list, sell_list)

    #pp.pprint(trading_dic)
    
    if input_type == 'train':
        with open("./data/Trajectory/Train/" + stock_id + "_" + "William_trajectory_50_train2.csv",  'w', encoding='utf8', newline='') as csvFile:
            writer = csv.writer(csvFile)
            for key in trading_dic.keys():
                #print(key)
                #print(trading_dic[key])
                writer.writerow([key, trading_dic[key][0], trading_dic[key][1]])
    elif input_type == 'test':
        with open("./data/Trajectory/Test/" + stock_id + "_" + "William_trajectory_50_test.csv",  'w', encoding='utf8', newline='') as csvFile:
            writer = csv.writer(csvFile)
            for key in trading_dic.keys():
                #print(key)
                #print(trading_dic[key])
                writer.writerow([key, trading_dic[key][0], trading_dic[key][1]])






    #Strategy 3: 布林通道
    print("BBAND")

    TaiwanStockPriceDay = api.taiwan_stock_daily(
        stock_id = stock_id,
        start_date = start_date,
        end_date = end_date
    )

    TaiwanStockPriceDay['TP'] = (TaiwanStockPriceDay['close'] + TaiwanStockPriceDay['min'] + TaiwanStockPriceDay['max']) / 3
    TaiwanStockPriceDay['std'] = TaiwanStockPriceDay['TP'].rolling(20).std(ddof=0)
    TaiwanStockPriceDay['MA-TP'] = TaiwanStockPriceDay['TP'].rolling(20).mean()
    TaiwanStockPriceDay['BOLU'] = TaiwanStockPriceDay['MA-TP'] + 2*TaiwanStockPriceDay['std']
    TaiwanStockPriceDay['BOLD'] = TaiwanStockPriceDay['MA-TP'] - 2*TaiwanStockPriceDay['std']

    flag = 0

    trading_info = []
    buy_sell_tuple = []
    buy_list = []
    sell_list = []
    return_list = []
    temp = []


    for i in range(len(TaiwanStockPriceDay[34:])):
        #if TaiwanStockPriceDay['close'][i+32] < TaiwanStockPriceDay['BOLD'][i+32] and TaiwanStockPriceDay['close'][i+33] > TaiwanStockPriceDay['BOLD'][i+33]:
        if TaiwanStockPriceDay['close'][i+33] > TaiwanStockPriceDay['BOLD'][i+33]:
            # 黃金交叉，買點
            if flag == 0:
                flag = 1
                trading_info.append([i, 'buy', TaiwanStockPriceDay['close'][i+34]])
                trading_dic[i] = ['buy', TaiwanStockPriceDay['close'][i+34]]
                temp.append([i, TaiwanStockPriceDay['close'][i+34]])
            else:
                trading_info.append([i, 'hold', TaiwanStockPriceDay['close'][i+34]])
                trading_dic[i] = ['hold', TaiwanStockPriceDay['close'][i+34]]
        elif TaiwanStockPriceDay['close'][i+33] < TaiwanStockPriceDay['MA-TP'][i+33]:
        #elif TaiwanStockPriceDay['close'][i+33] > TaiwanStockPriceDay['BOLD'][i+33]:
            # 死亡交叉，賣點
            if flag == 1:
                flag = 0
                trading_info.append([i, 'sell', TaiwanStockPriceDay['close'][i+34]])
                trading_dic[i] = ['sell', TaiwanStockPriceDay['close'][i+34]]
                temp.append([i, TaiwanStockPriceDay['close'][i+34]])
                temp.append((temp[1][1] - temp[0][1])/temp[0][1])
                return_list.append((temp[1][1] - temp[0][1])/temp[0][1])
                buy_sell_tuple.append(temp)
                temp = []
            else:
                trading_info.append([i, 'hold', TaiwanStockPriceDay['close'][i+34]])
                trading_dic[i] = ['hold', TaiwanStockPriceDay['close'][i+34]]
        else:
            trading_info.append([i, 'hold', TaiwanStockPriceDay['close'][i+34]])
            trading_dic[i] = ['hold', TaiwanStockPriceDay['close'][i+34]]

    if flag == 1:
        trading_info[-1][1] = 'sell'
        trading_dic[len(TaiwanStockPriceDay[34:])-1][0] = 'sell'
        temp.append([i, TaiwanStockPriceDay['close'][i+34]])
        temp.append((temp[1][1] - temp[0][1])/temp[0][1])
        return_list.append((temp[1][1] - temp[0][1])/temp[0][1])
        buy_sell_tuple.append(temp)
        temp = []





    for trade in buy_sell_tuple:
        buy_list.append(trade[0])
        sell_list.append(trade[1])


    #plot_stock(TaiwanStockPriceDay[34:], buy_list, sell_list)
    TaiwanStockPriceDay = api.taiwan_stock_daily(
        stock_id = stock_id,
        start_date = start_date,
        end_date = end_date
    )

    TaiwanStockPriceDay['TP'] = (TaiwanStockPriceDay['close'] + TaiwanStockPriceDay['min'] + TaiwanStockPriceDay['max']) / 3
    TaiwanStockPriceDay['std'] = TaiwanStockPriceDay['TP'].rolling(20).std(ddof=0)
    TaiwanStockPriceDay['MA-TP'] = TaiwanStockPriceDay['TP'].rolling(20).mean()
    TaiwanStockPriceDay['BOLU'] = TaiwanStockPriceDay['MA-TP'] + 2*TaiwanStockPriceDay['std']
    TaiwanStockPriceDay['BOLD'] = TaiwanStockPriceDay['MA-TP'] - 2*TaiwanStockPriceDay['std']

    if input_type == 'train':
        with open("./data/Trajectory/Train/" + stock_id + "_" + "BBAND_trajectory_all_train2.csv",  'w', encoding='utf8', newline='') as csvFile:
            writer = csv.writer(csvFile)
            for trade in trading_info:
                writer.writerow(trade)

    elif input_type == 'test':
        with open("./data/Trajectory/Test/" + stock_id + "_" + "BBAND_trajectory_all_test.csv",  'w', encoding='utf8', newline='') as csvFile:
            writer = csv.writer(csvFile)
            for trade in trading_info:
                writer.writerow(trade)

    total_num = len(buy_sell_tuple)
    total_sum = 0
    selected_num = 0
    selected_sum = 0

    profit_num = 0

    median = statistics.median(return_list)

    buy_list = []
    sell_list = []

    for trade in buy_sell_tuple:
        total_sum += trade[2]

        if trade[2] > 0:
            profit_num += 1

        if trade[2] >= median:
            selected_num += 1
            selected_sum += trade[2]
            buy_list.append(trade[0])
            sell_list.append(trade[1])
        else:
            trading_dic[trade[0][0]][0] = 'hold'
            trading_dic[trade[1][0]][0] = 'hold'

    print(median)

    print("【未篩選前】")
    print('賺錢的交易比例：', profit_num/total_num)
    print('平均報酬率：', total_sum/total_num)
    print('交易次數：',  total_num)
    print()


    print("【篩選後：保留一半】")
    print('平均報酬率：', selected_sum/selected_num)
    print('交易次數：',  selected_num)
    print()

    #plot_stock(TaiwanStockPriceDay[34:], buy_list, sell_list)

    #pp.pprint(trading_dic)
    
    if input_type == 'train':
        with open("./data/Trajectory/Train/" + stock_id + "_" + "BBAND_trajectory_50_train2.csv",  'w', encoding='utf8', newline='') as csvFile:
            writer = csv.writer(csvFile)
            for key in trading_dic.keys():
                #print(key)
                #print(trading_dic[key])
                writer.writerow([key, trading_dic[key][0], trading_dic[key][1]])
    elif input_type == 'test':
        with open("./data/Trajectory/Test/" + stock_id + "_" + "BBAND_trajectory_50_test.csv",  'w', encoding='utf8', newline='') as csvFile:
            writer = csv.writer(csvFile)
            for key in trading_dic.keys():
                #print(key)
                #print(trading_dic[key])
                writer.writerow([key, trading_dic[key][0], trading_dic[key][1]])







                



