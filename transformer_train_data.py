import pandas as pd
import numpy as np
import math
import csv

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

def stock_data(input_type,stock_id):
    if input_type == "train":
        #start_date = '2000-11-17'
        start_date = '2003-11-17'
        end_date = '2020-12-31'
        #start_bound = '2000-12-29'
        start_bound = '2003-12-31'
        start_index = 24

    elif input_type == "test":
        start_date = '2020-11-14'
        end_date = '2021-12-31'
        #start_bound = '2020-12-30'
        start_bound = '2020-12-31'
        start_index = 25
        
    # 股價日成交資訊
    TaiwanStockPriceDay = api.taiwan_stock_daily(
        stock_id = stock_id,
        start_date = start_date,
        end_date = end_date
    )



    # 融資融券表
    TaiwanStockMarginPurchaseShortSale = api.taiwan_stock_margin_purchase_short_sale(
        stock_id = stock_id,
        start_date = start_date,
        end_date = end_date
    )
    
    TaiwanStockPriceDay['ratio'] = TaiwanStockMarginPurchaseShortSale['ShortSaleTodayBalance'] / TaiwanStockMarginPurchaseShortSale['MarginPurchaseTodayBalance']
    

    # 法人買賣表
    TaiwanStockInstitutionalInvestorsBuySell = api.taiwan_stock_institutional_investors(
        stock_id = stock_id,
        start_date = start_date,
        end_date = end_date
    )

    Foreign_Investor = TaiwanStockInstitutionalInvestorsBuySell.loc[TaiwanStockInstitutionalInvestorsBuySell['name'] == 'Foreign_Investor']
    Investment_Trust = TaiwanStockInstitutionalInvestorsBuySell.loc[TaiwanStockInstitutionalInvestorsBuySell['name'] == 'Investment_Trust']

    # 加權指數
    TaiwanStockTotalReturnIndex = api.taiwan_stock_total_return_index(
        index_id="TAIEX",
        start_date = start_date,
        end_date = end_date
    )

    TaiwanStockTotalReturnIndex


    dateArray = TaiwanStockPriceDay['date']

    
    buysell_list = []

    for date in dateArray:
        Foreign_Investor = TaiwanStockInstitutionalInvestorsBuySell.loc[TaiwanStockInstitutionalInvestorsBuySell['date'] == date]
        if sum(Foreign_Investor['buy']) >= sum(Foreign_Investor['sell']):
            buysell_list.append(1)
        else:
            buysell_list.append(0)

    #piece['OverBoughtSold'] = np.array(buysell_list)[14:34]
    
    """
    for d in set(Foreign_Investor['date']) - set(dateArray):
        Foreign_Investor = Foreign_Investor.drop(Foreign_Investor[Foreign_Investor['date'] == d].index)
        Investment_Trust = Investment_Trust.drop(Investment_Trust[Investment_Trust['date'] == d].index)

    Foreign_Investor = Foreign_Investor.reset_index()
    Investment_Trust = Investment_Trust.reset_index()

    def get_max(buy, sell):
        if buy > sell:
            return buy 
        else:
            return sell

    ForeignInvestorMax = get_max(Foreign_Investor['buy'].max(), Foreign_Investor['sell'].max())
    InvestmentTrustMax = get_max(Investment_Trust['buy'].max(), Investment_Trust['sell'].max())

    Foreign_Investor['buy'] = Foreign_Investor['buy']/ForeignInvestorMax
    Foreign_Investor['sell'] = Foreign_Investor['sell']/ForeignInvestorMax
    Investment_Trust['buy'] = Investment_Trust['buy']/InvestmentTrustMax
    Investment_Trust['sell'] = Investment_Trust['sell']/InvestmentTrustMax
    """

    total_data = []
    next_day_price = []
    #print(TaiwanStockPriceDay[:36])
    #for i in range(len(dateArray[dateArray > '2012-12-27'])):
    #for i in range(len(dateArray[dateArray > '2020-12-30'])):
    #print(start_index)
    for i in range(len(dateArray[dateArray > start_bound])-1):
        print('============',TaiwanStockPriceDay.iloc[i+start_index+9][['date']],'====================')
        #print(i)

        piece = TaiwanStockPriceDay.iloc[i+start_index:i+start_index+10][['open', 'max', 'min', 'close','Trading_Volume']]
        piece2 = TaiwanStockPriceDay.iloc[i+start_index:i+start_index+11][['open', 'max', 'min', 'close','Trading_Volume']]
        #print(piece)
        max_value = 0.8
        min_value = 0.2

        Max_Volume = piece2['Trading_Volume'].max()
        Min_Volume = piece2['Trading_Volume'].min()
        Max_Price = piece2['max'].max()
        Min_Price = piece2['min'].min()

        piece['max'] = min_value + (max_value - min_value) * (piece['max'] - Min_Price) / (Max_Price - Min_Price)
        piece['min'] = min_value + (max_value - min_value) * (piece['min'] - Min_Price) / (Max_Price - Min_Price)
        piece['open'] = min_value + (max_value - min_value) * (piece['open'] - Min_Price) / (Max_Price - Min_Price)
        piece['close'] = min_value + (max_value - min_value) * (piece['close'] - Min_Price) / (Max_Price - Min_Price)
        piece['Trading_Volume'] = min_value + (max_value - min_value) * (piece['Trading_Volume'] - Min_Volume) / (Max_Volume - Min_Volume)

        piece['RSI'] = talib.RSI(TaiwanStockPriceDay.iloc[i:i+start_index+11]['close'], timeperiod=14)[start_index:]/100
        piece['EMA'] = talib.EMA(TaiwanStockPriceDay.iloc[i:i+start_index+11]['close'], timeperiod=10)[start_index:]
        piece['EMA'] = min_value + (max_value - min_value) * (piece['EMA'] - Min_Price) / (Max_Price - Min_Price)

        piece['OBV'] = talib.OBV(TaiwanStockPriceDay.iloc[i+start_index:i+start_index+11]['close'], TaiwanStockPriceDay.iloc[i+start_index:i+start_index+11]['Trading_Volume'])
        Max_OBV = piece['OBV'].max()
        Min_OBV = piece['OBV'].min()
        piece['OBV'] = min_value + (max_value - min_value) * (piece['OBV'] - Min_OBV) / (Max_OBV - Min_OBV)

        Max_TAIEX = TaiwanStockTotalReturnIndex[i+start_index:i+start_index+11]['price'].max()
        Min_TAIEX = TaiwanStockTotalReturnIndex[i+start_index:i+start_index+11]['price'].min()
        piece['TAIEX'] = min_value + (max_value - min_value) * (TaiwanStockTotalReturnIndex[i+start_index:i+start_index+11]['price'] - Min_TAIEX) / (Max_TAIEX - Min_TAIEX)

        #print(piece)     
        #piece['RGZRATIO'] = TaiwanStockPriceDay['ratio'][i+start_index:i+34]
        #piece['OverBoughtSold'] = np.array(buysell_list)[i+start_index:i+34]
        piece = piece.fillna(0)
        #piece['Next_day_closing_price'] = piece['close'].shift(-1).dropna()
        #print(piece)

        #test = piece.values.flatten()
        #print(test)
        total_data.append(piece.values)
        
        
        '''next day closing price'''
        n_price = TaiwanStockPriceDay.at[i+start_index+10,'close']
        #print(n_price)
        n_price = min_value + (max_value - min_value) * (n_price - Min_Price) / (Max_Price - Min_Price)
        next_day_price.append([n_price])
        
    total_data = np.array(total_data)
    next_day_price = np.array(next_day_price)
    return total_data, next_day_price