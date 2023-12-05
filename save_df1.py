import pandas as pd
from FinMind.data import DataLoader

api_token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJkYXRlIjoiMjAyMS0xMi0yNyAxNDo1OTowOSIsInVzZXJfaWQiOiJkdXJhbnQ3MTA5MTYiLCJpcCI6IjE0MC4xMjAuMTMuMjMwIn0.8-KIC3-OA4D6JcOtQ_fJBOVkyugx60t1Gy82c57TLz4"

api = DataLoader()
api.login_by_token(api_token = api_token)

# 設定股票標的和開始/結束日期

stock_list = ['2330', '2603', '2002','1301', '2801']

stock_id = '2002'
strategy = "SMA"
#start_date='2013-01-01'

#start_date='2001-01-01'
#end_date='2020-12-31'
start_date='2021-01-01'
end_date='2021-12-31'


for stock_id in stock_list:
    
    df = api.taiwan_stock_daily(
        stock_id = stock_id,
        start_date = start_date,
        end_date = end_date
    )
    
    df.to_csv("data/" + stock_id + "_test.csv")