import csv
import os
import pprint as pp

fee = 0.001425
tax = 0.003

stock_id = "2801"
strategy = 'BBAND'
#i=4


for stock_id in ['2330', '2603', '2002','1301', '2801']:
    for strategy in ['SMA', 'KD', 'BBAND']:
        num = 0
        average = 0
        total = 0
        for i in range(5):
            #print(i)

            folder = "./result/GAIL/Train/10Days/"

            #with open("render.csv", 'r', encoding = 'utf8', newline = '') as csvFile:
            with open(folder + stock_id + "_" + strategy + "_" + str(i) +".csv", 'r', encoding = 'utf8', newline = '') as csvFile:
                reader = csv.reader(csvFile)
                trajectory_list = [r for r in reader]


            MAX_ACCOUNT_BALANCE = 10000

            balance = MAX_ACCOUNT_BALANCE
            net_worth = MAX_ACCOUNT_BALANCE
            stock_num = 0
            stock_value = 0

            buy_sell_tuple = []
            temp = []

            for trajectory in trajectory_list[:]:
                if trajectory[1] == 'buy':
                    stock_num = int(balance / float(trajectory[2]))
                    stock_value = stock_num * float(trajectory[2])
                    balance = balance - stock_value - stock_value * fee
                    #print("Buy at", trajectory[2])
                    #print(balance+stock_value)
                    #print()

                    if len(temp) == 0:
                        temp.append(float(trajectory[2]))
                    else:
                        temp = []
                        temp.append(float(trajectory[2]))

                elif trajectory[1] == 'sell':
                    stock_value = stock_num * float(trajectory[2])
                    balance = balance + stock_value - stock_value * (fee + tax)
                    stock_num = 0
                    stock_value = 0
                    #print("Sell at", trajectory[2])
                    #print(balance+stock_value)
                    #print()
                    if len(temp) == 1:
                        temp.append(float(trajectory[2]))
                        r = (temp[1]-temp[0])/temp[0]
                        temp.append(r)
                        buy_sell_tuple.append(temp)
                        temp = []
                    else:
                        print(temp)

                elif trajectory[1] == 'hold':
                    stock_value = stock_num * float(trajectory[2])
                    #print(balance + stock_value)
                    #print()

            if len(temp) != 0:
                temp.append(float(trajectory[2]))
                r = (temp[1]-temp[0])/temp[0]
                temp.append(r)
                buy_sell_tuple.append(temp)
                temp = []

            total_num = len(buy_sell_tuple)
            total_sum = 0
            for t in buy_sell_tuple:
                total_sum += t[2] 

            #pp.pprint(buy_sell_tuple)
            #print(stock_id, strategy)
            num = num + total_num
            average = average  + total_sum/total_num
            total = total + balance+stock_value
            #print('平均報酬率：', total_sum/total_num)
            #print('交易次數：',  total_num)
            #print('最終收益', balance+stock_value)


            #os.remove("render.csv")
        print(stock_id, strategy)
        print(num/5)
        print(average/5)
        print(total/5)

        a = total/5

        print(pow(1+(a-10000)/10000, 0.05)-1)
        print()
