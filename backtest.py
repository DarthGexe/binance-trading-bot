import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from binance.client import Client
import binance_utils as utl
from config import *

client = Client(API_KEY, API_SECRET)
crypto = 'BTC'
ref = 'USDT'
symbol = f'{crypto}{ref}'
# moving average periods
ema_f = 5
ema_s = 15
sma = 200
# operation size ref
amount = 15
# % on take profit and stop loss
take_profit = 2 /100
stop_loss = 6 /100
# initial balances
initial_balance_crypto = 100
initial_balance_ref = 100
# historic data params
kline_interval = Client.KLINE_INTERVAL_15MINUTE
start = "5 Jul, 2022"
finish = "5 Jul, 2023"

def initialize_dataframe():
    candles = client.get_historical_klines(symbol, kline_interval, start, finish)
    df = pd.DataFrame(candles)
    df = df.drop([6, 7, 8, 9, 10, 11], axis=1)
    df.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
    df[['time', 'open', 'high', 'low', 'close', 'volume']] = df[['time', 'open', 'high', 'low', 'close', 'volume']].astype(float)
    df['time'] = pd.to_datetime(df['time'] * 1000000)
    return df

def parser():
    df = initialize_dataframe()
    df = pd.read_csv(f'./{symbol}_15M.csv')
    df['ema_s'] = df['close'].ewm(span=ema_s).mean()
    df['ema_f'] = df['close'].ewm(span=ema_f).mean()
    df['sma'] = df['close'].rolling(window=sma).mean()
    df['trend'] = np.nan
    df['operation'] = np.nan
    df['balance_crypto']= np.nan
    df['balance_ref']= np.nan
    df['profit']= np.nan
    return df

def backtest(df):
    balance_ref = initial_balance_ref
    balance_crypto = initial_balance_crypto
    opened_order = False
    i = sma
    while i < len(df):
        operation = None
        price = df['close'].iloc[i-1]
        # get trend, price 0.5% up or down SMA 
        if df['close'].iloc[i-1] > df['sma'].iloc[i-1]*1.005:
            df['trend'].iloc[i] = 'up'
        elif df['close'].iloc[i-1] < df['sma'].iloc[i-1]*0.995:
            df['trend'].iloc[i] = 'down'
        else:
            df['trend'].iloc[i] = None

        if opened_order is False:
            if utl.crossover((df['ema_f'].iloc[i-1], df['ema_f'].iloc[i]),(df['ema_s'].iloc[i-1], df['ema_s'].iloc[i])) and df['trend'].iloc[i] == 'up':
                quantity = utl.get_quantity(price, amount, min_qty, max_qty, max_float_qty)
                if  balance_ref >= amount:
                    operation = 'BUY'
                    balance_ref = balance_ref - (quantity * price)
                    balance_crypto = balance_crypto + quantity
                    opened_order = True
                    order_type = 'SELL'
                    sell_price = ((price *(1+take_profit)) // tick_size) * tick_size
                    stop_price = ((price*(1-stop_loss)) // tick_size) * tick_size
                    stop_limit_price = ((price*(1-stop_loss)) // tick_size) * tick_size

            elif utl.crossover((df['ema_s'].iloc[i-1], df['ema_s'].iloc[i]),(df['ema_f'].iloc[i-1], df['ema_f'].iloc[i])) and df['trend'].iloc[i] == 'down':
                quantity = utl.get_quantity(price, amount, min_qty, max_qty, max_float_qty)
                if quantity <= balance_crypto:
                    operation = 'SELL'
                    balance_crypto = balance_crypto - quantity
                    balance_ref = balance_ref + (quantity * price)
                    opened_order = True
                    order_type = 'BUY'
                    buy_price = ((price*(1-take_profit)) // tick_size) * tick_size
                    stop_price = ((price*(1+stop_loss)) // tick_size) * tick_size
                    stop_limit_price = ((price*(1+stop_loss)) // tick_size) * tick_size
                    quantity = utl.get_quantity(buy_price, amount, min_qty, max_qty, max_float_qty)

        elif opened_order:
            if order_type == 'SELL':
                if price >= sell_price:
                    balance_ref = balance_ref + (quantity * sell_price)
                    balance_crypto = balance_crypto - quantity
                    operation = 'SELL'
                    opened_order = False
                elif price <= stop_price:
                    balance_ref = balance_ref + (quantity * stop_limit_price)
                    balance_crypto = balance_crypto - quantity
                    operation = 'SELL'
                    opened_order = False
            elif order_type == 'BUY':
                if price <= buy_price:
                    balance_crypto = balance_crypto + quantity
                    balance_ref = balance_ref - (quantity * buy_price)
                    operation = 'BUY'
                    opened_order = False
                elif price >= stop_price:
                    balance_crypto = balance_crypto + quantity
                    balance_ref = balance_ref - (quantity * stop_limit_price)
                    operation = 'BUY'
                    opened_order = False

        df['operation'].iloc[i]= operation
        df['balance_crypto'].iloc[i]= balance_crypto
        df['balance_ref'].iloc[i]= balance_ref
        profit_crypto = round(balance_crypto - initial_balance_crypto,8)
        profit_ref = round(balance_ref - initial_balance_ref,2)
        total_profit = round(profit_ref + (profit_crypto * price),2)
        df['profit'].iloc[i]= total_profit
        i+=1

    df.to_csv(f'{symbol}_MF:{ema_f}_MS:{ema_s}_MA:{sma}_TP:{int(take_profit*100)}_SL:{int(stop_loss*100)}.csv')
  
    print("Backtesting Results:")
    print(f'EMAF: {ema_f} EMAS: {ema_s} SMA: {sma} TP: {int(take_profit*100)} SL: {int(stop_loss*100)}\n')
    print(f"Profit {crypto}: {round(profit_crypto, 8)} = {round(profit_crypto*price,2)} {ref}")
    print(f"Profit {ref}: {round(profit_ref,2)}")
    print(f"Total Profit : {total_profit} {ref}")

    df.plot(x="time", y="profit", figsize=(10,5), legend=False)
    plt.title(f'{symbol} EMAF:{ema_f} EMAS:{ema_s} SMA:{sma} TP:{int(take_profit*100)}% SL:{int(stop_loss *100)}%')
    plt.xlabel('Time')
    plt.ylabel('Profit')
    plt.xticks(rotation=45)
    plt.show()

if __name__ == "__main__":
    pd.options.mode.chained_assignment = None
    info = client.get_symbol_info(symbol)
    min_qty = float(info['filters'][1].get('minQty'))
    step_size = float(info['filters'][1].get('stepSize'))
    max_qty = float(info['filters'][1].get('maxQty'))
    max_float_qty = utl.get_max_float_qty(step_size)
    tick_size = float(info['filters'][0].get('tickSize'))

    df = parser()
    backtest(df)
