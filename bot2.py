import time
import datetime as dt
import pandas as pd
import numpy as np
import pytz
from binance.client import Client
from binance.enums import *
import binance_utils as ut
from config import *

crypto = 'BTC'
ref = 'USDT'
symbol = crypto + ref
amount = 15

# Eligible periods for candlesticks 1m, 3m, 5m, 15m, 1h, etc.
period = '15m'

# Moving Average Periods
ema_f = 15
ema_s = 50
sma = 200

# Functions
def synchronize():
    candles = client.get_klines(symbol=symbol,interval=period,limit=1)
    timer = pd.to_datetime(float(candles[0][0]) * 1000000)
    start = timer + dt.timedelta(minutes=step)
    print('Synchronizing .....')
    while dt.datetime.now(dt.timezone.utc) < pytz.UTC.localize(start):
        time.sleep(1)
    time.sleep(2)
    print('Bot synchronized')

def initialize_dataframe(limit):
    candles = client.get_klines(symbol=symbol,interval=period,limit=limit)
    df = pd.DataFrame(candles)
    df = df.drop([6, 7, 8, 9, 10, 11], axis=1)
    df.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
    df[['time', 'open', 'high', 'low', 'close', 'volume']] = df[['time', 'open', 'high', 'low', 'close', 'volume']].astype(float)
    df['time'] = pd.to_datetime(df['time'] * 1000000)
    return df

def parser():
    df = initialize_dataframe(sma+1)

    df['ema_s'] = df['close'].ewm(span=ema_s).mean()
    df['ema_f'] = df['close'].ewm(span=ema_f).mean()
    df['sma'] = df['close'].rolling(window=sma).mean()

    df['trend'] = np.nan
    df['operation'] = np.nan

    # get trend
    if df['close'].iloc[-1] > df['sma'].iloc[-1]*1.005:
        trend = 'up'
    elif df['close'].iloc[-1] < df['sma'].iloc[-1]*0.995:
        trend = 'down'
    else:
        trend = None

    df['trend'].iloc[-1]= trend

    return df

def operate(df):
    operation = None
    orders = client.get_open_orders(symbol=symbol)
    if len(orders) == 0:
        price = df['close'].iloc[-1]
        if ut.crossover(df.ema_f.values[-2:], df.ema_s.values[-2:]) and df['trend'].iloc[-1] == 'up':
            operation = 'BUY'

            quantity = ut.get_quantity(price, amount, min_qty, max_qty, max_float_qty)
            balances = client.get_account()['balances']
            balance = ut.get_balance(balances, ref)
            if not quantity:
                print('No Quantity available \n')
            elif balance <= amount:
                print('No {} to buy {}'.format(ref, crypto))
            else:
                order = client.order_limit_buy(
                    symbol=symbol,
                    quantity=quantity,
                    price=price)

                status='NEW'
                while status != 'FILLED':
                    time.sleep(5)
                    order_id = order.get('orderId')
                    order = client.get_order(
                        symbol=symbol,
                        orderId= order_id)
                    status=order.get('status')

                sell_price = np.round(price*1.02, max_float_qty)
                stop_price = np.round(price*0.991, max_float_qty)
                stop_limit_price = np.round(price*0.99, max_float_qty)

                order = client.order_oco_sell(
                    symbol=symbol,
                    quantity=quantity,
                    price=sell_price,
                    stopPrice=stop_price,
                    stopLimitPrice=stop_limit_price,
                    stopLimitTimeInForce='GTC')

        elif ut.crossover(df.ema_s.values[-2:], df.ema_f.values[-2:])and df['trend'].iloc[-1] == 'down':
            operation = 'SELL'

            quantity = ut.get_quantity(price, amount, min_qty, max_qty, max_float_qty)
            balances = client.get_account()['balances']
            balance = ut.get_balance(balances, crypto)
            if not quantity:
                print('No Quantity available \n')
            elif balance < quantity:
                print('No {} to sell for {}'.format(crypto, ref))
            else:
                order = client.order_limit_sell(
                    symbol=symbol,
                    quantity=quantity,
                    price=price)

                status='NEW'
                while status != 'FILLED':
                    time.sleep(3)
                    order_id = order.get('orderId')
                    order = client.get_order(
                        symbol=symbol,
                        orderId= order_id)
                    status=order.get('status')

                buy_price = np.round(price*0.98, max_float_qty)
                stop_price = np.round(price*1.009, max_float_qty)
                stop_limit_price = np.round(price*1.01, max_float_qty)

                order = client.order_oco_buy(
                    symbol=symbol,
                    quantity=quantity,
                    price=buy_price,
                    stopPrice=stop_price,
                    stopLimitPrice=stop_limit_price,
                    stopLimitTimeInForce='GTC')

    return operation

if __name__ == "__main__":
    pd.options.mode.chained_assignment = None

    client = Client(API_KEY, API_SECRET)
    info = client.get_symbol_info(symbol)
    min_qty = float(info['filters'][1].get('minQty'))
    step_size = float(info['filters'][1].get('stepSize'))
    max_qty = float(info['filters'][1].get('maxQty'))
    max_float_qty = ut.get_max_float_qty(step_size)

    df = initialize_dataframe(1)

    step = int(period[:2]) if len(period) > 2 else int(period[0])
    if 'h' in period:
        step = step * 60

    synchronize()

    while True:
        temp = time.time()
        try:
            temp_df = parser()
            i=1

            while df['time'].iloc[-1] != temp_df['time'].iloc[-i] and i < sma:
                i+=1
            df = pd.concat([df, temp_df.tail(i-1)], ignore_index=True)

            df['operation'].iloc[-1] = operate(df.tail(2))
            if df.shape[0] > 10000:
                df = df.tail(10000)
            print(f"{df['time'].iloc[-1]} | Price:{df['close'].iloc[-1]} | Trend:{df['operation'].iloc[-1]} | Operation:{df['operation'].iloc[-1]}")
            df.to_csv(f'./{symbol}_{period}.csv')
        except Exception as e:
            with open('./error.txt', 'a') as file:
                file.write(f'Time = {dt.datetime.now(dt.timezone.utc)}\n')
                file.write(f'Error = {e}\n')
                file.write('----------------------\n')

        delay = time.time() - temp
        idle = 60 * step - delay
        #print(f'idle = {idle} seconds')
        if idle > 0:
            time.sleep(idle)
        else:
            synchronize()
