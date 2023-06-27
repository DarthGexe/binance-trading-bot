import numpy as np

# Helper functions
def get_balance(balances, asset):
    for bal in balances:
        if bal.get('asset') == asset:
            return float(bal['free'])

def get_max_float_qty(step_size):
    max_float_qty = 0
    a = 10
    while step_size * a < 1:
      a = a*10**max_float_qty
      max_float_qty += 1
    return max_float_qty

def get_quantity(price, amount, min_qty, max_qty, max_float_qty):
    quantity = amount / price
    if (quantity < min_qty or quantity > max_qty):
        return False
    quantity = np.round(quantity, max_float_qty)
    return quantity

def crossover(ma_fast, ma_slow):
    if (ma_fast[0] < ma_slow[0] and ma_fast[1] >= ma_slow[1]):
        return True
    return False
