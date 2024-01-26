from typing import Dict, List

import numpy as np


def get_rank(data: List, target):
    rank = 0
    for val in data:
        if val < target:
            rank += 1

    return rank


def std(data):
    if len(data) < 2:
        raise ValueError("Standard deviation requires at least two data points")

    mean = sum(data) / len(data)
    variance = sum((x - mean) ** 2 for x in data) / (len(data) - 1)
    return variance ** 0.5


def get_price_avg(observation: Dict, vol_to_trade: float):
    """
    vol_to_trade > 0 : BUY
    vol_to_trade < 0 : SELL
    """

    abs_vol = abs(vol_to_trade)

    if abs_vol > 10 or vol_to_trade == 0:
        raise ValueError(f"avg_price_to_trade|{vol_to_trade}")

    ask_price_levels = [
        observation['ap0'],
        observation['ap1'],
        observation['ap2'],
        observation['ap3'],
        observation['ap4'],
    ]
    ask_vol_levels = [
        observation['av0'],
        observation['av1'],
        observation['av2'],
        observation['av3'],
        observation['av4'],
    ]
    bid_price_levels = [
        observation['bp0'],
        observation['bp1'],
        observation['bp2'],
        observation['bp3'],
        observation['bp4'],
    ]
    bid_vol_levels = [
        observation['bv0'],
        observation['bv1'],
        observation['bv2'],
        observation['bv3'],
        observation['bv4'],
    ]

    total_cost = 0
    margin_cost = 0
    if vol_to_trade > 0:
        if sum(ask_vol_levels) < 1e-5:
            return 0
        for ap, av in zip(ask_price_levels, ask_vol_levels):
            if vol_to_trade > av:
                vol_to_trade -= av
                total_cost += av * ap
            else:
                total_cost += vol_to_trade * ap
                margin_cost = ap
                break
        else:
            raise ValueError(f"Invalid vol_to_trade|{observation}|{vol_to_trade}")
        

    elif vol_to_trade < 0:
        vol_to_trade = -vol_to_trade
        if sum(bid_vol_levels) < 1e-5:
            return 0
        
        for bp, bv in zip(bid_price_levels, bid_vol_levels):
            if vol_to_trade > bv:
                vol_to_trade -= bv
                total_cost += bv * bp
            else:
                total_cost += vol_to_trade * bp
                margin_cost = bp
                break
        else:
            raise ValueError(f"Invalid vol_to_trade|{observation}|{vol_to_trade}")
        
    return total_cost / abs_vol, margin_cost


def log1p_abs(x):
    if x >= 0:
        return np.log1p(x)
    else:
        return -np.log1p(-x)
