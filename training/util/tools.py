from typing import Dict, List


def get_rank(data: List, target):
    
    rank = 0
    for val in data:
        if val < target:
            rank += 1

    return rank 
    

def get_price_avg(observation: Dict, vol_to_trade: int):
    """
    vol_to_trade > 0 : BUY
    vol_to_trade < 0 : SELL
    """

    abs_vol = abs(vol_to_trade)

    if abs_vol > 10 or vol_to_trade == 0:
        raise ValueError(f"Feature Error|avg_price_to_trade|{vol_to_trade}")
    
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
    if vol_to_trade > 0:
        for ap, av in zip(ask_price_levels, ask_vol_levels):
            if vol_to_trade > av:
                vol_to_trade -= av
                total_cost += av * ap
            else:
                total_cost += vol_to_trade * ap
                break

    elif vol_to_trade < 0:
        vol_to_trade = -vol_to_trade
        for bp, bv in zip(bid_price_levels, bid_vol_levels):
            if vol_to_trade > bv:
                vol_to_trade -= bv
                total_cost += bv * bp
            else:
                total_cost += vol_to_trade * bp
                break

    return total_cost / abs_vol

