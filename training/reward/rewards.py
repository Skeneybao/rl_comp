from typing import Dict

import numpy as np

from training.model_io.output_wrapper import ActionType
from training.reward.get_reward import register_reward
from training.util.tools import get_price_avg, log1p_abs


@register_reward('normalized_net_return')
def normalized_net_return(steps_done: int, obs_before: Dict, obs_after: Dict, action: ActionType, *args, **kwargs) -> float:
    """
    calculate reward by normalized net return

    SPECIAL NOTE on obs_before and obs_after:
    - obs_before contains the observation before the action is taken, while obs_after contains the observation after the
      action is taken
    - DIFFERENTLY, obs_after contains the information after the action is taken, and is useful to calculate the reward.
      However, obs_before contains the information after the last action is taken.


    columns that are in obs include 10 stages of prices and volumes, which are:
    - av0-4: 五档卖量
    - ap0-4: 五档卖价
    - bv0-4: 五档买量
    - bp0-4: 五档买价

    columns that are in info include the following, from the doc:
    - code_pnl: 某支股票每个 step 后的收益，初始为 0。
                'code_pnl' = 'code_cash_pnl' + 'code_positional_pnl' - 'code_handling_fee'
    - code_cash_pnl: 某支股票每个 step 后的手持现金，初始为 0，可以为负。
    - code_positional_pnl: 某支股票每个 step 后的仓位估值。
                           'code_positional_pnl' = (askPx1 + bidPx1) / 2 * 'code_net_position' * 10
    - code_handling_fee: 某支股票每个 step 后的累计交易手续费。所有股票每笔交易的 收费标准统一为每笔交易金额的万分之 0.7。
    - day_pnl: 某天所有已经交易完毕的股票的累计收益。
    - day_handling_fee: 某天所有已经交易完毕的股票的累计交易手续费。
    - code_net_position: 某支股票每个 step 后的仓位。
    - ap0_t0: 某支股票当天第一个 askPx1。

    :param steps_done:
    :param obs_before:
    :param obs_after:
    :param action:
    :return:
    """
    return (obs_after['code_pnl'] - obs_before['code_pnl']) / obs_after['ap0_t0']


@register_reward('scaled_net_return')
def scaled_net_return(steps_done: int, obs_before: Dict, obs_after: Dict, action: ActionType, scale=35e-4, *args, **kwargs):
    
    side, vol, price = action

    add_on_handling_fee = (obs_after['day_handling_fee'] - obs_before['day_handling_fee']) / obs_after['ap0_t0'] / 10
    before_mid_price = (obs_before['ap0'] + obs_before['bp0']) / 2
    after_mid_price = (obs_after['ap0'] + obs_after['bp0']) / 2
    net_pos = obs_after['code_net_position']
    
    # assume only take level1
    take_fee = abs(vol) * (obs_before['ap0'] - obs_before['bp0']) / 2 / obs_before['ap0_t0']

    interval_return = (after_mid_price - before_mid_price) / obs_after['ap0_t0']
    scaled_return = np.arctan(interval_return / scale) * scale
    
    return net_pos * scaled_return - add_on_handling_fee - take_fee


@register_reward('single600T')
def single_trade_return600TWAP(steps_done: int, obs_before: Dict, obs_after: Dict, action: ActionType, TWAP600: float, *args, **kwargs) -> float:
    
    side, vol, price = action
    if side == 1:
        return 0
    elif side == 0:
        Earn600 = (TWAP600 - obs_before['ap0']) / obs_before['ap0']
    elif side == 2:
        Earn600 = (obs_before['bp0'] - TWAP600) / obs_before['bp0']
    return Earn600


@register_reward('single600T_Mod')
def single_trade_return600TWAP(steps_done: int, obs_before: Dict, obs_after: Dict, action: ActionType, TWAP600: float, avg_spread: float, *args, **kwargs) -> float:
    
    net_pos = obs_before['code_net_position'] 
    before_mid_price = (obs_before['ap0'] + obs_before['bp0']) / 2
    side, vol, price = action
    pred_cleareance_cost = avg_spread / 2

    if side == 1:
        return 0
    elif side == 0:
        if net_pos >= 0:
            Earn600 = (TWAP600 - obs_before['ap0']) / obs_before['ap0_t0'] - pred_cleareance_cost
        else:
            Earn600 = (TWAP600 - before_mid_price) / obs_before['ap0_t0']
    elif side == 2:
        if net_pos <= 0:
            Earn600 = (obs_before['bp0'] - TWAP600) / obs_before['ap0_t0'] - pred_cleareance_cost
        else:
            Earn600 = (before_mid_price - TWAP600) / obs_before['ap0_t0']

    return Earn600


@register_reward('single600T_scaled')
def single_trade_return600TWAP_scaled(steps_done: int, obs_before: Dict, obs_after: Dict, action: ActionType, TWAP600: float, scale=100e-4, *args, **kwargs) -> float:
    
    side, vol, price = action
    if side == 1:
        return 0
    elif side == 0:
        Earn600 = (TWAP600 - obs_before['ap0']) / obs_before['ap0']
    elif side == 2:
        Earn600 = (obs_before['bp0'] - TWAP600) / obs_before['bp0']

    Earn600Scaled = np.arctan(Earn600 / scale) * scale
    return Earn600Scaled


@register_reward('short_sight_return')
def short_sight_return(steps_done: int, obs_before: Dict, obs_after: Dict, action: ActionType, *args, **kwargs) -> float:
    # Here we assume 'vol' can be traded given the 'price'
    side, vol, price = action

    before_mid_price = (obs_before['ap0'] + obs_before['bp0']) / 2
    after_mid_price = (obs_after['ap0'] + obs_after['bp0']) / 2
    # noop
    if side == 1:
        return 0
    trade_price_avg, _ = get_price_avg(obs_before, vol * (1 if side == 0 else -1))
    try:
        if side == 0:
            return vol * ((after_mid_price - trade_price_avg) / trade_price_avg - 0.0007)
        # sell
        elif side == 2:
            return vol * ((trade_price_avg - after_mid_price) / trade_price_avg - 0.0007)

        else:
            raise ValueError("Unknown trading side")
    except ZeroDivisionError as e:
        raise ValueError(f"Feature Error|short_sight_return|{trade_price_avg}|{after_mid_price}|{action}|{obs_before}") from e


@register_reward('long_short_sight_return')
def long_short_sight_return(steps_done: int, obs_before: Dict, obs_after: Dict, action: ActionType, divd=10, *args, **kwargs) -> float:
    long_sight_reward = normalized_net_return(steps_done, obs_before, obs_after, action)
    short_sight_reward = short_sight_return(steps_done, obs_before, obs_after, action)

    return long_sight_reward / divd + short_sight_reward


@register_reward('log_long_short_sight_return')
def log_long_short_sight_return(steps_done: int, obs_before: Dict, obs_after: Dict, action: ActionType, *args, **kwargs) -> float:
    long_sight_reward = normalized_net_return(steps_done, obs_before, obs_after, action)
    short_sight_reward = short_sight_return(steps_done, obs_before, obs_after, action)

    return log1p_abs(long_sight_reward) + log1p_abs(short_sight_reward)


