from typing import Dict

import numpy as np

from training.model_io.output_wrapper import ActionType
from training.reward.get_reward import register_reward
from training.util.tools import get_price_avg


@register_reward('normalized_net_return')
def normalized_net_return(steps_done: int, obs_before: Dict, obs_after: Dict, action: ActionType) -> float:
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


@register_reward('short_sight_return')
def short_sight_return(steps_done: int, obs_before: Dict, obs_after: Dict, action: ActionType) -> float:
    # Here we assume 'vol' can be traded given the 'price'
    side, vol, price = action

    before_mid_price = (obs_before['ap0'] + obs_before['bp0']) / 2
    after_mid_price = (obs_after['ap0'] + obs_after['bp0']) / 2
    # noop
    if side == 1:
        return 0
    trade_price_avg = get_price_avg(obs_before, vol * (1 if side == 0 else -1))
    try:
        if side == 0:
            return (after_mid_price - trade_price_avg) / trade_price_avg
        # sell
        elif side == 2:
            return (trade_price_avg - after_mid_price) / trade_price_avg

        else:
            raise ValueError("Unknown trading side")
    except ZeroDivisionError as e:
        raise ValueError(f"Feature Error|short_sight_return|{trade_price_avg}|{after_mid_price}|{action}|{obs_before}", e)


@register_reward('long_short_sight_return')
def long_short_sight_return(steps_done: int, obs_before: Dict, obs_after: Dict, action: ActionType, divd=10) -> float:
    long_sight_reward = normalized_net_return(steps_done, obs_before, obs_after, action)
    short_sight_reward = short_sight_return(steps_done, obs_before, obs_after, action)

    return long_sight_reward / divd + short_sight_reward


@register_reward('log_long_short_sight_return')
def log_long_short_sight_return(steps_done: int, obs_before: Dict, obs_after: Dict, action: ActionType) -> float:
    long_sight_reward = normalized_net_return(steps_done, obs_before, obs_after, action)
    short_sight_reward = short_sight_return(steps_done, obs_before, obs_after, action)

    return np.log(long_sight_reward + 1) + np.log(short_sight_reward + 1)
