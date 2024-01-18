from typing import Dict
import math

from training.model_io.output_wrapper import ActionType


def cal_reward(steps_done: int, obs_before: Dict, obs_after: Dict, action: ActionType) -> float:
    """
    return empty reward

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
    return 0
#     market_volatility = (obs_after['ap0']+obs_after['ap1']+obs_after['ap2']+obs_after['ap3']+obs_after['ap4'] - obs_before['ap0']-obs_before['ap1']-obs_before['ap2']-obs_before['ap3']-obs_before['ap4']) +
#     (obs_after['bp0']+obs_after['bp1']+obs_after['bp2']+obs_after['bp3']+obs_after['bp4'] - obs_before['bp0']-obs_before['bp1']-obs_before['bp2']-obs_before['bp3']-obs_before['bp4'])
#     valid_action, is_invalid = validate_action(obs_after: Dict, action: ActionType)
#     if is_invalid==True:
#         reward=-1
#     reward=obs_after['code_pnl']/(obs_after['code_positional_pnl']-obs_before['code_positional_pnl']+0.1*market_volatility)
#     return reward
#
#
# def cal_reward(steps_done: int, obs_before: Dict, obs_after: Dict, action: ActionType) -> float:
#     """
#     :param steps_done:
#     :param obs_before:
#     :param obs_after:
#     :param action:
#     :return:
#     """
#     return obs_after['code_pnl']/(obs_after['code_positional_pnl']-obs_before['code_positional_pnl'])
#
# def cal_reward(steps_done: int, obs_before: Dict, obs_after: Dict, action: ActionType) -> float:
#     """
#     :param steps_done:
#     :param obs_before:
#     :param obs_after:
#     :param action:
#     :return:
#     """
#     reward_log = math.log(obs_after['code_pnl'] / (obs_after['code_positional_pnl'] - obs_before['code_positional_pnl']))
#     return reward_log

# def cal_reward(steps_done: int, obs_before: Dict, obs_after: Dict, action: ActionType) -> float:
#     """
#     :param steps_done:
#     :param obs_before:
#     :param obs_after:
#     :param action:
#     :return:
#     """
#     valid_action, is_invalid = validate_action(obs_after: Dict, action: ActionType)
#     if is_invalid==True:
#         reward=-1
#     reward_log = math.log(obs_after['code_pnl'] / (obs_after['code_positional_pnl'] - obs_before['code_positional_pnl']))
#     return reward_log





