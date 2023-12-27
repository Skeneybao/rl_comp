from typing import List, Dict

from training.DQN.model import ActionType


def validate_action(obs: Dict, action: ActionType):
    side, vol, price = action

    # Extract data from the state
    ask_prices = [obs['ap0'], obs['ap1'], obs['ap2'],
                  obs['ap3'], obs['ap4']]
    bid_prices = [obs['bp0'], obs['bp1'], obs['bp2'],
                  obs['bp3'], obs['bp4']]
    ask_volumes = [obs['av0'], obs['av1'], obs['av2'],
                   obs['av3'], obs['av4']]
    bid_volumes = [obs['bv0'], obs['bv1'], obs['bv2'],
                   obs['bv3'], obs['bv4']]
    code_net_position = obs['code_net_position']

    # Constants
    MAX_POSITION = 300
    MIN_POSITION = -300

    # Initialize variables
    is_invalid = False

    # Ensure volume is positive
    if vol < 0:
        vol = 0
        is_invalid = True

    # Handle different sides
    if side == 0:  # Buy
        # Check for max position limit
        if code_net_position + vol > MAX_POSITION:
            vol = MAX_POSITION - code_net_position
            is_invalid = True

        # Adjust price and volume based on ask prices and volumes
        for i in range(len(ask_prices)):
            if price < ask_prices[i] and vol > sum(ask_volumes[:i]):
                vol = sum(ask_volumes[:i])
                is_invalid = True
                break
            elif price == ask_prices[i]:
                if vol > sum(ask_volumes[:i + 1]):
                    vol = sum(ask_volumes[:i + 1])
                    is_invalid = True
                break
        else:
            if price > ask_prices[-1] and vol > sum(ask_volumes):
                vol = sum(ask_volumes)
                is_invalid = True

    elif side == 2:  # Sell
        # Check for min position limit
        if code_net_position - vol < MIN_POSITION:
            vol = code_net_position - MIN_POSITION
            is_invalid = True

        # Adjust price and volume based on bid prices and volumes
        for i in range(len(bid_prices)):
            if price > bid_prices[i] and vol > sum(bid_volumes[:i]):
                vol = sum(bid_volumes[:i])
                is_invalid = True
                break
            elif price == bid_prices[i]:
                if vol > sum(bid_volumes[:i + 1]):
                    vol = sum(bid_volumes[:i + 1])
                    is_invalid = True
                break
        else:
            if price < bid_prices[-1] and vol > sum(bid_volumes):
                vol = sum(bid_volumes)
                is_invalid = True

    if vol == 0:
        side = 1
        price = 0.

    return (side, vol, price), is_invalid
