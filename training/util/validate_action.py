from typing import List, Dict, Tuple


def validate_action(state: List[Dict], action: Tuple[List, List, List]):
    state = state[-1]
    side, vol, price = action
    vol = vol[0]
    price = price[0]

    # Extract data from the state
    ask_prices = [state['observation']['ap0'], state['observation']['ap1'], state['observation']['ap2'],
                  state['observation']['ap3'], state['observation']['ap4']]
    bid_prices = [state['observation']['bp0'], state['observation']['bp1'], state['observation']['bp2'],
                  state['observation']['bp3'], state['observation']['bp4']]
    ask_volumes = [state['observation']['av0'], state['observation']['av1'], state['observation']['av2'],
                   state['observation']['av3'], state['observation']['av4']]
    bid_volumes = [state['observation']['bv0'], state['observation']['bv1'], state['observation']['bv2'],
                   state['observation']['bv3'], state['observation']['bv4']]
    code_net_position = state['observation']['code_net_position']

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
    if side[0] == 1:  # Buy
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

    elif side[2] == 1:  # Sell
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
        side = [0, 1, 0]
        price = 0

    return (side, [vol], [price]), is_invalid
