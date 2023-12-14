import unittest

from training.util.validate_action import validate_action


class UtilTestCase(unittest.TestCase):
    def test_validate_action(self):
        state = [{
            'observation': {
                'serverTime': 93004818.0,
                'eventTime': 93004880.0,
                'code': 2.0,
                'signal0': 0.06040532250349382,
                'signal1': -1.7849400355777338,
                'signal2': -3.4662585069980576,
                'ap0': 4606.900000000001,
                'bp0': 4606.049,
                'av0': 5.0,
                'bv0': 15.0,
                'ap1': 4607.912,
                'bp1': 4605.911,
                'av1': 1.0,
                'bv1': 1.0,
                'ap2': 4609.2,
                'bp2': 4599.816,
                'av2': 10.0,
                'bv2': 5.0,
                'ap3': 4611.5,
                'bp3': 4599.793000000001,
                'av3': 8.0,
                'bv3': 12.0,
                'ap4': 4613.57,
                'bp4': 4599.655000000001,
                'av4': 1.0,
                'bv4': 9.0,
                'code_net_position': 0,
                'ap0_t0': 4599.908
            },
            'new_game': False
        }]

        test_data = [
            [[0, 0, 1], [1], [4583.164]],
            [[1, 0, 0], [1], [4583.164]],
            [[0, 0, 1], [1000], [4583.164]],
            [[1, 0, 0], [1000], [4614]],
        ]

        for side, vol, price in test_data:
            val_side, val_vol, val_price = side, vol, price
            (side, vol, price), _ = validate_action(state, (side, vol, price))
            self.assertEqual(side, val_side)
            self.assertEqual(vol, val_vol)
            self.assertEqual(price, val_price)


if __name__ == '__main__':
    unittest.main()
