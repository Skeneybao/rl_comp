import unittest

from training.util.validate_action import validate_action


class UtilTestCase(unittest.TestCase):
    def test_validate_action(self):
        obs = {
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
        }

        test_data = [
            [2, 1, 4583.164],
            [0, 1, 4583.164],
            [2, 1000, 4583.164],
            [0, 1000, 4614],
        ]

        expected_data = [
            [2, 1, 4583.164],
            [1, 0, 0],
            [2, 42, 4583.164],
            [0, 25, 4614],
        ]

        for i, ((side, vol, price), (exp_side, exp_vol, exp_price)) in enumerate(zip(test_data, expected_data)):
            (side, vol, price), _ = validate_action(obs, (side, vol, price))
            fail_mesg = f'\nFailed on {i}, expected {exp_side, exp_vol, exp_price}, got {side, vol, price}'
            self.assertEqual(side, exp_side, msg=fail_mesg)
            self.assertEqual(vol, exp_vol, msg=fail_mesg)
            self.assertEqual(price, exp_price, msg=fail_mesg)


if __name__ == '__main__':
    unittest.main()
