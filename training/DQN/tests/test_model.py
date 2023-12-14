import unittest

from training.DQN.model import ActorConfig, Actor, ModelIOWrapper


class ModelIOWrapperTest(unittest.TestCase):
    state = [{'observation': {'serverTime': 93004818.0,
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
                              'ap0_t0': 4599.908},
              'new_game': False}]

    def test_inference(self):
        for i in range(100):
            model = Actor(ActorConfig(28, 64, 11))
            model_io_wrapper = ModelIOWrapper(model)
            action, model_input, model_output = model_io_wrapper.wrap_inference_single(self.state)
            self.assertEqual(len(action), 3)
            self.assertEqual(len(action[0]), 3)
            self.assertEqual(len(action[1]), 1)
            self.assertEqual(len(action[2]), 1)
            self.assertEqual(action[0][0] + action[0][1] + action[0][2], 1)
            action_id = model_output.argmax(-1).item()
            price_key = ['ap4', 'ap3', 'ap2', 'ap1', 'ap0', 'bp0', 'bp1', 'bp2', 'bp3', 'bp4', 'noop'][action_id]
            if price_key == 'noop':
                self.assertEqual(action[0], [0, 1, 0])
                self.assertEqual(action[1][0], 0)
                self.assertEqual(action[2][0], 0)
            else:
                if price_key.startswith('a'):
                    self.assertEqual(action[0], [1, 0, 0])
                else:
                    self.assertEqual(action[0], [0, 0, 1])
                price = self.state[-1]['observation'][f'{price_key}']
                self.assertEqual(action[1][0], 1)
                self.assertEqual(action[2][0], price)

    def test_random_action(self):
        model = Actor(ActorConfig(28, 64, 11))
        model_io_wrapper = ModelIOWrapper(model)
        actions = []
        iters = 10000
        for i in range(iters):
            action = model_io_wrapper.random_action(self.state)
            self.assertEqual(len(action), 3)
            self.assertEqual(len(action[0]), 3)
            self.assertEqual(len(action[1]), 1)
            self.assertEqual(len(action[2]), 1)
            self.assertEqual(action[0][0] + action[0][1] + action[0][2], 1)
            actions.append(action)
        for action in ['ap0', 'ap1', 'ap2', 'ap3', 'ap4', 'bp0', 'bp1', 'bp2', 'bp3', 'bp4']:
            price = self.state[-1]['observation'][action]
            percent = len([action for action in actions if action[2][0] == price]) / iters
            self.assertAlmostEqual(percent, 1 / 11, delta=0.01, msg=f'random action {action} percent error')


if __name__ == '__main__':
    unittest.main()
