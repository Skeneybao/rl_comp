import unittest

from training.DQN.model import Action11OutputWrapper
from training.env.featureEngine import FeatureEngineDummy
from training.model.DNN import DNNModelConfig, DNN


class ModelOutputWrapperTest(unittest.TestCase):
    obs = {'serverTime': 140806107.0,
           'eventTime': 140806190.0,
           'code': 1.0,
           'signal0': 0.46240370559544686,
           'signal1': 0.5974561054535862,
           'signal2': 0.44555716043968174,
           'ap0': 2568.042,
           'bp0': 2567.973,
           'av0': 16.0,
           'bv0': 47.0,
           'ap1': 2568.134,
           'bp1': 2567.9500000000003,
           'av1': 5.0,
           'bv1': 199.0,
           'ap2': 2568.295,
           'bp2': 2567.812,
           'av2': 10.0,
           'bv2': 1.0,
           'ap3': 2568.9159999999997,
           'bp3': 2567.03,
           'av3': 9.0,
           'bv3': 3.0,
           'ap4': 2569.33,
           'bp4': 2566.823,
           'av4': 40.0,
           'bv4': 5.0,
           'code_net_position': 300,
           'ap0_t0': 2566.34,
           'code_pnl': -64936.27882606555,
           'code_cash_pnl': -7742806.249999964,
           'code_positional_pnl': 7678342.424999999,
           'code_handling_fee': 26154.671455500677,
           'day_pnl': 0.0,
           'day_handling_fee': 0.0}

    def test_inference(self):
        feature_engine = FeatureEngineDummy()
        model = DNN(
            DNNModelConfig(feature_engine.get_input_shape(), [64], Action11OutputWrapper.get_output_shape()))
        model_output_wrapper = Action11OutputWrapper(model)
        for i in range(100):
            action, model_input, model_output = model_output_wrapper.select_action(
                self.obs, feature_engine.get_feature(self.obs))
            self.assertEqual(len(action), 3)
            self.assertIn(action[0], [0, 1, 2])
            action_id = model_output.argmax(-1).item()
            price_key = ['ap4', 'ap3', 'ap2', 'ap1', 'ap0', 'bp0', 'bp1', 'bp2', 'bp3', 'bp4', 'noop'][action_id]
            if price_key == 'noop':
                self.assertEqual(action[0], 1)
                self.assertEqual(action[1], 0)
                self.assertEqual(action[2], 0)
            else:
                if price_key.startswith('a'):
                    self.assertEqual(action[0], 0)
                else:
                    self.assertEqual(action[0], 2)
                price = self.obs[f'{price_key}']
                self.assertEqual(action[1], 1)
                self.assertEqual(action[2], price)

    def test_random_action(self):
        feature_engine = FeatureEngineDummy()
        model = DNN(
            DNNModelConfig(feature_engine.get_input_shape(), [64], Action11OutputWrapper.get_output_shape()))
        model_output_wrapper = Action11OutputWrapper(model)
        actions = []
        iters = 10000
        for i in range(iters):
            feature_engine = FeatureEngineDummy()
            action, model_input, model_output = model_output_wrapper.random_action(
                self.obs, feature_engine.get_feature(self.obs))
            self.assertEqual(model_output.size()[0], 11)
            self.assertEqual(len(action), 3)
            self.assertIn(action[0], [0, 1, 2])
            actions.append(action)
        for action in ['ap0', 'ap1', 'ap2', 'ap3', 'ap4', 'bp0', 'bp1', 'bp2', 'bp3', 'bp4']:
            price = self.obs[action]
            percent = len([action for action in actions if action[2] == price]) / iters
            self.assertAlmostEqual(percent, 1 / 11, delta=0.01, msg=f'random action {action} percent error')


if __name__ == '__main__':
    unittest.main()
