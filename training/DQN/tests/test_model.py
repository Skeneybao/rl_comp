import unittest

from training.model.Attn import Attn
from training.model.DNN import DNN, DNN_11_output, FullPosDNN
from training.model_io.featureEngine import FeatureEngineDummy, FeatureEngineVersion3_Simple
from training.model_io.output_wrapper import Action11OutputWrapper, Action3OutputWrapper


class ModelOutputWrapperTest(unittest.TestCase):
    obs = {'serverTime': 93001020.0, 'eventTime': 93001070.0, 'code': 511.0, 'signal0': -0.11349530868824634,
           'signal1': 0.0, 'signal2': 0.0, 'ap0': 2988.16, 'bp0': 2987.424, 'av0': 1.0, 'bv0': 76.0, 'ap1': 2988.367,
           'bp1': 2986.987, 'av1': 9.0, 'bv1': 30.0, 'ap2': 2992.047, 'bp2': 2985.8599999999997, 'av2': 2.0,
           'bv2': 12.0, 'ap3': 2992.2769999999996, 'bp3': 2985.791, 'av3': 7.0, 'bv3': 24.0, 'ap4': 2993.3579999999997,
           'bp4': 2984.963, 'av4': 2.0, 'bv4': 2.0, 'code_net_position': 0, 'ap0_t0': 2988.16, 'code_pnl': 0.0,
           'code_cash_pnl': 0.0, 'code_positional_pnl': 0.0, 'code_handling_fee': 0.0, 'day_pnl': 0.0,
           'day_handling_fee': 0.0, 'signal0_rank': 0.5, 'signal1_rank': 0.5, 'signal2_rank': 0.5, 'signal0_mean': 0,
           'signal1_mean': 0, 'signal2_mean': 0, 'mid_price_std': 1, 'warming-up': True, 'full_pos': 0}

    def test_dnn_inference(self):
        feature_engine = FeatureEngineDummy()
        model = DNN(input_dim=feature_engine.get_input_shape(), hidden_dim=[64],
                    output_dim=Action11OutputWrapper.get_output_shape())
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

    def test_output3_inference(self):
        feature_engine = FeatureEngineVersion3_Simple()
        model = DNN(input_dim=feature_engine.get_input_shape(), hidden_dim=[64],
                    output_dim=Action3OutputWrapper.get_output_shape())
        model_output_wrapper = Action3OutputWrapper(model)
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

    def test_output3_full_pos_dnn_inference(self):
        feature_engine = FeatureEngineVersion3_Simple()
        model = FullPosDNN(input_dim=feature_engine.get_input_shape(), hidden_dim=[64],
                           output_dim=Action3OutputWrapper.get_output_shape())
        model_output_wrapper = Action3OutputWrapper(model)

        # original action
        action, model_input, model_output = model_output_wrapper.select_action(
            self.obs, feature_engine.get_feature(self.obs))
        self.assertEqual(len(action), 3)
        self.assertIn(action[0], [0, 1, 2])
        action_id = model_output.argmax(-1).item()
        price_key = ['ap1', 'noop', 'bp1'][action_id]
        if price_key == 'noop':
            self.assertEqual(action[0], 1)
            self.assertEqual(action[1], 0)
            self.assertEqual(action[2], 0)
        else:
            if price_key.startswith('a'):
                self.assertEqual(action[0], 0)
            else:
                self.assertEqual(action[0], 2)

        # full pos action
        obs = self.obs.copy()
        obs['full_pos'] = 1
        action, model_input, model_output = model_output_wrapper.select_action(
            self.obs, feature_engine.get_feature(obs))
        self.assertLess(model_output[0], -1e10)

        # neg full pos action
        obs = self.obs.copy()
        obs['full_pos'] = -1
        action, model_input, model_output = model_output_wrapper.select_action(
            self.obs, feature_engine.get_feature(obs))
        self.assertLess(model_output[2], -1e10)



    def test_dnn_11_inference(self):
        feature_engine = FeatureEngineDummy()

        model = DNN_11_output(input_dim=feature_engine.get_input_shape(),
                              hidden_dim=[32, 32, 32],
                              output_dim=Action11OutputWrapper.get_output_shape())
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

    def test_attn_inference(self):
        feature_engine = FeatureEngineDummy()
        model = Attn(input_dim=feature_engine.get_input_shape(),
                     avg_price_dim=10,
                     hidden_dim=[32, 32],
                     output_dim=Action11OutputWrapper.get_output_shape())
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

    def test_random_action(self):
        feature_engine = FeatureEngineDummy()
        model = DNN(input_dim=feature_engine.get_input_shape(), hidden_dim=[64],
                    output_dim=Action11OutputWrapper.get_output_shape())
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
            percent = len([action for action in actions if action[0] == 1]) / iters
            self.assertAlmostEqual(percent, 1 / 11, delta=0.01, msg=f'random action {action} percent error')

    # def test_action_id_to_action(self):
    #     feature_engine = FeatureEngineDummy()
    #     model = DNN(feature_engine.get_input_shape(), [64], Action11OutputWrapper.get_output_shape())
    #     model_output_wrapper = Action11OutputWrapper(model)
    #
    #     action_ids = list(range(11))
    #     exp_action = [
    #         (0, 1., 2568.042),
    #         (0, 2., 2568.134),
    #         (0, 3., 2568.134),
    #         (0, 4., 2568.134),
    #         (0, 5., 2568.134),
    #         (2, 1., 2567.973),
    #         (2, 2., 2567.973),
    #         (2, 3., 2567.9500000000003),
    #         (2, 4., 2567.9500000000003),
    #         (2, 5., 2567.812),
    #         (1, 0., 0.)
    #     ]
    #     for action_id, exp in zip(action_ids, exp_action):
    #         self.assertEqual(
    #             model_output_wrapper.action_id_to_action(action_id, self.obs),
    #             exp,
    #             f'action_id={action_id}'
    #         )


if __name__ == '__main__':
    unittest.main()
