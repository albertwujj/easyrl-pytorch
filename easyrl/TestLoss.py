import unittest
from easyrl.ppo_parallel import *
import numpy as np

class TestLossCalculations(unittest.TestCase):


    def test_sample_and_sum_logits(self):
        rand_seq = np.asarray([[0.01,.099,.27,.43]]).T

        a_logits = np.asarray([[-6, -7, 8], [-5, -4, -3], [2, 9, 10],  [1, 1, 0]])


        a_i, value, a_logit, sum_exp_logits = Model.sample_and_sum_logits(np.asarray([5, 10, 11, 11]), a_logits, rand_seq = rand_seq)

        self.assertEqual(a_i.tolist(), [2, 1, 2, 1])
        self.assertTrue(np.allclose(a_logit, a_logits[np.arange(a_logits.shape[0]), a_i]))

if __name__ == '__main__':
    unittest.main()
