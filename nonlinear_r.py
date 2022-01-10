"""
** Paper **
Title: A NEW COEFFICIENT OF CORRELATION
Author: SOURAV CHATTERJEE
URL: https://arxiv.org/pdf/1909.10140.pdf
Abstract:
    Is it possible to define a coefficient of correlation which is
    (a) as simple as the classical coefficients like Pearson’s correlation or Spearman’s correlation, and yet
    (b) consistently estimates some simple and interpretable measure of the degree of dependence between the variables,
    which is 0 if and only if the variables are independent and 1 if and only if one is a measurable function of the
    other, and
    (c) has a simple asymptotic theory under the hypothesis of independence, like the classical coefficients?
    This article answers this question in the affirmative, by producing such a coefficient.
    No assumptions are needed on the distributions of the variables.
    There are several coefficients in the literature that converge to 0 if and only if the variables are independent,
    but none that satisfy any of the other properties mentioned above.

** Code **
Author: Ricardo Lemos
Date: Jan 9, 2022
Copyright: Apache 2.0
"""
import numpy as np
from scipy.stats import norm
import unittest


def nonlinear_r(x: np.ndarray, y: np.ndarray) -> dict:
    """
    Computes Sourav Chatterjee's nonlinear correlation coefficient for continuous variables

    :param x: sample of predictor variable
    :param y: sample of response variable (same length as x)
    :return: dict with correlation coefficient value and asymptotic p-value, assuming no ties
    """
    _check_inputs(x, y)
    rank_y = _get_rank(y[np.argsort(x)])
    antirank_y = _get_antirank(rank_y)
    numerator = _get_numerator(rank_y)
    denominator = _get_denominator(antirank_y)
    xi = 1 - numerator / denominator
    p_value = _nonlinear_p_value(xi, len(x))
    return {'correlation': xi, 'p_value': p_value}


###############################################
# Auxiliary functions #########################
###############################################

def _check_inputs(x: np.ndarray, y: np.ndarray) -> None:
    if len(x) != len(y):
        raise ValueError('the two arrays have different lengths: ' + str(len(x)) + ' vs ' + str(len(y)))


def _get_rank(z: np.ndarray) -> np.ndarray:
    temp = np.argsort(z)
    ranks = np.empty_like(temp)
    ranks[temp] = 1 + np.arange(len(z))
    return ranks


def _get_antirank(z: np.ndarray) -> np.ndarray:
    return len(z) - _get_rank(z) + 1


def _get_numerator(rank_y: np.ndarray) -> np.ndarray:
    return len(rank_y) * np.sum([np.abs(r_next - r) for r_next, r in zip(rank_y[1:], rank_y[:-1])])


def _get_denominator(antirank_y: np.ndarray) -> np.ndarray:
    return 2 * np.sum(antirank_y * (len(antirank_y) - antirank_y))


def _nonlinear_p_value(xi: float, n: int) -> float:
    return 1 - norm.cdf(xi * np.sqrt(n * 5 / 2))


###############################################
# Test class ##################################
###############################################

class NonlinearCorrelationTester(unittest.TestCase):

    x = np.array([5., 4., 7.])
    y = np.array([5., 6., 4.])
    x_long = np.array([-0.985, 1.963, 0.461, 0.512, 0.183, -1.443, 1.415, -0.792, 0.468, -0.905, -1.165, 0.008, -0.388,
                       0.33, -0.15, -0.916, 0.259, 0.079, 0.209, -0.565, 0.45, -0.091, 0.762, -1.404, 0.886, 1.579,
                       0.283, 0.439, -0.652, -0.998])
    y_long = np.array([0.946, 3.813, 0.289, 0.193, -0.219, 2.071, 2.003, 0.724, 0.154, 0.794, 1.242, 0.123, 0.489,
                       -0.329, 0.49, 0.915, 0.149, 0.148, 0.124, 0.359, 0.494, 0.021, 0.662, 1.938, 0.778, 2.513,
                       0.029, 0.136, 0.061, 1.289])

    def test_input_error(self):
        self.assertRaises(ValueError, _check_inputs, self.x, self.y[0:2])

    def test_input_ok(self):
        self.assertIsNone(_check_inputs(self.x, self.y))

    def test_rank(self):
        self.assertEqual(_get_rank(self.x).tolist(), [2, 1, 3])
        self.assertEqual(_get_rank(self.y).tolist(), [2, 3, 1])

    def test_antirank(self):
        self.assertEqual(_get_antirank(self.x).tolist(), [2, 3, 1])
        self.assertEqual(_get_antirank(self.y).tolist(), [2, 1, 3])

    def test_numerator(self):
        rank_y = _get_rank(self.y[np.argsort(self.x)])
        self.assertEqual(_get_numerator(rank_y), 6)

        rank_xx = _get_rank(self.x[np.argsort(self.x)])
        self.assertEqual(_get_numerator(rank_xx), 6)

        rank_yy = _get_rank(self.y[np.argsort(self.y)])
        self.assertEqual(_get_numerator(rank_yy), 6)

    def test_denominator(self):
        rank_y = _get_rank(self.y[np.argsort(self.x)])
        self.assertEqual(_get_denominator(rank_y), 8)

        rank_xx = _get_rank(self.x[np.argsort(self.x)])
        self.assertEqual(_get_denominator(rank_xx), 8)

        rank_yy = _get_rank(self.y[np.argsort(self.y)])
        self.assertEqual(_get_denominator(rank_yy), 8)

    def test_nonlinear_r(self):
        # For deterministic associations (i.e., no error), the max value of nonlinear_r is (n - 2) / (n - 1);
        # In this test case, n=3 and thus r=0.25
        self.assertEqual(nonlinear_r(self.x, self.y)['correlation'], 0.25)
        self.assertEqual(nonlinear_r(self.x, self.x)['correlation'], 0.25)
        self.assertEqual(nonlinear_r(self.y, self.y)['correlation'], 0.25)
        # the following correlation value was taken from R package XICOR, using the same input x_long and y_long
        self.assertAlmostEqual(nonlinear_r(self.x_long, self.y_long)['correlation'], 0.5995551)

    def test_nonlinear_p_value(self):
        self.assertAlmostEqual(nonlinear_r(self.x, self.y)['p_value'], 0.2467814)
        self.assertAlmostEqual(nonlinear_r(self.x, self.x)['p_value'], 0.2467814)
        self.assertAlmostEqual(nonlinear_r(self.y, self.y)['p_value'], 0.2467814)
        self.assertAlmostEqual(nonlinear_r(self.x_long, self.y_long)['p_value'], 1.038565e-07)
