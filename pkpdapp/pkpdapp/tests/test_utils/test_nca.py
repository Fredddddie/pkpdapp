#
# This file is part of PKPDApp (https://github.com/pkpdapp-team/pkpdapp) which
# is released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import unittest
import numpy as np
from pkpdapp.utils import NCA


class TestNCA(unittest.TestCase):
    def test_lin_trapz(self):
        times = np.array([0.0, 1.0, 2.0])
        concs = np.array([0.0, 1.0, 0.0])
        area = NCA.linlog_trapz(concs, times, linlog=False)
        expected = np.trapz(concs, times)
        self.assertAlmostEqual(area, expected)

    def test_half_life(self):
        times = np.array([0.0, 1.0, 2.0, 3.0])
        concs = np.array([1.0, 0.5, 0.25, 0.125])
        nca = NCA(times, concs, DM=1.0)
        nca.calculate_nca()
        expected = np.log(2) / nca.lambda_z
        self.assertAlmostEqual(nca._T_half(), expected)
        self.assertAlmostEqual(nca.t_half, expected)
