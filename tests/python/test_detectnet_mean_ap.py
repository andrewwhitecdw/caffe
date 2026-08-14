import unittest
import numpy as np

from caffe.layers.detectnet.mean_ap import score_det, MAX_BOXES


class TestScoreDet(unittest.TestCase):
    def test_truncate_past_max_boxes(self):
        """score_det must not crash when tp+fp+tn exceeds MAX_BOXES."""
        n = MAX_BOXES + 10
        gt = np.zeros((1, n, 5))
        gt[0, :, :4] = np.arange(n * 4).reshape(n, 4) % 100 + 1
        det = np.zeros((1, n, 5))
        det[0, :, :4] = np.arange(n * 4).reshape(n, 4) % 100 + 101

        scored = score_det(gt, det)
        self.assertEqual(scored.shape, (1, MAX_BOXES, 5))
        # No overlap => all detections are false positives, so first row is class 2.
        self.assertEqual(scored[0, 0, 4], 2)


