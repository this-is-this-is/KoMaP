"""Small behavioral checks for the new mask-sensitive training transforms."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from augmentations import (attenuate_connected_si, context_paste, fourier_style,
                           illumination_field, weak_blur)


class AugmentationTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(7)
        gray = np.full((96, 96), 185, np.uint8)
        gray[42:54, 42:54] = 35
        self.image = np.repeat(gray[..., None], 3, axis=2)
        self.mask = np.zeros((96, 96), np.uint8)
        self.mask[42:54, 42:54] = 2

    def check_image(self, value):
        self.assertEqual(value.shape, self.image.shape)
        self.assertEqual(value.dtype, np.uint8)
        self.assertTrue(np.array_equal(value[..., 0], value[..., 1]))
        self.assertTrue(np.array_equal(value[..., 1], value[..., 2]))

    def test_image_formation_changes_preserve_shape_and_grayscale(self):
        donor = np.full_like(self.image, 130)
        for output in (illumination_field(self.image, self.rng),
                       weak_blur(self.image, self.rng),
                       fourier_style(self.image, donor, self.rng),
                       attenuate_connected_si(self.image, self.mask, self.rng)):
            self.check_image(output)
        self.assertTrue(np.array_equal(self.mask[42:54, 42:54],
                                       np.full((12, 12), 2, np.uint8)))

    def test_fourier_identical_donor_is_nearly_identity(self):
        output = fourier_style(self.image, self.image, self.rng)
        self.assertLessEqual(np.abs(output.astype(int) - self.image.astype(int)).max(), 1)

    def test_context_paste_moves_image_and_mask_together(self):
        target = np.full_like(self.image, 185)
        target_mask = np.zeros_like(self.mask)
        image, mask = context_paste(target, target_mask, self.image, self.mask,
                                    self.rng, native_size=(48, 48))
        self.check_image(image)
        self.assertTrue(np.any(mask == 2))
        self.assertTrue(np.array_equal(image[mask == 2, 0],
                                       np.full(np.count_nonzero(mask == 2), 35, np.uint8)))
        self.assertTrue(np.array_equal(target_mask, np.zeros_like(target_mask)))


if __name__ == "__main__":
    unittest.main()
