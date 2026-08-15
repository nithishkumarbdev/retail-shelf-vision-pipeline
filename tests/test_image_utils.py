import os
import sys
import base64
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common.image_utils import (  # noqa: E402
    image_to_base64,
    base64_to_image,
    ImageDecodeError,
    allowed_filename,
)


class TestImageUtils(unittest.TestCase):
    def test_roundtrip_encode_decode(self):
        original = np.random.randint(0, 255, (50, 60, 3), dtype=np.uint8)
        b64 = image_to_base64(original, fmt="PNG")
        decoded = base64_to_image(b64)
        self.assertEqual(decoded.shape, original.shape)

    def test_data_url_prefix_is_stripped(self):
        original = np.random.randint(0, 255, (20, 20, 3), dtype=np.uint8)
        b64 = image_to_base64(original, fmt="PNG")
        data_url = "data:image/png;base64," + b64
        decoded = base64_to_image(data_url)
        self.assertEqual(decoded.shape, original.shape)

    def test_empty_string_raises(self):
        with self.assertRaises(ImageDecodeError):
            base64_to_image("")

    def test_garbage_base64_raises(self):
        with self.assertRaises(ImageDecodeError):
            base64_to_image("not-valid-base64-image-data!!!")

    def test_valid_base64_but_not_an_image_raises(self):
        garbage = base64.b64encode(b"hello world, this is not image data").decode()
        with self.assertRaises(ImageDecodeError):
            base64_to_image(garbage)

    def test_allowed_filename(self):
        self.assertTrue(allowed_filename("shelf.jpg"))
        self.assertTrue(allowed_filename("shelf.PNG"))
        self.assertFalse(allowed_filename("shelf.txt"))
        self.assertFalse(allowed_filename("noextension"))


if __name__ == "__main__":
    unittest.main()
