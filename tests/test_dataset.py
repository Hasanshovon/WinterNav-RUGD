"""Tests for RUGD image-mask pair discovery."""

from pathlib import Path
import tempfile
import unittest

from PIL import Image

from src.dataset import discover_image_mask_pairs


class DatasetDiscoveryTest(unittest.TestCase):
    def test_matching_image_mask_pair_discovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images_root = root / "images"
            masks_root = root / "masks"
            (images_root / "creek").mkdir(parents=True)
            (masks_root / "creek").mkdir(parents=True)

            Image.new("RGB", (2, 2), color=(1, 2, 3)).save(
                images_root / "creek" / "creek_00001.png"
            )
            Image.new("RGB", (2, 2), color=(0, 0, 0)).save(
                masks_root / "creek" / "creek_00001.png"
            )

            pairs = discover_image_mask_pairs(images_root, masks_root)

            self.assertEqual(len(pairs), 1)
            self.assertEqual(pairs[0].sequence, "creek")
            self.assertEqual(pairs[0].filename, "creek_00001.png")


if __name__ == "__main__":
    unittest.main()
