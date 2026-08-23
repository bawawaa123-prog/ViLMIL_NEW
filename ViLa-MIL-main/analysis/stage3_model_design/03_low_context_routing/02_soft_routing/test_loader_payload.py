import unittest

import torch

from utils.utils import collate_tranformer


class LoaderPayloadTest(unittest.TestCase):
    def test_mapping_round_trip(self):
        mapping = {
            "high_parent_ptr": torch.tensor([0, 1]),
            "parent_indices": torch.tensor([0]),
            "parent_weight": torch.tensor([1.0]),
        }
        item = (
            torch.zeros(1, 3), torch.zeros(1, 2), torch.zeros(1, 3),
            torch.zeros(1, 2), 1, "slide-x", mapping,
        )
        batch = collate_tranformer([item])
        self.assertEqual(len(batch), 7)
        self.assertEqual(batch[5], ["slide-x"])
        self.assertIs(batch[6][0], mapping)


if __name__ == "__main__":
    unittest.main()
