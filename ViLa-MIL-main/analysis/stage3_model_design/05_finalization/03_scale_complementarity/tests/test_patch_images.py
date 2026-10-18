"""Small PNG fixtures, including oversized metadata; no model inference."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image,ImageFile,PngImagePlugin

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))
from prepare_case_visualizations import load_patch_tile,omit_png_icc


class PatchImageTests(unittest.TestCase):
    def test_oversized_icc_preserves_pixels_source_and_pillow_limits(self):
        with tempfile.TemporaryDirectory() as root:
            plain=Path(root)/'plain.png';profiled=Path(root)/'profiled.png'
            pixels=np.random.default_rng(353).integers(0,256,size=(224,224,3),dtype=np.uint8)
            image=Image.fromarray(pixels)
            image.save(plain)
            image.save(profiled,icc_profile=b'x'*(PngImagePlugin.MAX_TEXT_CHUNK+1))
            before=profiled.read_bytes()
            with self.assertRaisesRegex(ValueError,'Decompressed Data Too Large'):
                with Image.open(profiled) as broken:broken.load()
            limits=(PngImagePlugin.MAX_TEXT_CHUNK,PngImagePlugin.MAX_TEXT_MEMORY,ImageFile.LOAD_TRUNCATED_IMAGES)
            actual,omitted=load_patch_tile(profiled)
            normal,normal_omitted=load_patch_tile(plain)
            self.assertTrue(omitted);self.assertFalse(normal_omitted)
            np.testing.assert_array_equal(np.asarray(actual),pixels)
            np.testing.assert_array_equal(np.asarray(actual),np.asarray(normal))
            self.assertEqual(profiled.read_bytes(),before)
            self.assertEqual(limits,(PngImagePlugin.MAX_TEXT_CHUNK,PngImagePlugin.MAX_TEXT_MEMORY,ImageFile.LOAD_TRUNCATED_IMAGES))
            self.assertEqual(actual.info,{})

    def test_other_image_errors_are_not_suppressed(self):
        with patch('PIL.Image.open',side_effect=ValueError('different image error')):
            with self.assertRaisesRegex(ValueError,'different image error'):
                load_patch_tile(Path('synthetic.png'))
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'invalid.png';path.write_bytes(b'not an image')
            with self.assertRaises(OSError):load_patch_tile(path)

    def test_malformed_png_fallback_is_rejected(self):
        for data in (b'bad',b'\x89PNG\r\n\x1a\n',b'\x89PNG\r\n\x1a\n\x00',
                     b'\x89PNG\r\n\x1a\n'+(100).to_bytes(4,'big')+b'iCCP'+b'short'):
            with self.assertRaises(ValueError):omit_png_icc(data)


if __name__=='__main__':unittest.main()
