import unittest

from mypkg import config
from mypkg import mymodule


class FileStructureTestCase(unittest.TestCase):
    def setUp(self):
        pass
    
    def test_module(self):
        assert True