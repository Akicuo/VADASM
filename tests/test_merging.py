import unittest
import torch
import torch.nn as nn
from vadasm.utils import ties_merge, dare_merge
from vadasm.alignment import hungarian_neuron_alignment

class TestVADASMMerging(unittest.TestCase):
    
    def test_ties_merge(self):
        """Test TIES parameter merging"""
        deltas = torch.randn(3, 100, 200)  # 3 models
        
        merged = ties_merge(deltas, drop_rate=0.3)
        self.assertEqual(merged.shape, (100, 200))
        
        # Check that some values are zero (sparsified)
        self.assertTrue((merged == 0).any())
    
    def test_dare_merge(self):
        """Test DARE sparsification"""
        deltas = torch.randn(50, 50)
        
        sparsified = dare_merge(deltas, drop_rate=0.2)
        
        # Should have same shape
        self.assertEqual(sparsified.shape, deltas.shape)
        
        # Should be sparser than original
        original_nonzero = (deltas != 0).sum()
        sparse_nonzero = (sparsified != 0).sum()
        self.assertLess(sparse_nonzero, original_nonzero)
    
    def test_hungarian_alignment(self):
        """Test neuron alignment"""
        model_a = torch.randn(100, 512) 
        model_b = torch.randn(100, 512)
        
        perm = hungarian_neuron_alignment(model_a, model_b)
        
        # Permutation should be valid indices
        self.assertEqual(len(perm), 100)
        self.assertTrue(all(0 <= i < 100 for i in perm.tolist()))

if __name__ == '__main__':
    unittest.main()