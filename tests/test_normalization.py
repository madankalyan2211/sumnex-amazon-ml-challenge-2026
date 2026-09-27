import unittest
from src.normalization import normalize_name_fast, normalize_address_fast

class TestNormalization(unittest.TestCase):
    def test_name_normalization(self):
        cleaned, base, compact, compact_base, tokens, base_tokens, char_3grams = normalize_name_fast("Team Air Pvt. Ltd.")
        self.assertEqual(base, "team air")
        self.assertEqual(compact_base, "teamair")
        
        c2, b2, comp2, cb2, t2, bt2, g2 = normalize_name_fast("TEAMAIR.COM")
        self.assertEqual(cb2, "teamair")
        self.assertEqual(compact_base, cb2)
        
        c3, b3, comp3, cb3, t3, bt3, g3 = normalize_name_fast("SERVICESCPMEDUSERVE.COM")
        self.assertEqual(cb3, "cpmeduserve")

    def test_address_normalization(self):
        cleaned1, tokens1, token_set1, numbers1 = normalize_address_fast("175 Boulevard du Président Franklin Roosevelt")
        cleaned2, tokens2, token_set2, numbers2 = normalize_address_fast("175 BD DU PRESIDENT FRANKLIN ROOSEVELT")
        
        self.assertEqual(numbers1, {'175'})
        self.assertEqual(numbers2, {'175'})
        self.assertEqual(cleaned1, cleaned2)

if __name__ == "__main__":
    unittest.main()
