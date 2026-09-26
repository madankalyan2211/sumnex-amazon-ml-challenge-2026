from src.normalization import normalize_name, normalize_address

def test_normalization():
    n1 = normalize_name("Team Air Pvt. Ltd.")
    n2 = normalize_name("TEAMAIR.COM")
    print("n1:", n1)
    print("n2:", n2)
    assert n1['compact'] == 'teamairpvtltd' or 'teamair' in n1['compact']
    assert n2['compact'] == 'teamair'
    
    n3 = normalize_name("LLC Moncada Léarning Center")
    print("n3:", n3)
    assert 'learning' in n3['cleaned']
    
    a1 = normalize_address("175 Boulevard du Président Franklin Roosevelt")
    a2 = normalize_address("175 BD DU PRESIDENT FRANKLIN ROOSEVELT")
    print("a1:", a1)
    print("a2:", a2)
    assert a1['numbers'] == {'175'}
    assert a2['numbers'] == {'175'}
    assert a1['cleaned'] == a2['cleaned']
    print("Normalization tests passed!")

if __name__ == "__main__":
    test_normalization()
