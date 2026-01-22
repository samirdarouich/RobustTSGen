period_group = {
    1: (1, 1),  2: (1, 18),                   # H, He
    3: (2, 1),  4: (2, 2),  5: (2, 13),  6: (2, 14),  7: (2, 15),  8: (2, 16),  9: (2, 17),  10: (2, 18), # Li-Ne
    11: (3, 1), 12: (3, 2), 13: (3, 13), 14: (3, 14), 15: (3, 15), 16: (3, 16), 17: (3, 17), 18: (3, 18),
    19: (4, 1), 20: (4, 2), 21: (4, 3), 22: (4, 4), 23: (4, 5), 24: (4, 6), 25: (4, 7), 26: (4, 8), 27: (4, 9), 28: (4, 10),
    29: (4, 11), 30: (4, 12), 31: (4, 13), 32: (4, 14), 33: (4, 15), 34: (4, 16), 35: (4, 17), 36: (4, 18),
    37: (5, 1), 38: (5, 2), 39: (5, 3), 40: (5, 4), 41: (5, 5), 42: (5, 6), 43: (5, 7), 44: (5, 8), 45: (5, 9), 46: (5, 10),
    47: (5, 11), 48: (5, 12), 49: (5, 13), 50: (5, 14), 51: (5, 15), 52: (5, 16), 53: (5, 17), 54: (5, 18),
    55: (6, 1), 56: (6, 2), # skip 57-71 lanthanoids
    72: (6, 4), 73: (6, 5), 74: (6, 6), 75: (6, 7), 76: (6, 8), 77: (6, 9), 78: (6, 10),
    79: (6, 11), 80: (6, 12), 81: (6, 13), 82: (6, 14), 83: (6, 15), 84: (6, 16), 85: (6, 17), 86: (6, 18),
    87: (7, 1), 88: (7, 2), # skip 89-103 actinoids
    104: (7, 4), 105: (7, 5), 106: (7, 6), 107: (7, 7), 108: (7, 8), 109: (7, 9), 110: (7, 10),
    111: (7, 11), 112: (7, 12), 113: (7, 13), 114: (7, 14), 115: (7, 15), 116: (7, 16), 117: (7, 17), 118: (7, 18)
}

own_periodic_group = {}
elements_per_row = [2, 8, 8, 18, 18, 17, 17]
z_0_per_row = [1, 3, 11, 19, 37, 55, 87, 119]
z_per_row = [range(z_0_per_row[i], z_0_per_row[i+1]) for i in range(len(z_0_per_row)-1)]
numbers_to_remove = range(57, 72)
z_per_row[-2] = [x for x in z_per_row[-2] if x not in numbers_to_remove]
numbers_to_remove = range(89, 104)
z_per_row[-1] = [x for x in z_per_row[-1] if x not in numbers_to_remove]
for i, z in enumerate(range(1, 119)):
    row = [j for j, r in enumerate(z_per_row) if z in r]
    if row:
        row = row[0] +1
        column = z_per_row[row-1].index(z) + 1
        if row == 1:
            if column == 2:
                column = 18
        elif row == 2 or row == 3:
            if column > 2:
                column += 10
        elif row > 5:
            if column > 2:
                column += 1
        own_periodic_group[z] = (row, column)
        ref = period_group.get(z)
        
        assert ref == own_periodic_group[z], f"Mismatch for Z={z}: {ref} vs {own_periodic_group[z]}"
        
for z in period_group:
    assert period_group[z] == own_periodic_group[z], f"Mismatch for Z={z}: {period_group[z]} vs {own_periodic_group[z]}"