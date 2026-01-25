#!/usr/bin/env python3
def summation_i_squared(n):
    if n >= 0 and type(n) is int:
        sum = n * (n+1) * (2*n+1) / 6
        return int(sum)
    else:
        return None
