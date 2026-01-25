#!/usr/bin/python3
def summation_i_squared(n):
    sum = 0
    if n >= 0 and type(n) is int:
        for i in range(1, n+1):
            sum += i ** 2
        return sum
    else:
        return None
