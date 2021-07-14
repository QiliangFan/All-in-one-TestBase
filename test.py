from typing import OrderedDict

if __name__ == "__main__":
    n, k = [int(i) for i in  input().strip().split(" ")]
    totals = []
    counts = []
    for i in range(n-1):
        a, b = [int(i) for i in input().strip().split(" ")]
        totals.append({a, b})
    res = []

    res.sort()
    print(*res[:k])

"""
6 3
2 1
2 6
4 2
5 6
2 3
"""