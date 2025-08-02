def inf_recur(x):
    return inf_recur(x ^ 1)

inf_recur(1)