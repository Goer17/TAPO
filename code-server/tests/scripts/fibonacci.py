def fibonacci(x: int):
    if x in [1, 2]:
        return 1
    return fibonacci(x - 1) + fibonacci(x - 2)

print(fibonacci(20))

# 6765\n