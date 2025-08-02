import sympy

x = sympy.symbols('x')

f = x ** 2

result = sympy.integrate(f, (x, 0, 1)).evalf()

print(f"{result :.3f}")

# 0.333\n