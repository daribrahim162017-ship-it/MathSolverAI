import sympy as sp

x = sp.symbols("x")
equation = sp.Eq(2*x + 5, 13)
solution = sp.solve(equation, x)

print("SymPy is working!")
print("Equation: 2*x + 5 = 13")
print("Solution:", solution)
print("Verification:", equation.subs(x, solution[0]))
