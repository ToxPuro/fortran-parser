import ast

equation = "-1+3+1+3+1"
root = ast.convert_equation_to_ast(equation)
ast.print_AST(root)
print(ast.simplify(equation))

