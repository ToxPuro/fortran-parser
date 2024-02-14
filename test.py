import ast

equation = "38-3"
root = ast.convert_equation_to_ast(equation)
ast.print_AST(root)
print(ast.simplify(equation))

