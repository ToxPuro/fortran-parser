multiplicative_ops = "*/"
additive_ops = "+-"
all_ops = "*/+-"
class ASTNode:
    def __init__(self,val,ast_type,lhs,rhs,parent):
        self.val = val
        self.type = ast_type
        self.lhs = lhs
        self.rhs = rhs
        self.parent = parent

def print_AST(root,nest_level=0):
    prefix = nest_level*" "
    if(root == None):
        return
    print(prefix,root.val, root.type)
    if(root.lhs):
        print(prefix,"<-")
        print_AST(root.lhs,nest_level+1)
    if(root.rhs):
        print(prefix, "->")
        print_AST(root.rhs,nest_level+1)


def is_number(val):
    return is_float(val) or val.isnumeric()
def is_float(string):
    if string[0] == "-":
        string = string[1:]
    return string.replace(".","").isnumeric()
def is_mult_node(node):
    return is_number(node.val.replace("*","").replace("/",""))
#used to reorder constant values to the right in the expression
#used .ef. if only additive ops
def reorder_mult_first(root):
    if(root is None or root.lhs is None):
        return root
    root.lhs = reorder_mult_first(root.lhs)
    if(not root.lhs.lhs):
        if(is_mult_node(root.rhs)):
            #swap number to left
            tmp_val = root.rhs.val
            root.rhs.val = root.lhs.val
            root.lhs.val =  tmp_val
        return root
    if(is_mult_node(root.rhs)):
        #swap number to left
        tmp_val = root.rhs.val
        root.rhs.val = root.lhs.rhs.val
        root.lhs.rhs.val = tmp_val
    return root

def build_ast_for_ops(equation,ops):
    root = ASTNode(None, None,None,None,None)
    last_index = 0
    for index, char in enumerate(equation):
        if char in ops:
            if(root.lhs == None):
                index_till_next_symbol = index+1
                while(index_till_next_symbol < len(equation) and equation[index_till_next_symbol] not in ops):
                    index_till_next_symbol += 1
                rhs = ASTNode(equation[index+1:index_till_next_symbol],None,None,None,root)
                root.rhs = rhs
                root.type = char
                lhs = ASTNode(equation[last_index:index],None,None,None,root)
                root.lhs = lhs
                root = ASTNode(None,None,root,None,None)
            else:
                index_till_next_symbol = index+1
                while(index_till_next_symbol < len(equation) and equation[index_till_next_symbol] not in ops):
                    index_till_next_symbol += 1
                rhs = ASTNode(equation[index+1:index_till_next_symbol],None,None,None,root)
                root.type = char
                root.rhs = rhs
                root = ASTNode(None,None,root,None,None)
            last_index = index+1

    return root.lhs

def expand_multiplicative_ops(root):
    if(root.lhs):
        root.lhs = expand_multiplicative_ops(root.lhs)
    if(root.rhs):
        root.rhs = expand_multiplicative_ops(root.rhs)
    if(not root.lhs and not root.rhs):
        if(any([x in root.val for x in multiplicative_ops])):
            return build_ast_for_ops(root.val, multiplicative_ops)
    return root
def convert_equation_to_ast(equation):
    if all([x not in equation for x in additive_ops]):
        return build_ast_for_ops(equation,multiplicative_ops)
    additive_ast_root = build_ast_for_ops(equation,additive_ops)
    if("-" not in equation):
        additive_ast_root = reorder_mult_first(additive_ast_root)
    if any([x in equation for x in multiplicative_ops]):
        return expand_multiplicative_ops(additive_ast_root)
    #only additive ops
    return additive_ast_root

def combine_values(root):
    if(root == None):
        return None
    if(root.rhs == None and root.lhs == None):
        return root.val
    lhs_val = combine_values(root.lhs)
    rhs_val = combine_values(root.rhs)
    if(lhs_val == None):
        return rhs_val
    if(rhs_val == None):
        return lhs_val
    if(is_float(rhs_val) and is_float(lhs_val)):
        return str(eval(f"{lhs_val}{root.type}{rhs_val}"))
    return f"{lhs_val}{root.type}{rhs_val}" 

    
   
def simplify(equation):
    if equation.isnumeric() or is_float(equation):
        return equation
    if all([x not in equation for x in all_ops]):
        return equation
    if "(" in equation and ")" in equation:
        range_start = 0
        while(equation[range_start] != "("):
            range_start += 1

        range_end = 0
        while(equation[range_end] != ")"):
            range_end += 1
        subexpression_value = simplify(equation[range_start+1:range_end])
        if(not subexpression_value.isnumeric()):
            return equation
        new_equation = equation[:range_start] + subexpression_value + equation[range_end+1:]
        return simplify(new_equation)
    root = convert_equation_to_ast(equation)
    val = combine_values(root)
    val_is_float = is_float(val)
    val_is_integer = val.isnumeric()
    if(val_is_float):
        #a way to check if it the floating point number is actually an integer
        if eval(val) == eval(str(int(eval(val)))):
            return str(int(eval(val)))
    return val


#equation = "mx+1*2+my"
#root = convert_equation_to_ast(equation)
#print_AST(root)
#print(simplify(equation))
