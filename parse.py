
import math
import re
import os
import csv
import argparse
import glob
import cProfile
import ctypes
##import my-ast

def format_lines(lines):
    tab = '  '
    n_tabs = 0
    res = []
    for line in lines:
        if "}" in line:
            n_tabs -= 1
        res.append(f"{n_tabs*tab}{line}")
        if "{" in line:
            n_tabs += 1
    return res


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
    if val == "":
        return False
    return is_float(val) or val.isnumeric()
def is_float(string):
    if string== "":
        return False
    if string[0] == "-":
        string = string[1:]
    return string.replace(".","").replace("e-","").replace("e+","").replace("e","").isnumeric()
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
    if equation == "":
        return equation
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



global_loop_y = "m__mod__cdata"
global_loop_z = "n__mod__cdata"
global_subdomain_range_x = "nx__mod__cparam"
global_subdomain_range_y = "ny__mod__cparam"
global_subdomain_range_z = "nz__mod__cparam"
nghost_val = "3"
global_subdomain_range_with_halos_x = f"{global_subdomain_range_x}+2*{nghost_val}"
global_subdomain_range_with_halos_y = f"{global_subdomain_range_y}+2*{nghost_val}"
global_subdomain_range_with_halos_z = f"{global_subdomain_range_z}+2*{nghost_val}"
global_subdomain_ranges = [global_subdomain_range_with_halos_x,global_subdomain_range_with_halos_y,global_subdomain_range_with_halos_z]
global_subdomain_range_x_upper = "l2__mod__cparam"
global_subdomain_range_x_lower= "l1__mod__cparam"
global_subdomain_range_x_inner = f"{global_subdomain_range_x_lower}:{global_subdomain_range_x_upper}"
impossible_val = ""

global_subdomain_range_y_upper = "m2__mod__cparam"
global_subdomain_range_y_lower= "m1__mod__cparam"

global_subdomain_range_z_upper = "n2__mod__cparam"
global_subdomain_range_z_lower= "n1__mod__cparam"
pc_parser = ""
# global_loop_y = ""
# global_loop_z = ""
# global_subdomain_range_x = ""
# global_subdomain_range_y = ""
# global_subdomain_range_z = ""
# nghost_val = ""
# global_subdomain_range_with_halos_x = ""
# global_subdomain_range_with_halos_y = ""
# global_subdomain_range_with_halos_z = ""
# global_subdomain_range_x_upper = ""
# global_subdomain_range_x_lower= ""

# global_subdomain_range_y_upper = ""
# global_subdomain_range_y_lower= ""

# global_subdomain_range_z_upper = ""
# global_subdomain_range_z_lower= ""

farray_register_funcs = ["farray_register_pde","farray_register_global","farray_register_auxiliary"]

number_of_fields = ""



def translate_fortran_ops_to_c(lines):
    lines = [line.replace(".false.","false") for line in lines]
    lines = [line.replace(".true.","true") for line in lines]
    lines = [line.replace(".and."," && ") for line in lines]
    lines = [line.replace(".or."," || ") for line in lines]
    lines = [line.replace(".not.","!") for line in lines]
    lines = [line.replace("/=","!=") for line in lines]
    return lines
def split_line_nml(line):
    if "=" not in line:
        return [line]
    if "&" not in line:
        return [line]
    res = []
    num_of_equals = 0
    last_start_index  = 0
    index = 0
    while(num_of_equals == 0):
        symbol = line[index]
        if(symbol == "="):
            num_of_equals += 1
            iter_index = 0
            symbol = line[iter_index]
            while(symbol != " "):
                iter_index += 1
                symbol = line[iter_index]
            res.append(line[last_start_index:iter_index].strip())
            last_start_index = iter_index + 1
            num_of_quotes = 0
            symbol = line[index]
            while(not (symbol == "," and num_of_quotes %2 == 0) ):
                symbol = line[index]
                if(symbol == '"'):
                    num_of_quotes += 1
                index += 1
            res.append(line[last_start_index:index].strip())
        index += 1
    last_start_index = index
    while(index < len(line)):
        char = line[index]
        if(char  == "="):
            iter_index = index+1
            symbol = line[index]
            num_of_quotes = 0
            while(line[iter_index] != "=" and iter_index < len(line)-1):
                iter_index += 1
            if(iter_index == len(line) -1 and line[iter_index] == "/"):
                res.append(f"{line[last_start_index:iter_index]},")
                break
                index = len(line) + 1
            while(line[iter_index] != ","):
                iter_index -= 1
            iter_index += 2
            res.append(line[last_start_index:iter_index])
            last_start_index = iter_index
            index = last_start_index
        index += 1

    if(res[-1][-1] == "/"):
        res[-1] = res[-1][:-1]
    res.append("/")
    return res

checked_local_writes = []
der_funcs = ["der","der2","der3","der4","der5","der6","der4i2j","der2i2j2k","der5i1j","der3i3j","der3i2j1k","der4i1j1k","derij"]
implemented_der_funcs = {}
der_func_map = {
  "der_main":
  {
    "1": "derx",
    "2": "dery",
    "3": "derz"
  },
  "der2_main":
  {
    "1": "derxx",
    "2": "deryy",
    "3": "derzz"
  },
  "der6_main":
  {
    "1": "der6x",
    "2": "der6y",
    "3": "der6z"
  }

}
def gen_field3(index):
    return f"Field3(Field({index}), Field({index}+1), Field({index}+2))"
def map_curl(func_call):
    params = func_call["parameters"]
    if len(params)>3:
        pexit("optional params not supported")
    return [f"{params[2]} = curl({gen_field3(params[1])})"]

def map_curl_mn(func_call):
    #for time being only support carteesian coords
    params = func_call["parameters"]
    #other_params = func_call["parameters"][:1] + func_call["parameters"][2:]
    #return f"{func_call['parameters'][1]}=curl_from_matrix({','.join(other_params)})"

    return [f"{func_call['parameters'][1]}=curl_from_matrix({func_call['parameters'][0]})"]

def map_dot_mn(func_call):
    params = func_call["parameters"]
    if len(params) == 3:
        return [f"{params[2]} = dot({params[0]},{params[1]})"]
    print("hmm",params)
    pexit("what to about ladd")

def map_del2_main(func_call):
    params = func_call["parameters"]
    return [f"{params[2]} = laplace(Field({params[1]}))"]

def map_del2v(func_call):
    params = func_call["parameters"]
    if len(params)>3:
        pexit("optional params not supported")
    return [f"{params[2]} = veclaplace({gen_field3(params[1])})"]

def map_del4v(func_call):
    params = func_call["parameters"]
    return [f"{params[2]} = vecdel4({gen_field3(params[1])})"]

def map_del6v(func_call):
    params = func_call["new_param_list"]
    if len(params)>3:
        if(params[-1][0] == ".true."):
            return [f"{params[2][0]} = del6v_strict({gen_field3(params[1][0])})"]
        print(params)
        pexit("optional params not supported")
    return [f"{params[2][0]} = del6v({gen_field3(params[1][0])})"]

def map_traceless_strain(func_call):
    params = func_call["new_param_list"]
    #all params are given in default order
    res = []
    line = f"{params[2][0]} = traceless_strain({params[0][0]},{params[1][0]})"
    res.append(line)
    if len(params)==5 and all(x[-1] is None for x in params):
        shear_param = params[4][0]
        if params[4][0]==".false.":
            pass
        else:
            if_line =    f"if (lshear__mod__cparam .and. {shear_param}) then"
            add_line_1 = f"{params[2][0]}(:,1,2) = {params[2][0]}(:,1,2) + Sshear__mod__shear"
            add_line_2 = f"{params[2][0]}(:,2,1) = {params[2][0]}(:,2,1) + Sshear__mod__shear"
            end_if_line = "endif"
            res.append(if_line)
            res.append(add_line_1)
            res.append(add_line_2)
            res.append(end_if_line)
    
    return res
    print(params)
    pexit("what to do?\n")
    

def map_gij(func_call):
    params = func_call["parameters"]
    if(params[3] == "1"):
        return [f"{params[2]} = gradients({gen_field3(params[1])})"]
    if(params[3] == "2"):
        return [f"{params[2]} = gradients_2({gen_field3(params[1])})"]
    if(params[3] == "3"):
        return [f"{params[2]} = gradients_3({gen_field3(params[1])})"]
    if(params[3] == "4"):
        return [f"{params[2]} = gradients_4({gen_field3(params[1])})"]
    if(params[3] == "5"):
        return [f"{params[2]} = gradients_5({gen_field3(params[1])})"]
    if(params[3] == "6"):
        return [f"{params[2]} = gradients_6({gen_field3(params[1])})"]
    print(params)
    pexit("what to do")

def map_grad_main(func_call):
    params = func_call["parameters"]
    return [f"{params[2]} = gradient(Field({params[1]}))"]
def map_div(func_call):
    params = func_call["parameters"]
    if len(params)>3:
        pexit("optional params not supported\n")
    return [f"{params[2]} = divergence({gen_field3(params[1])})"]
def map_div_mn(func_call):
    params = func_call["parameters"]
    if len(params)>3:
        pexit("optional params not supported\n")
    return [f"{params[1]} = divergence_from_matrix({params[0]})"]
def map_dot2_mn(func_call):
    params = func_call["parameters"]
    if len(params)>2:
        if params[2] == "fast_sqrt=.true.":
            return [f"{params[1]} = sqrt(dot({params[0]},{params[0]}))"]

        pexit("optional params not supported\n")
    return [f"{params[1]} = dot({params[0]},{params[0]})"]
def map_del2v_etc(func_call):
    params = func_call["new_param_list"]
    res = []
    for param in enumerate(params):
        if param[-1] == "del2":
            res.append[f"{param[0]} = veclaplace({gen_field3(params[1][0])})"]
        elif param[-1] == "graddiv":
            res.append[f"{param[0]} = gradient_of_divergence({gen_field3(params[1][0])})"]
    return res
def map_cross_mn(func_call):
    params = func_call["parameters"]
    return [f"{params[2]} = cross({params[0]},{params[1]})"]
def map_multm2_sym_mn(func_call):
    params = func_call["parameters"]
    return [f"{params[1]} = multm2_sym({params[0]})"]
def map_u_dot_grad_vec(func_call):
    params = func_call["new_param_list"]
    upwind = f"del6_upwd_vec({gen_field3(params[1][0])})"
    res = []
    add_line = ""
    if len(params) == 6 and params[5][-1] == "upwind" and params[5][0] == ".false.":
        pass
    elif len(params) == 6 and params[5][-1] == "upwind" and params[5][0] == ".true.":
        add_line = f"{params[4][0]} = {params[4][0]} + {upwind}"
    elif len(params) == 6 and params[5][-1] == "upwind":
        add_line = f"if ({params[5][0]}) {params[4][0]} = {params[4][0]} + {upwind}"
    elif len(params)>5:
        print("\n\n")
        print([x[0] for x in params])
        pexit("optional params not supported\n")
    main_line = f"{params[4][0]} = {params[2][0]}*{params[3][0]}"
    res.append(main_line)
    if len(add_line) > 0:
        res.append(add_line)
    return res
def map_u_dot_grad_scl(func_call):
    params = func_call["parameters"]
    add_line = None
    res = []
    start_line = "upwind_correction = 0.0"
    if len(params)>6:
        print(params)
        pexit("optional params not supported\n")
    if len(params) == 6 and params[5].split("=")[-1].strip() == ".false.":
        pass
    elif len(params) == 6 and params[5].split("=")[-1].strip() == ".true.":
        add_line = f"{params[4]} = {params[4]} - del6_upwd(Field({params[1]}))"
    elif len(params) == 6:
        param_name = params[5].split("=")[0]
        add_line = f"if ({param_name}) {params[4]} = {params[4]} - del6_upwd(Field({params[1]}))"
    else:
        pexit("don't know whether to do upwinding or not\n")
    main_line  = f"{params[4]} = dot({params[3]},{params[2]})"
    res.append(main_line)
    if add_line:
        res.append(add_line)
    return res
def map_del4(func_call):
    params = func_call["parameters"]
    if len(params)>3:
        pexit("optional params not supported\n")
    return [f"{params[2]} = del4(Field({params[1]})"]

def map_calc_del6_for_upwind(func_call):
    params = func_call["new_param_list"]
    print("DEL6 UPWIND params: ",params)
    if len(params) == 4:
        return [f"{params[2][0]} = del6(Field({params[1][0]}))"]
    else:
        return [f"{params[3][0]} = del6_masked(Field({params[1][0]}), {params[4][0]})"]

def map_del6(func_call):
    params = func_call["new_param_list"]
    print("DEL6 params: ",params)
    if len(params)>3:
        pexit("optional params not supported\n")
    return [f"{params[2][0]} = del6(Field({params[1][0]}))"]
def map_del2fi_dxjk(func_call):
    params = func_call["new_param_list"]
    return [f"{params[2][0]} = del2fi_dxjk(Field({params[1][0]}))"]
def map_d2fi_dxj(func_call):
    params = func_call["new_param_list"]
    return [f"{params[2][0]} = d2fi_dxj(Field({params[1][0]}))"]
def map_d2f_dxj(func_call):
    params = func_call["new_param_list"]
    return [f"{params[2][0]} = d2f_dxj(Field({params[1][0]}))"]
def map_g2ij(func_call):
    params = func_call["parameters"]
    return [f"{params[2]} = hessian(Field({params[1]}))"]
def map_multmv_mn(func_call):
    params = func_call["parameters"]
    if len(params)>3:
        pexit("optional params not supported\n")
    return [f"{params[2]} = {params[0]}*{params[1]}"]
def map_der_main(func_call):
    params = func_call["new_param_list"]
    if len(params)>4:
        pexit("optional params not supported\n")
    # j = self.evaluate_integer(params[3][0])
    j = int(params[3][0])
    if j == 1:
        return [f"{params[2][0]} = derx(Field({params[1][0]}))"]
    elif j == 2:
        return [f"{params[2][0]} = dery(Field({params[1][0]}))"]
    elif j == 3:
        return [f"{params[2][0]} = derz(Field({params[1][0]}))"]
    print("j=",j)
    pexit("don't know whic der to call")

def map_der2_main(func_call):
    params = func_call["new_param_list"]
    if len(params)>4:
        pexit("optional params not supported\n")
    j = self.evaluate_integer(params[3][0])
    if j == 1:
        return [f"{params[2][0]} = derxx(Field({params[1][0]}))"]
    elif j == 2:
        return [f"{params[2][0]} = deryy(Field({params[1][0]}))"]
    elif j == 3:
        return [f"{params[2][0]} = derzz(Field({params[1][0]}))"]
    print("j=",j)
    pexit("don't know whic der2 to call")

def map_der6_main(func_call):
    params = func_call["new_param_list"]
    ignoredx_str = ""
    upwind_str = ""
    if len(params) == 5 and params[-1][-1] == "ignoredx":
        if params[-1][0] == ".true.":
            ignoredx_str = "_ignore_spacing"
        elif params[-1][0] == ".false.":
            pass
        else:
            pexit("don't know if ignoredx")
    elif len(params) == 4:
        pass
    elif len(params) == 5 and params[-1][-1] == "upwind":
        if params[-1][0] == ".true.":
            upwind_str = "_upwd"
        elif params[-1][0] == ".false.":
            pass
        else:
            pexit("don't know if upwind")
    else: 
        print(params)
        pexit("optional params not supported\n")

    j = int(pc_parser.evaluate_integer(params[3][0]))
    if j == 1:
        return [f"{params[2][0]} = der6x{ignoredx_str}{upwind_str}(Field({params[1][0]}))"]
    elif j == 2:
        return [f"{params[2][0]} = der6y{ignoredx_str}{upwind_str}(Field({params[1][0]}))"]
    elif j == 3:
        return [f"{params[2][0]} = der6z{ignoredx_str}{upwind_str}(Field({params[1][0]}))"]
    print("j=",j)
    pexit("don't know whic der6 to call")
def map_der5(func_call):
    params = func_call["new_param_list"]
    ignoredx_str = ""
    if len(params) == 5 and params[-1][-1] == "ignoredx":
        if params[-1][0] == ".true.":
            ignoredx_str = "_ignore_spacing"
        elif params[-1][0] == ".false.":
            pass
        else:
            pexit("don't know if ignoredx")
    elif len(params) == 4:
        pass
    else: 
        print(params)
        pexit("optional params not supported\n")

    j = int(pc_parser.evaluate_integer(params[3][0]))
    if j == 1:
        return [f"{params[2][0]} = der5x{ignoredx_str}(Field({params[1][0]}))"]
    elif j == 2:
        return [f"{params[2][0]} = der5y{ignoredx_str}(Field({params[1][0]}))"]
    elif j == 3:
        return [f"{params[2][0]} = der5z{ignoredx_str}(Field({params[1][0]}))"]
    print("j=",j)
    pexit("don't know whic der6 to call")

def map_der6_pencil(func_call):
    #these have to be reformulated in Astaroth so return junk for now
    return ["assert(.false.)"]

def map_getderlnrho_z(func_call):
    #these have to be reformulated in Astaroth so return junk for now
    return ["assert(.false.)"]
def map_bval_from_neumann_scl(func_call):
    #these have to be reformulated in Astaroth so return junk for now
    return ["assert(.false.)"]
def map_bval_from_neumann_arr(func_call):
    #these have to be reformulated in Astaroth so return junk for now
    return ["assert(.false.)"]
def map_set_ghosts_for_onesided_ders(func_call):
    #these have to be reformulated in Astaroth so return junk for now
    return ["assert(.false.)"]

def map_gij_etc(func_call):
    params = func_call["new_param_list"]
    subroutine_lines = pc_parser.get_subroutine_lines(func_call["function_name"], f"{pc_parser.directory}/sub.f90")
    sub_parameters = pc_parser.get_parameters(subroutine_lines[0])
    mappings = pc_parser.get_parameter_mapping(sub_parameters,params)
    res = []
    for i, param in enumerate(params):
        mapping = mappings[i]
        #bij
        if mapping == 4:
            bij_line = f"{param[0]} = bij({gen_field3(params[1][0])})"
            res.append(bij_line)
        #del2
        elif mapping == 5:
            del2_line = f"{param[0]} = veclaplace({gen_field3(params[1][0])})"
            res.append(del2_line)
        #graddiv
        elif mapping == 6:
            graddiv_line = f"{param[0]} = gradient_of_divergence({gen_field3(params[1][0])})"
            res.append(graddiv_line)
        elif mapping > 6:
            return ['not_implemented("gij_etc with more than 6 params")']
            pexit("what to do?\n")
    return res

def map_bij_tilde(func_call):
    params = func_call["new_param_list"]
    print(params)
    return ['not_implemented("bij_tilde in sub.f90")']
    #pexit("what to do?\n")

def map_del4graddiv(func_call):
    params = func_call["parameters"]
    print(params)
    #TODO write
    return ["assert(.false.)"]
    ##pexit("what to do?\n")

def map_del6fj(func_call):
    params = func_call["parameters"]
    return [f"{params[3]}  = del6fj(Field({params[2]}), {params[1]})"]
def map_der5i1j(func_call):
    params = func_call["new_param_list"]
    i = int(pc_parser.evaluate_integer(params[3][0]))
    j = int(pc_parser.evaluate_integer(params[4][0]))
    if i == 1 and j == 1:
        return [f"{params[2][0]} = der6x(Field({params[1][0]}))"]
    elif i == 1 and j == 2:
        return [f"{params[2][0]} = der5x1y(Field({params[1][0]}))"]
    elif i == 1 and j == 3:
        return [f"{params[2][0]} = der5x1z(Field({params[1][0]}))"]

    elif i == 2 and j == 1:
        return [f"{params[2][0]} = der5x1y(Field({params[1][0]}))"]
    elif i == 2 and j == 2:
        return [f"{params[2][0]} = der6y(Field({params[1][0]}))"]
    elif i == 2 and j == 3:
        return [f"{params[2][0]} = der5y1z(Field({params[1][0]}))"]


    elif i == 3 and j == 1:
        return [f"{params[2][0]} = der5x1z(Field({params[1][0]}))"]
    elif i == 3 and j == 2:
        return [f"{params[2][0]} = der5y1z(Field({params[1][0]}))"]
    elif i == 3 and j == 3:
        return [f"{params[2][0]} = der6z(Field({params[1][0]}))"]
    print(params)
    pexit("what to do?")
def map_der2i2j2k(func_call):
    params = func_call["new_param_list"]
    return [f"{params[2][0]} = der2i2j2k(Field({params[1][0]}))"]
def map_der4i2j(func_call):
    params = func_call["new_param_list"]
    i = int(pc_parser.evaluate_integer(params[3][0]))
    j = int(pc_parser.evaluate_integer(params[4][0]))
    if i == 1 and j == 1:
        return [f"{params[2][0]} = der6x(Field({params[1][0]}))"]
    elif i == 1 and j == 2:
        return [f"{params[2][0]} = der4x2y(Field({params[1][0]}))"]
    elif i == 1 and j == 3:
        return [f"{params[2][0]} = der4x2z(Field({params[1][0]}))"]

    elif i == 2 and j == 1:
        return [f"{params[2][0]} = der4y2x(Field({params[1][0]}))"]
    elif i == 2 and j == 2:
        return [f"{params[2][0]} = der6y(Field({params[1][0]}))"]
    elif i == 2 and j == 3:
        return [f"{params[2][0]} = der4y2z(Field({params[1][0]}))"]


    elif i == 3 and j == 1:
        return [f"{params[2][0]} = der4z2x(Field({params[1][0]}))"]
    elif i == 3 and j == 2:
        return [f"{params[2][0]} = der4z2y(Field({params[1][0]}))"]
    elif i == 3 and j == 3:
        return [f"{params[2][0]} = der6z(Field({params[1][0]}))"]
    print(params)
    pexit("what to do?")

def map_multmm_sc_mn(func_call):
    params = func_call["parameters"]
    return [f"{params[2]} = multmm_sc_mn({params[0],params[1]})"]

def map_mult_matrix(func_call):
    params = func_call["parameters"]
    return [f"{params[2]} = {params[0]}*{params[1]}"]
def map_der_other(func_call):
    params = func_call["new_param_list"]
    print(params)
    #is actually a normal der call
    if(params[0][0] == 'f'):
        if(params[2][0] == '1'):
            return [f"{params[1][0]} = derx({params[0][0]},{params[2][0]})"]
        if(params[2][0] == '2'):
            return [f"{params[1][0]} = dery({params[0][0]},{params[2][0]})"]
        if(params[2][0] == '3'):
            return [f"{params[1][0]} = derz({params[0][0]},{params[2][0]})"]
        pexit("unknown dim")
    pexit("what to do?\n")

def map_calc_slope_diff_flux(func_call):
    return ['not_implemented("calc_slope_diff_flux")']

def map_multmv_mn_transp(func_call):
    params = func_call["parameters"]
    if len(params) > 3:
        pexit("WHAT TO DO?")
    return [f"{params[2]} = matmul_transpose(params[0],params[1])"]
    




# def map_bval_from_neumann_arr(func_call):
#     params = func_call["parameters"]
#     print(params)
#     pexit("what to do?\n")

# def map_bval_from_neumann_scl(func_call):
#     params = func_call["parameters"]
#     print(params)
#     pexit("what to do?\n")


sub_funcs = {
    "multmv_mn_transp":
    {
      "output_param_indexes": [2],
      "map_func": map_multmv_mn_transp
    },
    "calc_slope_diff_flux":
    {
      "output_param_indexes": [0,1,2,3,4],
      "map_func": map_calc_slope_diff_flux
    },
    # "bval_from_neumann_arr":{
    #     "output_params_indexes": [0],
    #     "map_func": map_bval_from_neumann_arr
    # },
    # "bval_from_neumann_scl":{
    #     "output_params_indexes": [0],
    #     "map_func": map_bval_from_neumann_scl
    # },
    "mult_matrix":{
        "output_params_indexes": [2],
        "map_func": map_mult_matrix
    },
    "multmm_sc_mn":{
        "output_params_indexes": [2],
        "map_func": map_multmm_sc_mn
    },
    "der5i1j":{
        "output_params_indexes": [2],
        "map_func": map_der5i1j
    },
    "del6fj":{
        "output_params_indexes": [3],
        "map_func": map_del6fj
    },
    "del4graddiv":
    {
        "output_params_indexes": [2],
        "map_func": map_del4graddiv
    },
    "bij_tilde":
    {
        "output_params_indexes": [2,3],
        "map_func": map_bij_tilde
    },
    "der_main":
    {
        "output_params_indexes": [2],
        "map_func": map_der_main
    },
    "der_other":
    {
        "output_params_indexes": [1],
        "map_func": map_der_other
    },
    "bval_from_neumann_scl":
    {
        "output_params_indexes": [0],
        "map_func": map_bval_from_neumann_scl
    },
    "set_ghosts_for_onesided_ders":
    {
        "output_params_indexes": [0],
        "map_func": map_set_ghosts_for_onesided_ders
    },
    "bval_from_neumann_arr":
    {
        "output_params_indexes": [0],
        "map_func": map_bval_from_neumann_arr
    },
    "getderlnrho_z":
    {
        "output_params_indexes":[2],
        "map_func":map_getderlnrho_z
    },
    "der2_main":
    {
        "output_params_indexes": [2],
        "map_func": map_der2_main
    },
    "der6_main":
    {
        "output_params_indexes": [2],
        "map_func": map_der6_main
    },
    "der5":
    {
        "output_params_indexes": [2],
        "map_func": map_der5
    },
    "der4i2j":
    {
        "output_params_indexes": [2],
        "map_func": map_der4i2j
    },
    "der2i2j2k":
    {
        "output_params_indexes": [2],
        "map_func": map_der2i2j2k
    },
    "der6_pencil":
    {
        "output_params_indexes": [2],
        "map_func": map_der6_pencil
    },
    "multmv_mn":
    {
        "output_params_indexes": [2],
        "map_func": map_multmv_mn
    },
    "g2ij":
    {
        "output_params_indexes": [2],
        "map_func": map_g2ij
    },
    "d2f_dxj":
    {
        "output_params_indexes": [2],
        "map_func": map_d2f_dxj
    },
    "del4":
    {
        "output_params_indexes": [2],
        "map_func": map_del4
    },
    "del6":
    {
        "output_params_indexes": [2],
        "map_func": map_del6
    },
    "calc_del6_for_upwind":
    {
        "output_params_indexes": [2],
        "map_func": map_calc_del6_for_upwind
    },
    "u_dot_grad_vec":
    {
        "output_params_indexes":  [4],
        "map_func": map_u_dot_grad_vec
    },
    "u_dot_grad_scl":
    {
        "output_params_indexes":  [4],
        "map_func": map_u_dot_grad_scl
    },
    "multm2_sym_mn":
    {
        "output_params_indexes": [1],
        "map_func": map_multm2_sym_mn
    },
    "cross_mn":
    {
        "output_params_indexes": [2],
        "map_func": map_cross_mn
    },
    "curl_mn":
    {
        "output_params_indexes": [1],
        "map_func": map_curl_mn
    },
    "dot_mn":
    {
        "output_params_indexes": [3],
        "map_func": map_dot_mn
    },
    "dot_0":
    {
        "output_params_indexes": [3],
        "map_func": map_dot_mn
    },
    "dot_mn_sv":
    {
        "output_params_indexes": [3],
        "map_func": map_dot_mn
    },
    "dot2_mn":
    {
        "output_params_indexes": [1],
        "map_func": map_dot2_mn
    },
    "del2_main":
    {
        "output_params_indexes": [2],
        "map_func": map_del2_main
    },
    "del2v":
    {
        "output_params_indexes": [2],
        "map_func": map_del2v
    },
    "del4v":
    {
        "output_params_indexes": [2],
        "map_func": map_del4v
    },
    "del6v":
    {
        "output_params_indexes": [2],
        "map_func": map_del6v
    },
    "traceless_strain":
    {
        #only one output, but because of optional params can't know which one
        "output_params_indexes": [1,2],
        "map_func": map_traceless_strain
    },
    "gij":
    {
        "output_params_indexes": [2],
        "map_func": map_gij
    },
    "grad_main":
    {
        "output_params_indexes": [2],
        "map_func": map_grad_main
    },
    "div":
    {
        "output_params_indexes": [2],
        "map_func": map_div
    },
    "div_mn":
    {
        "output_params_indexes": [1],
        "map_func": map_div_mn
    },
    "curl":{
        "output_params_indexes": [2],
        "map_func": map_curl
    },
    "del2v_etc":{
        "output_params_indexes": [2,3,4,5],
        "map_func": map_del2v_etc
    },
    "del2fi_dxjk":{
        "output_params_indexes": [2],
        "map_func": map_del2fi_dxjk
    },
    "d2fi_dxj":{
        "output_params_indexes": [2],
        "map_func": map_d2fi_dxj
    },
    "gij_etc":{
        "output_params_indexes": [4,5,6],
        "map_func": map_gij_etc
    },
}
mk_param_lines = [
    "logical, parameter, dimension(npencils):: lpenc_required  = .false.",
    "logical,            dimension(npencils):: lpenc_diagnos   = .false.",
    "logical,            dimension(npencils):: lpenc_diagnos2d = .false.",
    "logical,            dimension(npencils):: lpenc_video     = .false.",
    "logical,            dimension(npencils):: lpenc_requested = .false.",
    "logical,            dimension(npencils):: lpencil         = .false."
]

# c_helpers = ctypes.CDLL("./helpers.so") 
# c_helpers.hi_from_c.argtypes = [ctypes.c_int]
#transforms .5 -> 0.5 and etc.
#needed since the Astaroth grammar does not support e.g. .5
def normalize_reals(line):
  if line == "":
    return line
  add_indexes = []
  if line[0] == ".":
    line = "0" + line
  for i,char in enumerate(line):
    if i>0 and i<len(line)-1:
      if char in "." and line[i+1].isnumeric() and not line[i-1].isnumeric():
          add_indexes.append(i)
  res_line = ""
  for i,char in enumerate(line):
    if i in add_indexes:
      res_line += "0"
    res_line += char
  return res_line
def pexit(*args):
  for arg in args:
    print(arg)
  print("TP: DEBUG EXIT")
  assert(False)
  exit()
def remove_mod(x):
  remove_indexes = []
  buffer = ""
  start_index = 0
  #by setting it to lower works independent of lower/upper-case
  for j, char in enumerate(x.lower()):
    if char in " []':.;!,/*+-<>=()":
      if "__mod__" in buffer:
        end_index = start_index + len(buffer)-1
        len_after = len(buffer.split("__mod__")[1])
        remove_len = len_after + len("__mod__")
        for i in range(remove_len):
          remove_indexes.append(end_index-i)
        i = start_index
      start_index = j+1
      buffer = ""
    else:
      buffer += char
  if "__mod__" in buffer:
    end_index = start_index + len(buffer)-1
    len_after = len(buffer.split("__mod__")[1])
    remove_len = len_after + len("__mod__")
    for i in range(remove_len):
      remove_indexes.append(end_index-i)

  return "".join([y[1] for y in enumerate(x) if y[0] not in remove_indexes])
def get_mod_name(x,module):
    if "__mod__" not in x:
        return f"{x}__mod__{module}"
    return x
def get_mod_from_physics_name(x):
    if x in ["density","hydro","magnetic","viscosity","gravity","energy","chiral","cosmicray","forcing","shock","special"]:
        return x
    if x == "grav":
        return "gravity"
    if x == "entropy":
        return "energy"
    if x == "eos":
      return "equationofstate"
def unique_list(list):
    res = []
    for x in list:
        if x not in res:
            res.append(x)
    return res
def get_struct_name_from_init(line):
    if "::" in line:
        return line.split("::")[1].strip()
    return line.split(" ")[1].split(",")[0].strip()
def check_if_in_struct(line,current_val):
    if "end" not in line and (line.split(" ")[0].split(",")[0].strip() == "type" and len(line.split("::")) == 1) or (line.split(" ")[0].split(",")[0].strip() == "type" and "(" not in line.split("::")[0]):
        return True
    if current_val and ("endtype" in line.lower().strip()  or "end type" in line.lower().strip()):
        return False
    return current_val
    
def is_contains_line(line,next_line):
    if line.lower().strip() == "contains":
      return True 
    if not next_line:
      return 
    if next_line[0] == "!":
      return False
    return (re.match("function\s+.+\(.+\)",next_line) or re.match("subroutine\s+.+\(.+\)",next_line))
def parse_declaration_value(value):
    num_of_left_brackets = 0
    num_of_right_brackets = 0
    index = 0
    in_end = False
    index = 0
    while not in_end:
        char = value[index]
        if char in "(":
            num_of_left_brackets += 1
        if char in ")":
            num_of_right_brackets += 1
        if char in "," and num_of_left_brackets == num_of_right_brackets:
            in_end = True
        if index<len(value)-1:
            index += 1
        else:
            in_end = True
            index += 1
    return value[:index]

def parse_input_string(param,string_symbol):
        res = []
        char = param[0]
        index = 0
        while index<len(param)-1:
            while char in " ":
                index += 1
                char = param[index]
            #skip '
            if char in f"{string_symbol}":
                index += 1
                buffer = ""
                char = param[index]
                while char not in " ":
                    buffer += char
                    index += 1
                    char = param[index]
                res.append(buffer)
                while char in " ":
                    index += 1
                    char = param[index]
                if index<len(param)-1:
                    #skip '
                    index += 1
                    char = param[index]
                    if index<len(param)-1:
                        #skip ,
                        index += 1
                        char = param[index]

            elif char.isnumeric() and param[index+1] in "*":
                times = eval(char)
                #skip n*'
                index += 3
                buffer = ""
                char = param[index]
                while char not in " ":
                    buffer += char
                    index += 1
                    char = param[index]
                res.extend([buffer for i in range(times)])
                while char in " ":
                    index += 1
                    char = param[index]
                if index<len(param)-1:
                    #skip '
                    index += 1
                    char = param[index]
                    if index<len(param)-1:
                        #skip ,
                        index += 1
                        char = param[index]
            else:
                print(index)
                print(param[:index])
                pexit("huh",param,string_symbol,char)
        return [f"{string_symbol}{x}{string_symbol}" for x in res]
def parse_input_number_part(part):
  if len(part) >= 2 and part[0].isnumeric() and part[1] == "*":
    return [part[2:] for i in range(eval(part[0]))]
  return [part]
def parse_input_number(param):
  res = []
  lists = [parse_input_number_part(part.strip()) for part in param.split(",")]
  for x in lists:
    res.extend(x)
  return res

def parse_input_param(param):
    if param.strip() == "":
        return ["''"]
    if "'" in param and ("," in param or "*" in param):
      return parse_input_string(param,"'")
    if '"' in param and ("," in param or "*" in param):
      return parse_input_string(param,'"')
    if "." in param and ("," in param or "*" in param):
      return parse_input_number(param)
    if not any([x.isdigit() or x == "," for x in param]):
        return [f'"{x.strip()}"' for x in param.split(" ") if x.strip() != ""]
    return param
def get_func_from_trace(trace):
  if "->" not in trace:
    return trace.split()
  return trace.split("->")[-1].strip()
def opposite(value,orig_value):
    if value == ".true.":
        return ".false."
    if value == ".false.":
        return ".true."
    return orig_value 
def split_by_indexes(line,indexes):
    return [line[i:j] for i,j in zip(indexes, indexes[1:]+[None])]

def split_by_ops(line,op_chars):
    num_of_right_brackets = 0
    num_of_left_brackets = 0
    possible_var = ""
    split_indexes = [0]
    in_array = False
    for char_index,char in enumerate(line):
        if char == "(":
            num_of_left_brackets += 1
            if line[char_index-1]:
                back_index = char_index-1
                while(line[back_index] == " "):
                    back_index -= 1
                if line[back_index] not in "*-+-/(":
                    in_array = True
            else:
                if line[char_index-1] not in "*-+-/(":
                    # in_array = possible_var in local_variables or possible_var in self.static_variables 
                    in_array = True
            possible_var = ""
        elif char == ")":
            num_of_right_brackets += 1
            if num_of_left_brackets == num_of_right_brackets:
                in_array = False
            possible_var = ""
        elif (char in op_chars) and not in_array:
            split_indexes.append(char_index)
            possible_var = ""
        else:
            possible_var = possible_var + char
    res = [x for x in split_by_indexes(line,split_indexes) if x != ""]
    for i,val in enumerate(res):
        if val[0] in "+-*/":
            res[i] = val[1:]
    return [x for x in res if x != ""]
    
def okay_stencil_index(index,i):
    if index == ":":
        return True
    if i == 0:
        return index in [global_subdomain_range_x_inner]
        global_subdomain_range_x_
    if i==1:
        return index in [global_loop_y]
    if i==2:
        return index in [global_loop_z]
def is_vector_stencil_index(index):
    if ":" not in index:
        return False
    if index in ["iuxst__mod__cdata:iuzst__mod__cdata","ioxt__mod__cdata:iozt__mod__cdata","ioxst__mod__cdata:iozst__mod__cdata","iuu_sphr__mod__cdata:iuu_sphp__mod__cdata"]:
        return True
    lower,upper= [part.strip() for part in index.split(":")]
    if lower.isnumeric() and upper.isnumeric():
        return int(lower)+2  == int(upper)
    if upper == lower+"+2":
        return True
    lower = remove_mod(lower)
    upper = remove_mod(upper)
    if lower == "iuxt" and upper == "iuzt":
      return True
    if "(" in lower:
        lower_indexes = get_indexes(lower,lower.split("(",1)[0].strip(),0)
        upper_indexes= get_indexes(lower,lower.split("(",1)[0].strip(),0)
        lower = lower.split("(",1)[0].strip()
        upper = upper.split("(",1)[0].strip()
        assert(lower_indexes == upper_indexes)
        assert(len(lower_indexes) == 1)
        return lower[-1] == "x" and upper[-1] == "z"
    if lower[-1] == "x" and upper[-1] == "z":
        return True
    return False
def get_vtxbuf_name_from_index(prefix, index):
    if index.isnumeric():
        return f"{prefix}{pc_parser.pde_names[index]}"
    ## VEC informs that it is a vector access 
    if ":" in index:
        lower,upper= [part.strip() for part in index.split(":")]
        if lower.isnumeric() and upper.isnumeric():
            return f"{prefix}{pc_parser.pde_vec_names[index]}"
        if "(" in lower:
            lower_indexes = get_indexes(lower,lower.split("(",1)[0].strip(),0)
            upper_indexes= get_indexes(lower,lower.split("(",1)[0].strip(),0)
            assert(lower_indexes == upper_indexes)
            assert(len(lower_indexes) == 1)
            lower = lower.split("(",1)[0].strip()
            upper = upper.split("(",1)[0].strip()
            return f"{prefix}{(lower[1:-1]+lower_indexes[0]).upper()}VEC"
        return f"{prefix}{lower[1:-1].upper()}VEC"
    if "(" in index:
        indexes = get_indexes(index, index.split("(",1)[0].strip(),0)
        assert(len(indexes) == 1)
        index = index.split("(",1)[0].strip()
        return f"{prefix}{(index[1:]+indexes[0]).upper()}VEC"
    return f"{prefix}{index[1:].upper()}"

def get_function_call_index(function_call, lines):
    for i, line in enumerate(lines):
        if f"{function_call}(" in line or ("call" in line and function_call in line):
            return i
    pexit(f"Didn't find index for function_call: {function_call}")
def has_balanced_parens(line):
    return sum([char == "(" for char in line]) == sum([char == ")" for char in line])
def replace_exp_once(line):
    num_of_left_brackets = 0
    num_of_right_brackets = 0
    starting_index = 0
    it_index = 1
    while starting_index == 0:
        if line[it_index] == "*" and line[it_index-1] == "*":
            starting_index = it_index-1
        it_index += 1
    forward_index = starting_index + 2
    backward_index = starting_index - 1
    save_index = starting_index - 1
    exponent = ""
    num_of_left_brackets == 0
    num_of_right_brackets == 0
    parse = True
    while parse:
        if line[forward_index] in "([":
            num_of_left_brackets += 1
        if line[forward_index] in ")]":
            num_of_right_brackets += 1
        if (forward_index == len(line)-1 or line[forward_index+1] in " *+-;()/\{}") and num_of_left_brackets == num_of_right_brackets:
            parse = False
            exponent = exponent + line[forward_index]
        if parse:
            exponent = exponent + line[forward_index]
            forward_index += 1
    base = ""
    num_of_left_brackets == 0
    num_of_right_brackets == 0
    parse = True
    while parse:
        if line[backward_index] in "([":
            num_of_left_brackets += 1
        if line[backward_index] in ")]":
            num_of_right_brackets += 1
        if (backward_index == 0 or line[backward_index-1] in " *+-;()/\{}")  and num_of_left_brackets == num_of_right_brackets:
            parse = False
            base = line[backward_index] + base
        if parse:
            base = line[backward_index] + base
            backward_index -= 1
    res = ""
    for i,x in enumerate(line):
        if i<backward_index or i>forward_index:
            res = res + x
        elif i ==backward_index:
            res = res + f"pow({base},{exponent})"
    return res
def replace_exp(line):
    while "**" in line:
        line = replace_exp_once(line)
    return line
def translate_to_DSL(type):
    if type =="real":
        return "real"
    elif type =="logical":
        return "bool"
    elif type =="integer":
        return "int"
    elif type=="character":
        return "char*"
    else:
        pexit("WHAT is the translation for",type,"?")
def common_data(list1, list2):
    result = False
 
    # traverse in the 1st list
    for x in list1:
 
        # traverse in the 2nd list
        for y in list2:
   
            # if one common
            if x == y:
                result = True
                return result
                 
    return result
def get_segment_indexes(segment,line,dims):
    return get_indexes(line[segment[1]:segment[2]],segment[0],dims)
def get_indexes(segment,var,dim):
    index_search = re.search(f"{var}\((.*)\)",segment)
    indexes = [":" for i in range(dim)]
    if index_search:
        indexes = []
        index = ""
        index_line = index_search.group(1)
        num_of_left_brackets = 0
        num_of_right_brackets = 0
        for char in index_line:
            if char == "(": 
                num_of_left_brackets = num_of_left_brackets + 1
            if char == ")":
                num_of_right_brackets = num_of_right_brackets + 1
            if char == "," and num_of_left_brackets == num_of_right_brackets:
                indexes.append(index)
                index = ""
            else:
                index = index + char
        if index.strip() != "" and sum([char == "(" for char in index.strip()]) == sum([char == ")" for char in index.strip()]):
            indexes.append(index.strip())
        else:
            index = index[:-1]
            if index.strip() != "" and sum([char == "(" for char in index.strip()]) == sum([char == ")" for char in index.strip()]):
                indexes.append(index.strip())
            elif index.strip() != "":
                indexes.append(index.split("(")[-1].split(")")[0].strip())
    return indexes
def is_body_line(line):
    return not "::" in line and "subroutine" not in line and not line.split(" ")[0].strip() == "use" and "function" not in line
def is_init_line(line):
    return "::" in line
def add_splits(line):
    res = ""
    for line_part in line.split("\n"):
        res = res + add_splits_per_line(line_part) + "\n"
    return res
def add_splits_per_line(line):
    res = ""
    iterator = 0
    split_modulus = 80
    for char in line:
        res = res + char
        iterator = iterator + 1
        if char == " " and iterator >=split_modulus:
            iterator = 0
            res = res + "&\n"
    return res



def is_use_line(line):
    return "use" == line.split(" ")[0].strip()
def is_declaration_line(line):
    return "::" in line
def new_is_variable_line(line):
    parts = line.split("::")
    if "!" in parts[0]:
        return False
    return len(parts)>1
def is_variable_line(line_elem):
    parts = line_elem.split("::")
    if "!" in parts[0]:
        return False
    return len(parts)>1
def merge_dictionaries(dict1, dict2):
    merged_dict = {}
    merged_dict.update(dict1)
    merged_dict.update(dict2)
    return merged_dict
def get_var_name_segments(line,variables,structs=False):
  buffer = ""
  res = []
  start_index = 0
  num_of_single_quotes = 0
  num_of_double_quotes = 0
  num_of_left_brackets = 0
  num_of_right_brackets = 0
  end_index = -1
  for i,char in enumerate(line):
    if char == "'" and num_of_double_quotes %2 == 0:
      num_of_single_quotes += 1
    if char == '"' and num_of_single_quotes %2 == 0:
      num_of_double_quotes += 1
    if num_of_single_quotes %2 == 0 and num_of_double_quotes %2 == 0: 
      if char  == "(":
        num_of_left_brackets += 1
      if char  == ")":
        num_of_right_brackets += 1
      if char in " ':.;!,/*+-<>=()":
        if buffer and buffer.split("%")[0] in variables:
          buffer = buffer.split("%")[0]
          if structs:
            end_index = i
          else:
            end_index = start_index + len(buffer)
          #inside brackets
          if num_of_left_brackets != num_of_right_brackets:
            nsi = i
            while line[nsi] in " ":
              nsi += 1
            #a named param type so don't change it
            if line[nsi] not in "=":
              res.append((buffer,start_index,end_index))
            else:
              #if == then it is equaility check not named param type
              if line[nsi+1] in "=":
                res.append((buffer,start_index,end_index))
              #expections are do,if,where,forall
              elif len(line)>=3 and line[:3] == "do":
                res.append((buffer,start_index,end_index))
              elif len(line)>=len("forall") and line[:len("forall")] == "forall":
                res.append((buffer,start_index,end_index))
          else:
              res.append((buffer,start_index,end_index))
        start_index=i+1
        buffer = ""
      else:
        buffer += char
    else:
      buffer = ""
  if buffer.strip() in variables:
    res.append((buffer,start_index,len(line)))
  return res
def get_variable_segments(line, variables):
    check_string = ""
    accumulating = False
    start_index = 0
    var = ""
    res = []
    i = 0
    inside_indexes = False
    num_of_left_brackets = 0
    num_of_right_brackets = 0
    num_of_single_quotes = 0
    num_of_double_quotes = 0
    while i < len(line):
        char = line[i]
        if num_of_single_quotes %2 == 0 and char == "'":
          num_of_double_quotes += 1
        if num_of_double_quotes %2 == 0 and char == '"':
          num_of_single_quotes += 1
        if num_of_single_quotes %2 == 0 and num_of_double_quotes %2 == 0:
          if inside_indexes and char == "(":
              num_of_left_brackets += 1
          if inside_indexes and char == ")":
              num_of_right_brackets += 1
              if  num_of_left_brackets == num_of_right_brackets:
                  inside_indexes = False
          if char in " .!,/*+-<>=()" and not inside_indexes:
              if accumulating:
                  accumulating = False
                  if char in "),":
                      res.append((var,start_index,i+1))
                      i = i +1
                  else:
                      res.append((var,start_index,i))
              check_string = ""
              start_index = i + 1
          else:
              check_string = check_string + char
        if check_string in variables:
            if i >= len(line)-1:
                res.append((check_string,start_index,i+1))
                return res
            elif line[i+1] in " .!,/*+-<>=()":
                var = check_string
                if line[i+1] != "(":
                    res.append((var,start_index,i+1))
                elif var.strip() in variables:
                    inside_indexes = True
                    num_of_left_brackets = 1
                    num_of_right_brackets = 0
                    accumulating = True
                    i = i + 1
        i = i + 1
    return res

def replace_variable(line, old_var, new_var):
    check_string = ""
    indexes = []
    for i, char in enumerate(line):
        if char in  " !,/*+-<>=()[];:.":
            check_string = ""
        else:
            check_string = check_string + char
        if check_string == old_var:
            if i == len(line)-1:
                indexes.append((i+1-len(old_var),i+1))
            elif line[i+1] in   " !,/*+-<>=()[];:.":
                indexes.append((i+1-len(old_var),i+1))
    if len(indexes) == 0:
        return line
    res_line = ""
    last_index = 0
    for lower,upper in indexes:
        res_line= res_line+ line[last_index:lower]
        res_line= res_line+ new_var
        last_index = upper
    res_line = res_line + line[indexes[-1][1]:]
    search_var = new_var.split("(")[0].strip()
    new_var_segs = get_variable_segments(res_line,[search_var])
    if len(new_var_segs) == 0:
        return res_line

    seg_out = [res_line[seg[1]:seg[2]] for seg in new_var_segs]
    seg_indexes = [(seg[1],seg[2]) for seg in new_var_segs]
    for seg_index, seg in enumerate(new_var_segs):
        if seg[2] < len(res_line)-1:
            my_bool = res_line[seg[2]] == "(" and res_line[seg[2]-1] == ")"
            if my_bool:
                old_indexes = get_indexes(f"{res_line[seg[1]:seg[2]]})",search_var,-1)
                start_index = seg[2]+1
                end_index = start_index
                num_of_left_brackets = 0
                num_of_right_brackets = 0
                while res_line[end_index] != ")" or (num_of_left_brackets >= num_of_right_brackets + 1):
                    if res_line[end_index] == ")":
                        num_of_right_brackets = num_of_right_brackets + 1
                    if res_line[end_index] == "(":
                        num_of_left_brackets = num_of_left_brackets + 1
                    end_index = end_index + 1
                end_index = end_index + 1
                new_indexes = get_indexes(f"{search_var}({res_line[start_index:end_index]})",search_var,-1)
                combined_indexes = []
                new_indexes_iterator = 0 
                for old_index in old_indexes:
                    if ":" not in old_index:
                        combined_indexes.append(old_index)
                    elif old_index == ":":
                        combined_indexes.append(new_indexes[new_indexes_iterator])
                        new_indexes_iterator = new_indexes_iterator + 1
                    elif new_indexes[new_indexes_iterator] == ":":
                        combined_indexes.append(old_index)
                        new_indexes_iterator = new_indexes_iterator + 1
                    else:
                        old_lower, old_upper = [part.strip() for part in old_index.split(":")]
                        if ":" in new_indexes[new_indexes_iterator]:
                            new_lower, new_upper = [part.strip() for part in new_indexes[new_indexes_iterator].split(":")]
                            combined_indexes.append(f"{new_lower}+({old_lower}-1):{new_upper}+({old_lower}-1)")
                        else:
                            combined_indexes.append(f"{old_lower}+({new_indexes[new_indexes_iterator].strip()}-1)")
                        new_indexes_iterator = new_indexes_iterator + 1
                combined_indexes = [index.strip() for index in combined_indexes]
                sg_res = search_var + "(" 
                for i, index in enumerate(combined_indexes):
                    sg_res = sg_res + index
                    if i<len(combined_indexes)-1:
                        sg_res = sg_res + ","
                sg_res = sg_res + ")"
                seg_out[seg_index] = sg_res
                seg_indexes[seg_index] = (seg[1],end_index)
    last_res_line = ""
    last_index = 0
    for i,index in enumerate(seg_indexes):
        lower,upper = index
        last_res_line= last_res_line+ res_line[last_index:lower]
        last_res_line= last_res_line+ seg_out[i] 
        last_index = upper
    last_res_line = last_res_line + res_line[seg_indexes[-1][1]:]
    return last_res_line 

def second_val(x):
    return x[1]
def parse_variable(line_segment):
    iter_index = len(line_segment)-1
    end_index = iter_index
    start_index = 0
    num_of_left_brackets = 0
    num_of_right_brackets = 0
    while iter_index>0 and (line_segment[iter_index] in " ()" or num_of_left_brackets != num_of_right_brackets):
        elem = line_segment[iter_index]
        if elem == "(":
            num_of_left_brackets += 1
        elif elem == ")":
            num_of_right_brackets += 1
        iter_index -= 1
    end_index = iter_index+1

    while iter_index>0 and line_segment[iter_index] not in " *+-();":
        iter_index -= 1
        
    if iter_index == 0:
        res = line_segment[0:end_index]
    else:
        res = line_segment[iter_index+1:end_index]
    if "%" in res:
        end_index = 0
        while res[end_index] != "%":
            end_index += 1
        res = res[0:end_index]
    return res

def get_used_variables_from_line(line,parse=True):
    characters_to_space= ["/",";",":", ",","+","-","*","(",")","=","<","!",">","[","]","."]
    for character in characters_to_space:
        line = line.replace(character, " ")
    if not parse:
        return [x.strip() for x in line.split(" ")]
    return [parse_variable(part) for part in [x.strip() for x in line.split(" ")]]
def get_used_variables(lines):
    res = []
    for line in lines:
        res.extend(get_used_variables_from_line(line))
    return res
def get_var_segments_in_line(line,variables):
    vars = [var for var in get_used_variables_from_line(line) if var in variables]
    res = get_variable_segments(line, vars)
    return sorted(res, key = second_val)

def get_rhs_segment(line):
    res = []
    index = 0
    start_index = 0
    num_of_single_quotes = 0
    num_of_double_quotes = 0
    num_of_left_brackets = 0
    num_of_right_brackets = 0
    while index<len(line)-1:
        index += 1
        elem = line[index]
        if elem == "'":
            num_of_single_quotes += 1
        elif elem == '"':
            num_of_double_quotes += 1
        elif elem == "=" and line[index-1] not in "<>=!" and line[index+1] not in "<>=!" and num_of_single_quotes%2==0 and num_of_double_quotes%2==0 and num_of_left_brackets == num_of_right_brackets:
            return line[start_index:index]
def get_dims_from_indexes(indexes,rhs_var):
    dims = []
    num_of_looped_dims = 0
    for i,index in enumerate(indexes):
        if ":" in index:
            num_of_looped_dims += 1
        if index == ":":
            dims.append(f"size({rhs_var},dim={i+1})")
        elif ":" in index:
            lower, upper = [var.strip() for var in index.split(":")]
            dims.append(f"{upper}-{lower}+1")
    return (dims, num_of_looped_dims)


def get_new_indexes(segment,var,dim):
    indexes = get_indexes(segment,var,dim) 
    transformed_indexes = 0
    new_indexes = []
    for index in indexes: 
        if index == ":":
            new_indexes.append(f"explicit_index_{transformed_indexes}")
            transformed_indexes = transformed_indexes + 1
        elif ":" in index:
            lower, upper = [var.strip() for var in index.split(":")]
            new_indexes.append(f"{lower} + explicit_index_{transformed_indexes}")
            transformed_indexes = transformed_indexes + 1
        else:
            new_indexes.append(index)
    return new_indexes
def build_new_access(var,new_indexes):
    res = f"{var}("
    for i,index in enumerate(new_indexes):
        res = res + index
        if i < len(new_indexes)-1:
            res = res + ","
    res = res + ")"
    return res

def get_chosen_modules(makefile):
    chosen_modules = {}
    if makefile:
        lines = [line.strip().lower() for line in open(makefile,'r').readlines() if  line.strip() != "" and line.strip()[0] != "#" and line.split("=")[0].strip() != "REAL_PRECISION"] 
        for line in lines:
            if len(line.split("=")) == 2 and all([x not in line for x in ["@",">","_obj","override","flags","_src","$",".x","ld_"]]) and line[0] != "$":
                variable = line.split("=")[0].strip()
                value = line.split("=")[1].strip()
                chosen_modules[variable] = value
                if variable == "density":
                    chosen_modules["density_methods"] = f"{value}_methods"
                if variable == "eos":
                    chosen_modules["equationofstate"] = f"{value}"
                if variable == "entropy":
                    chosen_modules["energy"] = f"{value}"
    return chosen_modules


class Parser:

    def __init__(self, files,config):
        self.pde_index_counter = 1
        self.known_ints = {
            "iux__mod__cdata-iuu__mod__cdata": "0",
            "iuz__mod__cdata-iux__mod__cdata": "2",
            "iuz__mod__cdata-iux__mod__cdata+1": "3",
            "iuz__mod__cdata-iuu__mod__cdata": "2",
            "iuu__mod__cdata": "iux__mod__cdata",
            "iuu__mod__cdata+1": "iuy__mod__cdata",
            "iuu__mod__cdata+2": "iuz__mod__cdata",
            "iux__mod__cdata": "iux__mod__cdata",
            "iux__mod__cdata+1": "iuy__mod__cdata",
            "iux__mod__cdata+2": "iuz__mod__cdata",
            'iux__mod__cdata-iuu__mod__cdata+1': "1",
            'iuz__mod__cdata-iuu__mod__cdata+1': "3",
            f"{global_subdomain_range_x_upper}-{global_subdomain_range_x_lower}+1": global_subdomain_range_x,
            "n1": "1+nghost",
            "iux+1": "iuy",
            "nx": "nx__mod__cparam",
            "ny": "ny__mod__cparam",
            "nz": "nz__mod__cparam",
            "mx": "mx__mod__cparam",
            "my": "my__mod__cparam",
            "mz": "mz__mod__cparam",
            "nx__mod__cparam+2*3": "mx__mod__cparam",
            "ny__mod__cparam+2*3": "my__mod__cparam",
            "nz__mod__cparam+2*3": "mz__mod__cparam",
        }
        self.external_funcs = []
        self.pde_names = {}
        self.auxiliary_fields = []
        self.auxiliary_names = []
        self.pde_vec_names = {}
        self.select_eos_variable_calls = []
        self.farray_register_calls = []
        self.shared_flags_accessed = {}
        self.shared_flags_given = {}
        self.offloading = config["offload"]
        self.include_diagnostics = config["diagnostics"]
        self.known_bools = {
            "iux+0==ilnrho": ".false.",
            "iux+0==ilntt": ".false."
        }
        self.known_dims = {"beta_glnrho_scaled__mod__energy": ["3"]}
        self.test_to_c = config["to_c"]
        self.known_values = {}
        self.inline_num = 0
        self.sample_dir = config["sample_dir"]
        self.offload_type = None
        if config["stencil"]:
            self.offload_type = "stencil"
        elif config["boundcond"]:
            self.offload_type = "boundcond"
        #we don't want print out values
        self.ranges = []
        self.flag_mappings= {}
        self.default_mappings = {
            "headtt__mod__cdata":".false.",
            "ldebug__mod__cdata":".false.",
            "l1dphiavg__mod__cata":".false.",
            "lwrite_phiaverages__mod__cdata":".false.",
            # #for now set off
            # "ltime_integrals__mod__cdata":".false.",
        }
        self.safe_subs_to_remove = ["print","not_implemented","fatal_error","keep_compiler_quiet","warning"]
        self.func_info = {}
        self.file_info = {}
        self.module_info = {}
        self.static_variables = {}
        self.rename_dict = {}
        self.lines = {}
        self.parsed_files_for_static_variables = []
        self.parsed_modules = []
        self.parsed_subroutines = []
        self.loaded_files = []
        self.subroutine_order = 0
        self.used_files = []
        self.found_function_calls = []
        self.used_files = files 
        self.used_static_variables = []
        self.functions_in_file = {}
        self.static_writes = []
        self.directory = config["directory"]
        self.subroutine_modifies_param = {}
        self.struct_table = {}
        self.module_variables = {}
        ignored_files = []
        default_modules = get_chosen_modules(f"{config['sample_dir']}/src/Makefile.src")
        chosen_modules = get_chosen_modules(f"{config['sample_dir']}/src/Makefile.local")
        for mod in default_modules:
            if mod not in chosen_modules:
                chosen_modules[mod] = default_modules[mod]
        self.chosen_modules = chosen_modules
        self.ignored_modules = ["hdf5"]
	#TP: timestep_subcycle.f90 seems to be since it depends on a special module that it does not import
	#TP: lsode_for_chemistry.f90 has only bindings to lsode
        self.ignored_files = ["nodebug.f90","/boundcond_examples/","boundcond_alt.f90", "deriv_alt.f90","diagnostics_outlog.f90","pscalar.f90", "/cuda/", "/obsolete/", "test_field_compress_z.f90","test_flow.z","/inactive/", "/astaroth/", "/pre_and_post_processing/", "/scripts/","file_io_dist.f90","timestep_subcycle.f90","lsode_for_chemistry.f90","timestep_LSODE.f90","solid_cells_ogrid.f90"]
        # self.ignored_files = ["nodebug.f90","/boundcond_examples/","deriv_alt.f90","boundcond_alt.f90", "diagnostics_outlog.f90","pscalar.f90", "/cuda/", "/obsolete/", "/inactive/", "/astaroth/", "/initial_condition/", "/pre_and_post_processing/", "/scripts/"]
        self.ignored_files.append("magnetic_ffreeMHDrel.f90")
        self.ignored_files.append("photoelectric_dust.f90")
        self.ignored_files.append("interstellar_old.f90")
        self.ignored_files.append("spiegel.f90")
        self.ignored_files.append("fourier_fft.f90")
        self.used_files = [file for file in self.used_files if not any([ignored_file in file for ignored_file in self.ignored_files])  and ".f90" in file]
        self.main_program = f"{self.directory}/run.f90"
            
        self.not_chosen_files = []
        for file in self.used_files:
            self.get_lines(file)
        self.used_files = [file for file in self.used_files if not self.file_info[file]["is_program_file"] and file not in self.not_chosen_files]
        self.used_files.append(self.main_program)
        ##Intrinsic functions
        self.ignored_subroutines = ["alog10","count", "min1", "erf","aimag", "cmplx","len", "inquire", "floor", "matmul","ceiling", "achar", "adjustl", "index", "iabs","tiny","dble","float","nullify","associated","nint","open","close","epsilon","random_seed","modulo","nearest","xor","ishft","iand","ieor","ior","random_number","all","any","deallocate","cshift","allocated","allocate","case","real","int","complex","character","if","elseif","where","while","elsewhere","forall","maxval", "minval", "dot_product", "abs", "alog", "mod", "size",  "sqrt", "sum","isnan", "exp", "spread", "present", "trim", "sign","min","max","sin","cos","log","log10","tan","tanh","cosh","sinh","asin","acos","atan","atan2","write","read","char","merge","scan","precision","flush", "adjustr","transpose","repeat"]
        ##Ask Matthias about these
        self.ignored_subroutines.extend(["DCONST","fatal_error", "terminal_highlight_fatal_error","warning","caller","caller2", "coeffsx","coeffsy","coeffsz","r1i","sth1i","yh_","not_implemented","die","deri_3d","u_dot_grad_mat"])
        self.ignored_subroutines.extend(farray_register_funcs)
        self.safe_subs_to_remove.extend(farray_register_funcs)
        
        self.ignored_subroutines.extend(['keep_compiler_quiet','keep_compiler_quiet_r', 'keep_compiler_quiet_r1d', 'keep_compiler_quiet_r2d', 'keep_compiler_quiet_r3d', 'keep_compiler_quiet_r4d', 'keep_compiler_quiet_p', 'keep_compiler_quiet_bc', 'keep_compiler_quiet_sl', 'keep_compiler_quiet_i', 'keep_compiler_quiet_i1d', 'keep_compiler_quiet_i81d', 'keep_compiler_quiet_i2d', 'keep_compiler_quiet_i3d', 'keep_compiler_quiet_l', 'keep_compiler_quiet_l1d', 'keep_compiler_quiet_c', 'keep_compiler_quiet_c1d'])
        self.ignored_subroutines.extend(["helflux", "curflux_ds"])
        self.ignored_subroutines.extend(["cffti","cffti1","cfftf","cfftb","kx_fft","ky_fft"])
        self.ignored_subroutines.extend(["bc_aa_pot2"])
        self.ignored_subroutines.extend(["caller", "caller0", "caller1", "caller2", "caller3", "caller4", "caller5", "caller5_str5"])
        self.ignored_subroutines.append("output_penciled_scal_c")
        self.ignored_subroutines.append("output_penciled_vect_c")
        self.ignored_subroutines.append("output_pencil_scal")
        self.ignored_subroutines.append("output_pencil_vect")
        self.ignored_subroutines.append("output_pencil")
        self.safe_subs_to_remove.append("output_pencil")
        self.ignored_subroutines.append("output_profile")
        self.safe_subs_to_remove.append("output_profile")
        # self.ignored_subroutines.append("loptest")
        self.ignored_subroutines.append("result")
        self.ignored_subroutines.extend(["timing","inevitably_fatal_error"])
        #random functions anyways wonky for multithreading
        self.ignored_subroutines.extend(["random_number_wrapper","random_seed_wrapper"])
        #mpi subs
        self.ignored_subroutines.extend(["mpi_allreduce"])
        #omp subs
        self.ignored_subroutines.extend(["omp_get_thread_num"])
        self.modules_in_scope = {}
        self.file = config["file"]
        # for file in self.used_files:
        #     self.get_lines(file)
    def replace_variables_multi(self,line, new_vars,structs=False):
        segments = get_var_name_segments(line,new_vars,structs)
        res =  self.replace_segments(segments,line,self.map_val_func,{},{"map_val": [new_vars[x[0]] for x in segments]})
        return res
    def replace_variables_multi_array_indexing(self,line,new_vars):
        #TODO: change if is a local variable instead of global one
        vars = [x.split("(")[0].strip() for x in new_vars]
        segments = self.get_array_segments_in_line(line,{var: self.static_variables[var] for var in vars})
        return self.replace_segments(segments,line,self.map_val_func,{},{"map_val": [new_vars[line[x[1]:x[2]]] for x in segments]})


    def map_to_new_index(self,index,i,local_variables,line,possible_values=[],make_sure_indexes_safe=False):
        if ":" in index:
            self.ranges.append((index,i,line))
        if index == ":":
            if i==0:
                return "idx.x"
            elif i==1:
                return "idx.y"
            elif i==2:
                return "idx.z"
            else:
                pexit("whole range in last f index?",line)
        lower, upper = [part.strip() for part in index.split(":")]
        if index == "idx.x":
            return "idx.x"
        if index == "idx.y":
            return "idx.y"
        if index == "idx.z":
            return "idx.z"

        if index == "1:mx":
            pexit("what to do?")
            return "idx.x"
        if index == "1:nx":
            return "idx.x-NGHOST"
        if index == "1:nx__mod__cparam":
            return "idx.x-NGHOST"

        if index == "1:my":
            pexit("what to do?")
            return "idx.y"
        if index == "1:ny":
            return "idx.y-NGHOST"

        if index == "1:mz":
            pexit("what to do?")
            return "idx.z"
        if index == "1:nz":
            return "idx.z-NGHOST"



        elif index == global_subdomain_range_x_inner:
            return "idx.x"
        elif index == f"{global_subdomain_range_x_lower}+1:{global_subdomain_range_x_upper}+1":
            return "idx.x+1"
        elif index == f"{global_subdomain_range_x_lower}+2:{global_subdomain_range_x_upper}+2":
            return "idx.x+2"
        elif index == f"{global_subdomain_range_x_lower}+3:{global_subdomain_range_x_upper}+3":
            return "idx.x+3"
        elif index == f"{global_subdomain_range_x_lower}-1:{global_subdomain_range_x_upper}-1":
            return "idx.x-1"
        elif index == f"{global_subdomain_range_x_lower}-2:{global_subdomain_range_x_upper}-2":
            return "idx.x-2"
        elif index == f"{global_subdomain_range_x_lower}-3:{global_subdomain_range_x_upper}-3":
            return "idx.x-3"
        pexit("How to handle f index?: ",index,line)
    def write_world_out(self, variables):
        file = open("world.h", "w")
        file.write("typedef struct {\n")
        for var in variables:
            var_info = variables[var]
            c_type = translate_to_DSL(var_info["type"])
            dims_str = ""
            for dim in var_info["dims"]:
                dims_str += f"[{dim}]"
            file.write(f"{c_type} {var} {dims_str};\n")
        file.write("} WORLD;")
        file.write("constexpr WORLD world = {\n")
        for var in variables:
            var_info = variables[var]
            dims = var_info["dims"]
            if(len(dims) == 0):
                val = var_info["value"]
                file.write(f"{val},\n")
            elif(len(dims) == 1):
                res = "{"
                assert(dims[0].numeric())
                for i in range(int(dims[0])):
                    index = i+1
                    index_val = self.flag_mappings[f"{var}({index})"]
                    res += f"{index_val},"
                res += "},\n"
            else:
                pexit("") 

        file.write("};\n")

                

    def evaluate_leftover_pencils_as_true(self,lines,local_variables):
      #we assume that the leftover pencils are used to calculated df etc.
      for line_index,line in enumerate(lines):
        if "if" in line and "(" in line:
          if_calls = [call for call in self.get_function_calls_in_line(line,local_variables) if call["function_name"] == "if"]
          if len(if_calls) == 1 and len(if_calls[0]["parameters"]) == 1 and "lpencil__mod__cparam" in if_calls[0]["parameters"][0]:
            lines[line_index] = self.replace_func_call(line,if_calls[0],"if(.true.)")
      return lines

    def evaluate_integer(self,value):
      
        #pencil specific values known beforehand
        if value in self.known_ints:
            value = self.known_ints[value]
        for mapping in self.flag_mappings:
            if mapping in value:
                value = replace_variable(value,mapping,self.flag_mappings[mapping])
        if value in self.known_values:
            orig_value = value
            while value in self.known_values:
                value = self.known_values[value]
        return simplify(value)

    def evaluate_indexes(self,value):
        if value == ":":
            return ":"
        if ":" in value:
            lower,upper = [part.strip() for part in value.split(":")]
            lower = self.evaluate_integer(lower)
            upper = self.evaluate_integer(upper)
            #remove unnecessary ranges
            if lower == upper:
                return lower
            return f"{lower}:{upper}"
        return self.evaluate_integer(value)
    def evaluate_boolean(self,value,local_variables,func_calls):
        val = self.evaluate_boolean_helper(value,local_variables,func_calls)
        if val not in [".true.", ".false."]:
            return value
        return val
    def evaluate_boolean_helper(self,value,local_variables,func_calls):
        if value in self.known_bools:
            return self.known_bools[value]
        if value == f"{global_subdomain_range_with_halos_x}>{global_subdomain_range_x}":
            return ".true."
        #don't want to parse braces yet
        if "(" in value:
            #for now if multiple func calls skip
            if len(func_calls)>0:
                  variables = merge_dictionaries(self.static_variables,local_variables)
                  func_calls = [call for call in self.get_function_calls_in_line(value,variables) if call["function_name"] != "if"]
                  for call in func_calls:
                    if call["function_name"] == "minval":
                        new_value = self.replace_func_call(value,call,"ldo_not_know")
                        return self.evaluate_boolean_helper(new_value,local_variables,func_calls)
                  return value
            start_index = -1
            end_index = -1
            #gets the most nested braces
            for i,char in enumerate(value):
                if char == "(" and end_index<0 and (value[i-1] in ".( " or i==0) :
                    start_index = i
                if char == ")" and end_index<0 and start_index>-1:
                    end_index = i
            if start_index > -1 and end_index >-1:
                inside = self.evaluate_boolean_helper(value[start_index+1:end_index],local_variables,func_calls)
                res_value = value[:start_index]
                if inside in [".true.",".false."]:
                    res_value += inside
                else:
                    res_value += "ldo_not_know"
                res_value += value[end_index+1:]
                res = self.evaluate_boolean_helper(res_value,local_variables,func_calls)
                value = res
        if ".and." in value:
            parts = [self.evaluate_boolean_helper(part.strip(),local_variables,func_calls) for part in value.split(".and.")]
            all_true = all([part == ".true." for part in parts])
            some_false = any([part == ".false." for part in parts])
            if all_true:
                return ".true."
            elif some_false:
                return  ".false."
            else:
                return value
        elif ".or." in value:
            parts = [self.evaluate_boolean_helper(part.strip(),local_variables,func_calls) for part in value.split(".or.")]
            some_true= any([part == ".true." for part in parts])
            all_false= all([part == ".false." for part in parts])
            if some_true:
                return ".true."
            elif all_false:
                return  ".false."
            else:
                return value
        elif ".not." in value:
            return opposite(self.evaluate_boolean_helper(value.replace(".not.","").strip(),local_variables,func_calls),value)
        if ">=" in value and len(value.split(">=")) == 2:
            is_equal = self.evaluate_boolean_helper(value.replace(">=","==",1),local_variables,func_calls)
            is_more = self.evaluate_boolean_helper(value.replace(">=",">",1),local_variables,func_calls)
            if is_equal == ".true." or is_more == ".true.":
                return ".true."
            if is_equal == ".false." and is_more == ".false.":
                return ".false."
            return value
        if "<=" in value and len(value.split("<=")) == 2:
            return opposite(self.evaluate_boolean_helper(value.replace("<=",">",1),local_variables,func_calls),value)
        if ">" in value and len(value.split(">")) == 2:
            lhs, rhs = [part.strip() for part in value.split(">")]

            if lhs in self.known_values:
                lhs = self.known_values[lhs]
            if rhs in self.known_values:
                rhs = self.known_values[rhs]

            #integer and float comparison
            if lhs.replace(".","").replace("-","").isnumeric() and rhs.replace(".","").replace("-","").isnumeric():
                return ".true." if ((eval(lhs)-eval(rhs)) > 0) else ".false."
            else:
                return value
        if "<" in value and len(value.split("<")) == 2:
            return opposite(self.evaluate_boolean_helper(value.replace("<",">=",1),local_variables,func_calls),value)
        if "==" in value and len(value.split("==")) == 2:
            lhs, rhs = [part.strip() for part in value.split("==")]
            if lhs in self.known_values:
                lhs = self.known_values[lhs]
            if rhs in self.known_values:
                rhs = self.known_values[rhs]

            if lhs == rhs:
                return ".true."
            if lhs==".false." and rhs == ".true.":
              return ".false."
            if lhs == ".true." and rhs == ".false.":
              return ".false."
            if lhs == ".false." and rhs == ".false.":
              return ".true."
            if lhs == ".true." and rhs == ".true.":
              return ".true"
            elif lhs.isnumeric() and rhs.isnumeric() and lhs != rhs:
                return ".false."
            elif lhs == "0." and rhs == "0.0":
                return ".true."
            #integer and float comparison
            elif is_number(lhs) and is_number(rhs):
                if lhs in ["3.908499939e+37","3.90849999999999991e+37","3.90849994e+37","impossible__mod__cparam"]:
                    lhs = impossible_val
                if "e+37" in lhs:
                    lhs = impossible_val
                if "e37" in lhs:
                    lhs = impossible_val
                if rhs in ["3.908499939e+37","3.90849999999999991e+37","3.90849994e+37","impossible__mod__cparam"]:
                    rhs = impossible_val
                if "e+37" in rhs:
                    rhs = impossible_val
                if "e37" in rhs:
                    rhs = impossible_val
                if eval(rhs)-eval(lhs) == 0.0:
                    return ".true."
                else:
                    return ".false."
            #string comparison
            elif lhs[0] == "'" and lhs[-1] == "'" and rhs[0] == "'" and rhs[-1] == "'":
                if lhs == rhs:
                    return ".true."
                else:
                    return ".false."
            #TP: should not be needed anymore since we know the value of top and bot,nxgrid etc.
            # #TOP /= BOT
            # elif lhs == "top" and rhs == "bot":
            #     return ".false."
            #no less than 3d runs
            # elif lhs in ["nxgrid","nygrid","nzgrid","nwgrid__mod__cparam"] and rhs in ["1"]:
            #     return ".false."
            else:
                return value
        if "/=" in value and len(value.split("/=")) == 2:
            return opposite(self.evaluate_boolean_helper(value.replace("/=","==",1),local_variables,func_calls),value)
        return value
    def find_module_files(self, module_name):
        #external library
        if module_name not in self.module_info:
          return []
        return self.module_info[module_name]["files"]

    def find_subroutine_files(self, subroutine_name):
        #Workaround since don't have time to copypaste since some files say they support functions which they don't support
        if "interface_funcs" in self.func_info[subroutine_name]:
            return self.func_info[subroutine_name]["files"]
        res = [file for file in self.func_info[subroutine_name]["files"] if file in self.func_info[subroutine_name]["lines"]]
        res.extend([file for file in self.func_info[subroutine_name]["lines"]])
        return res
        # return [file for file in self.func_info[subroutine_name]["files"] if file in self.func_info[subroutine_name]["lines"]]

    def parse_module(self, module_name):
        if module_name not in self.parsed_modules:
            self.parsed_modules.append(module_name)
            file_paths = self.find_module_files(module_name)
            for file_path in file_paths:
                self.parse_file_for_static_variables(file_path)


    def get_array_segments_in_line(self,line,variables):
        array_vars = self.get_arrays_in_line(line,variables)
        res = get_variable_segments(line, array_vars)
        return sorted(res, key = second_val)
    def get_struct_segments_in_line(self,line,variables):
        search_vars = []
        save_var = False
        buffer = ""
        for char in line:
            if char == "%":
                save_var = True
            if not(char.isalpha() or char.isnumeric()) and char not in "%_":
                if save_var:
                    search_vars.append(buffer.strip())
                    save_var = False
                buffer = ""
            else:
                buffer = buffer + char
        if save_var:
            search_vars.append(buffer.strip())
        return [seg for seg in get_variable_segments(line,search_vars) if seg[0] != ""]
    def get_arrays_in_line(self,line,variables):
        variables_in_line= get_used_variables_from_line(line)
        res = []
        for var in variables_in_line:
            if var in variables and len(variables[var]["dims"]) > 0:
                res.append(var)
        return res


    def parse_line(self, line):
        if len(line) == 0:
            return line
        if line[0] == "!":
            return line.strip().lower()
        if "!" not in line:
            return line.strip().lower()
        ## remove comment at end of the line
        iter_index = len(line)-1;
        num_of_single_quotes = 0
        num_of_double_quotes = 0
        for iter_index in range(len(line)):
            if line[iter_index] == "'":
                num_of_single_quotes += 1
            if line[iter_index] == '"':
                num_of_double_quotes += 1
            if line[iter_index] == "!" and num_of_single_quotes%2 == 0 and num_of_double_quotes%2 == 0 and (iter_index+1==len(line) or line[iter_index+1] != "="):
                line = line[0:iter_index]
                break                          
        return line.strip().lower()
    def add_write(self, variable, line_num, is_local, filename, call_trace, line):
        self.static_writes.append({"variable": variable, "line_num": line_num, "local": is_local, "filename": filename, "call_trace": call_trace, "line": line})
        if is_local and variable:
          if get_func_from_trace(call_trace) not in ["div","get_reaction_rate"]:
            pass
          else:
            checked_local_writes.append({"variable": variable, "line_num": line_num, "local": is_local, "filename": filename, "call_trace": call_trace, "line": line})

    def set_optional_param_not_present(self,lines,param):
        for line_index,_ in enumerate(lines):
            lines[line_index] =  lines[line_index].replace(f"present({param})",".false.")
        return lines
    def get_boundcond_func_calls(self,boundcond_func,boundconds_map):
        mcom = int(self.static_variables["mcom__mod__cparam"]["value"])
        boundcond_module = self.chosen_modules["boundcond"]
        boundcond_file = f"{self.directory}/{boundcond_module}.f90"
        boundcond_lines = self.get_subroutine_lines(boundcond_func,boundcond_file)
        #stupid workaround to take the default ivar1,ivar2
        boundcond_lines = self.set_optional_param_not_present(boundcond_lines,"ivar1_opt")
        boundcond_lines = self.set_optional_param_not_present(boundcond_lines,"ivar2_opt")
        local_variables = {parameter:v for parameter,v in self.get_variables(boundcond_lines, {},boundcond_file,True).items() }
        boundcond_lines = self.rename_lines_to_internal_names(boundcond_lines,local_variables,boundcond_file)
        boundcond_lines = self.eliminate_while(boundcond_lines,unroll=True)

        all_are_shrear = True
        for index in range(1,mcom+1):
            all_are_shrear = all_are_shrear and self.flag_mappings[f"{boundconds_map}({index},1)"] == 'she'
            all_are_shrear = all_are_shrear and self.flag_mappings[f"{boundconds_map}({index},2)"] == 'she'
        if all_are_shrear:
            for line_index,_ in enumerate(boundcond_lines):
                boundcond_lines[line_index] = boundcond_lines[line_index].replace(f"all({boundconds_map}(1:{mcom},:)=='she')",".true.")
        else:
            for line_index,_ in enumerate(boundcond_lines):
                boundcond_lines[line_index] = boundcond_lines[line_index].replace(f"all({boundconds_map}(1:{mcom},:)=='she')",".false.")
        boundcond_lines = self.eliminate_while(boundcond_lines,unroll=True)
        boundcond_func_calls = self.get_function_calls(boundcond_lines,local_variables)
        file = open("boundcond-res.txt","w")
        for line in boundcond_lines:
            file.write(f"{line}\n")
        file.close()
        return boundcond_func_calls

    def get_pars_lines(self,suffix, name, lines):
        tmp_lines = []
        for line in lines:
            tmp_lines.extend(split_line_nml(line))
        lines = tmp_lines
        res = []
        in_pars = False
        for line in lines:
            line = line.strip().lower()
            if in_pars and line == "/":
                in_pars = False
            if in_pars:
                res.append(line)
            if f"&{name}{suffix}_pars" in line:
                in_pars = True
                res.append(line.replace(f"&{name}{suffix}_pars","").replace("/","").strip())
        #/ is replaced since it represents the end of line
        return [line.replace("/","").strip() for line in res]
    def get_flags_from_lines(self,lines):


        for module in ["grav","density","magnetic","hydro","entropy","viscosity","eos","chiral","cosmicray","forcing","shock","special",""]:
            if module:
                suffixes= ["_init","_run"]
            else:
                suffixes= ["init","run"]
            for suffix in suffixes :
                module_lines = self.get_pars_lines(suffix, module,lines)
                res_lines = []
                for line in module_lines:
                    res_line = line
                    if len(res_line) >0 and res_line[-1] == ",":
                      res_line = res_line[:-1]
                    res_lines.append(res_line)
                module_lines = res_lines
                writes = self.get_writes(module_lines,False)
                if module:
                    mod = get_mod_from_physics_name(module)
                    for write in writes:
                        if write["variable"] in self.rename_dict[mod]:
                            write["variable"] = self.rename_dict[mod][write["variable"]]
                        else:
                            pos_mod = self.get_module_where_declared(write["variable"],self.get_module_file(mod))
                            write["variable"] = self.rename_dict[pos_mod][write["variable"]]
                else:
                    for write in writes:
                        pos_mod = self.get_module_where_declared(write["variable"],f"{self.directory}/param_io.f90")
                        write["variable"] = self.rename_dict[pos_mod][write["variable"]]
                for write in writes:
                    if write["value"] == "t":
                        self.flag_mappings[write["variable"]] = ".true."
                    elif write["value"] == "f":
                        self.flag_mappings[write["variable"]] = ".false."
                    elif "'" in write["value"] and "," not in write["value"] and "*" not in write["value"]:
                        parsed_value = "'" + write["value"].replace("'","").strip() +"'"
                        self.flag_mappings[write["variable"]] = parsed_value
                    #impossible value
                    elif write["value"] in ["3.908499939e+37","3.90849999999999991e+37"]:
                        self.flag_mappings[write["variable"]] = "impossible__mod__cparam"
                    else:
                      self.flag_mappings[write["variable"]] = write["value"]
        for flag in self.flag_mappings:
            if(self.flag_mappings[flag] == ""):
                self.flag_mappings[flag] == "''"


    def get_variables_from_line(self, line,line_num,variables,filename,module,local,in_public):
        parts = [part.strip() for part in line.split("::")]
        start =  parts[0].strip()
        type = start.split(",")[0].strip()
        #public declaration not a variable declaration
        is_public_declaration = False
        if type == "public":
            is_public_declaration = True
        allocatable = "allocatable" in [x.strip() for x in start.split(",")]
        saved_variable = "save" in [x.strip() for x in start.split(",")]
        public = "public" in [x.strip() for x in start.split(",")]
        is_parameter = "parameter" in [x.strip() for x in start.split(",")]
        is_pointer = "pointer" in [x.strip() for x in start.split(",")]
        public_attribute = "public" in [x.strip() for x in start.split(",")]
        writes = []
        if is_parameter or "=" in line:
            #could be used to get the correct value for nwgrid but breaks other things for some reason
            # #done for example to get the correct value for nwgrid
            # int_calls = [call for call in self.get_function_calls_in_line(line,{}) if call["function_name"] == "int"]
            # if "kind=ikind8" in line and "int(" in line:
            #     print(line)
            #     print(int_calls)
            #     print(self.get_function_calls_in_line(line,{}))
            #     # pexit("dfd")
            # while(len(int_calls) > 0):
            #     line = self.replace_func_call(line,int_calls[0],int_calls[0]["parameters"][0])
            #     int_calls = [call for call in self.get_function_calls_in_line(line,{}) if call["function_name"] == "int"]
            #     # pexit(line)

            writes = self.get_writes_from_line(line)
        is_optional = "optional" in [x.strip() for x in start.split(",")]
        dimension = []
        if "dimension" in start:
            _, end = [(m.start(0), m.end(0)) for m in re.finditer("dimension", start)][0]
            while start[end] != "(":
                end = end + 1
            rest_of_line = start[end+1:]
            num_of_left_brackets = 1
            num_of_right_brackets= 0
            index = ""
            i = 0
            while num_of_left_brackets != num_of_right_brackets:
                char = rest_of_line[i]
                if char == "(":
                    num_of_left_brackets = num_of_left_brackets + 1
                if char == ")":
                    num_of_right_brackets = num_of_right_brackets + 1
                if char == "," and num_of_left_brackets == num_of_right_brackets + 1: 
                    dimension.append(index.strip())
                    index = ""
                else:
                    index = index + char
                i = i +1
            index = index[:-1]
            dimension.append(index.strip())
        line_variables = parts[1].strip()
        variable_names = [name.lower().strip().split("(")[0].strip() for name in self.get_variable_names_from_line(line_variables)]
        if is_public_declaration or in_public or public_attribute:
            if "public_variables" not in self.module_info[module]:
                self.module_info[module]["public_variables"] = []
            for name in variable_names:
                self.module_info[module]["public_variables"].append(name)
        if is_public_declaration:
          return
        #add internal names first for offloading
        if module and not local:
            for name in variable_names:
                self.rename_dict[module][name] = get_mod_name(name,module)
                # print(line)
                # print(filename)
                assert(module != "density" or name != "chi")
            #don't rewrite local variables
            variable_names = [get_mod_name(name,module) for name in variable_names]
            if writes:
              for x in writes:
                x["variable"] = get_mod_name(x["variable"],module)
        for i, variable_name in enumerate(variable_names):
            dims = dimension
            search = re.search(f"{remove_mod(variable_name)}\(((.*?))\)",line) 
            ## check if line is only specifying intent(in) or intent(out)
            if search:
                dims = [index.strip() for index in search.group(1).split(",")]
            #if "profx_ffree" in line:
            #    print(variable_name)
            #    print(line)
            #    print(dims)
            ## get first part of .split(" ") if f.e. character len(something)
            type = type.split(" ")[0].strip()
            #filter intent(in) and intent(inout)
            if "intent(" not in type and ".true." not in variable_name and ".false." not in variable_name:
                #if struct type parse more to get the type
                if "type" in type:
                    struct_type = re.search("\((.+?)\)",line).group(1)
                    type = struct_type
                if "(" in type:
                    type = type.split("(")[0].strip()
                if "rkind8" in line and type == "real":
                    type = "double"
                type = type.lower()
                if dims in [["my"],["ny"]]:
                    profile_type = "y"
                elif dims in [["mx"],["nx"]]:
                    profile_type = "x"
                elif dims in [["mz"],["nz"]]:
                    profile_type = "z"
                elif dims in [["mx","3"],["nx","3"]]:
                    profile_type = "x_vec"
                elif dims in [["my","3"],["ny","3"]]:
                    profile_type = "y_vec"
                elif dims in [["mz","3"],["nz","3"]]:
                    profile_type = "z_vec"
                elif dims in [["mx","my"],["nx","ny"]]:
                    profile_type = "xy"
                elif dims in [["mx","my","3"],["nx","ny","3"]]:
                    profile_type = "xy_vec"
                elif dims in [["mx","mz","3"],["mz","nz","3"]]:
                    profile_type = "xz_vec"
                elif dims in [["mx","mz","3"],["my","nz","3"]]:
                    profile_type = "xz_vec"
                elif dims in [["nx","ny","nz","3"]]:
                    profile_type = "glob_n_vec"
                else:
                    profile_type = None
                #var_object = {"type": type, "dims": dims, "allocatable": allocatable, "origin": [filename], "public": public, "threadprivate": False, "saved_variable": (saved_variable or "=" in line_variables.split(",")[i]), "parameter": is_parameter, "on_target": False, "optional": is_optional, "line_num": line_num, "profile_type": profile_type, "is_pointer": is_pointer}
                #TP for run_const analysis do not include saved_variables
                var_object = {"type": type, "dims": dims, "allocatable": allocatable, "origin": [filename], "public": public, "threadprivate": False, "saved_variable": False, "parameter": is_parameter, "on_target": False, "optional": is_optional, "line_num": line_num, "profile_type": profile_type, "is_pointer": is_pointer}
                if is_parameter or "=" in line:
                  var_writes = [x for x in writes if x["variable"] == variable_name and "kind" not in x["value"]]
                  if len(var_writes) == 1 and dims == []:
                    var_object["value"] = var_writes[0]["value"]
                  elif len(var_writes) == 1 and len(dims) == 1:
                    var_object["array_value"] = var_writes[0]["value"]
                  elif variable_name in ["nwgrid","nwgrid__mod__cparam"]:
                    var_object["value"] = "nxgrid__mod__cparam*nygrid__mod__cparam*nzgrid__mod__cparam"
                    # if type in ["integer","real"]:
                    #   var_object["value"] = f"({var_object['value']})"
                if variable_name not in variables:
                    variables[variable_name] = var_object
                else:
                    variables[variable_name]["origin"].append(filename)
                    #NOTE: unsafe hack rework later
                    #if variables with same name assume that the pointer will point to this later
                    if variables[variable_name]["dims"] == [":"]:
                        variables[variable_name]["dims"] = dims
                    variables[variable_name]["origin"] = unique_list(variables[variable_name]["origin"])
                if module and not local:
                    if module not in self.module_variables:
                        self.module_variables[module] = {}
                    if variable_name not in self.module_variables[module]:
                        self.module_variables[module][variable_name] = var_object
                    else:
                        self.module_variables[module][variable_name]["origin"].append(filename)
                        self.module_variables[module][variable_name]["origin"] = unique_list(variables[variable_name]["origin"])
                if filename and not local:
                    if filename not in self.file_info:
                        self.file_info[filename] = {}
                    if variable_name not in self.file_info[filename]["variables"]:
                        self.file_info[filename]["variables"][variable_name] = var_object

    def get_lines(self, filepath, start_range=0, end_range=math.inf, include_comments=False):
        if not os.path.isfile(filepath):
            return []
        if filepath not in self.lines.keys():
            in_sub_name = None 
            lines = []
            start = False
            has_module_line = False
            has_program_line = False
            in_static_variables_declaration = False
            in_struct = False
            in_public = True 
            struct_name = ""
            read_lines = []


            with open(filepath,"r") as file:
                for x in file:
                    line = self.parse_line(x)
                    read_lines.append(line)
                    match = re.search("include\s+(.+\.(h|inc))",line)
                    if match and line[0] != "!":
                        header_filename = match.group(1).replace("'","").replace('"',"")
                        header_filepath = filepath.rsplit("/",1)[0].strip() + "/" + header_filename
                        if os.path.isfile(header_filepath):
                            with open(header_filepath, 'r') as header_file:
                                for x in header_file:
                                    read_lines.append(self.parse_line(x))
            index = 0
            self.file_info[filepath] = {"is_program_file": False,"used_modules":{},"variables": {}}
            while index<len(read_lines):
                line = read_lines[index]
                start = False
                if len(line)>0:
                    #take care the case that the line continues after end of the line
                    if line[-1] == "&" and line[0] != "!":
                        while line[-1] == "&":
                            index += 1
                            next_line = read_lines[index].strip()
                            if len(next_line)>0 and next_line[0] != "!":
                                line = (line[:-1] + " " + self.parse_line(next_line)).strip()
                    # split multilines i.e. fsdfaf;fasdfsdf; into their own lines for easier parsing
                    if ";" in line:
                        parts = self.split_line(line)
                    else:
                        parts = [line]
                    for line in parts:
                        if line[0] != '!':
                            search_line = line.lower().strip()

                            if len(search_line.split(" ")) == 2 and search_line.split(" ")[0] == "external":
                                func_name = search_line.split(" ")[1].strip()
                                self.func_info[func_name] = {"files": [], "lines": {filepath: []}}
                                self.external_funcs.append(func_name)
                            if "program" in search_line:
                                has_program_line = True
                                self.file_info[filepath]["is_program_file"] = True
                                #don't need to rest since this file will not be used
                                if filepath != self.main_program:
                                    return
                            if search_line.split(" ")[0].strip() == "module" and search_line.split(" ")[1].strip() != "procedure":
                                has_module_line = True
                                in_static_variables_declaration = True
                                module_name = search_line.split(" ")[1].strip()
                                #choose only the chosen module files
                                file_end = filepath.split("/")[-1].split(".")[0].strip()
                                if module_name in ["special","density","energy","hydro","gravity","viscosity","poisson","weno_transport","magnetic","deriv"] and file_end != self.chosen_modules[module_name]:
                                  self.not_chosen_files.append(filepath)
                                  return
                                elif "deriv_8th" in filepath: 
                                  print("HMM: ",self.chosen_modules[module_name])
                                  pexit("WRONG", filepath)
                                if module_name not in self.module_info:
                                    self.module_info[module_name] = {"public_variables": []}
                                    if module_name not in self.rename_dict:
                                        self.rename_dict[module_name] = {}
                                if "files" not in self.module_info[module_name]:
                                    self.module_info[module_name]["files"] = []
                                self.module_info[module_name]["files"].append(filepath)
                                if filepath not in self.file_info:
                                    self.file_info[filepath] = {}
                                self.file_info[filepath]["module"] = module_name
                            if not in_sub_name:
                                search = re.search(f"\s?subroutine\s*([a-zA-Z0-9_-]*?)($|\s|\()", search_line)
                                if(search and "subroutine" in search_line):
                                    sub_name = search.groups()[0].strip()
                                    if sub_name not in self.func_info:
                                        self.func_info[sub_name] = {}
                                    if "lines" not in self.func_info[sub_name]:
                                        self.func_info[sub_name]["lines"] = {}
                                    self.func_info[sub_name]["lines"][filepath] = []
                                    if "files" not in self.func_info[sub_name]:
                                        self.func_info[sub_name]["files"] = []
                                    in_sub_name = sub_name
                                    start = True
                                search = re.search(f"\s?function\s*([a-zA-Z0-9_-]*?)($|\s|\()", search_line)
                                if(not in_sub_name and search and "function" in search_line):
                                    sub_name = search.groups()[0].strip()
                                    if sub_name not in self.func_info:
                                        self.func_info[sub_name] = {}
                                    if "lines" not in self.func_info[sub_name]:
                                        self.func_info[sub_name]["lines"] = {}
                                    self.func_info[sub_name]["lines"][filepath] = []
                                    if "files" not in self.func_info[sub_name]:
                                        self.func_info[sub_name]["files"] = []
                                    in_sub_name = sub_name
                                    start = True
                            if in_sub_name and not start:
                                if("end subroutine" in search_line or "endsubroutine" in search_line or "end function" in search_line or "endfunction" in search_line):
                                    self.func_info[in_sub_name]["lines"][filepath].append(self.parse_line(line))
                                    mod_name = get_mod_name(in_sub_name,module_name)
                                    if mod_name not in self.func_info:
                                      self.func_info[mod_name] = {"files": [], "lines": {}}
                                    if "lines" not in self.func_info[mod_name]:
                                      self.func_info[mod_name]["lines"] = {}
                                    if "files" not in self.func_info[mod_name]:
                                      self.func_info[mod_name]["files"] = []
                                    self.func_info[mod_name]["files"].append(filepath)
                                    self.func_info[mod_name]["lines"][filepath] = self.func_info[in_sub_name]["lines"][filepath].copy()
                                    self.func_info[mod_name]["lines"][filepath][0] = self.func_info[mod_name]["lines"][filepath][0].replace(in_sub_name,mod_name)
                                    self.func_info[mod_name]["lines"][filepath][-1] = self.func_info[mod_name]["lines"][filepath][-1].replace(in_sub_name,mod_name)
                                    in_sub_name = None
                            if "interface" in line:
                                iter_index = index
                                res_line = line.lower()
                                if res_line.split(" ")[0].strip() == "interface" and len(res_line.split(" "))>1:
                                    sub_name = res_line.split(" ")[1].strip()
                                    if sub_name not in self.func_info:
                                        self.func_info[sub_name] = {"files": []}
                                    if "interface_funcs" not in self.func_info[sub_name]:
                                        self.func_info[sub_name]["interface_funcs"] = {"files": [],"lines": {}}
                                    self.func_info[sub_name]["files"].append(filepath)
                                    self.func_info[sub_name]["interface_funcs"][filepath] = []
                                    cur_index = index+1
                                    cur_line = read_lines[cur_index].lower()
                                    find = True
                                    while not ("end" in cur_line and "interface" in cur_line):
                                        if("module procedure" in cur_line):
                                            self.func_info[sub_name]["interface_funcs"][filepath].append(self.parse_line(cur_line.split("module procedure ")[1].strip()))
                                        elif("function" in cur_line):
                                            self.func_info[sub_name]["interface_funcs"][filepath].append(self.parse_line(cur_line.split("function")[1].split("(")[0].strip()))
                                        cur_index += 1
                                        cur_line = read_lines[cur_index].lower()
                                    #add also mod name version
                                    if ".h" not in filepath:
                                        mod_sub_name = get_mod_name(sub_name,module_name)
                                        if mod_sub_name not in self.func_info:
                                            self.func_info[mod_sub_name] = {"files": []}
                                        if "interface_funcs" not in self.func_info[mod_sub_name]:
                                            self.func_info[mod_sub_name]["interface_funcs"] = {"files": [], "lines": {}}
                                        self.func_info[mod_sub_name]["files"].append(filepath)
                                        self.func_info[mod_sub_name]["interface_funcs"][filepath] = self.func_info[sub_name]["interface_funcs"][filepath]
                                    

                        
                        lines.append(line)
                        if line[0] != "!":
                            if index<len(read_lines)-1:
                                next_line = self.parse_line(read_lines[index].strip())
                            else:
                                next_line = ""
                            if not in_public:
                                in_public = line.strip() == "public"
                            else:
                                in_public = not line.strip() == "private"
                            in_static_variables_declaration = in_static_variables_declaration and not is_contains_line(line,next_line)
                            was_in_struct = in_struct
                            in_struct = check_if_in_struct(line,in_struct)
                            if not was_in_struct and in_struct:
                                struct_name = get_struct_name_from_init(line)
                                if struct_name not in self.struct_table:
                                    self.struct_table[struct_name] = {}
                            if in_static_variables_declaration:
                                if is_use_line(line):
                                    module,restrictions= self.get_used_module_and_restriction_from_line(line)
                                    self.file_info[filepath]["used_modules"][module] = restrictions
                                if not in_struct and line[0] != "!" and "::" in line:
                                    self.get_variables_from_line(line,index,self.static_variables,filepath,module_name,False,in_public)
                                if in_struct and was_in_struct:
                                    self.get_variables_from_line(line,index,self.struct_table[struct_name],filepath,"",True,False)
                            if in_sub_name and line[0] != '!':
                                self.func_info[in_sub_name]["lines"][filepath].append(self.parse_line(line))
                index += 1
            self.lines[filepath] = lines
            if filepath not in self.file_info:
                self.file_info[filepath] = {}
        res = [x[1] for x in enumerate(self.lines[filepath]) if x[0] >= start_range and x[0] <= end_range]
        if not include_comments:
            res = [x for x in res if x[0] != "!"] 
        return res


    def contains_subroutine(self, lines, function_name):
        for line in [x.lower() for x in lines]:
            if "(" in function_name or ")" in function_name:
                return False
            if re.search(f"\s?subroutine {function_name}[\s(]",line) or re.search(f"\s?function {function_name}[\s(]",line) or re.search(f"interface {function_name}(\s|$|,)",line) or line==f"subroutine {function_name}" or line==f"function {function_name}":
                return True
        return False

    def get_used_module_and_restriction_from_line(self,line):
        if line.strip().split(" ")[0].strip() == "use":
            module = line.strip().split(" ")[1].strip().split(",")[0].strip().lower()
            if "only:" in line:
                restrictions = [part.strip() for part in line.split("only:")[1].strip().split(",")]
            else:
                restrictions = None 
            return (module,restrictions)
        else: 
            return (None, None)
    def get_module_restriction_from_line(self,line):
      if "only:" in line:
        return [part.strip() for part in line.split("only:")[1].strip().split(",")]
      return None
    def get_used_modules_and_restrictions(self,lines):
        modules = []
        res_restrictions = []
        for line in lines:
            module,restrictions= self.get_used_module_and_restriction_from_line(line)
            if module:
                modules.append(module)
                res_restrictions.append(restrictions)
        return (modules,res_restrictions)
    def get_used_modules(self,lines):
       
      return self.get_used_modules_and_restrictions(lines)[0]
    def get_own_module(self,filename):
        return self.file_info[filename]["module"]
        

    def get_mem_access_or_function_calls(self,lines):
        res = []
        for line in filter(lambda x: not is_variable_line(x),lines):
            line = line.strip()
            matches = re.findall("[^'=' '\/+-.*()<>]+\(.+\)", line)
            if len(matches) > 0:
                res.extend(matches)
        return res


    def parse_variable(self, line_segment):
        iter_index = len(line_segment)-1
        end_index = iter_index
        start_index = 0
        num_of_left_brackets = 0
        num_of_right_brackets = 0
        while iter_index>0 and (line_segment[iter_index] in " ()" or num_of_left_brackets != num_of_right_brackets):
            elem = line_segment[iter_index]
            if elem == "(":
                num_of_left_brackets += 1
            elif elem == ")":
                num_of_right_brackets += 1
            iter_index -= 1
        end_index = iter_index+1

        if "(" in line_segment and "%" in line_segment:
            if "forall" in line_segment:
                return line_segment.split(")")[1].split("%")[0].split("(")[0].strip()
            else:
                return line_segment.split("%")[0].split("(")[0].strip()
        while iter_index>0 and line_segment[iter_index] not in " *+-();":
            iter_index -= 1
        
        if iter_index == 0:
            res = line_segment[0:end_index]
        else:
            res = line_segment[iter_index+1:end_index]
        if "%" in res:
            end_index = 0
            while res[end_index] != "%":
                end_index += 1
            res = res[0:end_index]
        return res.strip()



    def get_writes_from_line(self,line,count=0,to_lower=True):
        res = []
        index = 0
        start_index = 0
        num_of_single_quotes = 0
        num_of_double_quotes = 0
        num_of_left_brackets = 0
        num_of_right_brackets = 0
        variable_num = 0
        is_init_line = "::" in line
        line = line.split("::",1)[-1].strip()
        while index<len(line)-1:
            index += 1
            elem = line[index]
            if elem == "'":
                num_of_single_quotes += 1
            elif elem == '"':
                num_of_double_quotes += 1
            elif elem == "=" and line[index-1] not in "/<>=!" and (index>=len(line)-2 or line[index+1] not in "/<>=!") and num_of_single_quotes%2==0 and num_of_double_quotes%2==0 and num_of_left_brackets == num_of_right_brackets:
                write = self.parse_variable(line[start_index:index])
                variable = write
                if to_lower:
                    variable = variable.lower()
                ##is init line
                variable = variable.strip()
                if is_init_line:
                  val = parse_declaration_value(line.split("=")[variable_num+1].strip())
                  variable = variable.split(",")[-1].strip()
                  if val[-1] == ",":
                    val = val[:-1]
                else:
                    val = line.split("=",1)[1].strip()
                if variable == "":
                    pexit("empty variable: ",line)
                res.append({"variable": variable, "line_num": count, "line": line, "is_static": write in self.static_variables, "value": val})
                variable_num += 1
                start_index = index
            elif elem == "(":
                num_of_left_brackets += 1
            elif elem == ")":
                num_of_right_brackets += 1
        return res
    def get_idiag_vars(self,lines):
        local_variables = {parameter:v for parameter,v in self.get_variables(lines, {},"", True).items() }
        idiag_vars = []
        for line in lines:
          if "idiag_" in line:
            idiag_vars.extend([x[0] for x in get_var_name_segments(line,self.static_variables) if "idiag_" in x[0]])
        return unique_list(idiag_vars)

    def get_writes(self,lines,exclude_variable_lines=True):
        res = []
        if exclude_variable_lines:
            lines = list(filter(lambda x: not is_variable_line(x), lines))
        for line_index, line in enumerate(lines):
            res.extend(self.get_writes_from_line(line,line_index))
        return res
    def get_rhs_variable(self,line,to_lower=True):
        writes =  self.get_writes_from_line(line,0,to_lower)
        if len(writes) == 0:
            return None
        return writes[0]["variable"]
    def get_used_variables_from_line(self,line):
        characters_to_space= ["/",":", ",","+","-","*","(",")","=","<","!",">"]
        for character in characters_to_space:
            line = line.replace(character, " ")
        parts = [part.strip() for part in line.split(" ")]
        return [self.parse_variable(part) for part in parts if part]

    def get_used_variables(self,lines):
        res = []
        for line  in filter(lambda x: not is_variable_line(x),lines):
            res.extend(self.get_used_variables_from_line(line))
        return res


    def get_function_name(self,line):
        iter_index = len(line)-1
        while(iter_index > 0):
            if line[iter_index] == "%":
                return line[0:iter_index]
            iter_index -= 1
        return line
    def get_function_calls_in_line(self, line, local_variables,exclude_variable_lines=True):
        return self.get_function_calls([line],local_variables, exclude_variable_lines)
    def get_function_calls(self,lines, local_variables,exclude_variable_lines=True):
        function_calls = []
        if exclude_variable_lines:
            lines = filter(lambda x: not is_variable_line(x),lines)
        for line_index, line in enumerate(lines):
            line = line.lower()
            function_call_indexes = []
            num_of_single_quotes = 0
            num_of_double_quotes = 0
            #get normal function calls i.e. with parameters and () brackets
            for i in range(len(line)):
                if line[i] == "'":
                    num_of_single_quotes += 1
                if line[i] == '"':
                    num_of_double_quotes += 1
                if line[i] == "(" and (i==0 or line[i-1] not in "!^'=''\/+-.*\(\)<>^;:,") and num_of_double_quotes %2 == 0 and num_of_single_quotes %2 == 0:
                    function_call_indexes.append(i)
            for index in function_call_indexes:
                current_index = index-1
                if line[current_index] == " ":
                    while(line[current_index] == " "):
                        current_index -= 1
                while(line[current_index] not in " !^'=''\/+-.*\(\)<>^,;:" and current_index >= 0):
                    current_index -= 1
                current_index +=1
                function_name = self.get_function_name(line[current_index:index])
                save_index = current_index
                current_index = index
                function_name = function_name.strip()
                if function_name.lower() not in self.static_variables and function_name.lower() not in local_variables:
                    #step through (
                    current_index +=1
                    number_of_right_brackets = 1
                    number_of_left_brackets = 0
                    num_of_single_quotes = 0
                    num_of_double_quotes = 0
                    parameter_list_start_index = current_index
                    #print(line)
                    while(number_of_right_brackets>number_of_left_brackets):
                        parameter_list = line[parameter_list_start_index:current_index]
                        if line[current_index] == "'" and num_of_double_quotes %2 == 0:
                            num_of_single_quotes += 1
                        if line[current_index] == '"' and num_of_single_quotes %2 == 0:
                            num_of_double_quotes += 1
                        if line[current_index] == "(" and num_of_double_quotes %2 == 0 and num_of_single_quotes %2 == 0:
                            number_of_right_brackets += 1
                        elif line[current_index] == ")" and num_of_double_quotes %2 == 0 and num_of_single_quotes %2 == 0: 
                            number_of_left_brackets += 1
                        
                        current_index += 1
                     
                    parameter_list = line[parameter_list_start_index:current_index-1]
                    parameters = []
                    param ="" 
                    ##if inside brackets they are array indexes
                    num_of_left_brackets = 0
                    num_of_right_brackets= 0
                    num_of_single_quotes = 0
                    num_of_double_quotes = 0
                    for char in parameter_list:
                        if char == "'" and num_of_double_quotes %2 == 0:
                            num_of_single_quotes += 1
                        if char == '"' and num_of_single_quotes %2 == 0:
                            num_of_double_quotes += 1
                        if char == "(":
                            num_of_left_brackets = num_of_left_brackets + 1
                        if char == ")":
                            num_of_right_brackets = num_of_right_brackets + 1
                        if char == "," and num_of_left_brackets == num_of_right_brackets and num_of_double_quotes %2 == 0 and num_of_single_quotes %2 == 0: 
                            parameters.append(param.strip())
                            param = ""
                        else:
                            param = param + char
                    ## add last param
                    parameters.append(param.strip())
                    #if multithreading analysis
                    for i, param in enumerate(parameters):
                        if len(param) > 0 and param[0] == "-":
                            parameters[i] = param[1:]
                    function_name = function_name.strip()
                    if len(function_name) >0 and "%" not in function_name and not function_name.isnumeric():
                        function_calls.append({"function_name": function_name, "parameters": [param.strip() for param in parameters if len(param.strip()) > 0], "range": (save_index,current_index), "line": line, "line_num":line_index})
            
            #get function calls with call function i.e. call infront and no brackets
            function_call_indexes = []
            num_of_single_quotes = 0
            num_of_double_quotes = 0
            buffer = ""
            #get normal function calls i.e. with parameters and () brackets
            for i in range(len(line)):
                if line[i] == "'":
                    num_of_single_quotes += 1
                if line[i] == '"':
                    num_of_double_quotes += 1
                elif line[i] in "'!()[];-+*/=^":
                    buffer = ""
                elif line[i] == " ":
                    if buffer == "call" and num_of_single_quotes %2 == 0 and num_of_double_quotes %2 == 0:
                        function_call_indexes.append(i)
                    buffer = ""
                else:
                    buffer = buffer + line[i]
            for index in function_call_indexes:
                save_index = index-4
                #skip empty spaces
                while line[index] == " ":
                    index= index+1
                if line[index] == "=":
                    break
                start_index = index
                while index<len(line) and line[index] not in " (":
                    index = index+1
                function_name = line[start_index:index]
                #don't add duplicates, no need since no parameters
                if function_name not in [function_call["function_name"] for function_call in function_calls]:
                    function_name = function_name.strip()
                    ## % check is since the we can have for example p%inds(1) and %inds(1) would be marked as a function call.
                    if len(function_name) > 0 and "%" not in function_name and not function_name.isnumeric():
                        function_calls.append({"function_name": function_name, "parameters": [],"line":line,"range":(save_index,index), "line_num":line_index})
            
         
        res =  [function_call for function_call in function_calls if not function_call["function_name"].isspace()]
        return res




    def get_contains_line_num(self, filename):
        lines = self.get_lines(filename)
        for count, line in enumerate(lines):
            if count < len(lines)-1:
                next_line = lines[count+1].lower()
            else:
                next_line = ""
            if is_contains_line(line,next_line):
                return count
        #if no contains line return the end of line
        return len(lines)

    def add_public_declarations_in_file(self,filename,lines):
        in_public_block = False
        for count,line in enumerate(lines):
            if line == "public":
                in_public_block = True
            if line == "private":
                in_public_block = False
            parts = line.split("::")
            is_public_declaration = "public" == parts[0].split(",")[0].strip()
            if (is_public_declaration or in_public_block) and len(parts) == 2:
                variable_names = self.get_variable_names_from_line(parts[1])
                for variable in variable_names:
                    if variable in self.static_variables:
                        self.static_variables[variable]["public"] = True
            match = re.search("include\s+(.+\.h)",line)
            if match:
                header_filename = match.group(1).replace("'","").replace('"',"")
                directory = re.search('(.+)\/',filename).group(1)
                header_filepath = f"{directory}/{header_filename}"
                if os.path.isfile(header_filepath):
                    with open(header_filepath,"r") as file:
                        lines = file.readlines()
                        for line in lines:
                            parts = line.split("::")
                            is_public_declaration = "public" == parts[0].split(",")[0].strip()
                            if is_public_declaration:
                                variable_names = self.get_variable_names_from_line(parts[1])
                                for variable in variable_names:
                                    if variable in self.static_variables:
                                        self.static_variables[variable]["public"] = True

    def add_threadprivate_declarations_to_file(self,file, declaration):
        contents = []
        not_added = True
        for line in open(file, 'r').readlines():
            
            if (line.strip() == "contains" or "endmodule" in line) and not_added:
                not_added = False
                contents.append(f"{declaration}\n")
            elif not_added and re.match("function\s+.+\(.+\)",line) or re.match("subroutine\s+.+\(.+\)",line):
                contents.append(f"{declaration}\n")
                not_added = False
            contents.append(line)
        f = open(file, "w+")
        f.write("".join(contents))


    def get_variable_names_from_line(self,line_variables):
        res = []
        start_index=0
        end_index=0
        current_index=0
        parse_still = True
        num_of_left_brackets = 0
        num_of_right_brackets = 0
        while parse_still:
            current_elem = line_variables[current_index]
            if current_elem == "(":
                num_of_left_brackets += 1
            if current_elem == ")":
                num_of_right_brackets += 1
            if current_elem == "=":
            
                res.append(line_variables[start_index:current_index])
                parse_until_next_variable = True
                current_index +=1
                while parse_until_next_variable:
                    current_elem = line_variables[current_index]
                    not_inside_brackets = num_of_left_brackets == num_of_right_brackets
                    if current_elem == "(":
                        num_of_left_brackets += 1
                    if current_elem == ")":
                        num_of_right_brackets += 1
                    if current_elem == "!":
                        parse_until_next_variable = False
                        parse_still = False
                    if current_elem == "," and not_inside_brackets:
                        start_index = current_index+1
                        parse_until_next_variable=False
                    if parse_until_next_variable:
                        current_index +=1
                        if current_index >= len(line_variables):
                            start_index = current_index+1
                            parse_until_next_variable = False
                            parse_still = False
            elif current_elem == "," and num_of_left_brackets == num_of_right_brackets:
                res.append(line_variables[start_index:current_index])
                start_index = current_index + 1
            elif current_elem == "!":
                parse_still = False
                res.append(line_variables[start_index:current_index+1])
            current_index += 1
            if current_index >= len(line_variables):
                parse_still= False
                if current_index > start_index:
                    res.append(line_variables[start_index:current_index+1])
        return [x.replace("&","").replace("!","").replace("(:)","").strip() for x in res if x.replace("&","").replace("!","").strip() != ""]



    def make_function_file_known(self,line,filename):
        is_variable = False
        parts = [part.strip() for part in line.split("::")]
        if len(parts) == 2:
            variable_names = [name.lower() for name in self.get_variable_names_from_line(parts[1])]
            for variable in variable_names:
                if filename and variable in self.func_info:
                    if filename not in self.func_info[variable]["files"]:
                        self.func_info[variable]["files"].append(filename)
                        #in case is interface func add the interface funcs it has in the file
                        if "lines" not in self.func_info[variable]:
                            for interface_func in self.func_info[variable]["interface_funcs"][filename]:
                                if filename not in self.func_info[interface_func]["files"]:
                                    self.func_info[interface_func].append(filename)
    def get_variables(self, lines, variables,filename,local,in_public=False):
        module = None
        if filename and "module" in self.file_info[filename]:
            module = self.get_own_module(filename)
            if module not in self.module_variables:
                self.module_variables[module] = {}
        in_struct = False
        for i, line  in enumerate(lines):
            parts = [part.strip() for part in line.split("::")]
            start =  parts[0].strip()
            type = start.split(",")[0].strip()
            if type == "public":
                self.make_function_file_known(line,filename)
            else:
                in_struct = check_if_in_struct(line,in_struct)
                if not in_struct and len(parts)>1:
                    self.get_variables_from_line(line,i,variables,filename,module,local,in_public)
        for i, line  in enumerate(lines):
            for scalar_type in ["real","integer"]:
                if "function" in line and "result" in line and scalar_type in line:
                    var_name = line.split("result(")[1].split(")")[0]
                    if var_name not in variables:
                        variables[var_name] = {"type": scalar_type, "dimension": [], "is_pointer": False, "saved_variable": False}
        return variables




    def parse_file_for_static_variables(self, filepath):
        modules = self.get_always_used_modules(filepath)
        for module in modules:
            self.parse_module(module)
        self.load_static_variables(filepath)

    def add_threadprivate_declarations_in_file(self,filename):
        res = []
        lines = self.get_lines(filename, include_comments=True)
        index = 0
        while index<len(lines):
            line = lines[index]
            if "!$omp" in line and "threadprivate" in line.lower():
                if line[-1] == "&":
                    if line[-1] == "&":
                        while line[-1] == "&":
                            index += 1
                            next_line = lines[index][0].strip().replace("!$omp","")
                            if len(next_line)>0:
                                line = (line[:-1] + " " + next_line).strip()
                search = re.search("(threadprivate|THREADPRIVATE)\((.*)\)",line)
                if search:
                    variable_names = [variable.strip() for variable in search.group(2).split(",")]
                    for variable in variable_names:
                        res.append(variable)
                        if variable in self.static_variables:
                            self.static_variables[variable]["threadprivate"] = True
            index = index+1
        return res
    
    def function_is_already_declared(self,function,filename):
        res = []
        lines = self.get_lines(filename, include_comments=True)
        index = 0
        while index<len(lines):
            line  = lines[index]
            if "!$omp" in line and "declare target" in line.lower():
                if line[-1] == "&":
                    if line[-1] == "&":
                        while line[-1] == "&":
                            index += 1
                            next_line = lines[index][0].strip().replace("!$omp","")
                            if len(next_line)>0:
                                line = (line[:-1] + " " + next_line).strip()
                search = re.search("(declare target)\((.*)\)",line)
                if search:
                    function_names = [variable.strip() for variable in search.group(2).split(",")]
                    if function in function_names:
                        return True
            index = index+1
        return False
    def add_declare_target_declarations_in_file(self,filename):
        res = []
        lines = self.get_lines(filename, include_comments=True)
        index = 0
        while index<len(lines):
            line  = lines[index]
            if "!$omp" in line and "declare target" in line.lower():
                if line[-1] == "&":
                    if line[-1] == "&":
                        while line[-1] == "&":
                            index += 1
                            next_line = lines[index][0].strip().replace("!$omp","")
                            if len(next_line)>0:
                                line = (line[:-1] + " " + next_line).strip()
                search = re.search("(declare target)\((.*)\)",line)
                if search:
                    variable_names = [variable.strip() for variable in search.group(2).split(",")]
                    for variable in variable_names:
                        res.append(variable)
                        for var in self.static_variables:
                            if var == variable or f"{var}_offload" == variable:
                                self.static_variables[var]["on_target"] = True

            index = index+1
        return res


    def load_static_variables(self, filename, dst=None):
        if dst is None:
            dst = self.static_variables
        self.parsed_files_for_static_variables.append(filename)
        static_variables_end = self.get_contains_line_num(filename)
        self.get_variables(self.get_lines(filename, 1, static_variables_end), dst, filename, False)
        self.add_public_declarations_in_file(filename,self.get_lines(filename, 1, static_variables_end))
        self.add_threadprivate_declarations_in_file(filename)
        self.add_declare_target_declarations_in_file(filename)
        return dst

    def get_subroutine_variables(self, filename, subroutine_name):
        return self.get_variables(self.get_subroutine_lines(subroutine_name,filename))

    def get_always_used_modules(self, filename):
        return self.file_info[filename]["used_modules"]

    def get_subroutine_modules(self, filename, subroutine_name):
        return [module_name for module_name in  self.get_used_modules(self.get_subroutine_lines(subroutine_name,filename)) if module_name not in self.ignored_modules]

    def get_all_modules_in_file_scope(self,filename,modules):
        mod_dict = self.get_always_used_modules(filename)
        modules_to_add = []
        to_add_restrictions = []
        for x in mod_dict:
            modules_to_add.append(x)
            to_add_restrictions.append(mod_dict[x])
        modules_to_add.append(self.get_own_module(filename))
        to_add_restrictions.append(None)
        added_module_indexes = []
        for i,x in enumerate(modules_to_add):
            if x not in modules:
                modules[x] = to_add_restrictions[i]
                added_module_indexes.append(i)
            elif x in modules and to_add_restrictions[i] is None:
                modules[x] = None
        #if there is a restriction on the module then it is not recursed
        modules_to_recurse = [x[1] for x in enumerate(modules_to_add) if x[0] in added_module_indexes and not to_add_restrictions[i]]
        for module in modules_to_recurse:
            for file in self.find_module_files(module):
                self.get_all_modules_in_file_scope(file, modules)

    def update_used_modules(self):
        for filename in [x for x in self.used_files if "used_modules" in self.file_info[x] and "module" in self.file_info[x]]:
            tmp = {}
            self.get_all_modules_in_file_scope(filename, tmp)
            self.file_info[filename]["used_modules"] = tmp
            #for know remove external modules
            for mod in [x for x in self.file_info[filename]["used_modules"]]:
              if self.find_module_files(mod) == []:
                del self.file_info[filename]["used_modules"][mod]
    def update_static_var_dims(self):
      number_of_updated_values = 0
      for var in self.static_variables:
        for i, dim in enumerate(self.static_variables[var]["dims"]):
          file = self.static_variables[var]["origin"][0]
          self.static_variables[var]["dims"][i] = self.rename_line_to_internal_names(dim, {},self.file_info[file]["used_modules"], self.get_own_module(file))
          dim = self.static_variables[var]["dims"][i]
          known_parameters_in_dim= [x for x in self.static_variables if self.static_variables[x]["parameter"] and "value" in self.static_variables[x] and x in dim]
          replace_dict = {}
          for x in known_parameters_in_dim:
            replace_dict[x] = self.static_variables[x]["value"]
          new_dim = self.evaluate_integer(self.replace_variables_multi(dim, replace_dict))
          if(new_dim != self.static_variables[var]["dims"][i]):
            number_of_updated_values += 1
          self.static_variables[var]["dims"][i] = new_dim
            # for var in self.static_variables
            # if dim in self.static_variables and self.static_variables[dim]["parameter"] and "value" in self.static_variables[dim]:
            #   self.static_variables[var]["dims"][i] = self.static_variables[dim]["value"]

      for struct in self.struct_table:
        for field in self.struct_table[struct]:
          file = self.struct_table[struct][field]["origin"][0]
          for i, dim in enumerate(self.struct_table[struct][field]["dims"]):
            self.struct_table[struct][field]["dims"][i] = self.rename_line_to_internal_names(dim, {},self.file_info[file]["used_modules"], self.get_own_module(file))
            dim = self.struct_table[struct][field]["dims"][i]
            known_parameters_in_dim= [x for x in self.static_variables if self.static_variables[x]["parameter"] and "value" in self.static_variables[x] and x in dim]
            replace_dict = {}
            for x in known_parameters_in_dim:
                replace_dict[x] = self.static_variables[x]["value"]
            new_dim = self.evaluate_integer(self.replace_variables_multi(dim, replace_dict))
            if(new_dim != self.struct_table[struct][field]["dims"][i]):
                number_of_updated_values += 1
            self.struct_table[struct][field]["dims"][i] = new_dim
      return number_of_updated_values
    
    def update_value(self,value):
        #used to inline compile time params
        value_parts = [value]
        for char in [" ","+","-","*","/"]:
            tmp = []
            for part in value_parts:
                tmp.extend(part.split(char))
            value_parts = tmp
        known_parameters_in_value= [x for x in self.static_variables if self.static_variables[x]["parameter"] and "value" in self.static_variables[x] and x in value_parts]
        #if scientific number
        if "." in value and "e" in value and "'" in value:
            known_parameters_in_value = []
        #recurse until base known values
        while(len(known_parameters_in_value)>0):
            value_parts = []
            for char in [" ","+","-","*","/"]:
                tmp = []
                for part in value_parts:
                    tmp.extend(part.split(char))
                value_parts = tmp
            replace_dict = {}
            for x in known_parameters_in_value:
                replace_dict[x] = self.static_variables[x]["value"]
            value = self.replace_variables_multi(value, replace_dict)
            known_parameters_in_value= [x for x in self.static_variables if self.static_variables[x]["parameter"] and "value" in self.static_variables[x] and x in value_parts]
            if "." in value and "e" in value or ("'" in value):
                known_parameters_in_value = []
        return value
    def update_static_var_value(self,var):
        if "value" in self.static_variables[var] and "mpi" not in self.static_variables[var]["value"]:
          file = self.static_variables[var]["origin"][0]
          #remove all use guards from use statements for modules for this use case
          modules = self.file_info[file]["used_modules"]
          for mod in modules:
            modules[mod] = None
          self.static_variables[var]["value"] = self.rename_line_to_internal_names(self.static_variables[var]["value"], {},modules, self.get_own_module(file))

        if "array_value" in self.static_variables[var] and "mpi" not in self.static_variables[var]["array_value"]:
          file = self.static_variables[var]["origin"][0]
          #remove all use guards from use statements for modules for this use case
          modules = self.file_info[file]["used_modules"]
          for mod in modules:
            modules[mod] = None
          self.static_variables[var]["array_value"] = self.rename_line_to_internal_names(self.static_variables[var]["array_value"], {},modules, self.get_own_module(file))

        changed = False
        if "value" in self.static_variables[var] and "mpi" not in self.static_variables[var]["value"]:
          value = self.update_value(self.static_variables[var]["value"])
          if(self.static_variables[var]["value"] != value):
            changed = True
          if(self.static_variables[var]["type"] == "integer"):
            value = self.evaluate_integer(value)
          self.static_variables[var]["value"] = value

        if "array_value" in self.static_variables[var] and self.static_variables[var]["parameter"] and self.static_variables[var]["type"] == "logical" and self.static_variables[var]["dims"] == ["3"]:
            array_val = self.static_variables[var]["array_value"]
            if len(array_val) > 4 and array_val[0] == "(" and array_val[1] == "/" and array_val[-1] == ")" and array_val[-2] == "/":
                parts = [self.update_value(part.strip()) for part in array_val.split("/",1)[-1].split("/",1)[0].split(",")]
                first,second,third = (None,None,None)
                if self.static_variables[var]["type"] == "logical":
                    first,second,third = [self.evaluate_boolean(part,{},{}) for part in parts]
                self.flag_mappings[f"{var}(1)"] = first
                self.flag_mappings[f"{var}(2)"] = first
                self.flag_mappings[f"{var}(3)"] = first
            else:
                pexit("what do?",var, array_val)

        return changed

    def update_static_var_values(self):
      number_of_updated_values = 0
      for var in self.static_variables:
        if self.update_static_var_value(var):
            number_of_updated_values += 1
      return number_of_updated_values


    def update_static_vars(self):
      some_value_was_updated = True
      while(some_value_was_updated):
        number_of_updated_values = self.update_static_var_values()
        print("number of updated values:",number_of_updated_values)
        some_value_was_updated = number_of_updated_values > 0
      some_value_was_updated = True
      while(some_value_was_updated):
        number_of_updated_values = self.update_static_var_dims()
        print("number of updated values:",number_of_updated_values)
        some_value_was_updated = number_of_updated_values > 0


    def update_func_info(self):
        for func in self.func_info:
            if "interface_funcs" in self.func_info[func]:
                all_interface_funcs = []
                for key in self.func_info[func]["interface_funcs"]:
                    if key not in ["lines","files"]:
                        all_interface_funcs.extend(self.func_info[func]["interface_funcs"][key])
                self.func_info[func]["all_interface_funcs"] = all_interface_funcs


    def get_local_module_variables(self,filename,subroutine_name):
        res = {}
        #Variables in own file take precedence 
        self.load_static_variables(filename,res)

        print(self.func_info[subroutine_name].keys()) 
        for module in [x for x in self.get_all_modules_in_subroutine_scope(filename,subroutine_name) if x!=self.get_own_module(filename)]:
            if module not in self.module_variables and module in self.parsed_modules:
                pexit("module was parsed but not self.module_variables",module)
            for var in self.module_variables[module]:
                if var not in res:
                    res[var] = self.module_variables[module][var]
        return res


    def get_parameters(self,line):
        line = line.lower()
        check_subroutine= re.search("\s?subroutine.+\((.+)\)",line)
        if check_subroutine:
            res =  [parameter.split("=")[-1].split("(")[0].strip().lower() for parameter in check_subroutine.group(1).split(",")]
            return res
        else:
            ##The return parameter needs to be removed
            if "result" in line:
                line = line.split("result")[0].strip()

            param_search = re.search(".?function.+\((.+)\)",line)
            if not param_search:
                return []
            res =  [parameter.split("=")[-1].split("(")[0].strip().lower() for parameter in param_search.group(1).split(",")]
            return [res_part.lower() for res_part in res]

    def get_parameter_mapping(self, parameters, parameter_list):
        # return [x[0] for x in enumerate(parameter_list)]
        # Commented out for testing
        mapping = []
        for i, param in enumerate(parameter_list):
            if param[4] is None:
                mapping.append(i)
            else:
                for j,sub_param in enumerate(parameters):
                    if sub_param == param[4]:
                        mapping.append(j)
        return mapping

    def get_static_parameters(self,line,parameter_list):
        parameters = self.get_parameters(line)
        res = []
        mapping = self.get_parameter_mapping(parameters,parameter_list)
        already_done = []
        if len(parameter_list) < len(mapping) or len(parameters) < len(mapping):
            return res
        for i, map_index in enumerate(mapping):
            res.append((parameters[map_index], parameter_list[i][:-1]))
            already_done.append(map_index)
        for i in range(len(parameters)):
            if i not in already_done:
                res.append((parameters[i],("",False,"",[])))
        return res
    def get_var_info_from_array_access(self,parameter,local_variables,local_module_variables):
        var = parameter[0].split("(")[0].strip()
        if "%" in var:
            var_name, field = var.split("%",1)
            if var_name in local_variables:
                sizes = self.struct_table[local_variables[var_name]["type"]][field]["dims"]
            else:
                sizes = self.struct_table[self.static_variables[var_name]["type"]][field]["dims"]
        else:
            if var in local_variables:
                sizes = local_variables[var]["dims"]
            else:
                if(self.static_variables[var]["is_pointer"]):
                    var_name = remove_mod(var)
                    #again assume there is a single reference that is not a pointer and use its dims
                    pos_mods = []
                    for mod in self.rename_dict:
                        if var_name in self.rename_dict[mod]:
                            if not self.static_variables[f"{var_name}__mod__{mod}"]["is_pointer"]:
                                pos_mods.append(mod)
                    assert(len(pos_mods) == 1)
                    src_var = f"{var_name}__mod__{pos_mods[0]}"
                    sizes = self.static_variables[src_var]["dims"]
                else:
                    sizes = self.static_variables[var]["dims"]
        if parameter[0] == "(":
            parameter[0] == parameter[0][1:]
        if parameter[-1] == ")":
            parameter[0] == parameter[0][:-1]
        indexes = get_indexes(parameter[0],var,0)
        ##count the number of looped indexes
        dim = 0
        dims = []
        for i, index in enumerate(indexes):
            ##can have inline array as array range
            if ":" in index:
                if index == ":":
                    dims.append(sizes[i])
                else:
                    lower,upper = [part.strip() for part in index.split(":")]
                    if "(" not in index:
                      dims.append(self.evaluate_integer(f"{upper}-{lower}+1"))
                    else:
                      dims.append(self.evaluate_integer(":"))
            elif index.replace("(","").replace(")","").strip()[0] == '/' and index.replace("(","").replace(")","").strip()[-1] == '/':
                dims.append(":")
        source =  local_variables if var in local_variables else local_module_variables
        is_static = source[var]["saved_variable"] or var in local_module_variables if "%" not in var else False
        res_type = source[var]["type"] if "%" not in var else "pencil_case"
        return (var,is_static, res_type, dims)
    def get_param_info_recursive(self,parameter,local_variables,local_module_variables):
        if len(parameter[0][0]) == 0:
            pexit("INCORRECT PARAM",parameter)
        if parameter[0][0] == "(" and parameter[0][-1] == ")":
            return self.get_param_info_recursive((parameter[0][1:-1],parameter[1]),local_variables,local_module_variables)
        #is scientific number
        if ("e" in parameter[0] or parameter[0][0] == '.') and parameter[0].replace(".","").replace("-","").replace("e","").replace("+","").isnumeric():
            return (parameter[0],False,"real",[])
        if parameter[0] in local_variables:
            return (parameter[0],parameter[1],local_variables[parameter[0]]["type"],local_variables[parameter[0]]["dims"])
        if parameter[0] in local_module_variables:
            if(local_module_variables[parameter[0]]["is_pointer"]):
                var_name = remove_mod(parameter[0])
                #again assume there is a single reference that is not a pointer and use its dims
                pos_mods = []
                for mod in self.rename_dict:
                    if var_name in self.rename_dict[mod]:
                        if not self.static_variables[f"{var_name}__mod__{mod}"]["is_pointer"]:
                            pos_mods.append(mod)
                if(len(pos_mods) != 1):
                    if(len(pos_mods) == 0):
                        pexit(f"No possible modules for {parameter}")
                    else:
                        #if self.module_vars_the_same(pos_mods,var_name): pos_mods = [pos_mods[0]]
                        #else:
                            print(parameter[0])
                            print(pos_mods)
                            pexit("Too many possible modules")
                    
                src_var = f"{var_name}__mod__{pos_mods[0]}"
                dims = self.static_variables[src_var]["dims"]
            else:
                dims = local_module_variables[parameter[0]]["dims"]
            return (parameter[0],parameter[1],local_module_variables[parameter[0]]["type"],dims)
        if parameter[0] in self.static_variables:
            if(self.static_variables[parameter[0]]["is_pointer"]):
                var_name = remove_mod(parameter[0])
                #again assume there is a single reference that is not a pointer and use its dims
                pos_mods = []
                for mod in self.rename_dict:
                    if var_name in self.rename_dict[mod]:
                        if not self.static_variables[f"{var_name}__mod__{mod}"]["is_pointer"]:
                            pos_mods.append(mod)
                assert(len(pos_mods) == 1)
                src_var = f"{var_name}__mod__{pos_mods[0]}"
                dims = self.static_variables[src_var]["dims"]
            else:
                dims = self.static_variables[parameter[0]]["dims"]
            return (parameter[0],parameter[1],self.static_variables[parameter[0]]["type"],dims)
        if parameter[0][0] == "(":
            parameter = (parameter[0][1:],parameter[1])
        if parameter[0] == ".true." or parameter[0] == ".false.":
            return (parameter[0],False,"logical",[])
        is_sum = False
        is_product = False
        is_division = False
        is_difference = False
        num_of_left_brackets = 0
        num_of_right_brackets= 0
        possible_var = ""
        in_array = False
        for char_index,char in enumerate(parameter[0]):
            if char == "(":
                num_of_left_brackets += 1
                if parameter[0][char_index-1] == " ":
                    back_index = char_index-1
                    while(parameter[0][back_index] == " " and back_index>0):
                        back_index -= 1
                    if back_index != 0 and parameter[0][back_index] not in "*-+/(":
                        in_array = True
                else:
                    if char_index != 0 and parameter[0][char_index-1] not in "*-+-/(":
                        # in_array = possible_var in local_variables or possible_var in local_module_variables 
                        in_array = True
                possible_var = ""
            elif char == ")":
                num_of_right_brackets += 1
                if num_of_left_brackets == num_of_right_brackets:
                    in_array = False
                possible_var = ""
            elif char == "+" and not in_array:
                is_sum = True
                possible_var = ""
            elif char == "*" and not in_array:
                is_product= True
                possible_var = ""
            elif char == "-" and not in_array:
                is_difference= True
                possible_var = ""
            elif char == "/" and not in_array and char_index != 0 and char_index != len(parameter[0])-1 and parameter[0][char_index-1] != "(" and parameter[0][char_index+1] != ")":
                if char_index < len(parameter[0])-1:
                    is_division = parameter[0][char_index+1] not in ")!"
                else:
                    is_division= True
                possible_var = ""
            else:
                possible_var = possible_var + char
        operations = (is_sum,is_difference,is_product,is_division)
        #inline array
        if not is_division and parameter[0].replace("(","")[0] == "/" and parameter[0].replace(")","")[-1] == "/":
            par_str = parameter[0].strip()
            if par_str[0] != "(":
                par_str = "(" + par_str
            if par_str[1] == "/":
                par_str = par_str[0] + par_str[2:]
            if par_str[-1] != ")":
                par_str = par_str + ")"
            if par_str[-2] == "/":
                par_str = par_str[:-2] + par_str[-1]
            par_str = "inline_array" + par_str
            parameters = self.get_function_calls_in_line(par_str, local_variables)[0]["parameters"]
            info = self.get_param_info_recursive((parameters[0],False),local_variables,local_module_variables)

            new_dims = []
            for dim in info[3]:
                new_dims.append(dim)
            new_dims.append(":")
            return (parameter[0],"False",info[2],new_dims)
        func_calls = self.get_function_calls_in_line(parameter[0],local_variables)
        if len(func_calls)>0 and not any(operations):
            first_call = func_calls[0]
                #Functions that simply keep the type of their arguments
            if first_call["function_name"] in ["sqrt","alog","log","exp","sin","cos","log","abs"]:
                return self.get_param_info_recursive((first_call["parameters"][0],False),local_variables,local_module_variables)
            #Array Functions that return single value if single param else an array
            if first_call["function_name"] in ["sum"]:
                new_param = (first_call["parameters"][0],False)
                inside_info =  self.get_param_info_recursive(new_param,local_variables,local_module_variables)
                if len(first_call["parameters"]) == 1:
                    return (first_call["parameters"][0],False,inside_info[2],[])
                else:
                    return (first_call["parameters"][0],False,inside_info[2],[":"])
            #Array Functions that return scalar value, multiple params, return type is passed on first param
            if first_call["function_name"] in ["dot_product"]:
                
                inside_info =  self.get_param_info_recursive((first_call["parameters"][0],False),local_variables, local_module_variables)
                return (parameter[0],False,inside_info[2],[])
            #Array Functions that return the largest value in params, multiple params, return type is passed on first param
            if first_call["function_name"] in ["max","min"]:
                inside_info =  self.get_param_info_recursive((first_call["parameters"][0],False),local_variables, local_module_variables)
                return (parameter[0],False,inside_info[2],inside_info[3])
            
            if first_call["function_name"] in ["maxval","minval"]:
                inside_info =  self.get_param_info_recursive((first_call["parameters"][0],False),local_variables, local_module_variables)
                if(len(first_call["parameters"])) == 1:
                    return (parameter[0],False,inside_info[2],[])
                return (parameter[0],False,inside_info[2],inside_info[3][:-1])

            #SPREAD
            if first_call["function_name"] == "spread":
                inside_info =  self.get_param_info_recursive((first_call["parameters"][0],False),local_variables, local_module_variables)
                if first_call["parameters"][1] == "1":
                    return (parameter[0],False,inside_info[2],[":"])
            if first_call["function_name"] in ["char"]:
                return (parameter[0],False,"character",[])
            if first_call["function_name"] in ["real","float"]:
                inside_info = self.get_param_info_recursive((first_call["parameters"][0],False),local_variables, local_module_variables)
                return (parameter[0],False,"real",inside_info[3])
            if first_call["function_name"] in ["precision"]:
                inside_info = self.get_param_info_recursive((first_call["parameters"][0],False),local_variables, local_module_variables)
                return (parameter[0],False,"integer",inside_info[3])
            if first_call["function_name"] in ["mod","modulo"]:
                inside_info = self.get_param_info_recursive((first_call["parameters"][0],False),local_variables, local_module_variables)
                return (parameter[0],False,inside_info[2],inside_info[3])
            if first_call["function_name"] in ["count","size","len_trim","ubound","lbound"]:
                return (parameter[0],False,"integer",[])
            if first_call["function_name"] == "int":
                inside_info = self.get_param_info_recursive((first_call["parameters"][0],False),local_variables, local_module_variables)
                return (parameter[0],False,"integer",inside_info[3])
            if first_call["function_name"] == "trim":
                return (parameter[0],False,"character",[])
            if first_call["function_name"] == "len":
                return (parameter[0],False,"integer",[])
            #DCONST is Astaroth specific
            if first_call["function_name"] in ["merge","dconst"]:
                return self.get_param_info_recursive((first_call["parameters"][0],False),local_variables,local_module_variables)
            if first_call["function_name"] in ["cmplx"]:
                first_param_info = self.get_param_info_recursive((first_call["parameters"][0],False),local_variables,local_module_variables)
                second_param_info = self.get_param_info_recursive((first_call["parameters"][1],False),local_variables,local_module_variables)
                dims = first_param_info[3]
                if len(second_param_info[3]) > len(dims):
                    dims = second_param_info[3]
                return (parameter[0],False,"complex",dims)
            if first_call["function_name"] in ["transpose"]:
                first_param_info = self.get_param_info_recursive((first_call["parameters"][0],False),local_variables,local_module_variables)
                return (parameter[0],False,first_param_info[2],first_param_info[3])
            #if first_call["function_name"]  == "spread":
            #    inside_info = self.get_param_info_recursive((first_call["parameters"][0],False),local_variables, local_module_variables)
            #    if first_call["parameters"][1] == "1":
            #        return (parameter[0], False, inside_info[2], [":"])
            if first_call["function_name"] in self.func_info:
                file_path = self.find_subroutine_files(func_calls[0]["function_name"])[0]
                interfaced_call = self.get_interfaced_functions(file_path,func_calls[0]["function_name"])[0]
                _,type,param_dims = self.get_function_return_var_info(interfaced_call,self.get_subroutine_lines(interfaced_call, file_path), local_variables)
                return (parameter[0],False,type, param_dims)
            if first_call["function_name"] in ["sign"]:
                first_param_info = self.get_param_info_recursive((first_call["parameters"][0],False),local_variables,local_module_variables)
                return (parameter[0],False,first_param_info[2],first_param_info[3])
                
            
            print("unsupported func")
            pexit(first_call)
        elif parameter[0] in local_variables:
            return (parameter[0],parameter[1],local_variables[parameter[0]]["type"],local_variables[parameter[0]]["dims"])
        elif parameter[0] in local_module_variables:
            return (parameter[0],parameter[1],local_module_variables[parameter[0]]["type"],local_module_variables[parameter[0]]["dims"])
        elif "'" in parameter[0] or '"' in parameter[0]:
            return (parameter[0],parameter[1],"character",[])
        elif any(operations):
            op_chars = []
            if is_sum:
                 op_chars.append("+")
            if is_difference:
                 op_chars.append("-")
            if is_product:
                 op_chars.append("*")
            if is_division:
                 op_chars.append("/")
            parts = [part.strip() for part in split_by_ops(parameter[0].strip(),op_chars)]
            # print("PARTS",parts)
            parts_res = ("",False,"",[])
            for part in parts:
                if len(part) > 0  and part[0] == "(" and sum([char == "(" for char in part]) > sum([char == ")" for char in part]):
                    part = part[1:]
                if len(part) > 0  and part[-1] == ")" and sum([char == "(" for char in part]) < sum([char == ")" for char in part]):
                    part = part[:-1]
                part_res = self.get_param_info_recursive((part,False),local_variables,local_module_variables)
                if parts_res[2] == "" or len(part_res[3])>len(parts_res[3]):
                    parts_res = (parts_res[0],False,part_res[2],part_res[3])
                #int op real -> real
                if parts_res[2] == "integer" and part_res[2] == "real":
                    parts_res = (parts_res[0],parts_res[1],"real",parts_res[3])
            return parts_res
        elif "(" in parameter[0] and "%" not in parameter[0].split("(")[0] and ")" in parameter[0] and not any(operations) and ".and." not in parameter[0] and ".not." not in parameter[0]:
            var = parameter[0].split("(")[0].strip()
            if var in local_variables or var in local_module_variables or var in self.static_variables:
                return self.get_var_info_from_array_access(parameter,local_variables,self.static_variables)
            ##Boolean intrinsic funcs
            elif var in ["present","isnan","associated","allocated","all","any"]:
                return (parameter[0],False,"logical",[])
            elif var in ["trim"]:
                return (parameter[0],False,"character",[])
            else:
                #check if function in source code
                pexit(parameter,"how did I end up here?")
        elif "%" in parameter[0] and "'" not in parameter[0] and '"' not in parameter[0]:
            parts = [part.strip() for part in parameter[0].split("%")]
            if len(parts) == 2:
                var_name,field_name = parts
            elif len(parts) == 3:
                var_name,base_field_name,last_field_name = parts
                if var_name in local_variables:
                    struct = local_variables[var_name]["type"]
                else:
                    struct = local_module_variables[var_name]["type"]
                base_field = self.struct_table[struct][base_field_name]
                if("(" in last_field_name):
                        last_field_name = last_field_name.split("(")[0]
                        field = self.struct_table[base_field["type"]][last_field_name]
                        return (var_name,parameter[1],field["type"],[field["dims"][0]])
                else:
                	field = self.struct_table[base_field["type"]][last_field_name]
                	return (var_name,parameter[1],field["type"],field["dims"])
            else:
                pexit("More recursion",parameter[0])
            ##var_name can be array access if array of structures
            var_name = var_name.split("(")[0]
            struct = ""
            if var_name in local_variables:
                struct = local_variables[var_name]["type"]
            else:
                struct = local_module_variables[var_name]["type"]
            if "(" in field_name: 
                var_dims = self.get_var_info_from_array_access(parameter,local_variables,local_module_variables)[-1]
            else:
                field = self.struct_table[struct][field_name]
                var_dims = field["dims"]
            field_name = field_name.split("(")[0]
            field = self.struct_table[struct][field_name]
            return (var_name,parameter[1],field["type"],var_dims)
        elif ".and." in parameter[0] or ".not." in parameter[0] or ".or." in parameter[0]:
            return (parameter[0], False, "logical", [])
        else:
            type =""
            if "'" in parameter[0] or '"' in parameter[0]:
                type = "character"

            elif parameter[0].isnumeric() or parameter[0][1:].isnumeric():
                type = "integer"
            elif parameter[0].replace(".","").isnumeric() or parameter[0][1:].replace(".","").isnumeric():
                type = "real"
            elif "." in parameter:
                if all([part.isnumeric() or part[1:].isnumeric() for part in parameter[0].split(",")]):
                    type = "real"
            return (parameter[0],parameter[1],type,[])
    def get_param_info(self,parameter,local_variables,local_module_variables):
        info = self.get_param_info_recursive(parameter,local_variables,local_module_variables)
        new_dims = [self.evaluate_integer(dim) for dim in info[3]]
        return (info[0], info[1], info[2], new_dims)
    def get_static_passed_parameters(self,parameters,local_variables,local_module_variables):
        original_parameters = parameters
        parameters = [parameter.lower() for parameter in parameters]
        for i,param in enumerate(parameters):
            if len(param)>0 and param[0] == "-":
                parameters[i] = param[1:]
        res = list(zip(parameters,list(map(lambda x: (x in local_module_variables and x not in local_variables) or (x in local_variables and local_variables[x]["saved_variable"]),parameters))))
        for i,parameter in enumerate(res):
            param_name = parameter[0]
            func_calls = self.get_function_calls_in_line(param_name,local_variables)
            if len(func_calls) == 0:
               param_name = parameter[0].split("=")[-1].strip()
            info = self.get_param_info((param_name,parameter[1]),local_variables,local_module_variables)
            if len(parameter[0].split("=")) == 2:
                param_name,passed_param = [part.strip().lower() for part in parameter[0].split("=")]
                res[i] = (info[0],info[1],info[2],info[3],param_name)
            else:
                res[i] = (info[0],info[1],info[2],info[3],None)

        return res

    def get_interfaced_functions(self,file_path,subroutine_name):
        func_info = self.get_function_info(subroutine_name)
        if("interface_funcs" not in func_info):
            return [subroutine_name.lower()]
        if(file_path not in func_info["interface_funcs"]):
            #colliding names for a subroutine and an interface
            return [subroutine_name.lower()]
        if(len(func_info["interface_funcs"][file_path]) == 0):
             return [subroutine_name.lower()]
        res = [sub.lower() for sub in func_info["interface_funcs"][file_path]]
        if subroutine_name in res:
            pexit("subroutine in it's own interface",subroutine_name)
        return [sub for sub in res if file_path in self.func_info[sub]["lines"]]

    def generate_save_array_store(self,store_variable):
        res = f"{store_variable}_generated_array(imn"
        if len(self.static_variables[store_variable]['dims']) == 0:
            res += ",1"
        else:
            for i in range(len(self.static_variables[store_variable]['dims'])):
                res += ",:"
        res += f") = {store_variable}\n"
        return res

    def generate_read_from_save_array(self,store_variable):
        res = f"{store_variable} = {store_variable}_generated_array(imn"
        if len(self.static_variables[store_variable]['dims']) == 0:
            res += ",1"
        else:
            for i in range(len(self.static_variables[store_variable]['dims'])):
                res += ",:"
        res +=")\n"
        return res

    def generate_allocation_for_save_array(self,store_variable):
        res = f"{self.static_variables[store_variable]['type']}, dimension ("
        if self.static_variables[store_variable]["allocatable"]:
            res += ":"
        else:
            res += "nx*ny"
        if len(self.static_variables[store_variable]['dims']) == 0:
            if self.static_variables[store_variable]["allocatable"]:
                res += ",:"
            else:
                res += ",1"
        else:
            for dimension in self.static_variables[store_variable]["dims"]:
                res += f",{dimension}"
        res += ")"
        if self.static_variables[store_variable]["allocatable"]:
            res += ", allocatable"
        res += f" :: {store_variable}_generated_array\n"
        return res

    def save_static_variables(self):
        with open("static_variables.csv","w",newline='') as csvfile:
            writer = csv.writer(csvfile)
            for variable in self.static_variables.keys():
                writer.writerow([variable,self.static_variables[variable]["type"],self.static_variables[variable]["dims"],self.static_variables[variable]["allocatable"],self.static_variables[variable]["origin"],self.static_variables[variable]["public"],self.static_variables[variable]["threadprivate"]])
        
    def read_static_variables(self):
        with open("static_variables.csv","r",newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                self.static_variables[row[0]] = {"type": row[1], "dims": [dim for dim in row[2].replace("'","").strip('][').split(', ') if dim != ""], "allocatable": (row[3].lower() in ("yes", "true", "t", "1")), "origin": row[4], "public": (row[5].lower() in ("yes", "true", "t", "1")), "threadprivate": (row[6].lower() in ("yes", "true", "t", "1"))}

    def save_static_writes(self,static_writes):
        keys = static_writes[0].keys()
        with open("writes.csv","w",newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(static_writes)

    def read_static_writes(self):
        with open("writes.csv", mode="r") as infile:
            return [dict for dict in csv.DictReader(infile)]

    def get_threadprivate_declarations_and_generate_threadpriv_modules(self,files,threadprivate_variables,critical_variables):
        modules_file= open(f"{self.directory}/threadpriv_modules.inc","w")
        use_modules_file= open(f"{self.directory}/use_threadpriv_modules.inc","w")
        copy_in_call_file = open(f"{self.directory}/copyin.inc","w")
        threadprivate_declarations = {}
        exceptions = ["cdata.f90","cparam.f90"]
        for file in files:
            threadprivate_variables_to_add = []
            static_variables_end = self.get_contains_line_num(file)
            vars = list(self.get_variables(self.get_lines(file, 1, static_variables_end), {}, file).keys())
            vars_to_make_private = [variable for variable in vars if variable in self.static_variables and (variable in threadprivate_variables or  self.static_variables[variable]["threadprivate"]) and variable not in critical_variables and variable != ""]
            if(len(vars_to_make_private) > 0):
                module = self.get_own_module(file)
                copy_in_file = open(f"{file.replace('.f90','')}_copyin.inc","w")
                if not any([exp in file for exp in exceptions]):
                    copy_in_file.write(f"subroutine copyin_{module}\n")
                copy_in_file.write("!$omp parallel copyin( &\n")
                var_num = 0
                for var in vars_to_make_private:
                    if var_num > 0:
                        copy_in_file.write(f"!$omp ,{var}  &\n")
                    else:
                        copy_in_file.write(f"!$omp {var}  &\n")
                    var_num += 1
                res_lines = [f"!$omp threadprivate({var})" for var in vars_to_make_private]
                res_line = ""
                for x in res_lines:
                    res_line += x + "\n"
                threadprivate_declarations[file] = res_line
                modules_file.write(f"{self.get_own_module(file)}\n")
                use_modules_file.write(f"use {self.get_own_module(file)}\n")
        
                copy_in_file.write("!$omp )\n")
                copy_in_file.write("!$omp end parallel\n")
                if not any([exp in file for exp in exceptions]):
                    copy_in_file.write(f"end subroutine copyin_{module}\n")
                    copy_in_call_file.write(f"call copyin_{module}\n")
                copy_in_file.close()

        modules_file.close()
        copy_in_call_file.close()
        use_modules_file.close()
        return threadprivate_declarations

    def generate_copy_in(self,files,threadprivate_variables):
        variables = []
        for file in files:
            threadprivate_variables_to_add = []
            static_variables_end = self.get_contains_line_num(file)
            vars = list(self.get_variables(self.get_lines(file, 1, static_variables_end), {}, file).keys())
            private_vars = [variable for variable in vars if variable in self.static_variables and (variable in threadprivate_variables or self.static_variables[variable]["threadprivate"]) and variable != ""]
            variables.extend(private_vars)
            variables = unique_list(variables)
        print("Creating copyin.inc")
        split_str = ",\n!$omp"
        copyin_line = f"!$omp parallel copyin({split_str.join(variables)})\n"
        copyin_file = open("copyin.inc", "w")
        # copyin_file.write("subroutine copyin_func()\n")
        copyin_file.write(copyin_line)
        copyin_file.write("!$omp end parallel")
        copyin_file.close()

                    
    def add_atomic_declarations(self,variables):
        files = self.used_files
        for file in files:
            res_contents = []
            contents = self.get_lines(file,include_comments=True)
            for i,line in enumerate(contents):
                res_contents.append(line)
                if line[0] != "!":
                    writes = self.get_writes_from_line(line)

                    #handle where writes
                    if "if" in line or line.split("(")[0].strip() == "where":
                        for variable in variables:
                            if variable in [write["variable"] for write in writes]:
                                last_line = res_contents[-1]
                                res_contents[-1] = "!omp critical\n"
                                res_contents.append(last_line)
                                res_contents.append("!omp end critical\n")
                    #handle do loop writes
                    elif re.match("do\s+.=.+,\s?",line.split(";")[0]) and len(line.split(";"))==2:
                        for variable in variables:
                            if variable in [write["variable"] for write in writes]:
                                res_contents[-1] = (f"{line.split(';')[0]}\n")
                                res_contents.append("!omp critical\n")
                                res_contents.append(f"{line.split(';')[1]}\n")
                                res_contents.append("!omp end critical\n")
                                #let's see if there is a corresponding enddo
                                current_index = i+1
                                no_end_do = True
                                iter_line = contents[current_index][0]
                                while no_end_do and not re.match("do\s+.=.+,\s?",iter_line):
                                    no_end_do = not (re.match("enddo",iter_line) or re.match("end do",iter_line))
                                    current_index += 1
                                    iter_line = contents[current_index][0]
                                if no_end_do:
                                    res_contents.append("enddo")
                                
                    else:
                        for variable in variables:
                            if variable in [write["variable"] for write in writes]:
                                last_line = res_contents[-1]
                                res_contents[-1] = "!omp critical\n"
                                res_contents.append(last_line)
                                res_contents.append("!omp end critical\n")
                                #If one one's the more performant omp atomic
                                #last_line = res_contents[-1]
                                #res_contents[-1] = "!$omp atomic\n"
                                #res_contents.append(last_line)

            with open(f"./out/{file}","w") as f:
                f.write("\n".join(res_contents))

    def make_variables_public(self,variables):
        files = unique_list([self.static_variables[variable]["origin"] for variable in variables ])
        for file in files:
            res_contents = []
            contents = self.get_lines(file,include_comments=True)
            in_module_declaration = True
            in_struct = False
            for count,line in enumerate(contents):
                res_contents.append(line)
                if line.split(" ")[0] == "type":
                    in_struct = True
                if line.split(" ")[0] == "endtype":
                    in_struct = False
                if in_module_declaration and not in_struct:
                    if re.match("function\s+.+\(.+\)",line) or re.match("subroutine\s+.+\(.+\)",line) or line.strip() == "contains":
                        in_module_declaration = False
                    parts = line.split("::")
                    if len(parts) > 1:
                        line_variables = parts[1].strip()
                        variable_names = self.get_variable_names_from_line(line_variables)
                        res_contents.extend([f"public :: {variable}" for variable in variable_names if variable in variables and self.static_variables[variable]["origin"] == file and not self.static_variables[variable]["public"]])

            with open(f"./out/{file}","w") as f:
                f.write("\n".join(res_contents))
    def get_function_info(self,function):
        return self.func_info[function]
    def is_elemental(function):
        func_info = self.get_function_info(function)
        return func_info["is_elemental"]

    def choose_correct_interfaced_function(self,call,interfaced_functions,parameter_list,file_path):
        #Unfortunately different modules have different calling structures of the same function so can be 0 for a file.
        subroutine_name = call["function_name"]
        ##TODO: these use different views of pointers from the caller and function views i.e. pass a multidimensional array and the function takes a flat single-dimensional array
        ##Luckily these are not overloaded so do not have to take care of them
        if len(interfaced_functions) == 1:
            subroutine_lines = self.get_subroutine_lines(interfaced_functions[0], file_path)
            local_variables = {parameter:v for parameter,v in self.get_variables(subroutine_lines, {},file_path, True).items() }
            parameters = self.get_parameters(subroutine_lines[0])
            mandatory_vars = [var for var in parameters if not local_variables[var]["optional"]]
            print("HMM: ",interfaced_functions[0], parameters,parameter_list,mandatory_vars)
            if len(mandatory_vars) > len(parameter_list):
                return []
            #if "pic" in interfaced_functions[0] and len(parameters) == len(parameter_list):
            #if len(mandatory_vars) <= len(parameter_list):
            #    return interfaced_functions
            return interfaced_functions
            #if len(parameters) == len(parameter_list):
            #    return interfaced_functions
            #return []
        suitable_functions = []
        for function in interfaced_functions:
            is_suitable = True
            subroutine_lines = self.get_subroutine_lines(function, file_path)
            parameters = self.get_parameters(subroutine_lines[0])
            is_elemental = "elemental" in subroutine_lines[0]
            #if(local_variables is None): 
            local_variables = {parameter:v for parameter,v in self.get_variables(subroutine_lines, {},file_path, True).items() }
            #subroutine_lines = self.rename_lines_to_internal_names(subroutine_lines,local_variables,file_path,function)
            #local_variables = {parameter:v for parameter,v in self.get_variables(subroutine_lines, {},file_path, True).items() }
            mapping = self.get_parameter_mapping(parameters,parameter_list)
            #if mapping is less than parameter list than some named optional paramaters are not present in sub parameters
            is_suitable  = len(mapping) == len(parameter_list) and len(parameters) >= len(parameter_list)
            ##check if type and length of dims match between passed parameter and function parameter.
            ## All other parameters need to be optional 
            if is_suitable:
                for i, passed_param in enumerate(parameter_list):
                    is_suitable =  is_suitable and  passed_param[2] == local_variables[parameters[mapping[i]]]["type"] 
                    if not is_elemental:
                        is_suitable = is_suitable and len(passed_param[3]) == len(local_variables[parameters[mapping[i]]]["dims"])
                for i in [j for j in range(len(parameters)) if j not in mapping]:
                    is_suitable = is_suitable and local_variables[parameters[i]]["optional"]
            if is_suitable:
                num_of_needed_params = 0
                for param in parameters:
                    if not local_variables[param]["optional"]:
                        num_of_needed_params = num_of_needed_params + 1
                is_suitable = is_suitable and len(parameter_list) >= num_of_needed_params
            if is_suitable:
                suitable_functions.append(function)
        num_of_suitable_needed = 1 if self.offloading else 0
        if len(interfaced_functions) == 1 and interfaced_functions[0] == "ghost_particles_send":
            return interfaced_functions
        if len(suitable_functions) > 1 or len(suitable_functions)<num_of_suitable_needed:
            pexit(f"There are {len(suitable_functions)} suitable functions for the interface call: ",subroutine_name, "in file", file_path, "Params: ",parameter_list,"Original candidates: ", interfaced_functions ,"Possibilities", suitable_functions,call)
        return suitable_functions
    def get_module_file(self,module):
        possible_filepaths = []
        for x in self.chosen_modules:
            if module == x:
                return f"{self.directory}/{self.chosen_modules[module]}.f90"
        pexit("did not find a file for module: " + module)
    def choose_right_module(self,filepaths,call,original_file):
        filepaths = unique_list(filepaths)
        if len(filepaths) == 1:
            return filepaths[0]
        for i, path in enumerate(filepaths):
            for module in self.chosen_modules:
                if path.lower() == f"{self.directory}/{self.chosen_modules[module]}.f90".lower():
                    return filepaths[i]
        if original_file in self.func_info[call["function_name"]]["lines"]:
          return original_file
        pexit("did not found module in files",filepaths)
        
    def parse_subroutine_all_files(self, sub_call, call_trace, check_functions, offload,local_variables,file_called_from, layer_depth=math.inf, parameter_list=[], only_static=True):
        subroutine_name = sub_call["function_name"]
        if layer_depth<0:
            return []
        if subroutine_name in self.external_funcs:
            return []
        self.subroutine_order += 1
        if subroutine_name not in self.subroutine_modifies_param:
            self.subroutine_modifies_param[subroutine_name] = {}
        file_paths = self.find_subroutine_files(subroutine_name)
        print("Parse all files", subroutine_name, parameter_list)
        #if no files than it must be in the file_called_from
        if len(file_paths) == 0:
            file_paths = [file_called_from]
        global_modified_list = []
        param_types = [(param[2],len(param[3])) for param in parameter_list]
        all_functions = []
        all_found_functions = []
        for file_path in file_paths:
            interfaced_functions = self.get_interfaced_functions(file_path,subroutine_name)
            all_functions.extend(interfaced_functions)
            interfaced_functions = self.choose_correct_interfaced_function(sub_call,interfaced_functions,parameter_list,file_path)
            for function in interfaced_functions:
                all_found_functions.append(function)
                self.parse_subroutine_in_file(file_path, function, check_functions,offload,global_modified_list, layer_depth, call_trace,parameter_list,only_static)
        if len(all_found_functions) == 0 and len(all_functions)>0:
            pexit(f"There are no suitable function for the interface call: ",subroutine_name, "Params: ",parameter_list, "Original candidates: ", all_functions,sub_call)
        self.subroutine_modifies_param[subroutine_name][str(param_types)] = global_modified_list

    def replace_vars_in_lines(self,lines, new_names,exclude_variable_lines=False,structs=False):
        if exclude_variable_lines:
            res_lines = []
            for line in lines:
                if is_body_line(line):
                    res_lines.append(self.replace_variables_multi(line,new_names))
                else:
                    res_lines.append(line)
            return res_lines
        return [self.replace_variables_multi(line, new_names, structs) for line in lines]
    def replace_var_in_lines(self, lines, old_var, new_var):
        # return [add_splits(replace_variable(line,old_var,new_var)) for line in lines]
        return [replace_variable(line,old_var,new_var) for line in lines]
    def get_subroutine_lines(self,subroutine_name,filename):
        func_info = self.get_function_info(subroutine_name)
        return func_info["lines"][filename]
    def turn_f_size(self,params):
        if len(params) > 1:
            dim = params[1][0]
            if dim == "1":
                return global_subdomain_range_with_halos_x
            elif dim == "2":
                return global_subdomain_range_with_halos_y
            elif dim  == "3":
                return global_subdomain_range_with_halos_z
            elif dim == "4":
                return number_of_fields
            else:
                pexit("Weird dim in size",dim,params)
        else:
            return "mx*my*mz"
    def get_dim_info(self,param,local_variables,variables_in_scope,writes):
        if param in local_variables:
            src = local_variables
        else:
            src = self.static_variables
        if not any([":" in dim or "size" in dim for dim in src[param]["dims"]]):
            return src[param]["dims"]
        var_writes = []
        res_dims = [":" for i in range(len(src[param]["dims"]))]
        for write in writes:
            if write["variable"] == param:
                var_writes.append(write)
                vars_in_line = [var for var in get_used_variables_from_line(write["line"]) if var in self.static_variables or var in local_variables]
                vars_dims = []
                for var in vars_in_line:
                    if var in local_variables:
                        src = local_variables
                    else:
                        src = self.static_variables
                    vars_dims.append(src[var]["dims"])
                dims_index = 0
                for var_dims in vars_dims:
                    for i, var_dim in enumerate(var_dims):
                        if var_dim != ":" and "size" not in var_dim:
                            if len(res_dims) <= i:
                                res_dims.append(var_dim)
                            else:
                                res_dims[i] = var_dim
        return res_dims

    def get_size(self, func_call, local_variables, variables_in_scope, writes):
        params = func_call["parameters"]
        if params[0] == "f":
            return self.turn_f_size(params)
        elif params[0] in local_variables and all([":" not in dim and "size" not in dim for dim in local_variables[params[0]]['dims']]):
            sizes = local_variables[params[0]]["dims"]
        elif params[0] in local_variables and len(local_variables[params[0]]["dims"]) == 1 and "size" in local_variables[params[0]]["dims"][0]:
            size_func_call = self.get_function_calls_in_line(local_variables[params[0]]["dims"][0],local_variables)[0]
            sizes = self.get_size(size_func_call, local_variables, variables_in_scope, writes)
            local_variables[params[0]]["dims"] == ["*".join(sizes)]
        else:
            info = self.get_param_info((params[0],False),local_variables,self.static_variables)
            sizes = info[3]
        if len(func_call["parameters"]) == 1:
            return "*".join(sizes)
        else:
            dim = func_call["parameters"][1]
            return  sizes[int(dim)-1]
    def replace_func_call(self, line,func_call, replacement):
        return line[:func_call["range"][0]] + replacement + line[func_call["range"][1]:]
    def get_function_return_var_info(self,subroutine_name,subroutine_lines,local_variables):
            #has named return value
            local_variables = {parameter:v for parameter,v in self.get_variables(subroutine_lines, {},self.file, True).items() }
            result_func_calls = [call for call in self.get_function_calls_in_line(subroutine_lines[0],local_variables) if call["function_name"] == "result"]
            if "result" in subroutine_lines[0] and len(result_func_calls) == 1:
                return_var = result_func_calls[0]["parameters"][0]
            else:
            #if not named them default name is the function name
                return_var = subroutine_name
            #return value type is the local variable with it's name
            if return_var in local_variables:
                type = local_variables[return_var]["type"]
                return_dims = local_variables[return_var]["dims"]
            else:
                return_dims = []
                if "character" in subroutine_lines[0]:
                    type = "character"
                elif "real" in subroutine_lines[0]:
                    type = "real"
                elif "integer" in subroutine_lines[0]:
                    type = "integer"
                elif "logical" in subroutine_lines[0]:
                    type = "logical"
            return (return_var, type, return_dims)
    def get_replaced_body(self, filename, parameter_list, function_call_to_replace, variables_in_scope,global_init_lines,subs_not_to_inline,elim_lines):
        original_subroutine_name = function_call_to_replace["function_name"]
        ##in case is interfaced call get the correct subroutine
        # print("GETTING REPLACED BODY FOR: ", function_call_to_replace)
        if function_call_to_replace["function_name"][:3] == "fft":
          pexit("probably shouldn't inline fft func")
        interfaced_functions = self.get_interfaced_functions(filename,original_subroutine_name)
        if len(interfaced_functions)>1:
            interfaced_functions = self.choose_correct_interfaced_function(function_call_to_replace,interfaced_functions,parameter_list,filename)
        subroutine_name = interfaced_functions[0]
        if "inlined_lines" not in self.func_info[subroutine_name]:
            self.inline_all_function_calls(filename,subroutine_name,self.get_subroutine_lines(subroutine_name,filename),subs_not_to_inline,elim_lines)
        if filename not in self.func_info[subroutine_name]["inlined_lines"]:
            self.inline_all_function_calls(filename,subroutine_name,self.get_subroutine_lines(subroutine_name,filename),subs_not_to_inline,elim_lines)
        subroutine_lines = self.func_info[subroutine_name]["inlined_lines"][filename]
        #if "notanumber_" in subroutine_name:
        #    return ([".false."],True,"logical")

        print("FILENAME: ",filename,original_subroutine_name,subroutine_name)
        init_lines = [line for line in subroutine_lines if is_init_line(line)]

        lines = subroutine_lines
        local_variables = {parameter:v for parameter,v in self.get_variables(subroutine_lines, {},filename, True).items() }
        variables = merge_dictionaries(self.static_variables,local_variables)

        res_init_lines = []
        for line in init_lines:
            if len([part.strip() for part in line.split("::")[-1].split(",")]) > 1:
                res = line.split("::")[-1].strip()
                vars = []
                num_of_left_brackets = 0
                num_of_right_brackets = 0
                buffer  = ""
                for char in res:
                    if char == "(":
                        num_of_left_brackets += 1
                    if char == ")":
                        num_of_right_brackets += 1
                    if char == "," and num_of_right_brackets == num_of_left_brackets:
                        vars.append(buffer.strip())
                        buffer = ""
                    else:
                        buffer = buffer + char
                if buffer.strip() != "":
                    vars.append(buffer.strip())
                for var in vars:
                    res_init_lines.append(line.split("::")[0].strip() + "::" + var)
            else:
                res_init_lines.append(line)
        init_lines = res_init_lines
        params = self.get_parameters(subroutine_lines[0])
        remove_indexes = []
        for i,line in enumerate(init_lines):
            for param in params:
                if param in [part.strip() for part in line.split("::")[-1].split(",")]:
                    remove_indexes.append(i)
        init_lines = [x[1] for x in enumerate(init_lines) if x[0] not in remove_indexes]


        is_function = "function" in subroutine_lines[0]
        if is_function:
            return_var,type,_ = self.get_function_return_var_info(subroutine_name,subroutine_lines,local_variables) 
        else:
            type = None
        result_func_calls = [call for call in self.get_function_calls_in_line(subroutine_lines[0],local_variables) if call["function_name"] == "result"]
        has_named_return_value = "result" in subroutine_lines[0] and len(result_func_calls) == 1
        if is_function and not has_named_return_value:
            init_lines.append(f"{type} :: {return_var}")


        new_lines = [x.lower() for x in lines]
        mapping = self.get_parameter_mapping(params, parameter_list)

        passed_param_names = [x.split("=",1)[-1].strip() for x in function_call_to_replace["parameters"]]
        present_params = [x[1] for x in enumerate(params) if x[0] in mapping]
        optional_present_params = [x for x in present_params if local_variables[x]["optional"]]
        not_present_params = [param for param in params if param not in present_params]

        #remove_lines that write to parameters or read from parameters that are not passed
        remove_indexes = []
        for i,line in enumerate(new_lines):
            writes = self.get_writes_from_line(line)
            if(len(writes) == 1):
                if writes[0]["variable"] in not_present_params:
                    remove_indexes.append(i)
                local_var_segments = get_var_segments_in_line(line,local_variables)
                if any([param in [x[0] for x in local_var_segments] for param in not_present_params]):
                    remove_indexes.append(i)


        new_lines = [x[1] for x in enumerate(new_lines) if x[0] not in remove_indexes]

        #replace present with false if not present and true if present
        for i,line in enumerate(new_lines):
            #Done for speed optimization
            if "present(" in line or "where(" in line:
                func_calls = self.get_function_calls_in_line(line,local_variables)
                present_func_calls = [func_call for func_call in func_calls if func_call["function_name"] == "present"]
                present_func_call_segments = [(None, call["range"][0], call["range"][1]) for call in present_func_calls]
                present_map_val = [line[call["range"][0]:call["range"][1]] if call["parameters"][0] in optional_present_params else ".false." if call["parameters"][0] in not_present_params else ".true." for call in present_func_calls]
                func_call_segments = present_func_call_segments
                map_val = present_map_val

                info = {
                    "map_val": map_val
                }
                line = self.replace_segments(func_call_segments,line,self.map_val_func,local_variables,info)
                new_lines[i] = line


        #replace all local_params that are not passed params with function specific names to prevent name collision with other inlined funcs
        new_local_var_names = {}
        for local_variable in [x for x in local_variables if x not in present_params]:
            new_local_var_names[local_variable] = f"{local_variable}_{self.inline_num}"
        #Rename return var
        if is_function:
            new_local_var_names[return_var] = f"{return_var}_return_value_{self.inline_num}"

        init_lines = self.replace_vars_in_lines(init_lines, new_local_var_names)
        new_lines = self.replace_vars_in_lines(new_lines, new_local_var_names)

        # print("PARAMS:",subroutine_lines,params)
        #replace variables with passed values
        for i, passed_param in enumerate(function_call_to_replace["parameters"]):
            #in case was passed as a named param
            passed_param = passed_param.split("=",1)[-1].strip()
            if "(" in passed_param:
              init_lines = self.replace_var_in_lines(init_lines, params[mapping[i]], passed_param)
              new_lines = self.replace_var_in_lines(new_lines, params[mapping[i]], passed_param)
            else:
              init_lines = self.replace_vars_in_lines(init_lines, {params[mapping[i]] : passed_param})
              new_lines = self.replace_vars_in_lines(new_lines, {params[mapping[i]]: passed_param})
        #for line in new_lines:
        #    print("HMM LINE",line)
        #exit()



        init_variables= {parameter:v for parameter,v in self.get_variables(init_lines, {},filename,True).items() }
        global_init_lines.extend(init_lines)
        global_init_lines = unique_list(global_init_lines)

        lines = new_lines
        lines = self.eliminate_while(lines)
        #Todo has to evaluate whether return line is hit or not
        has_return_line = False 
        for count,line in enumerate(lines):
            if line.strip() == "return":
                has_return_line = True
        for count,line in enumerate(lines):
            if line.strip() == "return":
                print("don't want to inline this func")
                print("present params",present_params)
                print("not present params",not_present_params)
                print(init_lines)
                print(new_local_var_names)
                print(mapping)
                print(parameter_list)
                print(not_present_params)
                print(lines)
                print("iux__mod__cdata" in self.static_variables)
                print(local_variables.keys())
                print("\nlines:")
                print("\n\n")
                lines = self.eliminate_while(lines)
                for line in lines:
                    print(line)
                print("\n\n")
                print(subroutine_name, original_subroutine_name)
                pexit("don't want to return\n")
            #if "f_231" in line and not is_init_line(line) and "subroutine" not in line:
            #    pexit("WRONG LINE: ",line)
            #    print("wrong lines")
            #    for line in lines:
            #        print(line)
            #    print("present params")
            #    for param in present_params:
            #        print(param)
            #    print("call to replace")
            #    print(function_call_to_replace)
            #    pexit("Wrong")
        body_lines = [line for line in lines if is_body_line(line)]
        res_lines = [subroutine_lines[0].replace("subroutine","")]
        res_lines.append("{")
        for i in range(len(body_lines)):
            res_lines.append(body_lines[i])
        res_lines.append("}");
        #res_lines.append(subroutine_lines[-1].replace("end subroutine","").replace("endsubroutine", ""))
        #if(self.inline_num > 10):
        #  print("subroutine lines: ",subroutine_lines)
        #  print("res lines: ",res_lines)
        #  exit(0)
        return ([line for line in lines if is_body_line(line)],is_function,type)
    def parse_subroutine_in_file(self, filename, subroutine_name, check_functions, offload, global_modified_list = [], layer_depth=math.inf, call_trace="", parameter_list=[], only_static=True):
        print("parse_in_file", subroutine_name, filename,parameter_list)
        if layer_depth < 0:
            return []
        if subroutine_name not in self.subroutine_modifies_param:
            self.subroutine_modifies_param[subroutine_name] = {}
        subroutine_lines = self.get_subroutine_lines(subroutine_name, filename)
        local_variables = {parameter:v for parameter,v in self.get_variables(subroutine_lines, {},filename, True).items() }
        lines = subroutine_lines[1:]
        lines = self.rename_lines_to_internal_names(lines,local_variables,filename,subroutine_name)
        lines = self.normalize_if_calls(lines, local_variables)
        lines = self.normalize_where_calls(lines, local_variables)
        lines = self.unroll_forall(lines,local_variables,self.static_variables)
        own_module = self.get_own_module(filename)
        #used to parse module don't think it is needed anymore
        for var in local_variables:
            if local_variables[var]["saved_variable"] and not local_variables[var]["parameter"]:
            #skipping special modules for now
                if subroutine_name in ["bc_file_x","interpolate_linear_range","linear_interpolate","linear_interpolate_quadratic","yshift_block_bspline","bspline_interpolation","get_gas_density","interpolate_quadratic"] or var in ["nmig_max","t_foreign"] or any([x in filename for x in ["messages.f90","boundcond.f90"]]):
                #if subroutine_name in ["bc_file_x","div","bspline_interpolation","set_from_slice_x", "set_from_slice_y", "set_from_slice_z","bc_aa_pot_field_extrapol","linear_interpolate","linear_interpolate_quadratic"] or own_module in ["special","boundcond","particles_mpicomm","particles_map","messages","ghostfold"]:
                    continue
                #Commented out for now
                #pexit("saved variable",var,local_variables[var],f"in: {subroutine_name}",filename,"abort")
        #local_module_variables = self.get_local_module_variables(filename,subroutine_name)
        parameters = self.get_static_parameters(subroutine_lines[0], parameter_list)
        if self.subroutine_order == 0:
            call_trace = subroutine_name
        else:
            call_trace = f"{call_trace} -> {subroutine_name}"
        writes = self.get_writes(lines)
        param_is_modified= {}
        for (parameter,(passed_parameter,is_static,_,_)) in parameters:
            param_is_modified[parameter] = {"is modified": False, "line" : "", "filename": filename, "call trace": call_trace}
        for write in writes:
            ##Remember to check for function return values
            if (write["variable"] not in self.static_variables) and (write["variable"].strip() not in local_variables) and write["variable"].strip() != subroutine_name.strip():
                print("Variable: ", write["variable"], "In function: ", subroutine_name, "Didn't have an origin")
                print("Either a bug or a missing file!")
                print("LINE:",write["line"])
                pexit("FILE " + filename)
            write["is_static"] = (write["variable"] in self.static_variables and write["variable"] not in local_variables) or (write["variable"] in local_variables and local_variables[write["variable"]]["saved_variable"])
            for (parameter,(passed_parameter,is_static,_,_)) in parameters:
                if write["variable"] == parameter:
                    write["variable"] = passed_parameter
                    write["is_static"] = is_static
                    param_is_modified[parameter] = {"is modified": True, "line" : write["line"], "filename": filename, "call trace": call_trace}
        for i, (param,(passed_parameter,is_static,_,_)) in enumerate(parameters):
            if i < len(global_modified_list):
                if param_is_modified[param]["is modified"]:
                    global_modified_list[i] = param_is_modified[param]
            else:
                global_modified_list.append(param_is_modified[param])
        for write in writes: 
            if write["is_static"]:
                self.add_write(write["variable"], write["line_num"], write["variable"] in local_variables and local_variables[write["variable"]]["saved_variable"], filename, call_trace, write["line"])
        for function_call in self.get_function_calls(lines, local_variables):
            function_name = function_call["function_name"].lower()
            if function_name in self.external_funcs: 
                continue
            parse = True
            if function_name in check_functions or function_name.lower().startswith("mpi"):
                self.found_function_calls.append((call_trace, function_name, parameter_list))
                parse = False
            parse = parse and function_name.lower().strip() not in self.ignored_subroutines
            if parse:
                new_param_list = self.get_static_passed_parameters(function_call["parameters"],local_variables,self.static_variables)
                param_types = [(param[2],len(param[3])) for param in new_param_list]
                parse = (parse and ((function_name, str(param_types)) not in self.parsed_subroutines) and function_name != subroutine_name)
        # for i in range(len(new_param_list)):
            if parse:
                self.parsed_subroutines.append((function_name,str(param_types)))
                self.parse_subroutine_all_files(function_call,call_trace, check_functions, offload, local_variables, filename, layer_depth-1,new_param_list,only_static)
                #see if passed params were modified
                #if modified add them as writes and in case they were themselves passed params signal that they are modified for subroutines calling this subroutine
                for i in range(len(new_param_list)):
                    for j, (parameter,(passed_parameter,is_static,_,_)) in enumerate(parameters):
                        if new_param_list[i][0] == parameter:
                            if self.subroutine_modifies_param[function_name][str(param_types)][i]["is modified"]:
                                global_modified_list[j] = {"is modified": True, "filename": self.subroutine_modifies_param[function_name][str(param_types)][i]["filename"], "line": self.subroutine_modifies_param[function_name][str(param_types)][i]["line"], "call trace": self.subroutine_modifies_param[function_name][str(param_types)][i]["call trace"]}
                                if is_static:
                                    self.add_write(passed_parameter, 0, False, filename, self.subroutine_modifies_param[function_name][str(param_types)][i]["call trace"],self.subroutine_modifies_param[function_name][str(param_types)][i]["line"])
            elif function_name.lower().strip() not in self.ignored_subroutines and function_name not in check_functions and not function_name.lower().startswith("mpi"):

                # check if the subroutine name is itself in case of an recursive call
                if function_name == subroutine_name:
                    for i in range(len(new_param_list)):
                        if global_modified_list[i]["is modified"] and new_param_list[i][0] in self.static_variables or (new_param_list[i][0] in local_variables and local_variables[new_param_list[i][0]]["saved_variable"]):
                            self.add_write(new_param_list[i][0], 0, new_param_list[i][0] in local_variables, filename, call_trace, global_modified_list[i]["line"])
                        for j, (parameter,(passed_param,is_static,_,_)) in enumerate(parameters):
                            if new_param_list[i][0] == parameter and global_modified_list[i]["is modified"]:
                                global_modified_list[j] = {"is modified": True, "filename": global_modified_list[i]["filename"], "line": global_modified_list[i]["line"], "call trace": call_trace}
                                if is_static:
                                    self.add_write(passed_parameter, 0, False, filename, call_trace, global_modified_list[i]["line"])
                else:
                    new_param_list = self.get_static_passed_parameters(function_call["parameters"],local_variables,self.static_variables)
                    for i in range(len(new_param_list)):
                        if self.subroutine_modifies_param[function_name][str(param_types)][i]["is modified"] and ((new_param_list[i][0] in self.static_variables and new_param_list[i][0] not in local_variables) or (new_param_list[i][0] in local_variables and local_variables[new_param_list[i][0]]["saved_variable"])): 
                            self.add_write(new_param_list[i][0], 0, new_param_list[i][0] in local_variables, self.subroutine_modifies_param[function_name][str(param_types)][i]["filename"], self.subroutine_modifies_param[function_name][str(param_types)][i]["call trace"], self.subroutine_modifies_param[function_name][str(param_types)][i]["line"])
    def evaluate_ifs(self,lines,local_variables):
        for line_index, line in enumerate(lines):
            #Disregard lines that are not possible
            #Done for speed optimization
            if "if" in line and "(" in line:
                orig_line = line
                check_line = line.replace("else if","if",1).replace("elseif","if",1)
                func_calls = self.get_function_calls_in_line(check_line,local_variables)
                if_func_calls = list(filter(lambda func_call: func_call["function_name"] == "if", func_calls))
                replacement_value = None
                if len(if_func_calls) == 1:
                    func_call = if_func_calls[0]
                    if len(func_call["parameters"]) == 1 and func_call["parameters"][0] in self.known_values:
                        replacement_value = self.known_values[func_call["parameters"][0]]
                    if len(func_call["parameters"]) == 1 and not replacement_value:
                        replacement_value = func_call["parameters"][0]
                        variables_to_replace = [variable for variable in local_variables if "value" in local_variables[variable] and variable in func_call["parameters"][0]]
                        var_replacements = {}
                        for var in variables_to_replace:
                            var_replacements[var] = local_variables[var]["value"]
                        replacement_value = self.replace_variables_multi(replacement_value,var_replacements)
                    if not replacement_value:
                        replacement_value = self.evaluate_boolean(func_call["parameters"][0],local_variables,[call for call in func_calls if call["function_name"] != "if"])
                    else:
                        replacement_value = self.evaluate_boolean(replacement_value,local_variables,[call for call in func_calls if call["function_name"] != "if"])
                    if replacement_value:
                        if "else" not in line:
                            if_str = "if"
                        else:
                            if_str ="else if"
                        orig_line = line 
                        lines[line_index] = self.replace_func_call(check_line, func_call, f"{if_str}({replacement_value})").replace("call if","if").replace("call else if", "else if")

    def global_loop_replacer(self,segment,segment_index,line,local_variables,info):
        if segment[0] in local_variables:
            dims = local_variables[segment[0]]["dims"]
        else:
            dims = self.static_variables[segment[0]]["dims"]
        indexes = get_segment_indexes(segment,line,len(dims))
        for i, index in enumerate(indexes):
            if info["iterator"] in index:
                    if i!=info["replacement_index"]:
                        print("ITERATOR: ",info["iterator"])
                        print("ITERATOR: ",index)
                        pexit("wrong replacement_index",info,line)
                    new_lower= index.replace(info["iterator"],info["replacement_lower"])
                    new_upper= index.replace(info["iterator"],info["replacement_upper"])
                    if new_lower == "1" and new_upper == dims[i]:
                        indexes[i] = ":"
                    else:
                        indexes[i] = new_lower + ":" + new_upper
        return build_new_access(segment[0],indexes)
    def remove_global_loops(self, lines,local_variables,global_loop_lines,iterators):
        remove_indexes = []
        variables = merge_dictionaries(local_variables,self.static_variables)
        for loop_arr in global_loop_lines:
            remove_indexes.append(loop_arr[0][1])
            remove_indexes.append(loop_arr[-1][1])

            #this needed because funny business with nsave
            if self.offload_type == "boundcond":
                iterator_index = loop_arr[0][1] - 1
                rhs_var = self.get_rhs_variable(lines[iterator_index])
                while(rhs_var == global_loop_z or rhs_var == global_loop_y):
                    remove_indexes.append(iterator_index)
                    iterator_index -= 1
                    rhs_var = self.get_rhs_variable(lines[iterator_index])
                iterator_index = loop_arr[-1][1] + 1
                rhs_var = self.get_rhs_variable(lines[iterator_index])
                while(rhs_var == global_loop_z or rhs_var == global_loop_y):
                    remove_indexes.append(iterator_index)
                    iterator_index += 1
                    rhs_var = self.get_rhs_variable(lines[iterator_index])


        for loop_arr_index, loop_arr in enumerate(global_loop_lines):
            iterator,replacement_lower, replacement_upper, replacement_index = iterators[loop_arr_index]
            for line,line_index in loop_arr:
                lines[line_index] = self.replace_segments(self.get_array_segments_in_line(line,variables),line,self.global_loop_replacer,local_variables,{
                    "iterator": iterator, 
                    "replacement_lower": replacement_lower,
                    "replacement_upper": replacement_upper,
                    "replacement_index": replacement_index,
                })
        return [x[1] for x in enumerate(lines) if x[0] not in remove_indexes]
    def is_vector(self, lines, vector, local_variables):
        if vector in local_variables:
            src = local_variables
        elif vector in self.static_variables:
            src = self.static_variables
        if src[vector]["dims"] not in [[global_subdomain_range_x,"3"]]:
            return False
        return True
    def map_val_func(self,segment,segment_index, line, local_variables, info):
        return info["map_val"][segment_index]
    def rename_lines_to_internal_names(self,lines,local_variables,filename,subroutine=None):
        modules = self.file_info[filename]["used_modules"]
        if subroutine:
            sub_modules = self.get_subroutine_modules(filename,subroutine)
            for mod in [x for x in sub_modules if x not in modules]:
                modules[mod] = []
        for i, line in enumerate(lines):
            line_before = lines[i]
            lines[i] = self.rename_line_to_internal_names(line,local_variables,modules,self.get_own_module(filename))
        return lines

    def module_vars_the_same(self,found_modules,var):
        return ( not self.offloading and len(found_modules) > 1 and
             all([self.static_variables[self.rename_dict[mod][var]]["type"] == self.static_variables[self.rename_dict[found_modules[0]][var]]["type"] for mod in found_modules])
             and all([self.static_variables[self.rename_dict[mod][var]]["dims"] == self.static_variables[self.rename_dict[found_modules[0]][var]]["dims"] for mod in found_modules])
             )
    def rename_line_to_internal_names(self,line,local_variables,modules,own_module):
        if line.split(" ")[0].strip() in ["subroutine" ,"function"] or is_use_line(line):
          return line
        vars_in_modules = {}
        for mod in modules:
            vars_in_modules =  merge_dictionaries(vars_in_modules, self.rename_dict[mod])
        variables = merge_dictionaries(local_variables,vars_in_modules)
        #don't rename local variables
        for var in [x for x in variables]:
          if var in local_variables:
            del variables[var]
        var_segments = get_var_name_segments(line,variables)
        res = self.replace_segments(var_segments,line,self.rename_to_internal_module_name,local_variables,{"modules": modules,"own_module":own_module})
        if "qbc_xy" in line:
            print("SEGS: ",var_segments)
            print("RES: ",line, "--->",res)
        return res

    def rename_to_internal_module_name(self,segment,segment_index,line,local_variables,info):

        if segment[0] in local_variables:
            return line[segment[1]:segment[2]]
        if "__mod__" in segment[0]:
          return line[segment[1]:segment[2]]
        variables = merge_dictionaries(local_variables,self.static_variables)
        found_modules = []
        for i,mod in enumerate(info["modules"]):
            if segment[0] in self.module_info[mod]["public_variables"] and segment[0] in self.rename_dict[mod]:
              if info["modules"][mod]:
                if segment[0] in info["modules"][mod]:
                  found_modules.append(mod)
              else:
                found_modules.append(mod)
        #if didn't find to fallbacks
        if len(found_modules) == 0:
            if segment[0] in self.rename_dict[info["own_module"]]:
                found_modules.append(info["own_module"])
        
        ##For analysis it is okay if the variable is the same across modules
        if self.module_vars_the_same(found_modules,segment[0]):
            pass
        elif(len(found_modules) != 1):
            print(line[segment[1]:segment[2]])
            print(found_modules)
            print(len(found_modules))
            print(len(info["modules"]))
            print(info["modules"])
            print("ixbeam" in self.rename_dict["mpicomm"])
            pexit("should be only a single module")
        return line[segment[1]:segment[2]].replace(segment[0],self.rename_dict[found_modules[0]][segment[0]])

    def unroll_range(self,segment,segment_index,line,local_variables,info):
        variables = merge_dictionaries(local_variables,self.static_variables)
        sg_indexes = get_segment_indexes(segment,line,0)
        changed = False
        for i, index in enumerate(sg_indexes):
            if i==info["index_num"] and index==info["old_index"]:
                changed = True
                sg_indexes[i] = info["new_index"]
        if changed:
            return build_new_access(segment[0],sg_indexes)
        return line[segment[1]:segment[2]]

    def transform_pencils(self,lines,all_inlined_lines,local_variables):
        profile_replacements = {}
        #assuming all profiles are written in mn loop; if not use the lower
        for field in self.struct_table["pencil_case"]:
          new_name = f"ac_transformed_pencil_{field}"
          if new_name not in local_variables:
            local_variables[new_name] = self.struct_table["pencil_case"][field]
          profile_replacements[f"p%{field}"] = new_name

        ##get all profiles written to; can also assume that all profiles are calced at mn loop
        # for line_index,line in enumerate(lines):
        #     var_name = ""
        #     #do only for lines with profiles
        #     #done for speedoptim
        #     if "p%" in line:
        #         writes_in_line = self.get_writes_from_line(line,local_variables)
        #         if len(writes_in_line) == 1:
        #             write = writes_in_line[0]
        #             rhs_segment = get_variable_segments(line, [write["variable"]])
        #             if len(rhs_segment) == 0:
        #                 rhs_segment = self.get_struct_segments_in_line(line, [write["variable"]])
        #             rhs_segment  = rhs_segment[0]
        #             var_name = line[rhs_segment[1]:rhs_segment[2]].split("::",1)[-1].split("(",1)[0].strip()

        #         #do only for profiles
        #         if "p%" in var_name and var_name not in profile_replacements:
        #             var_info = self.get_param_info((var_name,False),local_variables,local_variables)
        #             new_name = var_name.replace("p%","ac_transformed_pencil_")
        #             if new_name not in local_variables:
        #                 local_variables[new_name] ={
        #                     "type":var_info[2],
        #                     "dims":var_info[3],
        #                     "saved_variable": False
        #                 }
        #             profile_replacements[var_name] = new_name
        #lines = self.replace_vars_in_lines(lines,profile_replacements, structs=True)
        res_lines = []
        for line_index,line in enumerate(lines):
            for replacement in profile_replacements:
                line = line.replace(replacement, profile_replacements[replacement])
            #add transformed profile lines
            if line_index == 3:
                for new_name in profile_replacements:
                    res_lines.append(f"real, dimension({','.join(local_variables[profile_replacements[new_name]]['dims'])}) :: {profile_replacements[new_name]}")
            res_lines.append(line)

        return res_lines
    def transform_sum_calls(self,lines,local_variables):
        variables = merge_dictionaries(local_variables,self.static_variables)
        for line_index,line in enumerate(lines):
                sum_calls =  [x for x in self.get_function_calls_in_line(line,variables) if x["function_name"] == "sum" ]
                modified_sum = True
                while(len(sum_calls)>0 and modified_sum):
                    call = sum_calls[0]
                    modified_sum = False 
                    if (len(call["parameters"]) == 2 and call["parameters"][1] == "2" and self.get_param_info((call["parameters"][0],False),local_variables,self.static_variables)[3] == [global_subdomain_range_x,"3"]):
                        line = self.replace_func_call(line,call,f"sum({call['parameters'][0]})")
                        lines[line_index] = line
                        modified_sum = True
                    sum_calls =  [x for x in self.get_function_calls_in_line(line,variables) if x["function_name"] == "sum" and len(x["parameters"]) == 2 and x["parameters"][1] == "2" and self.get_param_info((x["parameters"][0],False),local_variables,self.static_variables)[3] == [global_subdomain_range_x,"3"]]
        return lines
    def unroll_ranges(self,lines,local_variables):
        variables = merge_dictionaries(local_variables,self.static_variables)
        res_lines = []
        #if we have e.g. f(1:2) = x(1:2), where f is a three dimensional vector ->
        #f(1) = x(1)
        #f(2) = x(2)
        for line in lines:
            arr_segs_in_line = self.get_array_segments_in_line(line,variables)
            unroll = False
            lower = None
            upper = None
            for sg in arr_segs_in_line:
                indexes = get_segment_indexes(sg,line,0)
                if len(variables[sg[0]]["dims"]) > 0 and len(indexes) > 0:
                    unroll = unroll or variables[sg[0]]["dims"][-1]  == "3" and indexes[-1] in ["1:2", "2:3"]
                    if indexes[-1] in ["1:2"]:
                        assert(lower is None or lower == "1")
                        lower = "1"
                        upper = "2"
                    elif indexes[-1] in ["2:3"]:
                        assert(lower is None or lower == "2")
                        lower = "2"
                        upper = "3"
            if unroll:
                info = {
                    "index_num": len(indexes)-1,
                    "old_index": indexes[-1],
                    "new_index": lower,
                }
                line_lower = self.replace_segments(arr_segs_in_line,line,self.unroll_range,local_variables,info)
                info["new_index"] = upper
                line_upper = self.replace_segments(arr_segs_in_line,line,self.unroll_range,local_variables,info)
                res_lines.extend([line_lower,line_upper])
            else:
                res_lines.append(line)
        lines = res_lines
        res_lines = []

        #if Vector = Scalar spread into ->
        #Vector.x = Scalar
        #Vector.y = Scalar
        #Vector.z = Scalar

        for line in lines:
          writes = self.get_writes([line])
          if len(writes) == 1:
            write = writes[0]
            rhs_segment = get_variable_segments(line, [write["variable"]])[0]
            rhs_info =  self.get_param_info((line[rhs_segment[1]:rhs_segment[2]],False),local_variables,self.static_variables)
            if rhs_info[3] in [[global_subdomain_range_x,"3"],["3"]] :
              val_info = self.get_param_info((write["value"],False),local_variables,self.static_variables)
              rhs_var_dims = variables[rhs_segment[0]]["dims"]
              if val_info[3] == [] and rhs_var_dims[-1] in ["3"]:
                for dim in ["1","2","3"]:
                  if len(rhs_info[3]) == 2:
                    rhs_dims = [":",dim]
                  else:
                    rhs_dims = [dim]
                  res_lines.append(f"{build_new_access(rhs_segment[0],rhs_dims)} = {write['value']}") 
              else:
                res_lines.append(line)
            else:
              res_lines.append(line)
          else:
            res_lines.append(line)
        for line in res_lines:
            if "df(:,1)" in line:
                pexit("WRONG")
        return res_lines
    def inline_0d_replacer(self,segment,segment_index,line,local_variables,info):
        if segment_index== 0 and len(self.get_writes_from_line(line)) > 0:
            return line[segment[1]:segment[2]]
        if segment[0] in info["possible_values"]:
            return replace_variable(line[segment[1]:segment[2]], segment[0], f"({info['possible_values'][segment[0]]})")
        return line[segment[1]:segment[2]]
    def replace_segments(self,segments,line,map_func,local_variables,info):
        res_line = ""
        last_index = 0 
        for sg_index, segment in enumerate(segments):
            res_line = res_line + line[last_index:segment[1]]
            res_line = res_line + map_func(segment,sg_index,line,local_variables,info) 
            last_index = segment[2]
        res_line = res_line + line[last_index:]
        return res_line

    def inline_0d_writes(self, lines,local_variables):
            variables = merge_dictionaries(self.static_variables, local_variables)
            analyse_lines = self.get_analyse_lines(lines,local_variables)
            writes = self.get_writes([line[0] for line in analyse_lines])
            #only for logical and floats for the moment
            vars_to_check_safety = [var for var in variables if variables[var]["dims"] == [] and variables[var]["type"] in ["logical","real"]]
            safe_vars_to_inline = []
            for var in vars_to_check_safety:
                var_writes= [write for write in writes if write["variable"] == var]

                # for write in var_writes:
                #     lhs_local_vars = [var for var in local_variables if var in write["line"].split("=")[1]]
                #     no_unsafe_writes= no_unsafe_writes and lhs_local_vars == []

                if_nums = []
                nest_nums = []
                for write in var_writes:
                    for line in analyse_lines:
                        if line[3] == write["line_num"]:
                            if_nums.append(line[4])
                            nest_nums.append(line[1])
                all_are_in_the_same_if = all([if_num == if_nums[0] for if_num in if_nums])
                all_are_in_main_branch = all([nest_num == nest_nums[0] for nest_num in nest_nums])
                is_safe = all_are_in_the_same_if or all_are_in_main_branch
                #for time being all are safe
                if is_safe:
                    safe_vars_to_inline.append(var)
            possible_values = {}
            remove_indexes = []
            for line_index, line in enumerate(lines):
                res_line = ""
                last_index = 0
                line = self.replace_segments(get_var_segments_in_line(line,variables),line,self.inline_0d_replacer,local_variables,{
                    "possible_values": possible_values
                })
                lines[line_index] = line
                rhs_var = self.get_rhs_variable(line)
                if rhs_var in safe_vars_to_inline:
                    remove_indexes.append(line_index)
                    write = self.get_writes_from_line(line)[0]
                    possible_values[rhs_var] = write["value"]
            lines = [x[1] for x in enumerate(lines) if x[0] not in remove_indexes]
            lines = [line for line in lines if line != ""]
            return lines 


    def inline_1d_writes(self, lines,local_variables):
            analyse_lines = self.get_analyse_lines(lines,local_variables)
            writes = self.get_writes([line[0] for line in analyse_lines])
            # vars_to_check_safety = ["tmp1","tmp2"]
            vars_to_check_safety = local_variables
            safe_vars_to_inline = []
            for var in vars_to_check_safety:
                var_writes= [write for write in writes if write["variable"] == var]

                no_unsafe_writes = False 
                # for write in var_writes:
                #     lhs_local_vars = [var for var in local_variables if var in write["line"].split("=")[1]]
                #     no_unsafe_writes= no_unsafe_writes and lhs_local_vars == []

                is_safe = no_unsafe_writes
                if not no_unsafe_writes:
                    if_nums = []
                    nest_nums = []
                    for write in var_writes:
                        for line in analyse_lines:
                            if line[3] == write["line_num"]:
                                if_nums.append(line[4])
                                nest_nums.append(line[1])
                    all_are_in_the_same_if = all([if_num == if_nums[0] for if_num in if_nums])
                    all_are_in_main_branch = all([nest_num == nest_nums[0] for nest_num in nest_nums])
                    is_safe = all_are_in_the_same_if or all_are_in_main_branch
                if is_safe:
                    safe_vars_to_inline.append(var)
            variables = merge_dictionaries(self.static_variables, local_variables)
            possible_values = {}
            remove_indexes = []
            if_num = 0
            for line_index, line in enumerate(lines):
                line = line.strip()
                if "if" in line or "else" in line:
                    if_num += 1

                res_line = ""
                last_index = 0
                line = self.replace_segments(get_var_segments_in_line(line,variables),line,self.inline_replacer,local_variables,{
                    "possible_values": possible_values
                })
                lines[line_index] = line
                rhs_var = self.get_rhs_variable(line)
                if rhs_var in local_variables:
                    rhs_dim = len(variables[rhs_var]["dims"])
                    _ , num_of_looped_dims = get_dims_from_indexes(get_indexes(get_rhs_segment(line),rhs_var,rhs_dim),rhs_var)
                    if rhs_var in local_variables and rhs_dim <= 1 and rhs_dim == num_of_looped_dims and rhs_var in safe_vars_to_inline:
                        remove_indexes.append(line_index)
                        possible_values[rhs_var] = line.split("=")[1].replace("\n","")
            lines = [x[1] for x in enumerate(lines) if x[0] not in remove_indexes]
            lines = [line for line in lines if line != ""]
            return lines 

    def inline_known_parameters(self,lines,also_local_variables=True):
      inline_constants = {}
      remove_indexes = []
      local_variables = {parameter:v for parameter,v in self.get_variables(lines, {},self.file, True).items() }
      tmp_local_variables = {}
      #skip those local variables which value is not known or their value depend on their own value
      for var in local_variables:
        if("value" in local_variables[var] and var not in local_variables[var]["value"]):
            tmp_local_variables[var] = local_variables[var]
      local_variables = tmp_local_variables

      if also_local_variables:
        for var in local_variables:
            if local_variables[var]["parameter"] and local_variables[var]["dims"] == [] and "value" in local_variables[var]:
                inline_constants[var] = local_variables[var]["value"]
                #remove declarations for local parameters since they are not needed anymore
                remove_indexes.append(local_variables[var]["line_num"])
      #for var in self.static_variables:
      #    if self.static_variables[var]["parameter"] and self.static_variables[var]["dims"] == [] and "value" in self.static_variables[var] and var not in local_variables:
      #        inline_constants[var] = self.static_variables[var]["value"]
      return [x[1] for x in enumerate(self.replace_vars_in_lines(lines,inline_constants)) if x[0] not in remove_indexes]
    
    #used to translate l_a = l_a .or. l_b into:
    #if(l_b) l_a = .true.
    #this is easier for the transpiler to reason about since l_a's value does not depend it's previous value
    def translate_a_is_a_or_b(self,line,variables):
        writes_in_line = self.get_writes_from_line(line)
        if len(writes_in_line) != 1:
            return line
        write = writes_in_line[0]
        parts = [x.strip() for x in write["value"].split(".or.")]
        if len(parts) != 2:
            return line
        lhs = parts[0]
        rhs = parts[1]
        if lhs not in variables or variables[lhs]["type"] != "logical":
            return line
        if rhs not in variables or variables[rhs]["type"] != "logical":
            return line
        if lhs == write["variable"]:
            return f"if({rhs}) {lhs} = .true."
        elif rhs == write["variable"]:
            return f"if({lhs}) {rhs} = .true."
        return line
    def add_mapping(self,flag,val):
        self.flag_mappings[flag] = val
        if flag == "gravz__mod__gravity" and val == "grav_init_z(4)":
            pexit("added wrong val to gravz")
    def eliminate_while_body(self,lines,unroll,take_last_write_as_output):



        #if there are some writes to flagged params then not safe to substitue
        removed_flags = {}
        writes = self.get_writes(lines,False)
        flag_mappings = [x for x in self.flag_mappings]
        for flag_mapping in flag_mappings:
                if len([write for write in writes if write["variable"] == flag_mapping]) > 0:
                    removed_flags[flag_mapping] = self.flag_mappings[flag_mapping]
                    del self.flag_mappings[flag_mapping]
                #if we have something like a(1) and a=b, we want to remove all index mappings of a
                if "(" in flag_mapping and ")" in flag_mapping:
                    if len([write for write in writes if write["variable"] == flag_mapping.split("(")[0].strip()]) > 0:
                        removed_flags[flag_mapping] = self.flag_mappings[flag_mapping]
                        del self.flag_mappings[flag_mapping]


        ##Needed to remove size from variable dims
        local_variables = {parameter:v for parameter,v in self.get_variables(lines, {},self.file, True).items() }


        len_orig_removed_flags = len(removed_flags) + 1
        while(len_orig_removed_flags > len(removed_flags)):
          len_orig_removed_flags = len(removed_flags)
          lines = [self.expand_size_in_line(line,local_variables,writes) for line in lines]
          self.evaluate_ifs(lines,local_variables)
          orig_lines_len = len(lines)+1
          local_variables = {parameter:v for parameter,v in self.get_variables(lines, {},self.file, True).items() }
          while(orig_lines_len > len(lines)):
                  orig_lines_len = len(lines)
                  writes = self.get_writes(lines,False)
                  lines = self.eliminate_dead_branches(lines,local_variables)
                  self.try_to_deduce_if_params(lines,writes,local_variables,take_last_write_as_output)
                  self.evaluate_ifs(lines,local_variables)

          writes = self.get_writes(lines,False)
          analyse_lines = self.get_analyse_lines(lines,local_variables)
          func_calls = self.get_function_calls(lines,local_variables)
          self.try_to_deduce_if_params(lines,writes,local_variables,take_last_write_as_output)
          for flag in [x for x in removed_flags]:
            if flag in self.known_values:
              self.add_mapping(flag,self.known_values[flag])
              del removed_flags[flag]
          for flag in [x for x in removed_flags]:
            #if write in conditional branch not sure of value
            if [x for x in writes if x["variable"] == flag and analyse_lines[x["line_num"]][1] > 0] == []:
              if [x for x in func_calls if flag in x["parameters"] and x["function_name"] not in ["if","put_shared_variable","allocate"]] == []:
                self.add_mapping(flag,removed_flags[flag])
                del removed_flags[flag]
          for line_index in range(len(lines)):
            lines[line_index] = self.replace_variables_multi(lines[line_index],self.flag_mappings)

        # if "initialize_gravity" in lines[0]:
        #   # print([x for x in writes if x["variable"] == flag and analyse_lines[x["line_num"]][1] > 0])
        #   # print([x for x in func_calls if flag in x["parameters"] and x["function_name"] not in ["if"]])
        #   # print("lgravx__mod__cdata" in self.flag_mappings)
        #   file = open("init.txt","w")
        #   for line in lines:
        #     file.write(f"{line}\n")
        #   file.close()
        #   print(self.flag_mappings["lgravz__mod__cdata"])
        #   print(self.known_values["lgravz__mod__cdata"])
        #   exit()



          writes = self.get_writes(lines,False)
          analyse_lines = self.get_analyse_lines(lines,local_variables)
          func_calls = self.get_function_calls(lines,local_variables)
          for line_index in range(len(lines)):
            lines[line_index] = self.replace_variables_multi(lines[line_index],self.flag_mappings)
          self.evaluate_ifs(lines,local_variables)
          for val in self.known_values:
            if val[0] == "l" and val in self.static_variables and self.static_variables[val]["type"] == "logical":
              self.add_mapping(val,self.known_values[val])
          for flag in [x for x in removed_flags]:
            if flag in self.known_values:
              self.add_mapping(flag,self.known_values[flag])
              del removed_flags[flag]
          for flag in [x for x in removed_flags]:
            #if write in conditional branch not sure of value
            # if len([x for x in writes if x["variable"] == flag and analyse_lines[x["line_num"]][1] > 0]) == 0 and len([x for x in func_calls if flag in x["parameters"] and x["function_name"] not in ["if"]]) == 0:
            if [x for x in writes if x["variable"] == flag and analyse_lines[x["line_num"]][1] > 0] == []:
              if [x for x in func_calls if flag in x["parameters"] and x["function_name"] not in ["if","put_shared_variable"]] == []:
                self.add_mapping(flag,self.known_values[flag])
                del removed_flags[flag]
          self.try_to_deduce_if_params(lines,writes,local_variables,take_last_write_as_output)
          for flag in [x for x in removed_flags]:
            if flag in self.known_values:
              self.add_mapping(flag,self.known_values[flag])
              del removed_flags[flag]
          #the flag param value was mutated and was not able to deduce it
          for flag in removed_flags:
            if flag in self.default_mappings:
              del self.default_mappings[flag]

          # if "initialize_energy" in lines[0]:
          #   file = open("init.txt","w")
          #   for line in lines:
          #     file.write(f"{line}\n")
          #   file.close()
          #   print("bye")
          #   print("lheatc_kprof__mod__energy" in self.known_values)
          #   print(self.flag_mappings["lheatc_kprof__mod__energy"])
          #   exit()


                


        #   print(self.flag_mappings["lgravx__mod__cdata"])
        #   exit()

        return lines
    def pretty_print(self,lines,filename,local_variables):
        file = open(filename,"w")
        analyse_lines = self.get_analyse_lines(lines,local_variables)
        for line in analyse_lines:
            if ("if" in line and "(" in line) or (line == "else") or ("else" in line and "if" in line):
                line_prefix = "  "*(line[1]-1)
            else:
                line_prefix = "  "*line[1]
            line_middle = line[0]
            line_end = f": if num={line[2]}\tnest num={line[1]}\n"
            file.write(f"{line_prefix}{line_middle}{line_end}")
        file.close()
    def map_mappings(self,lines,mappings,local_variables):
        writes = self.get_writes(lines,False)
        #will also include function calls
        index_of_first_write_to_flag_mapping = {}
        for line_index in range(len(lines)):
            line_writes = [write for write in writes if write["line_num"] == line_index] 
            line_func_calls = [call for call in self.get_function_calls_in_line(lines[line_index],local_variables) if call["function_name"] not in ["if"]]
            for flag in [flag for flag in mappings if flag not in index_of_first_write_to_flag_mapping]:
                for write in line_writes:
                    if write["variable"] == flag:
                        index_of_first_write_to_flag_mapping[flag] = line_index
                for call in line_func_calls:
                    for param in call["parameters"]:
                        if param == flag:
                            index_of_first_write_to_flag_mapping[flag] = line_index

        for line_index in range(len(lines)):
            mappings_active = [flag for flag in mappings if (flag not in index_of_first_write_to_flag_mapping or index_of_first_write_to_flag_mapping[flag] > line_index)]
            lines[line_index] = self.replace_variables_multi(lines[line_index],{flag : mappings[flag] for flag in mappings_active})
            array_flags = [flag for flag in mappings_active if "(" in flag and ")" in flag and flag in lines[line_index]]
            lines[line_index] = self.replace_variables_multi_array_indexing(lines[line_index],{flag: mappings[flag] for flag in array_flags})
        # done twice since can have lflag_a -> lflag_b .or. lflag_c
        for line_index in range(len(lines)):
            mappings_active = [flag for flag in mappings if (flag not in index_of_first_write_to_flag_mapping or index_of_first_write_to_flag_mapping[flag] > line_index)]
            lines[line_index] = self.replace_variables_multi(lines[line_index],{flag : mappings[flag] for flag in mappings_active})
            array_flags = [flag for flag in mappings_active if "(" in flag and ")" in flag and flag in lines[line_index]]
            lines[line_index] = self.replace_variables_multi_array_indexing(lines[line_index],{flag: mappings[flag] for flag in array_flags})
        return lines
    def eliminate_while(self,lines,unroll=False,take_last_write_as_output=False,extra_mappings = None):
        orig_lines = lines
        return orig_lines
        local_variables = {parameter:v for parameter,v in self.get_variables(lines, {},self.file, True).items() }
        lines = self.normalize_where_calls(lines, local_variables)

        variables = merge_dictionaries(local_variables,self.static_variables)
        for line_index,_ in enumerate(lines):
            lines[line_index] = self.translate_a_is_a_or_b(lines[line_index],variables)


        lines = self.transform_case(lines)
        lines = self.normalize_if_calls(lines, local_variables)
        lines = self.normalize_where_calls(lines, local_variables)
        lines = self.normalize_if_calls(lines, local_variables)
        self.normalize_impossible_val()

        remove_indexes = []
        #Matthias said that lcoarse_mn is broken for now
        #No communication
        for symbol in ["lcoarse_mn__mod__cdata","lcommunicate"]:
          for line_index,line in enumerate(lines):
            writes = self.get_writes_from_line(line)
            if len(writes) == 1 and (writes[0]["variable"] == symbol or writes[0]["variable"] == ".false."):
              remove_indexes.append(line_index)
          lines = self.replace_var_in_lines(lines,symbol,".false.")
        lines = [x[1] for x in enumerate(lines) if x[0] not in remove_indexes]
        for line_index, _ in enumerate(lines):
            lines[line_index] = lines[line_index].replace(".false)",".false.)").replace(".true)",".true.)")


        local_variables = {parameter:v for parameter,v in self.get_variables(lines, {},self.file, True).items() }
        lines = self.inline_known_parameters(lines,local_variables)
        if unroll:
          lines = self.unroll_constant_loops(lines,local_variables)

        for line_index in range(len(lines)):
            for x in [x for x in self.flag_mappings if "(" in x and ")" in x]:
                orig_line = lines[line_index]
                lines[line_index]= lines[line_index].replace(x,self.flag_mappings[x])

        if(extra_mappings):
            lines = self.map_mappings(lines,extra_mappings,local_variables)

        start_line_len = len(lines)+1
        while(start_line_len > len(lines)):
            start_line_len = len(lines)

            #replace with flag mappings until the first write to it
            lines = self.map_mappings(lines,self.flag_mappings,local_variables)
            self.evaluate_ifs(lines,local_variables)
            orig_lines_len = len(lines) + 1
            while(orig_lines_len > len(lines)):
                    orig_lines_len = len(lines)
                    writes = self.get_writes(lines,False)
                    lines = self.eliminate_dead_branches(lines,local_variables)
                    self.try_to_deduce_if_params(lines,writes,local_variables,take_last_write_as_output)
                    self.evaluate_ifs(lines,local_variables)

        
        lines = self.eliminate_while_body(lines,unroll,take_last_write_as_output)
        return orig_lines
        #return lines

            # if "initialize_energy" in lines[0]:

            #     lines = self.map_mappings(lines)
            #     self.try_to_deduce_if_params(lines,writes,local_variables,take_last_write_as_output)
            #     self.evaluate_ifs(lines,local_variables)

            #     filename = f"res-initialize-energy-{iter_num}.txt"
            #     self.pretty_print(lines,filename,local_variables)
            #     iter_num += 1

            #     # orig_lines_len = len(lines)
            #     # writes = self.get_writes(lines,False)
            #     lines = self.eliminate_dead_branches(lines,local_variables)
            #     # self.try_to_deduce_if_params(lines,writes,local_variables,take_last_write_as_output)
            #     # self.evaluate_ifs(lines,local_variables)

            #     filename = f"res-initialize-energy-{iter_num}.txt"
            #     self.pretty_print(lines,filename,local_variables)
            #     print(self.flag_mappings["hcond0__mod__energy"])
            #     print(self.flag_mappings["kbot__mod__energy"])
            #     print(self.flag_mappings["bcz__mod__cdata(5)"])
            #     print(self.flag_mappings["bcz12__mod__cdata(5,1)"])
            #     print(self.flag_mappings["gravz__mod__gravity"])
            #     if iter_num == 2:
            #         pexit("check res-initialize-energy")
        
        
    def get_ac_matrix_res(self,segment,indexes):
      #read/write to matrix indexes
      if ":" not in indexes[0] and ":" not in indexes[1]:
        return f"{segment[0]}.data[{indexes[0]}-1][{indexes[1]}-1]"
      #Reading matrix row
      elif ":" not in indexes[0] and indexes[1] == ":":
        return f"{segment[0]}.row({indexes[0]}-1)"
      #Reading matrix col
      elif indexes[0] == ":" and ":" not in indexes[1]:
        return f"{segment[0]}.col({indexes[1]}-1)"
      else:
          print("unsupported matrix read/write")
          print(line[segment[1]:segment[2]])
          print(indexes)
    def transform_line_stencil(self,line,num_of_looped_dims, local_variables, array_segments_indexes,rhs_var,vectors_to_replace, writes, loop_indexes):
        variables = merge_dictionaries(self.static_variables, local_variables)
        last_index = 0
        res_line = ""
        #for testing remove later
        # if "prof_lnt" in line:
        #     return ""
        make_vector_copies = False
        #whole nx writes become scalar writes 
        #3 dim writes are vectors
        print("line",line)
        print("rhs",rhs_var)
        if rhs_var is not None:
          rhs_segment = get_variable_segments(line, [rhs_var])
          if len(rhs_segment) == 0:
              rhs_segment = self.get_struct_segments_in_line(line, [rhs_var])
          rhs_segment  = rhs_segment[0]
          rhs_info = self.get_param_info((line[rhs_segment[1]:rhs_segment[2]],False),local_variables,self.static_variables)
          rhs_dim = [self.evaluate_indexes(dim) for dim in rhs_info[3]]
          print(rhs_dim)
        bundle_dims = ["npscalar","n_forcing_cont_max"]
        #can't have writes to a profile
        if rhs_var is not None:
          if rhs_var in self.static_variables:
            if(self.static_variables[rhs_var]["profile_type"]):
                self.static_variables[rhs_var]["profile_type"] = None
          elif local_variables[rhs_var]["profile_type"]:
                local_variables[rhs_var]["profile_type"] = None
        if (
            rhs_var is None or
            ((rhs_var in local_variables and
                (
                    (num_of_looped_dims == 0 and len(rhs_dim) == 0)
                    or (num_of_looped_dims == 1 and rhs_dim == ["3"])
                    or (num_of_looped_dims == 1 and rhs_dim in [[global_subdomain_range_x,"3"],[global_subdomain_range_x]])
                    or (num_of_looped_dims == 2 and rhs_dim in [[global_subdomain_range_x,"3"]])
                    or (num_of_looped_dims == 2 and rhs_dim[0] == global_subdomain_range_x and rhs_dim[1] in bundle_dims)
                    or (num_of_looped_dims == 3 and rhs_dim[:-1] == [global_subdomain_range_x,"3"] and rhs_dim[-1] in bundle_dims)
                    or (num_of_looped_dims == 3 and rhs_dim in [[global_subdomain_range_x,"3","3"]])
                    or (num_of_looped_dims == 4 and rhs_dim in [[global_subdomain_range_x,"3","3","3"]])
                ))
                or (rhs_var in ["df","f"] or rhs_var in vectors_to_replace))
            ): 
            for i in range(len(array_segments_indexes)):
                segment = array_segments_indexes[i]
                if segment[0] in local_variables:
                    src = local_variables
                else:
                    src = self.static_variables
                if segment[0] == "df" or segment[0] == "f":
                    orig_indexes = get_segment_indexes(segment,line, len([src[segment[0]]["dims"]]))
                    indexes = [self.evaluate_indexes(index) for index  in orig_indexes]
                    if not all([okay_stencil_index(self.evaluate_indexes(x[1]),x[0]) for x in enumerate(orig_indexes[:-1])]):
                        print("How how to handle stencil indexes?")
                        print(line[segment[1]:segment[2]])
                        print(indexes)
                        print([self.evaluate_indexes(index) for index in orig_indexes])
                        pexit(line)
                    if ":" in indexes[-1]:
                        lower,upper = indexes[-1].split(":")
                        print("HMM LINE: ",line)
                        if is_vector_stencil_index(indexes[-1]) or indexes[-1] == "icc__mod__cdata:icc__mod__cdata+npscalar__mod__cparam-1" or upper == f"2+{lower}" or indexes[-1] in ["ibb_sphr__mod__cdata:ibb_sphp__mod__cdata","iglobal_ax_ext__mod__cdata:iglobal_az_ext__mod__cdata","ibxt__mod__cdata:ibzt__mod__cdata","ijxt__mod__cdata:ijzt__mod__cdata"""]:
                            make_vector_copies = True
                        else:
                            print("range in df index 3")
                            print(line[segment[1]:segment[2]])
                            print(indexes)
                            print(orig_indexes)
                            print("INDEX PAIR: ",lower,upper,upper == f"2+{lower}")
                            pexit("LINE: ",line)
                    if segment[0] == "df":
                        vtxbuf_name = get_vtxbuf_name_from_index("DF_", remove_mod(indexes[-1]))
                        if "VEC" in vtxbuf_name:
                          res = vtxbuf_name.replace("VEC",vtxbuf_name[-4])
                        else: 
                          res = vtxbuf_name
                    elif segment[0] == "f":
                        #split in case range
                        vtxbuf_name = get_vtxbuf_name_from_index("F_",remove_mod(indexes[-1]))
                        if "VEC" in vtxbuf_name:
                          vtxbuf_name = vtxbuf_name.replace("VEC",vtxbuf_name[-4])
                          res = f"vecvalue({vtxbuf_name})"
                        else:
                          if i > 0:
                            res = f"value({vtxbuf_name})"
                          else:
                            #write to f variable, presumably this is because auxiliary variables
                            #reuse DF_variable for now
                            res = f"D{vtxbuf_name}"
                else:
                    var_dims = src[segment[0]]["dims"]
                    if var_dims == ['nx__mod__cparam,9']:
                        var_dims = ['nx__mod__cparam','9']
                    indexes = [self.evaluate_indexes(index) for index  in get_indexes(line[segment[1]:segment[2]],segment[0],0)]
                    is_profile = (
                        segment[0] in self.static_variables
                        and var_dims in [[global_subdomain_range_x],[global_subdomain_range_with_halos_x],[global_subdomain_range_y],[global_subdomain_range_with_halos_y],[global_subdomain_range_z],[global_subdomain_range_with_halos_z],[global_subdomain_range_x,global_subdomain_range_with_halos_y]]
                        and len([x for x in writes if x["variable"] == segment[0]])  == 0
                      )
                    is_pointer_to_profile = var_dims == [":"] and indexes in [["n__mod__cdata"],["m__mod__cdata"],[":"],[]]
                    if is_pointer_to_profile:
                        prof_type = "x"
                        if(indexes == ["m__mod__cdata"]):
                            prof_type = "y"
                        elif(indexes  == ["n__mod__cdata"]):
                            prof_type = "z"
                        src[segment[0]]["profile_type"] = prof_type
                    if src[segment[0]]["profile_type"]:
                        prof_type = src[segment[0]]["profile_type"]
                        res_index = None
                        if len(indexes) == 0:
                          for dim in ["x", "y", "z"]:
                            if prof_type == dim and var_dims in [[f"m{dim}__mod__cparam"],[":"]]:  
                              res_index = "[vertexIdx.{dim}]"
                            elif prof_type == dim and var_dims == [f"n{dim}__mod__cparam"]:  
                              res_index = "[vertexIdx.{dim}-NGHOST_VAL]"
                            elif prof_type == f"{dim}_vec" and var_dims[0] == f"n{dim}__mod__cparam":  
                              res_index = "[vertexIdx.{dim}-NGHOST_VAL]"
                            elif prof_type == f"{dim}_vec" and var_dims[0] == f"m{dim}__mod__cparam":  
                              res_index = "[vertexIdx.{dim}]"
                        elif len(indexes) == 1:
                          index = indexes[0].replace(global_loop_z,"vertexIdx.z").replace(global_loop_y,"vertexIdx.y").replace(global_subdomain_range_x_inner,"vertexIdx.x")
                          res_index = "[" + index + "]"
                        elif len(indexes) == 2:
                          first_index = indexes[0].replace(global_loop_z,"vertexIdx.z").replace(global_loop_y,"vertexIdx.y").replace(global_subdomain_range_x_inner,"vertexIdx.x")
                          second_index = indexes[1].replace(global_loop_z,"vertexIdx.z").replace(global_loop_y,"vertexIdx.y").replace(global_subdomain_range_x_inner,"vertexIdx.x")
                          res_index = "[" + first_index + "]" + "[" + second_index + "]"
                        elif len(indexes) == 3:
                          first_index  = indexes[0].replace(global_loop_z,"vertexIdx.z").replace(global_loop_y,"vertexIdx.y").replace(global_subdomain_range_x_inner,"vertexIdx.x")
                          second_index = indexes[1].replace(global_loop_z,"vertexIdx.z").replace(global_loop_y,"vertexIdx.y").replace(global_subdomain_range_x_inner,"vertexIdx.x")
                          third_index  = indexes[2].replace(global_loop_z,"vertexIdx.z").replace(global_loop_y,"vertexIdx.y").replace(global_subdomain_range_x_inner,"vertexIdx.x")
                          res_index = "[" + first_index + "]" + "[" + second_index + "]" + "[" + third_index + "]"
                        elif len(indexes) == 4:
                          first_index  = indexes[0].replace(global_loop_z,"vertexIdx.z").replace(global_loop_y,"vertexIdx.y").replace(global_subdomain_range_x_inner,"vertexIdx.x")
                          second_index = indexes[1].replace(global_loop_z,"vertexIdx.z").replace(global_loop_y,"vertexIdx.y").replace(global_subdomain_range_x_inner,"vertexIdx.x")
                          third_index  = indexes[2].replace(global_loop_z,"vertexIdx.z").replace(global_loop_y,"vertexIdx.y").replace(global_subdomain_range_x_inner,"vertexIdx.x")
                          fourth_index = indexes[3].replace(global_loop_z,"vertexIdx.z").replace(global_loop_y,"vertexIdx.y").replace(global_subdomain_range_x_inner,"vertexIdx.x")
                          res_index = "[" + first_index + "]" + "[" + second_index + "]" + "[" + third_index + "]" + "[" + fourth_index + "]"
                        
                        if res_index is None:
                          print("INDEXES: ",indexes)
                          print("DIMS : ",var_dims)
                          print("PROFILE TYPE: ",src[segment[0]]["profile_type"])
                          pexit("WHAT TO DO?")
                        res = f"{segment[0]}{res_index}"
                    #assume that they are auxiliary variables that similar to pencils but not inside pencil case
                    elif segment[0] in self.static_variables and var_dims in [[global_subdomain_range_x],[global_subdomain_range_with_halos_x]]:
                      if indexes  in [[":"]]:
                        res = segment[0]
                      else:
                        res = line[segment[1]:segment[2]]
                    #becomes a vector write for all of npscalar vals
                    elif segment[0] in local_variables and var_dims[:-1] == [global_subdomain_range_x] and var_dims[-1] in bundle_dims and indexes in [[],[":",":"]]:
                        res = segment[0]
                    elif segment[0] in local_variables and  var_dims[:-1] == [global_subdomain_range_x] and var_dims[-1] in bundle_dims and indexes[:-1] in [[],[":"]]:
                        res = f"{segment[0]}.data[{indexes[-1]} -1]"
                    #becomes a vector to vector indexes
                    elif segment[0] in local_variables and var_dims[:-1] ==  [global_subdomain_range_x,"3"] and var_dims[-1] in bundle_dims and indexes[:-1] in [[],[":",":"]]:
                        res = segment[0]
                    elif segment[0] in local_variables and  var_dims[:-1] == [global_subdomain_range_x,"3"] and var_dims[-1] in bundle_dims and indexes[:-1] in [[],[":","1"]]:
                        res = f"{segment[0]}.data[{indexes[-1]} -1].x"
                    elif segment[0] in local_variables and  var_dims[:-1] == [global_subdomain_range_x,"3"] and var_dims[-1] in bundle_dims and indexes[:-1] in [[],[":","2"]]:
                        res = f"{segment[0]}.data[{indexes[-1]} -1].y"
                    elif segment[0] in local_variables and  var_dims[:-1] == [global_subdomain_range_x,"3"] and var_dims[-1] in bundle_dims and indexes[:-1] in [[],[":","3"]]:
                        res = f"{segment[0]}.data[{indexes[-1]} -1].z"
                    elif segment[0] in local_variables and var_dims[:-1] == [global_subdomain_range_x,"3","3"] and var_dims[-1] in bundle_dims and indexes[:-1] in [[],[":",":",":"]]:
                        res = f"{segment[0]}.data[{indexes[-1]} -1]"
                    #these turn to scalar read/writes
                    #for pointers assume they are pointing to n[x|y|z] variables
                    elif segment[0] in local_variables and self.evaluate_indexes(src[segment[0]]["dims"][0]) in [global_subdomain_range_x,":"] and indexes in [[],[":"]]:
                        res = segment[0]
                    elif segment[0] in local_variables and self.evaluate_indexes(src[segment[0]]["dims"][0]) == global_subdomain_range_with_halos_x and indexes in [f"{global_subdomain_range_x_lower}:{global_subdomain_range_x_upper}"]:
                        res = segment[0]
                    #global vec
                    elif segment[0] in self.static_variables and src[segment[0]]["dims"] == ["3"] and indexes in [["1"],["2"],["3"],[]]:
                        #do nothing
                        res = line[segment[1]:segment[2]]
                    elif var_dims == ["nx__mod__cparam","my__mod__cparam"] and indexes == [":",global_loop_y]:
                        res = f"{segment[0]}[vertexIdx.x][vertexIdx.y]"
                    elif src[segment[0]]["dims"] == [global_subdomain_range_x,"3"]:
                        indexes = [self.evaluate_indexes(index) for index in indexes]
                        if indexes[0] == ":":
                          if indexes[1] in [":","1:3"]:
                            res = segment[0]
                          elif indexes[1] not in ["1","2","3"]:
                            print("what to do?")
                            print(line)
                            print(indexes)
                            print(line[segment[1]:segment[2]])
                            assert(False)
                          elif indexes[1]  == "1":
                            res = f"{segment[0]}.x"
                          elif indexes[1] == "2":
                            res = f"{segment[0]}.y"
                          elif indexes[1] == "3":
                            res = f"{segment[0]}.z"
                        else:
                          print("what to do?")
                          print(indexes)
                          pexit(line)
                    #constant local array
                    elif len(src[segment[0]]["dims"]) == 1 and src[segment[0]]["dims"][0].isnumeric() and len(indexes) == 1:
                        if segment[0] == "hcond_prof__mod__energy":
                            pexit("wrong!\n")
                        res = line[segment[1]:segment[2]]
                    #can simply do the lookup normally for lpencil
                    elif segment[0] == "lpencil":
                        res = line[segment[1]:segment[2]]
                    #AcMatrix
                    elif src[segment[0]]["dims"] == ["3","3"]:
                      res = self.get_ac_matrix_res(segment,indexes)
                    #nx var -> AcMatrix
                    elif src[segment[0]]["dims"] == [global_subdomain_range_x,"3","3"] and indexes[0] == ":": 
                      res = self.get_ac_matrix_res(segment,indexes[1:])
                      # #read/write to matrix indexes
                      # if indexes[0] ==":" and ":" not in indexes[1] != ":" and ":" not in indexes[2]:
                      #   res =f"{segment[0]}.data[{indexes[1]}][{indexes[2]}]"
                      # #Reading matrix row
                      # elif indexes[0] == ":" and ":" not in indexes[1] and indexes[2] == ":":
                      #   res =f"{segment[0]}.row({indexes[1]})"
                      # #Reading matrix col
                      # elif indexes[0] == ":" and indexes[1] == ":" and ":" not in indexes[2]:
                      #   res =f"{segment[0]}.col({indexes[1]})"
                      # else:
                      #     print("unsupported matrix read/write")
                      #     print(line[segment[1]:segment[2]])
                      #     print(num_of_looped_dims)
                      #     print(indexes)
                      #     pexit(line)
                    #AcTensor
                    elif var_dims == [global_subdomain_range_x,"3","3","3"]:
                        var_info = self.get_param_info((line[segment[1]:segment[2]],False),local_variables,self.static_variables)
                        #read/write to tensor indexes:
                        if len(var_info[3]) == 1 and indexes[0] == ":":
                            res = f"{segment[0]}.data[{indexes[1]}][{indexes[2]}][{indexes[3]}]"
                        else:
                          print("unsupported tensor read/write")
                          print("NUM of looped dims: ",num_of_looped_dims)
                          print("first index",indexes[0])
                          print("var dims",var_dims)
                          pexit(line[segment[1]:segment[2]])
                    elif var_dims == [global_subdomain_range_x,"9"]:
                        assert(num_of_looped_dims == None or (num_of_looped_dims == 1 and indexes[0] == ":"))
                        if len(indexes) == 0:
                          res = f"{segment[0]}"
                        elif len(indexes) == 2:
                          res = f"{segment[0]}[{indexes[1]}]"
                        else:
                          pexit("HMM: ",res)
                    elif len(var_dims) == 1 and num_of_looped_dims == 0:
                        res = f"{line[segment[1]:segment[2]]}"
                    elif len(var_dims) == 2 and num_of_looped_dims == 1 and indexes[1] == ":" and var_dims[1] == "nx__mod__cparam":
                        res = f"{line[segment[1]:segment[2]]}"
                    #2d profile
                    elif var_dims == ["mx__mod__cparam", "mz__mod__cparam"] and i != 0 and indexes == ["l1__mod__cparam:l2__mod__cparam","n__mod__cdata"]:
                        res = f"{segment[0]}[vertexIdx.x+NGHOST][vertexIdx.y+NGHOST]"
                    elif var_dims == ["3"] and num_of_looped_dims == 1 and indexes in [[":"],[]]:
                      res = line[segment[1]:segment[2]]
                    elif rhs_var is None: 
                      res = line[segment[1]:segment[2]]
                    elif len(var_dims) == 1 and len(indexes) == 1:
                      if indexes[0] == global_loop_y:
                        res = f"{segment[0]}[vertexIdx.y]"
                      elif indexes[0] == global_loop_z:
                        res = f"{segment[0]}[vertexIdx.z]"
                      elif indexes == ":":
                        pexit("HMM")
                      else:
                        res = f"{segment[0]}[{indexes[0]}-1]"
                    elif len(var_dims) == 2 and len(indexes) == 2:
                        res = f"{segment[0]}[{indexes[0]}][{indexes[1]}]"
                    elif len(var_dims) == 4 and len(indexes) == 4:
                        res = f"{segment[0]}[{indexes[0]}][{indexes[1]}]{indexes[2]}[{indexes[3]}]"
                    else:
                        print("what to do?")
                        print(line[segment[1]:segment[2]])
                        print(segment[0])
                        print("is static: ",segment[0] in self.static_variables)
                        print("is local: ",segment[0] in local_variables)
                        print("var dims",var_dims)
                        print("num of looped dims",num_of_looped_dims)
                        print("indexes",indexes)
                        print(src[segment[0]])
                        print(indexes)
                        pexit(line)
                res_line += line[last_index:segment[1]] + res
                last_index = segment[2]
            # res_line += line[last_index:] + ";"
            res_line += line[last_index:]
            #res_line = self.replace_fortran_indexing_to_c(res_line,variables)
            return res_line

        else:
            print("NO case for",line)
            print("RHS VAR",rhs_var)
            print(rhs_var in local_variables)
            print(num_of_looped_dims)
            print(rhs_dim)
            pexit(local_variables[rhs_var]["dims"])
    def map_to_new_indexes(self,indexes,line,i,src,segment,local_variables):
        res = []
        num_of_looping_dim = -1
        for i,index in enumerate(indexes):
            if ":" in index:
                num_of_looping_dim += 1
            if index == ":":
                possible_index = src[segment[0]]["dims"][i]
                if possible_index == ":":
                    possible_index = local_variables[rhs_var]["dims"][num_of_looping_dim]
                dims = src[segment[0]]["dims"][i]
                print(possible_index)
                parts = possible_index.split(",")
                print(indexes)
                print("PARTS: ",parts)
                print("DIMS: ",dims)
                print("possible_index: ",possible_index)
                print("same: ",possible_index == dims)
                if len(parts) == 2 and possible_index == dims and ":" in indexes[0] and ":" not in indexes[1]:
                    res_indexes  = []
                    res_indexes.append(self.map_to_new_index("1:" + parts[0],i,local_variables,line))
                    res_indexes.append(parts[1])
                    return res_indexes
                    print(res_indexes)
                    pexit("what to do?")
                else:
                    res.append(self.map_to_new_index("1:" + possible_index,i,local_variables,line))
            elif ":" in index:
                res.append(self.map_to_new_index(indexes[i],local_variables))
            else:
                res.append(index)
        return res
    def transform_line_boundcond_DSL(self,line,num_of_looped_dims, local_variables, array_segments_indexes,rhs_var,vectors_to_replace, writes,loop_indexes):
        last_index = 0
        res_line = ""
        assumed_boundary = None
        if (num_of_looped_dims==0 or num_of_looped_dims==2) and (rhs_var in local_variables or rhs_var == "f"): 
            for i in range(len(array_segments_indexes)):
                segment = array_segments_indexes[i]
                var = segment[0]
                if var in local_variables:
                    src = local_variables
                else:
                    src = self.static_variables
                indexes = [self.evaluate_indexes(index) for index in get_segment_indexes(segment,line, len(src[var]["dims"]))]
                print(num_of_looped_dims)
                print(indexes)
                if(num_of_looped_dims == 2 and indexes in [[":",":"],[global_subdomain_range_x_inner,":"]]):
                    res = segment[0]
                elif(var == "f" and (assumed_boundary is None or assumed_boundary == "z") and len(indexes) == 4 and indexes[:2] in [[":",":"], [global_subdomain_range_x_inner,":"]]):
                    f_index = f"[vertexIdx.x][vertexIdx.y][{indexes[2]}-1]"
                    res = f"{indexes[3]}{f_index}"
                    assumed_boundary = "z"
                elif(var == "f" and (assumed_boundary is None or assumed_boundary == "y") and len(indexes) == 4 and indexes[0] == ":" and indexes[2] == ":"):
                    f_index = f"[vertexIdx.x][{indexes[1]}-1][vertexIdx.z]"
                    res = f"{indexes[3]}{f_index}"
                    assumed_boundary = "y"
                elif(var == "f" and (assumed_boundary is None or assumed_boundary == "x") and len(indexes) == 4 and indexes[1] == ":" and indexes[2] == ":"):
                    f_index = f"[{indexes[0]}-1][vertexIdx.y][vertexIdx.z]"
                    res = f"{indexes[3]}{f_index}"
                    assumed_boundary = "x"
                elif(var == "f" and (assumed_boundary is None or assumed_boundary == "z") and len(indexes) == 4 and len(loop_indexes) >= 2 and all([x[1] == True for x in loop_indexes[:2]])):
                    f_index = f"[vertexIdx.x][vertexIdx.y][{indexes[2]}-1]"
                    res = f"{indexes[3]}{f_index}"
                    assumed_boundary = "z"
                elif(len(indexes) == 1 and indexes != [":"] and len(src[var]["dims"]) == 1):
                    ##-1 since from 1 to 0-based indexing
                    res = f"{var}[{indexes[0]}-1]"
                #comes from spread statement
                elif(len(indexes) == 2 and num_of_looped_dims == 2 and indexes[0] == ":" and indexes[1] != ":"):
                    res = f"{var}[{indexes[1]}]"
                else:
                    print(loop_indexes)
                    pexit("What to do",segment)
                res_line = res_line + line[last_index:segment[1]]
                res_line = res_line + res 
                last_index = segment[2]
            res_line = res_line + line[last_index:]
            res_line = res_line + ";"
            if ":" in res_line or "explicit_index" in res_line or not has_balanced_parens(res_line) or ";" not in res_line or "()" in res_line:
                print("NEED to transform more")
                print("LINE",line)
                print("RES_LINE",res_line)
                pexit("RES: " + res)
            return res_line

        else:
            print("NO case for",line)
            exit()

    def transform_line_boundcond(self,line,num_of_looped_dims, local_variables, array_segments_indexes,rhs_var,vectors_to_replace, writes):
        last_index = 0
        res_line = ""
        if (num_of_looped_dims==0 or num_of_looped_dims==2) and (rhs_var in local_variables or rhs_var == "f"): 
            for i in range(len(array_segments_indexes)):
                segment = array_segments_indexes[i]
                if segment[0] in local_variables:
                    src = local_variables
                else:
                    src = self.static_variables
                if segment[0] == "f":
                    indexes = [self.evaluate_indexes(index) for index in get_segment_indexes(segment,line, len(src[segment[0]]["dims"]))]
                    vtxbuf_name = get_vtxbuf_name_from_index("VTXBUF_", indexes[-1])
                    new_var = f"vba.in[{vtxbuf_name}]"
                    indexes= indexes[:-1]
                    for i,index in enumerate(indexes):
                        if ":" in index:
                            indexes[i] = self.map_to_new_index(indexes[i],i,local_variables,line)
                        else:
                            #convert from 1 to 0 based index
                            indexes[i] = self.evaluate_integer(f"{indexes[i]}-1")
                    res = f"{new_var}[DEVICE_VTXBUF_IDX({','.join(indexes)})]"
                ##Value local to kernel i.e. from the viewpoint of a thread a scalar
                elif segment[0] in local_variables and len(local_variables[segment[0]]["dims"]) == 2 and num_of_looped_dims == 2:
                    res = segment[0] 
                else:
                    indexes = [self.evaluate_indexes(index) for index in get_segment_indexes(segment,line,len(src[segment[0]]["dims"]))]
                    indexes = self.map_to_new_indexes(indexes,line,i,src,segment,local_variables)
                    #for now consider only 1d real arrays
                    if(len(indexes) == 1):
                        assert(src[segment[0]]["type"] == "real")
                        #translate to zero based indexing by decreasing by one
                        new_indexes = [self.evaluate_integer(f"{index}-1") for index in indexes]
                        #1d real arrays are found in real_arrays
                        res = f"vba.real_arrays[{segment[0]}][{new_indexes[0]}]"
                    elif len(indexes) == 2:
                        new_indexes = [self.evaluate_integer(f"{index}-1") for index in indexes]
                        res = f"vba.real_arrays[{segment[0]}][{new_indexes[0]}][{new_indexes[1]}]"
                    else:
                        pexit("WHAT")
                res_line = res_line + line[last_index:segment[1]]
                res_line = res_line + res 
                if ":" in res or "explicit_index" in res or ("(" in res and "DEVICE" not in res) or "'" in res:
                    print("NEED to transform more earlier")
                    print("LINE",line)
                    print("RES_LINE",res_line)
                    print("RES",res)
                    print("indexes",)
                    print("indexes",indexes)
                    print("seg",segment)
                    pexit("seg line " + line[segment[1]:segment[2]])
                last_index = segment[2]
            res_line = res_line + line[last_index:]
            res_line = res_line + ";"
            if ":" in res_line or "explicit_index" in res_line or not has_balanced_parens(res_line) or ";" not in res_line or "()" in res_line:
                print("NEED to transform more")
                print("LINE",line)
                print("RES_LINE",res_line)
                pexit("RES: " + res)
            return res_line

        else:
            print("NO case for",line)
            exit()
    def unroll_constant_loops(self,lines,local_variables):
        found_constant_loop = True
        while(found_constant_loop):
            lines,found_constant_loop = self.unroll_constant_loops_once(lines,local_variables)
        return lines
    def unroll_constant_loops_once(self,lines,local_variables):
        constant_loops_indexes = []
        in_constant_loop = False
        found_constant_loop = False
        loop_start_index  = 0
        do_nest_num = 0
        for line_index, line in enumerate(lines):
            if not found_constant_loop:
                if in_constant_loop:
                    constant_loops_indexes.append(line_index)
                if line[:3] == "do " and "while" not in line and "end" not in line:
                    do_nest_num += 1
                    write = self.get_writes_from_line(line)[0]
                    parts = [self.evaluate_integer(part.strip()) for part in write["value"].split(",")]
                    if len(parts) == 2:
                        lower = parts[0]
                        upper = parts[1]
                        # print("DO LINE",line)
                        # print("PARTS:",lower,upper)

                        #for the time being only loops smaller then the grid dimensios
                        #i.e. lower=1;upper=3
                        if lower.isnumeric() and upper.isnumeric():
                            loop_index = write["variable"]
                            in_constant_loop = True
                            do_nest_num = 1
                            constant_loops_indexes = []
                            replacements = []
                            loop_start_index = line_index
                            add_replacement = int(lower)
                            while add_replacement <= int(upper):
                                replacements.append(str(add_replacement))
                                add_replacement += 1

                if "do" in line and "end" in line and in_constant_loop:
                    do_nest_num -= 1
                    if do_nest_num == 0:
                        in_constant_loop = False
                        constant_loops_indexes.pop()
                        found_constant_loop = True
                        lines_to_add = []
                        remove_indexes = [loop_start_index,line_index]
                    # for replacement in replacements:
                    #     for x in lines_to_unroll:
                    #         lines_to_add.append(replace_variable(x,loop_index,replacement))
                    # for x in lines_to_add:
                    #     res_lines.append(x)


        if not found_constant_loop:
            return (lines,False)
        if len(constant_loops_indexes) == 0:
            return ([x[1] for x in enumerate(lines) if x[0] not in remove_indexes],True)
        
        res_lines = []
        unrolled_lines = []
        for replacement in replacements:
            for x in constant_loops_indexes:
                unrolled_lines.append(replace_variable(lines[x],loop_index,replacement))
        for line_index,line in enumerate(lines):
            if line_index not in constant_loops_indexes and line_index not in remove_indexes:
                res_lines.append(line)
            if line_index == constant_loops_indexes[0]:
                res_lines.extend(unrolled_lines)
        return (res_lines,found_constant_loop)
    def replace_fortran_indexing_to_c(self,line,variables,loop_indexes):
        array_segments_indexes = self.get_array_segments_in_line(line,variables)
        map_vals = []
        for seg in array_segments_indexes:
            indexes = get_segment_indexes(seg,line,0)
            if seg[0] == 'f' and len(indexes) == 4 and len(loop_indexes) >= 2 and all([x[1] == True for x in loop_indexes]):
                    f_index = f"[vertexIdx.x][vertexIdx.y][{indexes[2]}-1]"
                    res = f"{indexes[3]}{f_index}"
                    map_vals.append(res)
            else:
                if(indexes == []):
                    map_vals.append(line[seg[1]:seg[2]])
                else:
                    new_indexes = [f"{x}-1" for x in indexes]
                    new_val = build_new_access(seg[0],new_indexes).replace("(","[").replace(")","]")
                    map_vals.append(new_val)
        info = {"map_val": map_vals}
        line = self.replace_segments(array_segments_indexes ,line,self.map_val_func,{},info)
        if "xyz0" in line:
            print(array_segments_indexes)
            print(self.static_variables["profy_ffree__mod__hydro"])
            pexit(line)
        return line
    def transform_line(self,i,lines,local_variables,loop_indexes,symbol_table,initialization_lines,orig_params,transform_func,vectors_to_replace,writes):
        line = lines[i]
        if line == "assert(.false.)":
            return "static_assert(false)"
        #we disregard some mn_loop setup lines
        if line in ["n__mod__cdata=nn__mod__cdata(imn__mod__cdata)","m__mod__cdata=mm__mod__cdata(imn__mod__cdata)","enddo mn_loop","headtt=.false.","lfirstpoint=.false."]:
            return ""
        if "mn_loop:" in line and "do" in line:
            return ""
        variables = merge_dictionaries(self.static_variables, local_variables)
        if is_init_line(line):
            vars_in_line = {}
            self.get_variables_from_line(line,i,vars_in_line, self.file,"",True, False)
            vars_to_declare = []
            for var in vars_in_line:
                if var.strip() in local_variables and var.strip() not in orig_params:
                    vars_to_declare.append(var)
            if self.offload_type == "stencil":
                if len(vars_to_declare) == 0 or local_variables[vars_to_declare[0]]["type"] != "real":
                    return ""
                if local_variables[vars_to_declare[0]]["dims"] == [global_subdomain_range_x]:
                    return "real " + ", ".join(vars_to_declare)
                if local_variables[vars_to_declare[0]]["dims"] == [global_subdomain_range_x,"3","3"]:
                    return "Matrix " + ", ".join(vars_to_declare)
                if local_variables[vars_to_declare[0]]["dims"] == [global_subdomain_range_x,"3"]:
                    return "real3 " + ", ".join(vars_to_declare)
                if local_variables[vars_to_declare[0]]["dims"] == [global_subdomain_range_x,"3","3","3"]:
                    #tensors are not yet supported
                    return ""
                    return "Tensor " + ", ".join(vars_to_declare)
                return ""
            ##No local declaration of pointers since they are supposed to be global
            if len(vars_to_declare) == 0 or local_variables[vars_to_declare[0]]["is_pointer"]:
                return ""
            if local_variables[vars_to_declare[0]]["type"] != "pencil_case":
                DSL_type = translate_to_DSL(local_variables[vars_to_declare[0]]["type"])
                res = [f"{DSL_type} {var}" for var in vars_to_declare]
                return  ";\n".join(res) +";"
            else:
                return ""
        array_segments_indexes = self.get_array_segments_in_line(line,variables)
        if line.strip()  == "exit":
            return "continue"
        if "else" in line and "if" in line:
            res_line =  "}\n" + line.replace("then","{").replace("elseif","else if")
            ##res_line = res_line.replace("if ","if constexpr")
            return res_line
        if "else" in line:
            return "}\nelse {"
        if "if" in line and "then" in line:
             line = line.replace("then","{")
        if "end" in line and "select" in line:
            return "}\n"
        if "select case" in line:
            select_case_var = line.split("(")[1].split(")")[0].lower()
            return f"switch({select_case_var})"+"{\n"
        if "case" in line and "default" in line:
            return "default:"
        if "case" in line:
            select_case_var = line.split("(")[1].split(")")[0]
            return f"case {select_case_var}:\n"
        if "end" in line and "do" in line:
            is_global_index = loop_indexes.pop()[1]
            if(is_global_index): return ""
            return "}\n"
        if "subroutine" in line and "end" in line:
            if self.test_to_c:
                return ""
            else:
                return "}\n"
        if "subroutine" in line:
            func_call = self.get_function_calls_in_line(line,local_variables)[0]
            function_name = func_call["function_name"]
            params = func_call["parameters"]
            param_strings = []
            for param in params:
                print(local_variables[param]['type'])
                if param == "j":
                    param_strings.append(f"VtxBuffer j")
                elif param != "f":
                    param_strings.append(f"{param}")
            if self.test_to_c:
                return ""
            elif self.offload_type == "boundcond":
                return f"{function_name}({','.join(param_strings)})\n"+"{\n"
            elif self.offload_type == "stencil":
              return "Kernel rhs(){"
        if is_use_line(line):
            return ""

        if "do" in line[:2]:
            print("HMM LINE: ",line)
            loop_index = self.get_writes_from_line(line)[0]["variable"]
            lower,upper= [part.strip() for part in line.split("=")[1].split(",",1)]
            loop_indexes.append((loop_index, upper in global_subdomain_ranges))
            if upper in global_subdomain_ranges:
                return ""
            #to convert to C loops
            #return f"for(int {loop_index} = {lower};{loop_index}<={upper};{loop_index}++)" +"{"
            #to convert to DSL loops
            return f"for {loop_index} in {lower}:{upper}+1 " + "{"
        if "endif" in line:
            return "}"
        original_line = line
        if new_is_variable_line(line):
            return ""

        # func_calls = self.get_function_calls_in_line(line,local_variables)
        # if len(func_calls) == 1 and func_calls[0]["function_name"] in der_funcs:
        #       file_path = self.find_subroutine_files(func_calls[0]["function_name"])[0]
        #       interfaced_call = self.get_interfaced_functions(file_path,func_calls[0]["function_name"])[0]
        #       #derij_main will do nothing if i==
        #       if interfaced_call == "derij_main" and func_calls[0]["parameters"][3] == func_calls[0]["parameters"][4]:
        #         return ""
        #       if interfaced_call not in implemented_der_funcs:
        #         print("implement der func:",interfaced_call, "in DSL")
        #         print(func_calls[0])
        #         exit()
        #       else:
        #         new_param_list = self.get_static_passed_parameters(func_calls[0]["parameters"],local_variables,self.static_variables)
        #         param_types = [(param[2],param[3]) for param in new_param_list]
        #         if str(param_types) not in implemented_der_funcs[interfaced_call]:
        #           #for der4 if ignoredx is not on then it a normal ignoredx call
        #           # if interfaced_call in ["der4","der6_main"] and len(func_calls[0]["parameters"]) == 5 and self.evaluate_boolean(func_calls[0]["parameters"][4],local_variables,[]) == ".false.":
        #           #   pass
        #           # else:
        #           print(interfaced_call)
        #           print("not implemented for these param types")
        #           print(param_types)
        #           print(func_calls[0])
        #           exit()
        #       rest_params = func_calls[0]["parameters"][:2] + func_calls[0]["parameters"][3:]
        #       if interfaced_call in ["der_main","der2_main"]:
        #         res = f"{func_calls[0]['parameters'][2]} = {der_func_map[interfaced_call][func_calls[0]['parameters'][3]]}({self.get_der_index(func_calls[0]['parameters'][1])})"
        #       elif interfaced_call in ["der6_main"]:
        #         if len(new_param_list) == 4:
        #           res = f"{func_calls[0]['parameters'][2]} = {der_func_map[interfaced_call][func_calls[0]['parameters'][3]]}({self.get_der_index(func_calls[0]['parameters'][1])})"
        #         else:
        #           if new_param_list[4][-1] == "upwind":
        #             res = f"{func_calls[0]['parameters'][2]} = {der_func_map[interfaced_call][func_calls[0]['parameters'][3]]}_upwd({self.get_der_index(func_calls[0]['parameters'][1])})"
        #           else:
        #             print("hmm is it ignoredx?")
        #             print(new_param_list)
        #             assert(False)
        #       else:
        #         print("no der case for ", interfaced_call)
        #         print(func_calls[0])
        #         assert(False)
        #       return res

        rhs_segment = get_rhs_segment(line)
        if "not_implemented(" in line:
          return line

        rhs_segment = get_rhs_segment(line)
        if rhs_segment is None:
          rhs_var = None
        else:
          rhs_var = self.get_rhs_variable(line)
          if rhs_var is not None:
            rhs_var = rhs_var.lower()
        #if rhs_var is None:
        #    print("rhs var is none")
        #    pexit("LINE: ",line)
        #This is done since after inlining some global variables can be considered as good as local variables
        if rhs_var is not None and rhs_var not in local_variables:
            if rhs_var in [".false.",".true."]:
              pexit("WRONG")
            print("LINE: ",line)
            local_variables[rhs_var] = self.static_variables[rhs_var]
            # print("WHAT TO DO rhs not in variables",line)
            # print(rhs_var)

            # # #for the time being simply assume they are diagnostics writes so can simply remove them
            # # return ""
            
            # print(rhs_var in self.static_variables)
            # if rhs_var == global_loop_z:
            #     print("IS n_save in local_variables?","n_save" in local_variables)
            # exit()
        if rhs_var:
          dim = len(variables[rhs_var]["dims"])
          indexes = get_indexes(get_rhs_segment(line),rhs_var,dim)
          dims, num_of_looped_dims = get_dims_from_indexes(indexes,rhs_var)
        else:
          num_of_looped_dims = None

        #line = self.transform_spread(line,[f":" for i in range(num_of_looped_dims)],local_variables)
        #line = self.transform_spread(line,[f":" for i in range(num_of_looped_dims)],local_variables)
        return transform_func(line,num_of_looped_dims, local_variables, array_segments_indexes,rhs_var,vectors_to_replace,writes, loop_indexes)
    def transform_spreads(self,lines,local_variables,variables):
        res_lines = []
        for line_index, line in enumerate(lines):
            spread_calls= [call for call in self.get_function_calls_in_line(line,variables) if call["function_name"] == "spread"]
            if len(spread_calls) > 1:
                print("multiple spread calls",line)
                exit()
            elif len(spread_calls) == 1:
                call = spread_calls[0]
                lhs = call["parameters"][0]
                redundant_index = call["parameters"][1]
                rhs_var = self.get_rhs_variable(line)
                res_line = self.replace_func_call(line,call,lhs)
                rhs_segment = get_variable_segments(line, [rhs_var])
                if len(rhs_segment) == 0:
                    rhs_segment = self.get_struct_segments_in_line(line, [rhs_var])
                rhs_segment  = rhs_segment[0]
                rhs_info = self.get_param_info((line[rhs_segment[1]:rhs_segment[2]],False),local_variables,self.static_variables)
                var_name = line[rhs_segment[1]:rhs_segment[2]]
                rhs_indexes = get_segment_indexes(rhs_segment,res_line,0)
                dims, num_of_looped_dims = get_dims_from_indexes(rhs_indexes,rhs_segment[0])
                if rhs_indexes == [] and len(rhs_info[3]) == 1:
                    new_rhs = f"{var_name}"
                    res_line = new_rhs + res_line[rhs_segment[2]:]
                    res_lines.append(res_line)
                #spreading scalar to vectors
                elif call["parameters"][1] == "2" and call["parameters"][2] == "3":
                    if "spread_index" not in local_variables:
                        local_variables["spread_index"] = {
                            "type": "integer",
                            "dims": []
                        }
                    res_lines.append("do spread_index=1,3")
                    new_rhs = f"{var_name}(:,spread_index)"
                    res_lines.append(new_rhs + res_line[rhs_segment[2]:])
                    res_lines.append("enddo")

                #not really a spread since would be legal without the spread
                elif num_of_looped_dims == 1:
                       res_line = self.replace_func_call(line,call,call["parameters"][0])
                       res_lines.append(res_line)
                #not really a spread
                elif call["parameters"][2] == "1":
                    res_lines.append(res_line)
                elif call["parameters"][1] == "2" and len(rhs_info[3]) == 2:
                    param_info = self.get_param_info((call["parameters"][0],False),local_variables,self.static_variables)
                    if(len(param_info[3]) == 1 and param_info[3][0] == rhs_info[3][0] and rhs_info[3][1] == call["parameters"][2]):
                       print("SAFE TO REMOVE SPREAD")
                       res_line = self.replace_func_call(line,call,call["parameters"][0])
                       print("RES: ",res_line)
                       res_lines.append(res_line)
                    #pexit("spread: what to do?")
                #if spreading scalar to 1d there is only a single way to do it
                #elif len(rhs_info[3]) == 1:
                #       print("SAFE TO REMOVE SPREAD")
                #       res_line = self.replace_func_call(line,call,call["parameters"][0])
                #       res_lines.append(res_line)
                else:
                    param_info = self.get_param_info((call["parameters"][0],False),local_variables,self.static_variables)
                    print("have to append it to")
                    print("rhs var",rhs_var)
                    print("indexes",rhs_indexes)
                    print(variables[rhs_var]["dims"])
                    print(call["parameters"])
                    print(param_info)
                    print(rhs_info)
                    pexit(line)
            
            else:
                res_lines.append(line)
        return res_lines
    def elim_empty_dos(self,lines,local_variables):
        orig_line_len = len(lines)+1
        while(orig_line_len > len(lines)):
            orig_line_len = len(lines)
            remove_indexes = []
            for line_index, line in enumerate(lines[:-2]):
                if len(line) > 3:
                  if line[:3] == "do " and "while" not in line and "end" not in line:
                    if any([x == lines[line_index+1] for x in ["enddo", "end do"]]):
                      remove_indexes.append(line_index)
                      remove_indexes.append(line_index+1)
            lines = [x[1] for x in enumerate(lines) if x[0] not in remove_indexes]
        return lines
    def elim_empty_branches(self,lines,local_variables):

        orig_line_len = len(lines)+1
        while(orig_line_len > len(lines)):
            orig_line_len = len(lines)
            remove_indexes = []
            for line_index, line in enumerate(lines):
                if "if" in line and "then" in line and "else" not in line:
                    if_calls = [call for call in self.get_function_calls_in_line(line,local_variables) if call["function_name"] == "if"]
                    if len(if_calls) == 1:
                        if any([x == lines[line_index+1] for x in ["endif", "end if"]]):
                            remove_indexes.append(line_index)
                            remove_indexes.append(line_index+1)
                        elif lines[line_index+1] == "else" and any([x == lines[line_index+2] for x in ["endif", "end if"]]):
                            remove_indexes.extend([line_index,line_index+1,line_index+2])
                # if "else if" in line and (lines[line_index+1] == "endif" or "else if" in lines[line_index+1]):
                #     remove_indexes.append(line_index)
                # elif "if" in line and "then" in line and "else" in line:
                #     if lines[line_index+1] == "endif":
                #         remove_indexes.append(line_index)
            lines = [x[1] for x in enumerate(lines) if x[0] not in remove_indexes]
        return lines

    def get_var_use_info(self,lines,local_variables,variables):
        var_dict = {}
        writes = self.get_writes(lines,False)
        needed_func_call_lines = []
        #the first line is the subroutine we are transforming so skip it
        func_calls = self.get_function_calls(lines,local_variables,False)
        for call in func_calls:
            if call["function_name"] in der_funcs:
                output_param = call["parameters"][2].split("(")[0].strip()
                if output_param not in var_dict:
                    var_dict[output_param] = {
                        "indexes": [],
                        "is_used": False
                    }
                var_dict[output_param]["indexes"].append(call["line_num"])
            if call["function_name"] in sub_funcs:
                for i,_ in enumerate(call["parameters"]):
                    param = call["parameters"][i].split("=")[-1].strip()
                    if param not in var_dict:
                        var_dict[param] = {
                            "indexes": [],
                            "is_used": False
                        }
                    var_dict[param]["indexes"].append(call["line_num"])
                    if i not in sub_funcs[call["function_name"]]["output_params_indexes"]:
                        var_dict[param]["is_used"] = True

                    #if multiple output params things get weird if one is used but another one is not
                    #so for now keep those func calls
                    if len(sub_funcs[call["function_name"]]["output_params_indexes"]) > 1:
                        needed_func_call_lines.append(call["line_num"])

        for write in [x for x in writes if x["variable"] not in [".false.",".true."]]:
                line = write["line"]
                line_index = write["line_num"]
                rhs_segment = get_variable_segments(line, [write["variable"]])
                if len(rhs_segment) == 0:
                    rhs_segment = self.get_struct_segments_in_line(line, [write["variable"]])
                rhs_segment  = rhs_segment[0]
                var_name = line[rhs_segment[1]:rhs_segment[2]].split("::",1)[-1].split("(",1)[0].strip()
                if var_name not in var_dict:
                    var_dict[var_name] = {
                        "indexes": [],
                        "is_used": False
                    }
                    #some calls imply a write so redundant to check for it
                    for call in [call for call in func_calls if call["function_name"] not in ["max","min","maxval","cos","sin","cosh","sinh"] and var_name in call["line"]]:
                        if var_name in call["parameters"]:
                            var_dict[var_name]["is_used"] = True

                for var_name_in_dict in [x for x in var_dict if not var_dict[x]["is_used"]]:
                    if (("%" in var_name_in_dict and var_name_in_dict in write["value"]) or (var_name_in_dict in write["value"] and var_name_in_dict in get_used_variables_from_line(write["value"],False))):
                        if var_name_in_dict != var_name:
                            var_dict[var_name_in_dict]["is_used"] = True
                var_dict[var_name]["indexes"].append(line_index)
        return (var_dict, needed_func_call_lines)
            
                

    def elim_unnecessary_writes_and_calls(self,lines,local_variables,variables):
        orig_lines_len = len(lines)+1
        while(orig_lines_len>len(lines)):
            orig_lines_len = len(lines)
            #elim empty if branches
            lines = self.elim_empty_branches(lines,local_variables)
            lines = self.eliminate_while(lines)
            var_dict, needed_func_call_lines = self.get_var_use_info(lines,local_variables,variables) 
            remove_indexes = []
            if "mass_per_proc" in var_dict:
                var_dict["mass_per_proc"]["is_used"] = False
            for var in var_dict:
                if not var_dict[var]["is_used"]:
                    if self.get_param_info((var,False),local_variables,variables)[2] != "integer":
                        for index in var_dict[var]["indexes"]:
                            if index not in needed_func_call_lines:
                                remove_indexes.append(index)
            lines = [x[1] for x in enumerate(lines) if x[0] not in remove_indexes]
        return lines
    def evaluate_rest_pencil_flags(self,lines,local_variables):
        variables = merge_dictionaries(local_variables,self.static_variables)
        var_dict, _ = self.get_var_use_info(lines,local_variables,variables)
        pencil_flag_mappings = {}
        for var in var_dict:
            if "ac_transformed_pencil_" in var:
                pencil_name = var.replace("ac_transformed_pencil_","")
                flag_name = f"lpencil__mod__cparam(i_{pencil_name}__mod__cparam)"
                if var_dict[var]["is_used"]:
                    pencil_flag_mappings[flag_name] = ".true."
                else:
                    pencil_flag_mappings[flag_name] = ".false."
        #if pencil not in used info then it has been already eliminated completely, thus the flag is .false.
        for var in local_variables:
            if "ac_transformed_pencil_" in var and var not in var_dict:
                pencil_name = var.replace("ac_transformed_pencil_","")
                flag_name = f"lpencil__mod__cparam(i_{pencil_name}__mod__cparam)"
                if flag_name not in pencil_flag_mappings:
                    pencil_flag_mappings[flag_name] = ".false."
                

        for line_index,_ in enumerate(lines):
            for flag in pencil_flag_mappings:
                lines[line_index] = lines[line_index].replace(flag,pencil_flag_mappings[flag])
        return lines

    def get_module_where_declared(self,static_var,filename):
        possible_modules = []
        modules = self.file_info[filename]["used_modules"]
        for mod in modules:
            if modules[mod]:
                if static_var in modules[mod]:
                    possible_modules.append(mod)
            elif static_var in self.rename_dict[mod]:
                possible_modules.append(mod)
        if len(possible_modules) == 1:
            return possible_modules[0]
        print("did not found module for var: ",static_var)
        print("len possible modules:", len(possible_modules))
        print("possible modules:",possible_modules)
        print("modules: ",modules)
        #Note: if in wrong order in general.f90 won't parse correctly
        #
        #print("general vars: ",self.rename_dict["general"])
        for x in possible_modules:
            print(x, static_var in self.rename_dict[x])
            print(x, static_var in self.rename_dict[x])
        exit()
    def transform_get_shared_variable_line(self,line,local_variables,lines):
        variables = merge_dictionaries(local_variables,self.static_variables)
        func_calls = self.get_function_calls_in_line(line,variables)
        if len(func_calls) == 1 and func_calls[0]["function_name"] == "get_shared_variable":
              
            shared_variable = func_calls[0]["parameters"][1].replace("'","").replace('"',"'")
            rhs = self.get_pointer_target(shared_variable)
            lhs = f"{func_calls[0]['parameters'][1]}"
            #if the name is already the instead of assignment change the reference to the src globally
            if(lhs == remove_mod(rhs)):
                lines = self.replace_var_in_lines(lines,lhs,rhs)
                return ""
            return f"{lhs} = {rhs}"
        return line


    def replace_interfaced_calls(self,lines,variables):
        for line_index,_ in enumerate(lines):
            func_calls = self.get_function_calls_in_line(lines[line_index],variables)
            #might not be in func info as in the case of intrinsic funcs
            if len(func_calls) == 1 and func_calls[0]["function_name"] in self.func_info and "interface_funcs" in self.func_info[func_calls[0]["function_name"]]:
                file_path = self.find_subroutine_files(func_calls[0]["function_name"])[0]
                interfaced_functions = self.get_interfaced_functions(file_path,func_calls[0]["function_name"])
                parameter_list = self.get_static_passed_parameters(func_calls[0]["parameters"],variables,variables)
                interfaced_calls = self.choose_correct_interfaced_function(func_calls[0],interfaced_functions,parameter_list,file_path)
                if len(interfaced_calls) == 1:
                    interfaced_call = interfaced_calls[0]
                    lines[line_index] = self.replace_func_call(lines[line_index],func_calls[0],f"{interfaced_call}({','.join(func_calls[0]['parameters'])})")
                else:
                    print(lines[line_index])
                    for call in interfaced_calls:
                        print(call)
                    pexit("too many possible calls\n")
        return lines


    def translate_func_call_to_astaroth(self,func,lines,local_variables,variables):
        res = []
        for line_index,_ in enumerate(lines):
            func_calls = self.get_function_calls_in_line(lines[line_index],variables)
            if(len(func_calls) == 1 and func_calls[0]["function_name"] == func):
                func_calls[0]["new_param_list"] = self.get_static_passed_parameters(func_calls[0]["parameters"],local_variables,self.static_variables)
                #get local var info to map funcs
                func_calls[0]["local_variables"] = local_variables
                res.extend(sub_funcs[func_calls[0]["function_name"]]["map_func"](func_calls[0]))
            else:
                res.append(lines[line_index])
        return res

    def translate_func_calls_to_astaroth(self,lines,local_variables, variables):
        for func in sub_funcs: 
            lines = self.translate_func_call_to_astaroth(func,lines,local_variables,variables)
        return lines
    def transform_any_calls(self,lines,local_variables, variables):
        for line_index,line in enumerate(lines):
                all_calls = self.get_function_calls_in_line(line,variables)
                any_calls=  [x for x in all_calls if x["function_name"] == "any"]
                other_calls =  [x for x in all_calls if x["function_name"] not in ["any","if"]]
                #for now consider only singular any calls
                print(line)
                if len(any_calls) == 1 and len(other_calls) == 0:
                    while(len(any_calls)>0):
                        call = any_calls[0]
                        assert(len(call["parameters"]) == 1)
                        comp_symbol = None
                        for symbol in ["/=","==","<",">"]:
                            if not comp_symbol and symbol in call["parameters"][0]:
                                comp_symbol = symbol
                        parts = [part.strip() for part in call["parameters"][0].split(comp_symbol)]
                        if(len(parts) == 2):
                            lower,upper = parts
                            dims = self.get_param_info((lower,False), local_variables, self.static_variables)[3]
                            if lower in variables and len(dims) == 1 and dims[0].isnumeric():
                                replacement_indexes = [str(i) for i in range(1,int(dims[0])+1)]
                                lines[line_index] = self.replace_func_call(line,call," .or. ".join([f"{lower}({index}) {comp_symbol} {upper}" for index in replacement_indexes]))
                                any_calls=  [x for x in self.get_function_calls_in_line(lines[line_index],variables) if x["function_name"] == "any"]
                                assert(len(any_calls) == 0)
                            else:
                                print(lower)
                                print(variables[lower]["is_pointer"])
                                print(dims)
                                pexit("WHAT TO DO?")
                        elif(len(parts) == 1):
                            print(parts[0])
                            dims = variables[parts[0]]["dims"]
                            assert(len(dims) == 1)
                            res = f"any_AC({parts[0]},{dims[0]})"
                            lines[line_index] = self.replace_func_call(line,call,res)
                            any_calls=  [x for x in self.get_function_calls_in_line(lines[line_index],variables) if x["function_name"] == "any"]
                            assert(len(any_calls) == 0)
                        else:
                            pexit("WHAT TO DO?")
        return lines

    def transform_tiny(self,lines,local_variables,variables):
        for line_index,line in enumerate(lines):
            for call in [x for x in self.get_function_calls_in_line(line,variables) if x["function_name"] == "tiny"]:
                type = self.get_param_info((call["parameters"][0],False), local_variables,self.static_variables)[2]
                if type == "real":
                    line = self.replace_func_call(line,call,"AC_tiny_val")
                    lines[line_index] = line
                else:
                    print("what to do with this tiny type?",type)
                    pexit(type)
        return lines
    def transform_array_calls_into_scalar_calls(self,array_func, scalar_func, lines,local_variables,variables):
        for line_index,_ in enumerate(lines):
            array_calls = [call for call in self.get_function_calls_in_line(lines[line_index],variables) if call["function_name"] == array_func]
            while(len(array_calls) > 0):
                first_param_info = self.get_param_info((array_calls[0]["parameters"][0],False),local_variables,self.static_variables)
                #max of nx (possibly vector components), is safe to take max
                if ((len(array_calls[0]["parameters"]) == 1) or (len(array_calls[0]["parameters"]) == 2 and array_calls[0]["parameters"][1] in ["dim=2","2"])) and first_param_info[3] in [[global_subdomain_range_x], [global_subdomain_range_x,"3"]]:
                    lines[line_index] = self.replace_func_call(lines[line_index],array_calls[0],f"{scalar_func}({array_calls[0]['parameters'][0]})")
                else:
                    print("first param info",first_param_info)
                    print("no case")
                    pexit(lines[line_index])
                array_calls = [call for call in self.get_function_calls_in_line(lines[line_index],variables) if call["function_name"] == array_func]
            if "minval" in lines[line_index]:
                print(lines[line_index])
                print(self.get_function_calls_in_line(lines[line_index],variables))
                print(array_func)
                pexit("sfdfgd")
        return lines
    def transform_random_number_calls(self,lines,variables):
        for line_index,line in enumerate(lines):
                func_calls = [x for x in self.get_function_calls_in_line(line,variables) if x["function_name"] == "random_number_wrapper"]
                if len(func_calls)>0:
                    assert(len(func_calls) == 1)
                    output_param = func_calls[0]["parameters"][0]
                    assert(output_param in variables)
                    lines[line_index] = self.replace_func_call(line,func_calls[0],f"{output_param} = rand_uniform()")
        return lines

    def transform_conversion_calls(self,lines,variables,func_name):
        for line_index,line in enumerate(lines):
                func_calls = [x for x in self.get_function_calls_in_line(line,variables) if x["function_name"] == func_name]
                while(len(func_calls)>0):
                    call = func_calls[0]
                    if len(call["parameters"]) == 1:
                        line = self.replace_func_call(line,call,call['parameters'][0])
                        lines[line_index] = line
                        func_calls = [x for x in self.get_function_calls_in_line(line,variables) if x["function_name"] == func_name]
                    else:
                        print("multiple params in: ", func_name)
                        pexit(line)
        return lines

    def translate_alog_to_log(self,lines,variables):
        for line_index,line in enumerate(lines):
                alog_calls =  [x for x in self.get_function_calls_in_line(line,variables) if x["function_name"] == "alog"]
                while(len(alog_calls)>0):
                    call = alog_calls[0]
                    line = self.replace_func_call(line,call,f"log({','.join(call['parameters'])})")
                    lines[line_index] = line
                    alog_calls =  [x for x in self.get_function_calls_in_line(line,variables) if x["function_name"] == "alog"]
        return lines
    
    def unroll_forall(self,lines,local_variables,variables):
        for line_index,line in enumerate(lines):
            search_line = line.replace("forall","call forall")
            func_calls = self.get_function_calls_in_line(search_line,local_variables)
            if len(func_calls) == 1 and func_calls[0]["function_name"] == "forall" and len(func_calls[0]["parameters"]) <= 2:
                write = self.get_writes_from_line((func_calls[0]["parameters"])[0])[0]
                if len(func_calls[0]["parameters"]) == 2:
                    if "/=" in func_calls[0]["parameters"][1]:
                        prefix_if = f"  if({func_calls[0]['parameters'][1]})  "
                    else:
                        #WE simply do not unroll in this case
                        continue
                        pexit("what to do?\n","LINE: ",line)
                else:
                    prefix_if = ""
                iterator = write["variable"]
                if ":" not in write["value"]:
                    print("should unroll for all")
                    print("don't know how to do it")
                    pexit(line)
                replacement_lower, replacement_upper = [part.strip() for part in write["value"].split(":")]

                segments = self.get_array_segments_in_line(search_line, variables)
                if len(segments) == 0:
                    continue
                first_segment = segments[0]
                indexes = get_segment_indexes(first_segment, search_line, 0)
                replacement_index = -1
                for i,index in enumerate(indexes):
                    if index == iterator:
                        replacement_index = i
                search_line = search_line[:func_calls[0]["range"][1]] + prefix_if + search_line[func_calls[0]["range"][1]:]

                res = self.replace_segments(self.get_array_segments_in_line(search_line,variables),search_line,self.global_loop_replacer,local_variables,{
                    "iterator": iterator, 
                    "replacement_lower": replacement_lower,
                    "replacement_upper": replacement_upper,
                    "replacement_index": replacement_index, 
                })
                save_var = False 
                #copypaste
                search_index = 0
                if "%" in res:
                    search_var = ""
                    save_var = False
                    buffer = ""
                    for i,char in enumerate(res):
                        if char == "%":
                            save_var = True
                        if not(char.isalpha() or char.isnumeric()) and char != "%":
                            if save_var:
                                search_var = buffer
                                search_index = i
                                save_var = False
                            buffer = ""
                        else:
                            buffer = buffer + char
                    if save_var:
                        search_var = buffer
                        search_index = None
                    search_var = search_var.strip()
                    last_index = 0
                    res_final = ""
                    struct_segs = [seg for seg in get_variable_segments(res,search_var) if seg[0] != ""]
                    for seg in struct_segs:
                        var_name,field = [part.strip() for part in seg[0].split("%",1)]
                        if var_name in local_variables:
                            src = local_variables
                        elif var_name in self.static_variables:
                            src = self.static_variables
                        if src[var_name]["type"] != "pencil_case":
                            print("what to do non pencil_case struct ?")
                            pexit("struct seg", seg[0], res_line[seg[1]:seg[2]])
                        dims = self.struct_table[src[var_name]["type"]][field]["dims"]
                        indexes = get_segment_indexes(seg, res, 0)
                        for i,index in enumerate(indexes):
                            if iterator in index:
                                new_lower= index.replace(iterator,replacement_lower)
                                new_upper= index.replace(iterator,replacement_upper)
                                if new_lower == "1" and new_upper == dims[i]:
                                    indexes[i] = ":"
                                else:
                                    indexes[i] = new_lower + ":" + new_upper
                        res_final = res_final + res[last_index:seg[1]]
                        res_final= res_final + build_new_access(seg[0],indexes)
                        last_index = seg[2]
                    res_final = res_final + res[last_index:]
                    res = res_final

                func_call = self.get_function_calls_in_line(res,local_variables)[0]
                res = self.replace_func_call(res,func_call,"").replace("call","")
                lines[line_index] = res
        return lines

    def trans_to_normal_indexing(self,segment,segment_index,line,local_variables,info):
        dims = info["dims"]
        first_part = dims[0].split(":")[0]
        assert(first_part[0] == "-")
        negative_offset = first_part[1:]
        if segment[0] == info["var"]:
            indexes = get_segment_indexes(segment,line,len(dims))
            assert(len(indexes) == 1)
            #don't want to consider it for now
            assert(":" not in indexes[0])
            #plus 1 since normal Fortran indexing is 1 based indexing
            indexes = [f"{indexes[0]}+{negative_offset}+1"]
            res = build_new_access(segment[0],indexes)
            return res
        return line[segment[1]:segment[2]]
    def elim_leftover_lines(self,lines):
        res = []
        for line in lines:
            if "=" in line:
                lhs, rhs = line.split("=",1)
                if not is_number(lhs):
                    res.append(line)
            else:
                res.append(line)
        return res

    def translate_negative_indexing_to_normal_indexing(self,lines,local_variables, variables):
        for var in variables:
            dims = variables[var]["dims"]
            if len(dims) == 1 and all([x in dims[0] for x in "-:"]):
                for line_index,_ in enumerate(lines):
                    arr_segs_in_line = self.get_array_segments_in_line(lines[line_index],variables)
                    lines[line_index] = self.replace_segments(arr_segs_in_line,lines[line_index],self.trans_to_normal_indexing,local_variables,{"var": var, "dims":dims})
        return lines
    def get_pointer_target(self,pointer):
            pointer = remove_mod(pointer)
            possible_modules = [mod for mod in self.module_info if mod in self.shared_flags_given and f"{pointer}__mod__{mod}" in self.shared_flags_given[mod]]
            print(pointer)
            #If the target module is not gotten from the put/get_shared_variable then we get it from the source assuming that it is the only non pointer refence to the variable
            if(possible_modules == []):
                for mod in self.rename_dict:
                    if pointer in self.rename_dict[mod]:
                        mod_var = f"{pointer}__mod__{mod}"
                        if(not self.static_variables[mod_var]["is_pointer"]):
                          possible_modules.append(mod)
            if(len(possible_modules) != 1):
              print(len(possible_modules))
              assert(len(possible_modules) == 1)
            mod = possible_modules[0]
            return f"{pointer}__mod__{mod}"
    def transform_pointers(self,lines,variables):
        res_lines = []
        for line in lines:
          var_segs_in_line = get_var_name_segments(line,variables)
          targets = []
          for seg in var_segs_in_line:
            var = seg[0]
            if var in self.static_variables and self.static_variables[var]["is_pointer"]:
              targets.append(self.get_pointer_target(var))
            else:
              targets.append(var)
          res_lines.append(self.replace_segments(var_segs_in_line,line,self.map_val_func,{},{"map_val": targets}))
        #for line in res_lines:
        #  print(line)
        #exit()
        return res_lines
    def transform_lines(self,lines,all_inlined_lines,local_variables,transform_func):
        lines = self.transform_pointers(lines,merge_dictionaries(local_variables,self.static_variables))
        lines = self.transform_case(lines)
        lines = self.normalize_if_calls(lines, local_variables)
        lines = self.normalize_where_calls(lines, local_variables)
        lines = self.normalize_if_calls(lines, local_variables)

        #move written profiles to local_vars since we know their values
        if self.offload_type == "stencil":
            lines = self.transform_pencils(lines,all_inlined_lines,local_variables)
            variables = merge_dictionaries(local_variables,self.static_variables)

        self.normalize_impossible_val()
        for i,line in enumerate(lines):
            if not has_balanced_parens(line) and "print*" not in line:
                print("not balanced")
                print("--->")
                pexit(line)
        for var in self.known_dims:
            self.static_variables[var]["dims"] = self.known_dims[var]
        symbol_table = {}
        #mapping that can be deduced for global flags
        lines = [line.replace("\n","") for line in lines]
        #transform cases to if statements to be able to analyse them together with other if cases
        orig_params = [param for param in self.get_function_calls_in_line(lines[0],local_variables)[0]["parameters"] if local_variables[param]["type"] != "pencil_case"]
        loop_indexes = []
        initialization_lines = []
        variables = merge_dictionaries(self.static_variables, local_variables)
        writes = self.get_writes(lines)
        remove_indexes = []
        for line_index,line in enumerate(lines):
            if len([call for call in self.get_function_calls_in_line(line,local_variables) if call["function_name"] in self.safe_subs_to_remove]) == 1:
                remove_indexes.append(line_index)
            #print and write are somewhat special functions
            if "print*" in line or "write(*,*)" in line or "print *" in line:
                remove_indexes.append(line_index)
        lines = [x[1] for x in enumerate(lines) if x[0] not in remove_indexes]
        self.known_values = {}
        lines = self.eliminate_while(lines)
        #remove allocate and at the same time make sure the allocate is safe to remove i.e. refers to local variables
        remove_indexes = []
        for i,line in enumerate(lines):
            has_allocate = False 
            for func_call in self.get_function_calls_in_line(line,local_variables):
                if func_call["function_name"] == "allocate":
                    remove_indexes.append(i)
                    for param in [param.split("(")[0].strip() for param in func_call["parameters"] if "=" not in param and ("stat" not in param or "src" not in param or "source" not in param)]:
                        if param not in local_variables:
                            print("Subroutine allocates global variable", param)
                            pexit("So can't generate cuda code for it")

        lines = [x[1] for x in enumerate(lines) if x[0] not in remove_indexes]

        file = open(f"res-eliminated.txt","w")
        for line in lines:
            file.write(f"{line}\n")
        file.close()

        lines = self.unroll_forall(lines,local_variables,variables)

        variables = merge_dictionaries(self.static_variables, local_variables)
        print("Unrolling constant loops")
        lines = [line.strip() for line in lines]
        file = open("res-unroll.txt","w")
        for line in lines:
            file.write(f"{line}\n")
        file.close()
        print("Check unroll file")

        #get local variables back to get actual dims not size dims
        local_variables = {parameter:v for parameter,v in self.get_variables(lines, {},self.file, True).items() }

        lines = [line.strip() for line in lines if line.strip() != ""]
        file = open("res-inlined.txt","w")
        for line in lines:
            file.write(f"{line}\n")
        file.close()
        print("Check inlined file")

        #try to transform maxvals into max and minval into min before transform 
        #TP: this doesn't of course work but maybe don't remove for now in case we can reuse array -> scalar calls later
        #TP have to simply try to elim all maxval and minval calls before we are finished

        lines  = self.transform_array_calls_into_scalar_calls("minval","min",lines,local_variables,variables)
        lines  = self.transform_array_calls_into_scalar_calls("maxval","max",lines,local_variables,variables)
        #replace all calls to tiny with predefined float 
        lines = self.transform_tiny(lines,local_variables,variables)
        lines = self.translate_alog_to_log(lines,variables)
        #for calls [real] just return the param
        lines = self.transform_conversion_calls(lines,variables,"real")
        lines = self.transform_conversion_calls(lines,variables,"dble")







        file = open("res-before-interfaced.txt","w")
        for line in lines:
            file.write(f"{line}\n")
        file.close()
        lines = self.replace_interfaced_calls(lines,variables)
        #lines = self.elim_leftover_lines(lines)
        #lines = self.elim_unnecessary_writes_and_calls(lines,local_variables,variables)
        #remove writes to fname,lfirstpoint and mass_per_proc

        #expand sizes
        writes = self.get_writes(lines,False)
        lines = [self.expand_size_in_line(line,local_variables,writes) for line in lines]
        local_variables = {parameter:v for parameter,v in self.get_variables(lines, {},"",True).items() }
        variables = merge_dictionaries(local_variables,self.static_variables)
        file = open("res-before.txt","w")
        for line in lines:
            file.write(f"{line}\n")
        file.close

        remove_indexes = []
        writes = self.get_writes(lines,False)
        for x in [x for x in writes if x["variable"] in ["fname","mass_per_proc","lfirstpoint"]]:
            remove_indexes.append(x["line_num"])
        lines = [x[1] for x in enumerate(lines) if x[0] not in remove_indexes]
        lines = self.unroll_constant_loops(lines,local_variables)
        file = open("res-before-inlined-spread.txt","w")
        for line in lines:
            file.write(f"{line}\n")
        file.close()
        #transform spreads into do loops
        lines = self.transform_spreads(lines,local_variables,variables)
        file = open("res-inlined-spread.txt","w")
        for line in lines:
            file.write(f"{line}\n")
        file.close()
        self.known_values = {}
        lines = self.eliminate_while(lines)

        writes = self.get_writes(lines)
        self.try_to_deduce_if_params(lines,writes,local_variables)
        #replace 1d vecs with 3 1d arrays
        vectors_to_try_to_replace = []
        for var in local_variables:
            vectors_to_try_to_replace.append(var)
        #Make sure are actually vectors
        vectors_to_replace = []
        for vector in vectors_to_try_to_replace:
            if self.is_vector(lines, vector, local_variables):
                vectors_to_replace.append(vector)
        vectors_to_replace.append("dline_1")

        file = open("res-inlined-profiles.txt","w")
        for line in lines:
            file.write(f"{line}\n")
        file.close()

        #for dim=2 sums for AcReal3 replace with simpl sum, which will sum components
        lines = self.transform_sum_calls(lines,local_variables)

        writes = self.get_writes(lines)

                    

        self.known_values = {}
        lines = self.eliminate_while(lines)
        # lines = self.inline_0d_writes(lines,local_variables)
        #rewrite some ranges in AcVectors and AcMatrices
        variables = merge_dictionaries(local_variables,self.static_variables)
        file = open("res-inlined-profiles-before-any.txt","w")
        for line in lines:
            file.write(f"{line}\n")
        file.close()


        file = open("res-inlined-profiles.txt","w")
        for line in lines:
            file.write(f"{line}\n")
        file.close()
        #lines = self.evaluate_rest_pencil_flags(lines,local_variables)
        variables = merge_dictionaries(local_variables,self.static_variables)
        #any -> normal if 
        lines = self.transform_any_calls(lines,local_variables,variables)



        lines = self.elim_empty_branches(lines,local_variables)
        lines = self.elim_empty_dos(lines,local_variables)




        lines = self.eliminate_while(lines)

        lines = self.elim_empty_branches(lines,local_variables)
        lines = self.elim_empty_dos(lines,local_variables)
        local_variables = {parameter:v for parameter,v in self.get_variables(lines, {},"",True).items() }
        lines = self.unroll_constant_loops(lines,local_variables)
        file = open("res-before-unroll-ranges.txt","w")
        for line in lines:
            file.write(f"{line}\n")
        file.close()
        lines = self.unroll_ranges(lines,local_variables)


        #lines = self.elim_unnecessary_writes_and_calls(lines,local_variables,variables)

        #translate how sub funcs are called in PC to how they would be called in Astaroth
        lines = self.translate_func_calls_to_astaroth(lines,local_variables,variables)

        #lines = self.elim_unnecessary_writes_and_calls(lines,local_variables,variables)
        lines = self.eliminate_while(lines)
        #lines = self.elim_unnecessary_writes_and_calls(lines,local_variables,variables)

        

        #if we have vars that have the dimension (-nghost:nghost) f.e. dz2_bounds these have to be translated to C style 0 based indexing
        lines = self.translate_negative_indexing_to_normal_indexing(lines,local_variables,variables)
        lines = self.unroll_forall(lines,local_variables,variables)
        lines = self.eliminate_while(lines)

        file = open("res-inlined-profiles.txt","w")
        for line in lines:
            file.write(f"{line}\n")
        file.close()

        #add some static vars to local vars
        for var in ["dline_1__mod__cdata","lcoarse_mn__mod__cdata"]:
            local_variables[var] = self.static_variables[var]
        self.known_values = {}
        self.try_to_deduce_if_params(lines,writes,local_variables)
        #no known values for n and m
        for x in self.static_variables:
          if "reference_state__mod__" in x:
            self.static_variables[x]["dims"] = [f"{global_subdomain_range_x},9"]
        writes = self.get_writes(lines)
        self.known_values = {}
        self.try_to_deduce_if_params(lines,writes,local_variables)
        for x in ["n__mod__cdata","n__mod__cdata"]:
          if x in self.known_values:
            del self.known_values[x]

        file = open("res.txt","w")
        for line in lines:
          file.write(f"{line}\n")
        file.close()
        for i,line in enumerate(lines):
            res = self.transform_line(i,lines,local_variables,loop_indexes,symbol_table,initialization_lines,orig_params, transform_func,vectors_to_replace,writes)
            lines[i] = res
        file = open("res-transform.txt","w")
        for line in lines:
            file.write(f"{line}\n")
        file.close()
        lines = [line.replace("iuu__mod__cdata","F_UX") for line in lines]
        lines = [line.replace("F_UX","F_UU.x").replace("F_UY","F_UU.y").replace("F_UZ","F_UU.z") for line in lines]

        lines = translate_fortran_ops_to_c(lines)


        lines = [replace_exp(line) for line in lines]
        file = open("res-replace_exp.txt","w")
        for line in lines:
            file.write(f"{line}\n")
        file.close()
 
        #for testing vtxbuf_ss -> vtxbuf_entropy
        lines = [line.replace("VTXBUF_SS","VTXBUF_ENTROPY") for line in lines]
        #close empty defaults
        for i,line in enumerate(lines):
            if "default:" in line and i<len(lines)-1 and "}" in lines[i+1]:
                lines[i] = line + ";"
        res_lines = []
        for line in lines:
            res_lines.extend([x for x in line.split("\n") if x != ""])
        lines = res_lines
        static_vars = []
        for i,line in enumerate(lines):
            static_variables_in_line= unique_list([var.lower() for var in get_used_variables_from_line(line) if var.lower() in self.static_variables])
            static_vars.extend(static_variables_in_line)
        static_vars = unique_list(static_vars)
        file = open("ac_declarations.h","w")
        for var in static_vars:
            type = ""
            if self.static_variables[var]["type"] == "real":
                type = "real"
            if self.static_variables[var]["type"] == "integer":
                type = "int"
            if self.static_variables[var]["type"] == "logical":
                type = "int"
            if len(self.static_variables[var]["dims"]) == 1:
                type = type + "Array"
            if len(self.static_variables[var]["dims"]) == 2:
                type = type + "2dArray"
            if "intArray" not in type and type != "" and type != "Array":
                file.write(f"{type} AC_{var}\n")
        file.close()

        # replace_dict = {
        #   global_subdomain_range_x:"AC_nx",
        #   global_subdomain_range_y:"AC_ny",
        #   global_subdomain_range_z:"AC_nz",
        #   global_subdomain_range_with_halos_x:"AC_nx+2*NGHOST",
        #   global_subdomain_range_with_halos_y:"AC_ny+2*NGHOST)",
        #   global_subdomain_range_with_halos_z:"AC_nz+2*NGHOST)",
        #   # global_subdomain_range_x_lower:"AC_nx_min",
        #   global_subdomain_range_x_upper:"AC_nx_max",
        #   global_subdomain_range_y_lower:"AC_ny_min",
        #   global_subdomain_range_y_upper:"AC_ny_max",
        #   global_subdomain_range_z_lower:"AC_nz_min",
        #   global_subdomain_range_z_upper:"AC_nz_max",
        # }
        # for x in replace_dict:
        #   lines = [line.replace(x,replace_dict[x]) for line in lines]


        ##Delete pointers because they are actually global scope variables
        for var in list(local_variables.keys()):
            if local_variables[var]["is_pointer"]:
                del local_variables[var]



        orig_lines = lines.copy()
        for i,line in enumerate(lines):
            static_variables_in_line= unique_list([var for var in get_used_variables_from_line(line) if var.lower() in self.static_variables])
            res_line = line
            for var in static_variables_in_line:
                #pow is a function in this context not a parameter
                if var.lower() not in ["bot","top","nghost","pow"] and var.lower() not in local_variables and var.upper() != var:
                    res_line = replace_variable(res_line, var, f"AC_{var.lower()}")
            lines[i] = res_line


        if self.test_to_c:
            idx_line = "const int3 idx = {i, j, k};"
            lines = [line.replace("vba.in","mesh_test.vertex_buffer") for line in lines]
            res_lines = [idx_line] + lines
            return res_lines
        static_vars = []
        # rest_params = ""
        # for param in orig_params:
        #     rest_params = rest_params + "," + translate_to_DSL(local_variables[param]["type"]) + " " + param
        # lines[1] = lines[1].replace("[rest_params_here]",rest_params)
        lines = [line.replace("nghost","NGHOST_VAL") for line in lines]
        file = open("res-3.txt","w")
        for line in lines:
            file.write(f"{line}\n")
        file.close()
        allowed_func_calls = ["constexpr", "&&", "||","all","sqrt","abs","sinh","cosh","tanh","min","max","pow","DEVICE_VTXBUF_IDX".lower(),"DCONST".lower(),"exp","log","if","else","for","sin","cos","tan","atan2"]
        astaroth_funcs = ["der","der2","der3","der4","der5","der6","col","row","derx","dery","derz","derxx","deryy","derzz","der6x","der6y","der6z","der6x_upwd","der6y_upwd","der6z_upwd","sum","dot","gradient","gradients","laplace","divergence","veclaplace","value","vecvalue","field","field3","divergence_from_matrix","curl_from_matrix","traceless_strain","traceless_strain_without_divu","multm2_sym","del6_upwd","del6_upwd_vec","gradient_of_divergence","cross","bij","mult","multmm_sc_mn","vecdel4","del6v","del6v_strict","static_assert","d2fi_dxj","del2fi_dxjk","der5x1y","der5x1z","der5y1z","gradients_5","der6x_ignore_spacing","der6y_ignore_spacing","der6z_ignore_spacing","del6","der4x2y","der4y2z","der4x2z","hessian"]
        allowed_func_calls.extend(astaroth_funcs)


        #for i,line in enumerate(lines):
        #    func_calls = self.get_function_calls_in_line(line,local_variables)
        #    if i>1:
        #        for func_call in func_calls:
        #            #Acvector sums are okay
        #            if func_call["function_name"].lower() not in allowed_func_calls and not "[" in func_call["function_name"]:
        #                # if func_call["function_name"] == "sum" and self.get_param_info((func_call["parameters"][0],False),local_variables,self.static_variables)[3] == [global_subdomain_range_x,"3"]:
        #                #     pass
        #                if func_call["function_name"].lower() not in der_funcs:
        #                    print("STILL FUNC CALLS in line:",line,i)
        #                    pexit(func_call)

        dx_lines = []
        for name in self.pde_names:
            dx_lines.append(f"DF_{name} = 0.0")
        for name in self.pde_vec_names:
            dx_lines.append(f"DF_{name}")
            dx_lines.append(f"DF_{name}.x = 0.0")
            dx_lines.append(f"DF_{name}.y = 0.0")
            dx_lines.append(f"DF_{name}.z = 0.0")

        #for now simply write DX out, normally would write out RHS3 substep
        write_fields_lines = []
        for name in self.pde_names:
            write_fields_lines.append(f"write(F_{name},DF_{name})")
        for name in self.pde_vec_names:
            write_fields_lines.append(f"write_vector(F_{name},DF_{name})")


        declarations_line = ""
        for var in unique_list(initialization_lines):
            declarations_line = declarations_line + translate_to_DSL(local_variables[var.lower()]["type"]) + " " + var + ";\n"
        if self.offload_type == "boundcond":
            lines = res_lines
        elif self.offload_type == "stencil":
            res_lines = lines[:1] + dx_lines + lines[1:-1] + write_fields_lines + [lines[-1]]
            lines = res_lines
        lines = [line.replace("nghost","NGHOST_VAL") for line in lines]
        formatted_lines = format_lines(lines)
        file = open("res.txt","w")
        for line in formatted_lines:
            file.write(f"{remove_mod(line)}\n")
        file.close()
        return lines
    def add_known_value(self,variable,val):
        self.known_values[variable] = val
        if variable == "gravz__mod__gravity" and val == "grav_init_z(4)":
            pexit("added wrong val to gravz")
    def add_deduced_value(self,variable,write,src,analyse_lines,local_variables):
        is_scalar = src[variable]["dims"] == []
        if variable in local_variables:
            if is_scalar:
                src[variable]["value"] = write["value"]
                self.add_known_value(variable,write["value"])
            else:
                if src[variable]["dims"] in [["3"]] and src[variable]["type"] in ["real"] and is_number(write["value"]):
                    for index in ["1","2","3"]:
                        self.add_known_value(f"{variable}({index})",write["value"])
        else:
            if analyse_lines[write["line_num"]][1] == 0:
                if is_scalar:
                    src[variable]["value"] = write["value"]
                    self.add_known_value(variable,write["value"])
                else:
                    if src[variable]["dims"] in [["3"]] and src[variable]["type"] in ["real"] and is_number(write["value"]):
                        for index in ["1","2","3"]:
                            self.add_known_value(f"{variable}({index})",write["value"])
    def deduce_value(self,variable,writes,local_variables,analyse_lines,take_last_write_as_output=False):
        var_writes = [write for write in writes if write["variable"] == variable and write["line"].split(" ")[0].strip() != "do"]
        if variable in local_variables:
            src = local_variables
        else:
            src = self.static_variables
        if len(var_writes) > 1:
            if all([write["value"] == var_writes[0]["value"] for write in var_writes]):
                write = var_writes[0]
                self.add_deduced_value(variable,write,src,analyse_lines,local_variables)

            #all writes are on the main branch then the value will be the last write after this subroutine
            elif take_last_write_as_output and all([analyse_lines[x["line_num"]][1] == 0 for x in var_writes]) and variable not in var_writes[-1]["value"]:
                src[variable]["value"] = var_writes[-1]["value"]
                # for write in var_writes:
                #     print(write)
                #     print(analyse_lines[write["line_num"]])
                # for line in analyse_lines:
                #     print(line)
                self.add_known_value(variable,var_writes[-1]["value"])

        elif len(var_writes) == 1:
            self.add_deduced_value(variable,var_writes[0],src,analyse_lines,local_variables)
        #can't have x = x + y
        #since it will cause infinite recursion
        if variable in self.known_values:
            if variable in self.known_values[variable]:
                del self.known_values[variable]
        return
    
    def expand_size_in_line(self, line,local_variables,writes):
        func_calls = self.get_function_calls_in_line(line,local_variables,False)
        need_to_transform_size = any([func_call["function_name"] == "size" for func_call in func_calls])
        while need_to_transform_size:
            func_call = list(filter(lambda func_call: func_call["function_name"] == "size", func_calls))[0]
            replacement = self.get_size(func_call,local_variables,local_variables,writes)
            if replacement != ":":
              line = self.replace_func_call(line, func_call,replacement)
            #don't replace if will can't return enough info
            else:
              return line
            func_calls =self.get_function_calls_in_line(line,local_variables,False)
            need_to_transform_size = any([func_call["function_name"] == "size" for func_call in func_calls])
        return line
    def get_global_loop_lines(self,lines,local_variables):
        variables = merge_dictionaries(local_variables,self.static_variables)
        writes = self.get_writes(lines)
        global_loop_lines = []
        in_global_loop = False
        number_of_dos = 0
        number_of_enddos= 0
        lines_in_loop = []
        iterators = []
        for i,line in enumerate(lines):
            if in_global_loop:
                lines_in_loop.append((line,i))
            if "end" in line and "do" in line and in_global_loop:
                number_of_enddos += 1
                in_global_loop = not number_of_enddos == number_of_dos
                if not in_global_loop:
                    global_loop_lines.append(lines_in_loop)
                    lines_in_loop = []
                    number_of_dos = 0
                    number_of_enddos = 0
            elif self.offload_type == "stencil" and "do" in line and ("1,nx" in line or f"{global_subdomain_range_x_lower},{global_subdomain_range_x_upper}" in line or f"{global_subdomain_range_x_lower}-2,{global_subdomain_range_x_upper}+2" in line):
                write = self.get_writes_from_line(line,local_variables)[0]
                #not supported if used as write value
                if all([write["variable"] not in [x[0] for x in get_variable_segments(write["value"],variables)] for x in writes]):
                    if "1,nx" in line:
                        iterators.append((write["variable"],"1",global_subdomain_range_x,0))
                    elif f"{global_subdomain_range_x_lower},{global_subdomain_range_x_upper}" in line:
                        iterators.append((write["variable"],global_subdomain_range_x_lower,global_subdomain_range_x_upper,0))
                    elif f"{global_subdomain_range_x_lower}-2,{global_subdomain_range_x_upper}+2" in line:
                        iterators.append((write["variable"],f"{global_subdomain_range_x_lower}-2","{global_subdomain_range_x_upper}+2",0))
                    if in_global_loop:
                        pexit("Can't handle nested global loops")
                    in_global_loop = True
                    number_of_dos = 1
                    number_of_enddos = 0
                    lines_in_loop = []
                    lines_in_loop.append((line,i))
            elif self.offload_type == "boundcond" and "do" in line and "1,my" in line:
                write = self.get_writes_from_line(line,local_variables)[0]
                iterators.append((write["variable"],"1",global_subdomain_range_with_halos_y,1))
                if in_global_loop:
                    pexit("Can't handle nested global loops")
                in_global_loop = True
                number_of_dos = 1
                number_of_enddos = 0
                lines_in_loop = []
                lines_in_loop.append((line,i))
        return (global_loop_lines,iterators)
    def get_default_flags_from_file(self,filename,dst):
        module = self.get_own_module(filename)
        for x in self.file_info[filename]["variables"]:

            vals = {}
            if x[0] == "l" and self.file_info[filename]["variables"][x]["type"] == "logical" and "value" in self.file_info[filename]["variables"][x]:
                vals[get_mod_name(x,module)] = self.file_info[filename]["variables"][x]["value"]
            elif self.file_info[filename]["variables"][x]["type"] in ["real","integer"] and len(self.file_info[filename]["variables"][x]["dims"]) == 0 and "value" in self.file_info[filename]["variables"][x]:
                vals[get_mod_name(x,module)] = self.file_info[filename]["variables"][x]["value"]
            elif self.file_info[filename]["variables"][x]["type"] in ["real","integer","logical"] and self.file_info[filename]["variables"][x]["dims"] == ["3"] and "array_value" in self.file_info[filename]["variables"][x] and "," not in self.file_info[filename]["variables"][x]["array_value"]:
                for index in ["1","2","3"]:
                    vals[f"{get_mod_name(x,module)}({index})"] = self.file_info[filename]["variables"][x]["array_value"]

            
            # if x == "beta_glnrho_scaled__mod__density":
            #     print("vals",vals)
            #     print("value" in self.file_info[filename]["variables"][x])
            #     pexit("")
            for key in vals:
                val = vals[key]
                dst[key] = val



    def update_which_shared_flags_given_and_used(self,mod,filename,new_lines,local_variables,analyse_lines):
        #done after elimination since which flags are accessed depend in input params
        for call in self.get_function_calls(new_lines,local_variables,False):
          #check if in main branch
          if analyse_lines[call["line_num"]][1] == 0:
            if call["function_name"] == "get_shared_variable" and len(call["parameters"]) >= 2:
              flag = call["parameters"][1]
              if flag in self.file_info[filename]["variables"]:
                self.shared_flags_accessed[mod].append(flag)
            if call["function_name"] == "put_shared_variable" and len(call["parameters"]) >= 2:
              flag = call["parameters"][1]
            #try removing this
            #   if (flag[0] == "l" and flag in self.file_info[filename]["variables"] and self.file_info[filename]["variables"][flag]["type"] == "logical") or remove_mod(flag) in ["beta_glnrho_scaled"]:
              self.shared_flags_given[mod].append(flag)
    def get_flags_from_initialization_func(self,subroutine_name, filename):
        orig_lines = self.get_subroutine_lines(subroutine_name,filename)

        file = open("res-original-magnetic.txt","w")
        for line in orig_lines:
            file.write(f"{line}\n")
        file.close()
        # if subroutine_name == "initialize_magnetic":
        #     print(self.flag_mappings["lforcing_cont__mod__cdata"])
        #     print(self.flag_mappings["lforcing_cont_aa__mod__magnetic"])

        mod_default_mappings = {}
        self.get_default_flags_from_file(filename,mod_default_mappings)
        local_variables = {parameter:v for parameter,v in self.get_variables(orig_lines, {},filename,True).items() }
        orig_lines = self.rename_lines_to_internal_names(orig_lines,local_variables,filename,subroutine_name)
        local_variables = {parameter:v for parameter,v in self.get_variables(orig_lines, {},filename,True).items() }
        orig_lines = self.transform_any_calls(orig_lines,local_variables,merge_dictionaries(local_variables,self.static_variables))
        local_variables = {parameter:v for parameter,v in self.get_variables(orig_lines, {},filename,True).items() }


        mod = self.get_own_module(filename)
        if mod not in self.shared_flags_accessed:
          self.shared_flags_accessed[mod] = []
          self.shared_flags_given[mod] = []




        for flag in self.flag_mappings:
            if self.flag_mappings[flag] == "":
                self.flag_mappings[flag] = "''"
        new_lines = self.eliminate_while(orig_lines,True,True)
        new_lines = self.unroll_constant_loops(new_lines,{})
        self.known_values = {}
        new_lines = self.eliminate_while(new_lines,True,True)
        analyse_lines = self.get_analyse_lines(new_lines,local_variables)
        local_variables = {parameter:v for parameter,v in self.get_variables(new_lines, {},filename,True).items() }


        self.update_which_shared_flags_given_and_used(mod,filename,new_lines,local_variables,analyse_lines)

        #if init func then which flags are accessed depend on previously shared flags from other mods
        if subroutine_name.split("_")[0].strip() == "initialize":
            self.update_shared_flags(mod)
            new_lines = self.eliminate_while(new_lines,True,True)
            analyse_lines = self.get_analyse_lines(new_lines,local_variables)
            self.update_which_shared_flags_given_and_used(mod,filename,new_lines,local_variables,analyse_lines)


        

        #variables that can be deduced after elimination
        self.known_values = {}
        self.try_to_deduce_if_params(new_lines,self.get_writes(new_lines,False), local_variables,True)
        for x in [x for x in self.known_values if x[0] == "l" and x in self.static_variables and self.static_variables[x]["type"] == "logical"]:
            self.add_mapping(x,self.known_values[x])

        # new_lines = self.eliminate_while(new_lines,True,True,mod_default_mappings)
        new_lines = self.eliminate_while(new_lines,True,True,self.default_mappings)
        self.known_values = {}
        self.try_to_deduce_if_params(new_lines,self.get_writes(new_lines,False), local_variables,True)
        for x in [x for x in self.known_values if x[0] == "l" and x in self.static_variables and self.static_variables[x]["type"] == "logical"]:
            self.add_mapping(x,self.known_values[x])
        for x in [x for x in self.known_values if x in self.static_variables and self.static_variables[x]["type"] == "integer" and self.static_variables[x]["dims"] == []]:
            self.add_mapping(x,self.known_values[x])
        #if eliminated all writes then to flag than we take the default val
        # writes = self.get_writes(new_lines)
        # for var in [x for x in mod_default_mappings if x not in self.flag_mappings]:
        #     if len([x for x in writes if x["variable"] == var]) == 0:
        #         self.flag_mappings[var] = mod_default_mappings[var]

        analyse_lines = self.get_analyse_lines(new_lines,local_variables)
        local_variables = {parameter:v for parameter,v in self.get_variables(new_lines, {},filename,True).items() }
        func_calls = self.get_function_calls(new_lines,local_variables,False)
        for call in func_calls:
          #check if in main branch
          if analyse_lines[call["line_num"]][1] == 0:
            if call["function_name"] in farray_register_funcs:
              self.farray_register_calls.append(call)
            if call["function_name"] == "get_shared_variable" and len(call["parameters"]) >= 2:
              flag = call["parameters"][1]
              if flag in self.file_info[filename]["variables"]:
                self.shared_flags_accessed[mod].append(flag)
            if call["function_name"] == "put_shared_variable" and len(call["parameters"]) >= 2:
              flag = get_mod_name(call["parameters"][0].replace("'","").replace('"',""),mod)
               #   if (flag[0] == "l" and flag in self.file_info[filename]["variables"] and self.file_info[filename]["variables"][flag]["type"] == "logical") or remove_mod(flag) in ["beta_glnrho_scaled"]:
              if flag in self.file_info[filename]["variables"]:
                self.shared_flags_given[mod].append(flag)

        #if init func then allocate allocated global variables (i.e. get dim info)
        if subroutine_name.split("_")[0].strip() == "initialize":
            self.inline_all_function_calls(filename,subroutine_name,new_lines,self.ignored_subroutines) 
            new_lines = self.func_info[subroutine_name]["inlined_lines"][filename]
            func_calls = self.get_function_calls(new_lines,local_variables)
            allocate_calls = [call for call in func_calls if call["function_name"] == "allocate"]
            for call in allocate_calls:
                for param in call["parameters"]:
                    param_name = param.split("(")[0].strip()
                    param_dim = param.split("(")[-1].split(")")[0].strip()
                    if param_name in self.static_variables and len(self.static_variables[param_name]["dims"]) == 1:
                        self.static_variables[param_name]["dims"] = [param_dim]
                        if param_dim in [self.static_variables["nz__mod__cparam"]["value"], self.static_variables["mz__mod__cparam"]["value"]]:
                            self.static_variables[param_name]["profile_type"] = "z"
                        elif param_dim in [self.static_variables["nx__mod__cparam"]["value"], self.static_variables["mx__mod__cparam"]["value"]]:
                            self.static_variables[param_name]["profile_type"] = "x"
                        elif param_dim in [self.static_variables["ny__mod__cparam"]["value"], self.static_variables["my__mod__cparam"]["value"]]:
                            self.static_variables[param_name]["profile_type"] = "y"
            for call in func_calls:
                if call["function_name"] == "select_eos_variable":
                    self.select_eos_variable_calls.append(call)



        # if subroutine_name == "initialize_forcing":
            
        #     self.pretty_print(new_lines,"res-init-energy.txt",local_variables)
        #     print("ifff__mod__forcing" in self.known_values)
        #     print(self.known_values["ifff__mod__forcing"])
        #     print(self.default_mappings["ifff__mod__forcing"])
        #     pexit("HMM")
        #     for flag in self.shared_flags_accessed["energy"]:
        #         print(flag)
        #     print("\n")
        #     print(self.flag_mappings["lreduced_sound_speed__mod__density"])
        #     print(self.flag_mappings["lreduced_sound_speed__mod__energy"])
        #     print(self.flag_mappings["hcond0__mod__energy"])
        #     for flag in self.shared_flags_given["density"]:
        #         print(flag)
        #     print(self.static_variables["hcond_prof__mod__energy"])
        #     pexit("hmm")

        #     print(self.flag_mappings["lheatc_kramers__mod__energy"])
        #     print(self.shared_flags_given["energy"])
        #     pexit("look at res-register-magnetic.txt")
        #     print(self.flag_mappings["lresi_hyper3__mod__magnetic"])
        #   file = open("init.txt","w")
        #   for line in new_lines:
        #       file.write(f"{line}\n")
        #   file.close()
        #   print(self.flag_mappings["lconservative__mod__density"])
        #   exit()
        #   print(self.shared_flags_accessed[mod])
        #   print(self.shared_flags_given[mod])
        #   print(self.flag_mappings["lheatc_sqrtrhochiconst__mod__energy"])
        #   exit()
        #   print(self.flag_mappings["lpressuregradient_gas__mod__hydro"])
        #   print("wrote init.txt")
        #   exit()
        #   print(self.flag_mappings["gravz_profile__mod__gravity"])
        #   exit()
        #   print("res in init.txt")
        #   print(self.flag_mappings["omega__mod__cdata"])
        #   print(self.flag_mappings["ltime_integrals__mod__cdata"])
        #   print("lgradu_as_aux__mod__hydro" in mod_default_mappings)
        #   print("lgradu_as_aux__mod__hydro" in self.file_info[filename]["variables"])
        #   print("lgradu_as_aux__mod__hydro" in self.static_variables)
        #   print(self.static_variables["lgradu_as_aux__mod__hydro"])
        #   self.file_info[filename]["variables"]["lgradu_as_aux__mod__hydro"]
        #   print(self.file_info[filename]["variables"]["lgradu_as_aux__mod__hydro"])
        #   print(self.flag_mappings["lgradu_as_aux__mod__hydro"])
        #   print("wrote init.txt to use internal names")
        #   exit()


    def try_to_deduce_if_params(self,lines,writes,local_variables,take_last_write_as_output=False):
        return
        analyse_lines = self.get_analyse_lines(lines,local_variables)
        writes = self.get_writes(lines,False)
        variables = merge_dictionaries(local_variables,self.static_variables)
        for var in variables:
            self.deduce_value(var,writes,local_variables,analyse_lines,take_last_write_as_output)
        # for line in lines:
        #     check_line = line.replace("then","").replace("elseif","if").replace("if","call if")
        #     if_func_calls = list(filter(lambda func_call: func_call["function_name"] == "if", self.get_function_calls_in_line(check_line,local_variables)))
        #     if len(if_func_calls) == 1:
        #         if len(if_func_calls[0]["parameters"]) == 1:
        #             param = if_func_calls[0]["parameters"][0]
        #             params = [part.replace(".not.","").strip() for part in param.split(".and.")]
        #             for par in params:
        #                 for var in [var for var in local_variables if "value" not in local_variables[var]]:
        #                     if var == par:
        #                         self.deduce_value(var,writes,local_variables)

    def update_shared_flags(self,mod):
        for flag in self.shared_flags_accessed[mod]:
            cleaned_flag = remove_mod(flag)
            possible_modules = []
            for mod_second in self.shared_flags_given:
                if get_mod_name(cleaned_flag,mod_second) in self.shared_flags_given[mod_second]:
                    possible_modules.append(mod_second)
            if len(possible_modules)>1:
                if self.module_vars_the_same(possible_modules,parameter[0]):
                    possible_modules = [possible_modules[0]]
                else:
                    print("Too many possible modules")
                    print(cleaned_flag)
                    print(possible_modules)
            if len(possible_modules) == 0:
                print("no possible module for:")
                print(cleaned_flag)
                print("accessed from: ",mod)
            assert(len(possible_modules) == 1)
            flag_from = get_mod_name(cleaned_flag,possible_modules[0])
            flag_into = get_mod_name(cleaned_flag,mod)
            if flag_from in self.flag_mappings:
                self.flag_mappings[flag_into] = self.flag_mappings[flag_from]
            for i in ["1","2","3"]:
                if f"{flag_from}({i})" in self.flag_mappings:
                    self.flag_mappings[f"{flag_into}({i})"] = self.flag_mappings[f"{flag_from}({i})"]
    def choose_correct_if_lines(self,lines):
        if_lines = []
        for line_index,line in enumerate(lines):
            if "if" in line or "else" in line:
                if_lines.append((line,line_index))
        possibilities = []
        found_true = False 
        for x_index, x in enumerate(if_lines):
            line = x[0]
            if "if" in line and ".false." in line and ".true." not in line and ".and." not in line and ".or." not in line:
                pass
            elif "if" in line and "else" not in line and ".true." in line and ".false." not in line and ".and." not in line and ".or." not in line:
                found_true = True
                possibilities= [x_index]
            elif not found_true and "end" not in line: 
                possibilities.append(x_index)
        if len(possibilities) == 1 and len(if_lines) > 2:
            correct_index = possibilities[0] 
            lower = if_lines[correct_index][1] + 1
            upper = if_lines[correct_index+1][1]
            return lines[lower:upper]
        elif len(possibilities) == 1:
            correct_index = possibilities[0] 
            line = lines[if_lines[correct_index][1]]
            if "if" in line and "else" not in line and "then" in line and ".true." in line and ".false." not in line and ".and." not in line and ".or." not in line:
                lower = if_lines[correct_index][1] + 1
                upper = if_lines[correct_index+1][1]
                return lines[lower:upper]
            elif "if" in line and "else" not in line and "then" in line and ".false." in line and ".true." not in line and ".and." not in line and ".or." not in line:
                return []
            else:
                return lines
        elif len(possibilities) == 0:
            return []
        return lines
        
    def eliminate_dead_branches(self,lines,local_variables):
        orig_lines = lines.copy()
        orig_lines.append("one more")
        while(len(orig_lines) > len(lines)):
            orig_lines = lines.copy()
            lines = self.eliminate_dead_branches_once(lines,local_variables)
            file = open("res-eliminated.txt","w")
            for line in lines:
                file.write(f"{line}\n") 
            file.close()
        return lines
    def get_analyse_lines(self,lines,local_variables):
        if_num = 0
        analyse_lines = []
        remove_indexes = []
        done = False
        max_if_num = 0
        nest_num = 0
        case_num = 0
        if_nums = [0]
        for line_index,line in enumerate(lines):
            if "if" in line and "then" in line and line[:4] != "elif"  and line[:4] != "else" and len([call for call in self.get_function_calls_in_line(line,local_variables) if call["function_name"] == "if"])>0:
                max_if_num += 1
                if_num = max_if_num 
                if_nums.append(if_num)
                nest_num += 1
            elif "if" and "then" in line and line[:4] != "elif"  and line[:4] != "else":
                pexit("missed this one",line)
            if "if" in line or line=="else":
                case_num += 1
            analyse_lines.append((line,nest_num, if_num,line_index,case_num))
            if line in ["endif","end if"]:
                nest_num -= 1
                if_nums.pop()
                assert(len(if_nums)>0)
                if_num = if_nums[-1]
        assert(len(analyse_lines) == len(lines))
        return analyse_lines
    def has_no_ifs(self,lines):
        return all([x[1] == 0 for x in self.get_analyse_lines(lines,local_variables)])
    def eliminate_dead_branches_once(self,lines,local_variables):
        remove_indexes = []
        done = False

        analyse_lines = self.get_analyse_lines(lines,local_variables)
        max_if_num = max([x[2] for x in analyse_lines])
        for if_num in range(1,max_if_num+1):
            if_lines = [line for line in analyse_lines if line[2] == if_num and line[1] > 0]
            choices = []
            for line in if_lines:
                if ("if" in line[0] and ("then" in line[0] or "end" in line[0] or "else" in line[0])) or line[0] == "else":
                    choices.append(line)
            possibilities = []
            found_true = False
            res_index = 0
            for choice_index, choice in enumerate(choices):
                line = choice[0]
                if "if" in line and ".false." in line and ".true." not in line and ".and." not in line and ".or." not in line:
                    pass
                #if multiple ifs that evaluate true the program would take the first one
                elif "if" in line and ".true." in line and ".false." not in line and ".and." not in line and ".or." not in line and not found_true:
                    found_true = True
                    possibilities = [choice]
                    res_index = choice_index
                elif not found_true and "end" not in line:
                    possibilities.append(choice)
                    res_index = choice_index
            if len(possibilities) == 0:
                starting_index = choices[0][3]
                ending_index = choices[-1][3]
                for index in range(starting_index,ending_index+1):
                    remove_indexes.append(index)
            if len(possibilities) == 1 and (len(choices) > 2 or found_true):
                starting_index = choices[0][3]
                ending_index = choices[-1][3]

                keep_starting_index = possibilities[0][3]+1
                #till next conditional or end
                file = open("res-debug.txt","w")
                for line in lines:
                    file.write(f"{line}\n")
                file.close()
                keep_ending_index = choices[res_index+1][3]-1
                for index in range(starting_index, ending_index+1):
                    if not (index >= keep_starting_index and index<=keep_ending_index):
                        remove_indexes.append(index)
        return [x[1] for x in enumerate(lines) if x[0] not in remove_indexes]

    
    def split_line(self, line):
        if line[0] == "!":
            return [line]
        lines = []
        split_indexes = []
        num_of_single_quotes = 0
        num_of_double_quotes = 0
        num_of_left_brackets = 0
        num_of_right_brackets = 0
        for iter_index in range(len(line)):
            if line[iter_index] == "'":
                num_of_single_quotes += 1
            if line[iter_index] == '"':
                num_of_double_quotes += 1
            if line[iter_index] == "(":
                num_of_left_brackets += 1
            if line[iter_index] == ")":
                num_of_right_brackets += 1
            if line[iter_index] == ";" and num_of_single_quotes%2 == 0 and num_of_double_quotes%2 == 0 and num_of_left_brackets == num_of_right_brackets:
                split_indexes.append(iter_index)
        start_index = 0
        for split_index in split_indexes:
            lines.append(line[start_index:split_index])
            start_index=split_index+1
        lines.append(line[start_index:])
        return filter(lambda x: x != "",lines)

    def get_function_declarations(self, lines, function_declarations, filename):
        ## Somewhat buggy, also returns variables in case of a header file but this is not a problem for the use case
        in_struct = False
        for count,line in enumerate(lines):
            parts = line.split("::")
            is_variable = True
            start =  parts[0].strip()
            type = start.split(",")[0].strip()
            if type == "public":
                is_variable = False
            if line.split(" ")[0] == "type" and len(line.split("::")) == 1:
                in_struct = True
            if line.split(" ")[0] == "endtype":
                in_struct = False
            if not is_variable and not in_struct and len(parts)>1:
                allocatable = "allocatable" in [x.strip() for x in start.split(",")]
                saved_variable = "save" in [x.strip() for x in start.split(",")]
                public = "public" in [x.strip() for x in start.split(",")]
                dimension = re.search("dimension\s*\((.+?)\)", start)
                if dimension is None:
                    dimension = []
                else:
                    dimension = [x.strip() for x in dimension.group(1).split(",")]
                line_variables = parts[1].strip()
                variable_names = self.get_variable_names_from_line(line_variables)
                for i, variable_name in enumerate(variable_names):
                    function_declarations.append(variable_name)
        return function_declarations
    def expand_function_call_main(self, subroutine_lines, subroutine_name, filename, calls_to_expand,global_init_lines,sub_modules,subs_not_to_inline,elim_lines,local_variables):
        call_to_expand = calls_to_expand[0]
        # local_variables = {parameter:v for parameter,v in self.get_variables(subroutine_lines, {},filename).items() }
        replaced_function_call_lines,is_function,res_type = self.expand_function_call(subroutine_lines, subroutine_name, filename, call_to_expand,local_variables,global_init_lines,subs_not_to_inline,elim_lines,local_variables)
        subroutine_lines = [line for line in subroutine_lines if not is_init_line(line)]
        subroutine_lines = [line for line in subroutine_lines if not is_use_line(line)]
        init_var_names = []
        global_init_lines = unique_list(global_init_lines)
        if is_function:
            subroutine_lines = subroutine_lines[:1] +  ["use {module}" for module in sub_modules] + global_init_lines + [f"{res_type} :: {call_to_expand['function_name']}_return_value_{self.inline_num}"] + subroutine_lines[1:]
        else:
            subroutine_lines = subroutine_lines[:1] +  [f"use {module}" for module in sub_modules] + global_init_lines + subroutine_lines[1:]
        has_replaced_call = False
        line_num = 0
        for line_num in range(len(subroutine_lines)):
          if not has_replaced_call:
            line = subroutine_lines[line_num]
            # func_calls = [call for call in self.get_function_calls_in_line(subroutine_lines[line_num][0],local_variables) if call["function_name"] == call_to_expand["function_name"]]
            # if len(func_calls)>0:
            if line == call_to_expand["line"] or f"{call_to_expand['function_name']}(" in line or f"call {call_to_expand['function_name']}" == line and "print*" not in line:
                has_replaced_call = True
                func_call = [call for call in self.get_function_calls_in_line(subroutine_lines[line_num],local_variables) if call["function_name"] == call_to_expand["function_name"]][0]
                # func_call = call_to_expand
                if is_function:

                    subroutine_lines[line_num] = self.replace_func_call(subroutine_lines[line_num], func_call, f"{call_to_expand['function_name']}_return_value_{self.inline_num}") 
                    res_lines = subroutine_lines[:line_num] + replaced_function_call_lines +subroutine_lines[line_num:]
                else:
                    res_lines = subroutine_lines[:line_num] + replaced_function_call_lines + subroutine_lines[line_num+1:]
                subroutine_lines = res_lines
        if not has_replaced_call:
          print("DID not replace it")
          print(call_to_expand)
          print("lines:")
          for x in subroutine_lines:
            print(x)
          exit()
        return subroutine_lines
    def populate_pencil_case(self):
        self.struct_table["pencil_case"] = {}
        for file in self.used_files:
            for count,line in enumerate(self.get_lines(file,include_comments=True)):
                if line[0] == "!" and "pencils provided" in line:
                    line = line.replace("!","").replace("pencils provided","")
                    fields = []
                    in_struct = False
                    field = ""
                    for char in line:
                        if char == " " or char == "," or char == ";":
                            if not in_struct:
                                fields.append(field)
                                field = ""
                            else:
                                field += char
                        elif char == "(":
                            in_struct = True
                            field += char
                        elif char == ")":
                            in_struct = False
                            fields.append(field)
                            field = ""
                        else:
                            field += char
                    fields.append(field)
                    fields = [field.strip().lower() for field in fields if field != ""]
                    for field in fields:
                        field_name = field.split("(")[0].strip()
                        field_dims = [global_subdomain_range_x]
                        if "(" in field:
                            field_dims += (field.split("(",1)[1].split(")")[0].split(","))
                        if field_name in self.struct_table["pencil_case"]:
                            if not len(self.struct_table["pencil_case"][field_name]["dims"]) == len(field_dims):
                                print("disagreeing pencils provided")
                                print("file", file)
                                print(field,self.struct_table["pencil_case"][field_name])
                                pexit(line)
                        else:
                            self.struct_table["pencil_case"][field_name] = {"type": "real", "dims": field_dims, "origin": [file], "allocatable": False, "saved_variable": False, "threadprivate": False, "saved_variable": False, "parameter": False, "optional": False, "profile_type": None, "is_pointer": False }
        for field in [x for x in self.struct_table["pencil_case"]]:
          if "." in field:
            del self.struct_table["pencil_case"][field]

    def expand_function_call(self,lines,subroutine_name, filename, call_to_expand,variables_in_scope,global_init_lines,subs_not_to_inline,elim_lines,local_variables):

        function_to_expand = call_to_expand["function_name"]
        mpi_calls = ["mpi_send","mpi_barrier","mpi_finalize","mpi_wait"]
        if function_to_expand in mpi_calls:
            pexit("MPI call not safe :(")
        file_paths = self.find_subroutine_files(function_to_expand)
        #if file_paths is [] then the function is only present in the current file and not public
        if file_paths == []:
            file_paths = [filename]
        modules = self.get_used_modules(lines)
        for line in lines:
                if line == call_to_expand["line"] or f"{call_to_expand['function_name']}(" in line:
                    if not(line == call_to_expand["line"] or f"{call_to_expand['function_name']}(" in line):
                        pexit("WRONG!")
                    function_call = [call for call in self.get_function_calls_in_line(line,local_variables) if call["function_name"] == call_to_expand["function_name"]][0]
                    parameter_list = self.get_static_passed_parameters(function_call["parameters"],local_variables,self.static_variables)
                    function_to_expand_filename = self.choose_right_module(file_paths,function_call,filename)
                    #return on the first call
                    return self.get_replaced_body(function_to_expand_filename,parameter_list, function_call,variables_in_scope,global_init_lines,subs_not_to_inline, elim_lines)
        return ([],False,None)
    def transform_case(self,lines):
        found_case = False 
        remove_indexes = []
        case_indexes = []
        case_param = ""
        case_params = []
        end_index = 0
        in_case = False
        for i, line in enumerate(lines):
            #Consider only lines having case
            #Done for speed optimization
            if "case" in line:
                func_calls = self.get_function_calls_in_line(line,{})
                if in_case and len(func_calls) == 1 and func_calls[0]["function_name"] == "case":
                    case_indexes.append(i)
                    case_params.append(func_calls[0]["parameters"])
                if in_case and "default" in line and "case" in line:
                    case_indexes.append(i)
                if  "select" in line:
                    in_case = True
                    found_case = True
                    remove_indexes = [i]
                    case_indexes = []
                    case_param = func_calls[0]["parameters"][0]
                    case_params = []
            if "endselect" in line or "end select" in line:
                end_index = i
                in_case = False
                break
        if not found_case:
            return lines
        res_lines = []
        for i,x in enumerate(case_params):
            if i == 0:
                inside = ".or.".join([f"{case_param} == {y}" for y in x])
                res = f"if({inside}) then" 
            else:
                inside = ".or.".join([f"{case_param} == {y}" for y in x])
                res = f"else if({inside}) then" 
            res_lines.append(res)
        #default case is handled separately
        res_lines.append("else")
        for j,i in enumerate(case_indexes):
            lines[i] = res_lines[j]
        lines[end_index] = "endif"
        lines = [x[1] for x in enumerate(lines) if x[0] not in remove_indexes]
        return self.transform_case(lines)
    def remove_name_after_keyword(self,line):
        if len(line)>=len("else") and line[:len("else")].strip() == "else" and "if" not in line and "where" not in line:
            return "else"
        if len(line)>=len("endif") and line[:len("endif")].strip() == "endif":
            return "endif"
        return line

    def remove_named_ifs(self,lines,local_variables):
        res_lines = []
        for line in lines:
            if(len([call for call in self.get_function_calls_in_line(line,local_variables) if call["function_name"] == "if"]) >=1):
                has_name = False
                scope_has_began = False
                for char in line:
                    if char == ":" and not scope_has_began:
                        has_name = True
                    if char == "(":
                        scope_has_began = True
                start_index = 0
                if has_name:
                    while(line[start_index] != ":"):
                        start_index +=1
                    res_lines.append(line[start_index+1:])
                    if ":" in res_lines[-1]:
                        print("TP HMM", line[start_index+1:])
                        pexit(line)
                else:
                    res_lines.append(line)
            else:
                res_lines.append(self.remove_name_after_keyword(line))
        return res_lines
    def normalize_impossible_val(self):
        for flag in self.flag_mappings:
            self.flag_mappings[flag] = self.flag_mappings[flag].strip()
        for val  in ["3.908499939e+37","3.90849999999999991e+37","3.90849994e+37","impossible__mod__cparam"]:
            for flag in self.flag_mappings:
                self.flag_mappings[flag] = self.flag_mappings[flag].replace(val,impossible_val)
            for var in self.static_variables:
                if "value" in self.static_variables[var]:
                    self.static_variables[var]["value"] = self.static_variables[var]["value"].replace(val,impossible_val)
    def normalize_if_calls(self,lines,local_variables):
        #lines = self.remove_named_ifs(lines,local_variables)
        res_lines = []
        for line_index, line in enumerate(lines):
            #Consider only possible lines
            #Done for speed optim
            if "if" in line and "(" in line:
                if_calls = [call for call in self.get_function_calls_in_line(line,local_variables) if call["function_name"] == "if"]
                if len(if_calls) == 1 and " then" not in line and ")then" not in line:
                    if_line = self.replace_func_call(line,if_calls[0],line[if_calls[0]["range"][0]:if_calls[0]["range"][1]] + " then ")
                    if_line, middle_line =  [part.strip() for part in if_line.split(" then ")]
                    middle_line = self.replace_func_call(line,if_calls[0],"")
                    lines[line_index] = line
                    res_lines.append(if_line + " then")
                    res_lines.append(middle_line)
                    res_lines.append("endif")
                else:
                    res_lines.append(line)
            else:
                res_lines.append(line)
        return res_lines
    def normalize_where_calls(self,lines,local_variables):
      res_lines = []
      for line_index, line in enumerate(lines):
        line = line.strip()
        if "where" in line and "(" in line:
          where_calls= [call for call in self.get_function_calls_in_line(line,local_variables) if call["function_name"] == "where"]
          if len(where_calls) == 1 and line[-1] != ")":
            where_line = line[where_calls[0]["range"][0]:where_calls[0]["range"][1]]
            middle_line = line[where_calls[0]["range"][1]:] 
            lines[line_index] = where_line
            res_lines.append(where_line)
            res_lines.append(middle_line)
            res_lines.append("endwhere")
          else:
            res_lines.append(line)
        else:
          res_lines.append(line)
      return res_lines

    def add_vtxbuf_indexes(self,func_name,auxiliaries=False):
        calls = [call for call in self.farray_register_calls if call["function_name"] == func_name]
        for call in calls:
            param_list = self.get_static_passed_parameters(call["parameters"],self.static_variables,self.static_variables)
            index = param_list[1][0]
            self.flag_mappings[index] = str(self.pde_index_counter)
            if auxiliaries:
                self.auxiliary_fields.append(index)
            is_vector_register = any([x[-1] == "vector" and x[0] == "3" for x in param_list])
            if is_vector_register:
                index_base = remove_mod(index)[:-1]
                for dim in ["x","y","z"]:
                   self.flag_mappings[get_mod_name(f"{index_base}{dim}","cdata")] = str(self.pde_index_counter)
                   if auxiliaries:
                    self.auxiliary_fields.append(get_mod_name(f"{index_base}{dim}","cdata"))
                   self.pde_index_counter += 1
            else:
                self.pde_index_counter += 1

    def inline_all_function_calls(self,filename,subroutine_name,new_lines,subs_not_to_inline=None,elim_lines = True):
        self.known_values = {}
        if subs_not_to_inline == None:
          subs_not_to_inline = self.ignored_subroutines
        writes = self.get_writes(new_lines,False)
        local_variables = {parameter:v for parameter,v in self.get_variables(new_lines, {},filename,True).items() }
        new_lines = self.rename_lines_to_internal_names(new_lines,local_variables,filename,subroutine_name)
        local_variables = {parameter:v for parameter,v in self.get_variables(new_lines, {},filename,True).items() }
        
        init_lines = [line for line in new_lines if is_init_line(line)]
        global_init_lines = init_lines
        global_init_lines = unique_list(global_init_lines)
        #normalize .eq. => == and .ne. => /=
        #.lt. => < and .gt. => >
        new_lines = [x.replace(".eq.","==").replace(".ne.","/=").replace(".lt.","<").replace(".gt.",">") for x in new_lines]
        # if subroutine_name == "get_prof_pencil":
        #     file = open("res-get-prof-pencil.txt","w")

        #     new_lines = self.eliminate_while(new_lines)
        #     for line in new_lines:
        #         file.write(f"{line}\n")
        #     file.close()
        #     pexit("HMM\n")

            # file = open("res-get-prof-pencil-elim.txt","w")
            # new_lines = self.eliminate_dead_branches(new_lines,local_variables)
            # for line in new_lines:
            #     file.write(f"{line}\n")
            # file.close()
            # print(self.flag_mappings["lmultilayer__mod__energy"])
            # pexit("HMM\n")
        if elim_lines:
            new_lines = self.eliminate_while(new_lines)
        local_variables = {parameter:v for parameter,v in self.get_variables(new_lines, {},filename,True).items() }

        #TP uncomment to exclude saved local variables
        # for var in local_variables:
        #   if local_variables[var]["saved_variable"] and not local_variables[var]["parameter"]:
        #     if subroutine_name not in ["set_from_slice_x", "set_from_slice_y", "set_from_slice_z","bc_aa_pot_field_extrapol","div","get_reaction_rate"] and self.get_own_module(filename) not in ["special","boundcond"]:
        #       print("saved variable",var)
        #       print(local_variables[var])
        #       print("in:",subroutine_name,filename)
        #       print(self.static_variables["lpscalar__mod__cparam"])
        #       pexit("abort")

        #function name is not in func info if is library call like mpi_bcast
        func_calls_to_replace = [call for call in self.get_function_calls(new_lines,local_variables) if call["function_name"] != subroutine_name and call["function_name"] not in subs_not_to_inline and call["function_name"] in self.func_info and ("interface_funcs" not in self.func_info[call["function_name"]] or all([x not in subs_not_to_inline for x in self.func_info[call["function_name"]]["all_interface_funcs"]]) )]
        for call in func_calls_to_replace:
            if call["function_name"] == "dot":
                pexit("wrong")
        res_func_calls_to_replace = []
        for call in func_calls_to_replace:
            replace_call = True
            if "interface_funcs" in self.func_info[call["function_name"]]:
                replace_call = not any([x in subs_not_to_inline for x in self.func_info[call["function_name"]]["interface_funcs"]])
            if replace_call:
                res_func_calls_to_replace.append(call)
        func_calls_to_replace = res_func_calls_to_replace

        sub_modules = self.get_subroutine_modules(filename,subroutine_name)
        while len(func_calls_to_replace) != 0:
            new_lines = self.expand_function_call_main(new_lines, subroutine_name, filename,func_calls_to_replace,global_init_lines,sub_modules,subs_not_to_inline,elim_lines,local_variables)
            # local_variables = {parameter:v for parameter,v in self.get_variables(new_lines, {},filename).items() }
            func_calls_to_replace.pop(0)
            self.inline_num += 1
        if "inlined_lines" not in self.func_info[subroutine_name]:
            self.func_info[subroutine_name]["inlined_lines"] = {}

        print("subname",subroutine_name)
        file = open("save.txt","w")
        for line in new_lines:
          file.write(f"{line}\n")
        file.close()
        if elim_lines:
            new_lines = self.eliminate_while(new_lines)


        self.func_info[subroutine_name]["inlined_lines"][filename] = new_lines
        file = open(f"res.txt","w")
        for line in new_lines:
            file.write(f"{line}\n")
        file.close()
        where_lines = []
        endwhere_lines = []
        #for line in new_lines:
        #  if "where" in line and "(" in line:
        #    where_lines.append(line)
        #  if line == "endwhere":
        #    endwhere_lines.append(line)
        #if len(where_lines) != len(endwhere_lines): 
        #  print("WRONG")
        #  print(filename,subroutine_name)
        #  print(where_lines)
        #  print(endwhere_lines)
        #  for line in new_lines:
        #    print(line)
        #  assert(False)
    def profile_parse_line(self, line):
        for i in range(1000000):
            self.parse_line(line)

def get_used_files(make_output,directory):
    files = []
    # if make_output is not None:
    #     with open(make_output, mode="r") as file:
    #             lines = file.readlines()
    #             for line in lines:
    #                 search = re.search("([^\s]+\.f90)", line)
    #                 if search is not None:
    #                     files.append(f"{directory}/{search.group(1)}")
    #     return files
    return glob.glob(f"{directory}/**/*.f90",recursive=True)

                    

def print_ranges(parser):
        print("Ranges:")
        has_x_range = False
        has_y_range = False
        has_z_range = False
        for boundcond_range,index,line in parser.ranges:
            if boundcond_range not in [":","1:mx","1:my","1:mz"]:
                print(boundcond_range)
                pexit("NOT full range check what to do?")
            has_x_range = has_x_range or (index == 0)
            has_y_range = has_y_range or (index == 1)
            has_z_range = has_z_range or (index == 2)
        x_dim= global_subdomain_range_with_halos_x if has_x_range else "1"
        y_dim= global_subdomain_range_with_halos_y if has_y_range else "1"
        z_dim= global_subdomain_range_with_halos_z if has_z_range else "1"
        print(f"{x_dim},{y_dim},{z_dim}")
def main():
    argparser = argparse.ArgumentParser(description="Tool to find static writes in Fortran code",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("-f", "--function", help="function to be parsed", required=True)
    argparser.add_argument("-F", "--file", help="File where to parse the function", required=True)
    argparser.add_argument("-m", "--use-make-output",type=str, help="Pass a file path if want to only analyze the files used in build process. Takes in the print output of make. If not given traverses the file structure to find all .f90 files in /src")
    argparser.add_argument("-c", "--communication",default=True,action=argparse.BooleanOptionalAction, help="Whether to check for mpi calls, can be used to check if code to be multithreaded is safe")
    argparser.add_argument("-o", "--offload",default=False,action=argparse.BooleanOptionalAction, help="Whether to offload the current code. Cannot multithread and offload at the same time")
    argparser.add_argument("-t", "--test",default=False,action=argparse.BooleanOptionalAction, help="Whether to generate an inlined version of the given subroutine (mainly used for testing)")
    argparser.add_argument("-d", "--directory",required=True, help="From which directory to look for files")
    argparser.add_argument("--sample-dir", required=False, help="Sample")
    # argparser.add_argument("-od", "--out-directory",required=True, help="To which directory write include files")
    argparser.add_argument("-M", "--Makefile", help="Makefile.local from which used modules are parsed")
    argparser.add_argument("-b", "--boundcond",default=False,action=argparse.BooleanOptionalAction, help="Whether the subroutine to offload is a boundcond or not")
    argparser.add_argument("-s", "--stencil",default=False,action=argparse.BooleanOptionalAction, help="Whether the subroutine to offload is a stencil op e.g. RK3 or not")
    argparser.add_argument("--to-c",default=False,action=argparse.BooleanOptionalAction, help="Whether to translate offloadable function to single threaded C for testing")
    argparser.add_argument("--diagnostics",default=False,action=argparse.BooleanOptionalAction, help="Whether to include diagnostics calculations in rhs calc")
    
    args = argparser.parse_args()
    config = vars(args)
    filename = config["file"]
    subroutine_name = config["function"]
    directory_name = config["directory"]
    files = get_used_files(config["use_make_output"],directory_name)
    files = [file for file in files if os.path.isfile(file)]
    parser = Parser(files,config)
    for var in parser.static_variables:
        
        if parser.static_variables[var]["dims"] == ['nx__mod__cparam,9']:
            parser.static_variables[var]["dims"] = ['nx__mod__cparam','9']
        dims = parser.static_variables[var]["dims"]
        new_dims = []
        for dim in dims:
            new_dims.append(parser.evaluate_integer(dim))
        parser.static_variables[var]["dims"] = new_dims

    global pc_parser
    pc_parser = parser
    global impossible_val

    # lines = ["coeff_grid__mod__cdata = 0.0"]
    # local_variables = {}
    # lines = parser.unroll_ranges(lines,local_variables)
    # file = open("res-after-inlining.txt","w")
    # for line in lines:
    #     file.write(f"{line}\n")
    #     print(line)
    # file.close()
    # exit()

    # print(parser.evaluate_boolean("8.00000038e-03==3.9085e37",{},{}))
    # exit()

    # filename = f"{parser.directory}/{parser.chosen_modules[get_mod_from_physics_name('density')]}.f90"
    # res = {}
    # parser.get_default_flags_from_file(filename,res)
    # print(res["beta_glnrho_scaled__mod__density(1)"])
    # print(res["beta_glnrho_scaled__mod__density(2)"])
    # print(res["beta_glnrho_scaled__mod__density(3)"])
    # # for var in parser.file_info[filename]["variables"]:
    # #     print(var)
    # exit()

    # lines = ["if(.true.) then","endif"]
    # print(parser.elim_empty_branches(lines,{}))
    # exit()


    #PENCIL SPECIFIC
    parser.populate_pencil_case()
    lines = parser.get_lines(f"{parser.sample_dir}/src/cparam.inc")
    lines = parser.get_lines(f"{parser.sample_dir}/src/cparam.local")
    #used to load sample specific values
    parser.get_variables(lines,parser.static_variables,f"{parser.directory}/cparam.f90",False,True)
    parser.update_func_info()
    parser.update_used_modules()
    #done so we are not dependent on cparam.local

    #parser.update_static_vars()
    if not config["offload"]:
        parser.ignored_subroutines.append("rhs_cpu")
        parser.ignored_subroutines.append("calc_diagnostics_particles_rad")
        parser.ignored_subroutines.append("gen_output")
        parser.ignored_subroutines.append("prints")
        parser.ignored_subroutines.append("addforce")
        parser.ignored_subroutines.append("der_onesided_4_slice")
        parser.ignored_subroutines.append("bc_nsbc_prf")
        parser.ignored_subroutines.append("boundcond_shear")
        parser.ignored_subroutines.append("minloc")
        parser.ignored_subroutines.append("integrate_diffusion")
        parser.ignored_subroutines.append("zsum_yy")
        parser.ignored_subroutines.append("maxloc")
        parser.ignored_subroutines.append("huge")
        parser.ignored_subroutines.append("len_trim")
        parser.ignored_subroutines.append("ubound")
        parser.ignored_subroutines.append("lbound")
        parser.ignored_subroutines.append("backspace")
        parser.ignored_subroutines.append("date_and_time")
        parser.ignored_subroutines.append("file_open_hdf5")
        parser.ignored_subroutines.append("h5screate_simple_f")
        parser.ignored_subroutines.extend(["init_hdf5", "initialize_hdf5", "finalize_hdf5", "file_open_hdf5", "file_close_hdf5", "create_group_hdf5"])
        parser.ignored_subroutines.extend(["exists_in_hdf5", "input_hdf5", "output_hdf5", "output_hdf5_double", "wdim", "input_dim"])
        parser.ignored_subroutines.extend(["index_append",      "particle_index_append", "pointmass_index_append", "index_get", "index_reset"])
        parser.ignored_subroutines.extend(["input_profile",     "output_profile"])
        parser.ignored_subroutines.extend(["hdf5_input_slice",  "hdf5_output_slice", "hdf5_output_slice_position"])
        parser.ignored_subroutines.extend(["output_timeseries", "output_settings"])
        parser.ignored_subroutines.extend(["output_average",    "trim_average"])
        ##FUNCTIONS THAT CALL EXTERNAL FUNCTIONS
        parser.ignored_subroutines.append("rhs_gpu")
        parser.ignored_subroutines.append("load_farray_to_gpu")
        parser.ignored_subroutines.append("get_env_var")
        parser.ignored_subroutines.append("copy_farray_from_gpu")
        parser.ignored_subroutines.append("special_calc_particles")
        parser.ignored_subroutines.append("parallel_read")
        parser.ignored_subroutines.append("read_zaver")
        parser.ignored_subroutines.append("read_namelist")
        parser.ignored_subroutines.append("swap")
        parser.ignored_subroutines.append("reload")


    #gsl functions
    parser.ignored_subroutines.extend(["sp_harm_real","sp_harm_imag","sp_harm_imag_costh","sp_harm_real_costh","sp_bessely_l","cyl_bessel_jnu","cyl_bessel_ynu","sp_besselj_l","legendre_pl"])
    parser.ignored_subroutines.append("set_from_slice_x")
    parser.ignored_subroutines.append("set_from_slice_y")
    parser.ignored_subroutines.append("set_from_slice_z")
    parser.ignored_subroutines.append("output_crash_files")
    parser.ignored_subroutines.append("format")
    parser.ignored_subroutines.append("find_index_by_bisection")
    parser.ignored_subroutines.append("svn_id")
    ##TODO is in chemistry.f90 look into why does not work
    parser.ignored_subroutines.append("indgen")
    ##TODO uses the .in. operator that does not work well at the moment
    parser.ignored_subroutines.append("count_lines")
    parser.ignored_subroutines.append("get_pid")
    parser.ignored_subroutines.append("rewind")
    parser.ignored_subroutines.append("get_shared_variable")
    parser.ignored_subroutines.append("put_shared_variable")
    parser.ignored_subroutines.append("find_variable")
    parser.ignored_subroutines.append("farray_use_variable")
    parser.ignored_subroutines.append("farray_use_global")
    parser.ignored_subroutines.append("getderlnrho_x")
    parser.ignored_subroutines.append("getderlnrho_y")
    parser.ignored_subroutines.append("getderlnrho_z")
    parser.ignored_subroutines.append("getderlnrho")


    parser.ignored_subroutines.extend(["lun_input", "lun_output", "lcollective_IO", "IO_strategy"])
    parser.ignored_subroutines.extend(["register_io", "finalize_io"])
    parser.ignored_subroutines.extend(["output_snap", "output_snap_finalize", "output_pointmass", "output_ode"])
    parser.ignored_subroutines.extend(["output_part_snap", "output_part_rmv"])
    parser.ignored_subroutines.extend(["output_average_2D"])
    parser.ignored_subroutines.extend(["output_stalker_init", "output_stalker", "output_part_finalize"])
    parser.ignored_subroutines.extend(["input_snap", "input_snap_finalize", "input_part_snap", "input_pointmass", "input_ode"])
    parser.ignored_subroutines.extend(["output_globals", "input_globals"])
    parser.ignored_subroutines.extend(["input_slice", "output_slice", "output_slice_position"])
    parser.ignored_subroutines.extend(["init_write_persist", "write_persist", "write_persist_id"])
    parser.ignored_subroutines.extend(["init_read_persist", "read_persist", "read_persist_id", "persist_exists"])
    parser.ignored_subroutines.extend(["read_persist_logical_0D", "read_persist_logical_1D"])
    parser.ignored_subroutines.extend(["read_persist_int_0D", "read_persist_int_1D", "read_persist_real_0D"])
    parser.ignored_subroutines.extend(["read_persist_real_1D", "read_persist_torus_rect"])
    parser.ignored_subroutines.extend(["write_persist_logical_0D", "write_persist_logical_1D"])
    parser.ignored_subroutines.extend(["write_persist_int_0D", "write_persist_int_1D", "write_persist_real_0D"])
    parser.ignored_subroutines.extend(["write_persist_real_1D", "write_persist_torus_rect"])
    parser.ignored_subroutines.extend(["wgrid", "rgrid"])
    parser.ignored_subroutines.extend(["wproc_bounds", "rproc_bounds"])
    parser.ignored_subroutines.extend(["directory_names", "log_filename_to_file"])

    parser.safe_subs_to_remove.append("get_shared_variable")
    parser.safe_subs_to_remove.append("put_shared_variable")
    parser.ignored_subroutines.append("get_shared_variable")
    parser.ignored_subroutines.append("put_shared_variable")
    parser.safe_subs_to_remove.append("find_variable")
    # parser.safe_subs_to_remove.append("farray_use_variable")
    # parser.safe_subs_to_remove.append("farray_use_global")

    # print(parser.static_variables["nwgrid__mod__cparam"])
    # exit()
    if config["offload"]:
        impossible_val = parser.static_variables["impossible__mod__cparam"]["value"]
        

        line = "lheatc_kprof__mod__energy = lheatc_kprof__mod__energy .or. lheatc_kconst__mod__energy"
        if(parser.static_variables["lspherical_coords__mod__cdata"]["value"] == ".true."):
            pexit("transpiler not supported for spherical coords at the moment!\n")
        if(parser.static_variables["lcylindrical_coords__mod__cdata"]["value"] == ".true."):
            pexit("transpiler not supported for cylindrical coords at the moment!\n")
    # parser.static_variables["l2__mod__cparam"]["value"] = "35"
    # parser.static_variables["l1__mod__cparam"]["value"] = "4"
    # exit()
    #global global_subdomain_range_x
    #global global_subdomain_range_y
    #global global_subdomain_range_z

    #global global_subdomain_range_with_halos_x
    #global global_subdomain_range_with_halos_y
    #global global_subdomain_range_with_halos_z


    #global global_subdomain_range_x_lower
    #global global_subdomain_range_x_upper

    #global global_subdomain_range_y_lower
    #global global_subdomain_range_y_upper

    #global global_subdomain_range_z_lower
    #global global_subdomain_range_z_upper

    #global implemented_der_funcs
    #global number_of_fields


    #number_of_fields = parser.static_variables[get_mod_name("mfarray","cparam")]["value"]

    #global_subdomain_range_x = parser.static_variables[get_mod_name("nx","cparam")]["value"]
    #global_subdomain_range_y = parser.static_variables[get_mod_name("ny","cparam")]["value"]
    #global_subdomain_range_z = parser.static_variables[get_mod_name("nz","cparam")]["value"]
    #nghost_val = parser.static_variables[get_mod_name("nghost","cparam")]["value"]

    #global_subdomain_range_with_halos_x = parser.evaluate_integer(f"{global_subdomain_range_x}+2*{nghost_val}")
    #global_subdomain_range_with_halos_y = parser.evaluate_integer(f"{global_subdomain_range_y}+2*{nghost_val}")
    #global_subdomain_range_with_halos_z = parser.evaluate_integer(f"{global_subdomain_range_z}+2*{nghost_val}")

    #global_subdomain_range_x_lower= parser.static_variables[get_mod_name("l1","cparam")]["value"]
    #global_subdomain_range_x_upper = parser.static_variables[get_mod_name("l2","cparam")]["value"]

    #global_subdomain_range_y_lower= parser.static_variables[get_mod_name("l1","cparam")]["value"]
    #global_subdomain_range_y_upper = parser.static_variables[get_mod_name("l2","cparam")]["value"]

    #global_subdomain_range_z_lower= parser.static_variables[get_mod_name("l1","cparam")]["value"]
    #global_subdomain_range_z_upper = parser.static_variables[get_mod_name("l2","cparam")]["value"]

    implemented_der_funcs = {
        "der_main": str([("real",[global_subdomain_range_with_halos_x,global_subdomain_range_with_halos_y,global_subdomain_range_with_halos_z,number_of_fields]), ("integer", []), ("real",[global_subdomain_range_x]),("integer",[])]),
        "derij_main": str([("real",[global_subdomain_range_with_halos_x,global_subdomain_range_with_halos_y,global_subdomain_range_with_halos_z,number_of_fields]), ("integer", []), ("real",[global_subdomain_range_x]),("integer",[]),("integer",[])]),
        "der2_main": str([("real",[global_subdomain_range_with_halos_x,global_subdomain_range_with_halos_y,global_subdomain_range_with_halos_z,number_of_fields]), ("integer", []), ("real",[global_subdomain_range_x]),("integer",[])]),
        "der6_main": [
                str([("real",[global_subdomain_range_with_halos_x,global_subdomain_range_with_halos_y,global_subdomain_range_with_halos_z,number_of_fields]), ("integer", []), ("real",[global_subdomain_range_x]),("integer",[])]),
                str([("real",[global_subdomain_range_with_halos_x,global_subdomain_range_with_halos_y,global_subdomain_range_with_halos_z,number_of_fields]), ("integer", []), ("real",[global_subdomain_range_x]),("integer",[]),("logical",[])]),
        ],
        "der5": str([("real",[global_subdomain_range_with_halos_x,global_subdomain_range_with_halos_y,global_subdomain_range_with_halos_z,number_of_fields]), ("integer", []), ("real",[global_subdomain_range_x]),("integer",[])]),
        "der4": str([("real",[global_subdomain_range_with_halos_x,global_subdomain_range_with_halos_y,global_subdomain_range_with_halos_z,number_of_fields]), ("integer", []), ("real",[global_subdomain_range_x]),("integer",[])]),
        "der4i2j": str([("real",[global_subdomain_range_with_halos_x,global_subdomain_range_with_halos_y,global_subdomain_range_with_halos_z,number_of_fields]), ("integer", []), ("real",[global_subdomain_range_x]),("integer",[]),("integer",[])]),
    }

    #for our purposes l2,n2,m2 and etc. can be parameters since their values depend on MPI what we don't care about
    if config["offload"]:
        for param in ["l2","n2","m2","l2i","n2i","m2i"]:
            parser.static_variables[get_mod_name(param,"cparam")]["parameter"] = True
        line = "dx_1(l1__mod__cparam:l2__mod__cparam)"
        lines = [line]
        lines  = parser.inline_known_parameters(lines,also_local_variables=True)
        lines  = parser.inline_known_parameters(lines,also_local_variables=True)


    #for Pencil Code get variables added by scripts
    parser.get_variables(mk_param_lines,parser.static_variables,f"{parser.directory}/cparam.f90",False,True)




    variables = {}
    header = f"{parser.directory}/mpicomm.h"
    # cparam_pencils.inc might not be included
    parser.load_static_variables(f"{parser.directory}/cparam.f90")
    #start is always off since we are transpiling the run version
    parser.flag_mappings["lstart__mod__cdata"] = ".false."
    # parser.static_variables["lpencil"] = {"type": "logical", "dims": ["npencils"], "threadprivate": False,"parameter": False}
    # parser.static_variables["nghost"] = {"type": "integer", "dims": [],"threadprivate": False,"parameter": True, "value": "3"}

    if config["test"] or config["offload"]:
        #get flag mappings from cparam.inc
        #lines = parser.get_lines(f"{parser.sample_dir}/src/cparam.inc")
        #writes = parser.get_writes(lines,False)
        #for write in writes:
        #    if write["variable"][0] == "l" and (write["value"] == ".true." or write["value"] == ".false."):
        #        parser.flag_mappings[write["variable"]] = write["value"]
        #    elif write["variable"] == "nghost":
        #        parser.flag_mappings[write["variable"]] = write["value"]

        ##currently only done for npscalar
        #lines = parser.get_lines(f"{parser.sample_dir}/src/cparam.inc")
        #res_lines = []
        #for x in lines:
        #    res_lines.extend([part.strip() for part in x.split(",")])
        #lines = res_lines
        #writes = parser.get_writes(lines,False)
        #for write in writes:
        #    if write["variable"] not in parser.default_mappings and write["variable"] in ["npscalar"]:
        #        parser.default_mappings[write["variable"]] = write["value"]


        ##get flags from params.log
        #parser.get_flags_from_lines(parser.get_lines(f"{parser.sample_dir}/data/param.nml"))
        #parser.get_flags_from_lines(parser.get_lines(f"{parser.sample_dir}/data/params.log"))
        #parser.get_flags_from_lines(parser.get_lines(f"{parser.sample_dir}/data/param2.nml"))
        ##we are always transpiling the running version
        #parser.flag_mappings["lrun__mod__cdata"] = ".true."
        #print("\nMappings\n")
        #for x in parser.flag_mappings:
        #    print(x)

        #for map_param in parser.default_mappings:
        #    if map_param not in parser.flag_mappings:
        #        parser.flag_mappings[map_param] = parser.default_mappings[map_param]
        parser.ignored_subroutines.append("get_shared_variable")
        parser.ignored_subroutines.append("write_xprof")
        parser.ignored_subroutines.append("write_yprof")
        parser.ignored_subroutines.append("write_zprof")
        parser.ignored_subroutines.append("select_eos_variable")
        parser.ignored_subroutines.append("find_by_name")
        parser.ignored_subroutines.append("farray_index_append")
        parser.ignored_subroutines.append("save_analysis_info")
        parser.ignored_subroutines.append("save_diagnostics_controls")
        parser.ignored_subroutines.append("information")
        for func in sub_funcs:
            parser.ignored_subroutines.append(func)
        #for testing
        # parser.flag_mappings["topbot"] = "top"
        # parser.flag_mappings["lone_sided"] = ".false."
        # parser.flag_mappings["loptest(lone_sided)"] = ".false."
        #print(f"bcy",parser.flag_mappings["bcy__mod__cdata"])
        #don't need it and it breaks parsing function
        #if "cvsid__mod__cdata" in parser.flag_mappings:
        #    del parser.flag_mappings["cvsid__mod__cdata"]
    
        #for x in [y for y in parser.flag_mappings]:
        #    parser.flag_mappings[x] = parser.flag_mappings[x].strip()
        #    #remove arrays and in place but each val
        #    if any([char in parser.flag_mappings[x] for char in "*,"]):
        #        if "'" in parser.flag_mappings[x] or '"':
        #            arr = parse_input_param(parser.flag_mappings[x])
        #            for i in [x[0] for x in enumerate(arr)]:
        #                parser.flag_mappings[f"{x}({i+1})"] = arr[i]
        #        del parser.flag_mappings[x]


        #mcom = int(parser.static_variables["mcom__mod__cparam"]["value"])
        ##if no boundary condition specified then periodic
        #for boundcond_param in ["bcx","bcy","bcz"]:
        #    for index in range(1,mcom+1):
        #        boundcond_param_full = f"{boundcond_param}__mod__cdata"
        #        if f"{boundcond_param_full}{index}" not in parser.flag_mappings:
        #            parser.flag_mappings[f"{boundcond_param_full}({index})"] = "'p'"
        ##get used boundary conditions for each face and dimension
        #for boundcond_param in ["bcx","bcy","bcz"]:
        #    boundcond_param_full = f"{boundcond_param}__mod__cdata"
        #    boundcond_top_bot = f"{boundcond_param}12__mod__cdata"
        #    for index in range(1,mcom+1):
        #        val = parser.flag_mappings[f"{boundcond_param_full}({index})"]
        #        if ":" not in val:
        #            parser.flag_mappings[f"{boundcond_top_bot}({index},1)"] = val
        #            parser.flag_mappings[f"{boundcond_top_bot}({index},2)"] = val
        #        else:
        #            parser.flag_mappings[f"{boundcond_top_bot}({index},1)"] = f"{val.split(':')[0].strip()}'"
        #            parser.flag_mappings[f"{boundcond_top_bot}({index},2)"] = f"'{val.split(':')[1].strip()}"
        #find if some specific boundconds are called that affect the RHS calc
        #all_boundconds_func_calls = []

        #boundconds_x_func_calls = parser.get_boundcond_func_calls("boundconds_x","bcx12__mod__cdata")
        #all_boundconds_func_calls.extend(boundconds_x_func_calls)

        #boundconds_y_func_calls = parser.get_boundcond_func_calls("boundconds_y","bcy12__mod__cdata")
        #all_boundconds_func_calls.extend(boundconds_y_func_calls)

        #boundconds_z_func_calls = parser.get_boundcond_func_calls("boundconds_z","bcz12__mod__cdata")
        #all_boundconds_func_calls.extend(boundconds_z_func_calls)



        #parser.get_default_flags_from_file(f"{parser.directory}/cdata.f90",parser.default_mappings)
        #for mod in ["eos","viscosity","hydro","gravity","density","energy","magnetic","chiral","cosmicray","forcing","shock","special"]:
        #  parser.get_default_flags_from_file(f"{parser.directory}/{parser.chosen_modules[get_mod_from_physics_name(mod)]}.f90",parser.default_mappings)


        #if parser.chosen_modules["forcing"] == "noforcing":
        #  parser.flag_mappings["lforcing_cont__mod__cdata"] = ".false."


        #TODO handle if called
        #for time being simply set to false if not called
        #if "bc_frozen_in_bb" not in [call["function_name"] for call in all_boundconds_func_calls]:
        #    for index in ["1","2","3"]:
        #        parser.flag_mappings[f"lfrozen_bb_bot__mod__magnetic({index})"] = ".false."
        #        parser.flag_mappings[f"lfrozen_bb_top__mod__magnetic({index})"] = ".false."
        #print(parser.flag_mappings["lfrozen_bb_bot__mod__magnetic(1)"])
        #print(parser.flag_mappings["lfrozen_bb_bot__mod__magnetic(2)"])
        #print(parser.flag_mappings["lfrozen_bb_bot__mod__magnetic(3)"])

        #print(parser.flag_mappings["lfrozen_bb_top__mod__magnetic(1)"])
        #print(parser.flag_mappings["lfrozen_bb_top__mod__magnetic(2)"])
        #print(parser.flag_mappings["lfrozen_bb_top__mod__magnetic(3)"])

        #parser.flag_mappings["ioo__mod__cdata"] = "0"
        #parser.get_flags_from_initialization_func("set_coorsys_dimmask",f"{parser.directory}/grid.f90")
        #for mod in ["eos","shock","viscosity","hydro","gravity","density","forcing","energy","magnetic","chiral","cosmicray","special"]:
        #  parser.get_flags_from_initialization_func(f"register_{mod}",f"{parser.directory}/{parser.chosen_modules[get_mod_from_physics_name(mod)]}.f90")
        #for pde_index in ["iphiuu","iee","ietat","icc"]:
        #  index = f"{pde_index}__mod__cdata"
        #  if len([call for call in parser.farray_register_calls if call["parameters"][1] == index]) == 0:
        #    parser.flag_mappings[index] = "0"
        #for func in farray_register_funcs:
        #    parser.add_vtxbuf_indexes(func)

        ## for index in ["iuu","iux","iuy","iuz","ilnrho","iss"]:
        ##     print(f"{index} = ",parser.flag_mappings[get_mod_name(index,"cdata")])








        ##scalar global indexes
        #for global_index in ["bx_ext","by_ext","bz_ext","ax_ext","ay_ext","az_ext","ss0","lnrho0"]:
        #    index = get_mod_name(f"iglobal_{global_index}","cdata")
        #    if len([call for call in parser.farray_register_calls if call["parameters"][1] == index]) == 0:
        #        parser.flag_mappings[index] = "0"
        ##vector global indexes
        #for global_index in ["iglobal_eext__mod__cdata","iglobal_jext__mod__cdata"]:
        #  if len([call for call in parser.farray_register_calls if call["parameters"][1] == index]) == 0:
        #    for index in ["1","2","3"]:
        #        parser.flag_mappings[f"{global_index}({index})"] = ".false."
        #for index in ["iee"]:
        #    if f"{index}__mod__cdata" in parser.flag_mappings and parser.flag_mappings[get_mod_name(index,"cdata")] == "0":
        #        index_prefix = index[:-1]
        #        for dim in ["x","y","z"]:
        #            parser.flag_mappings[f"{index_prefix}{get_mod_name(dim,'cdata')}"] = "0"
        #for mod in ["eos","gravity","density","hydro","forcing","energy","magnetic","chiral","cosmicray","shock","viscosity","special"]:
        #  parser.get_flags_from_initialization_func(f"initialize_{mod}",f"{parser.directory}/{parser.chosen_modules[get_mod_from_physics_name(mod)]}.f90")
        #
        #parser.pde_index_counter = 1
        #for func in farray_register_funcs:
        #    parser.add_vtxbuf_indexes(func, func == "farray_register_auxiliary")

        #for index in ["iss__mod__cdata","ilnrho__mod__cdata","iux__mod__cdata","iuy__mod__cdata","iuz__mod__cdata","iax__mod__cdata","iay__mod__cdata","iaz__mod__cdata","ixx_chiral__mod__chiral","iyy__mod__chiral","iecr__mod__cdata"]:
        #    if index in parser.flag_mappings:
        #        parser.pde_names[parser.flag_mappings[index]] = remove_mod(index[1:]).upper()
        #        parser.pde_names[index] = remove_mod(index[1:]).upper()
        #for index_pair in [("iux__mod__cdata","iuz__mod__cdata"), ("iax__mod__cdata","iaz__mod__cdata")]:
        #    start,end = index_pair
        #    if start in parser.flag_mappings:
        #        parser.pde_vec_names[f"{parser.flag_mappings[start]}:{parser.flag_mappings[end]}"] = (remove_mod(start)[1] + remove_mod(start)[1]).upper()
        #        parser.pde_vec_names[f"{start}:{end}"] = (remove_mod(start)[1] + remove_mod(start)[1]).upper()
        #

        #if "iglobal_gg__mod__gravity" in parser.flag_mappings:
        #    gg = int(parser.flag_mappings["iglobal_gg__mod__gravity"])
        #    ggz = gg+2
        #    parser.pde_vec_names[f"{gg}:{ggz}"] = "F_GG"
        #    parser.pde_names[f"{parser.flag_mappings['iglobal_gg__mod__gravity']}"] = "F_GX"
        #    parser.pde_names[parser.evaluate_integer(f"{parser.flag_mappings['iglobal_gg__mod__gravity']}+1")] = "F_GY"
        #    parser.pde_names[parser.evaluate_integer(f"{parser.flag_mappings['iglobal_gg__mod__gravity']}+2")] = "F_GZ"



        ##TP: comes from density_before_boundary
        #parser.flag_mappings["lupdate_mass_source__mod__density"] =  "lmass_source__mod__density .and. t>=tstart_mass_source .and. (tstop_mass_source==-1.0 .or. t<=tstop_mass_source)"
        #if parser.chosen_modules[get_mod_from_physics_name("eos")] != "noeos":
        #    #replicate the eos var choosing logic
        #    assert(len(parser.select_eos_variable_calls) == 2)
        #    lnrho_val = 2**0
        #    rho_val = 2**1
        #    ss_val = 2**2
        #    lntt_val = 2**3
        #    tt_val = 2**4
        #    cs2_val = 2**5
        #    pp_val = 2**6
        #    eth_val = 2**7
        #    select_val = 0
        #    for i, call in enumerate(parser.select_eos_variable_calls):
        #        if call["parameters"][0] == "'ss'":
        #            select_val += ss_val
        #        elif call["parameters"][0] == "'lnrho'":
        #            select_val += lnrho_val
        #        elif call["parameters"][0] == "'cs2'":
        #            select_val += cs2_val
        #        else:
        #            pexit(f"add eos var value: {call['parameters'][0]}")
        #        parser.flag_mappings[f"ieosvar{i+1}__mod__equationofstate"] = call["parameters"][1]
        #    if select_val == lnrho_val + ss_val:
        #        parser.flag_mappings["ieosvars__mod__equationofstate"] = parser.static_variables["ilnrho_ss__mod__equationofstate"]["value"]
        #    elif select_val == rho_val + ss_val:
        #        parser.flag_mappings["ieosvars__mod__equationofstate"] = parser.static_variables["irho_ss__mod__equationofstate"]["value"]
        #    elif select_val == lnrho_val + lntt_val:
        #        parser.flag_mappings["ieosvars__mod__equationofstate"] = parser.static_variables["ilnrho_lntt__mod__equationofstate"]["value"]
        #    elif select_val == lnrho_val + cs2_val:
        #        parser.flag_mappings["ieosvars__mod__equationofstate"] = parser.static_variables["ilnrho_cs2__mod__equationofstate"]["value"]
        #    else:
        #        pexit(f"add eos select val mapping: {parser.select_eos_variable_calls}")

        #    print("ieos1",parser.flag_mappings["ieosvar1__mod__equationofstate"])
        #    print("ieos2",parser.flag_mappings["ieosvar2__mod__equationofstate"])
        #print("\n flag mappings: \n")
        #for flag in parser.flag_mappings:
        #    print(flag,parser.flag_mappings[flag])
        ## print("lread",parser.flag_mappings["lread_hcond__mod__energy"])
        #for flag in parser.default_mappings:
        #  if flag not in parser.flag_mappings:
        #    parser.flag_mappings[flag] = parser.default_mappings[flag]

        #for mod in parser.shared_flags_accessed:
        #  parser.update_shared_flags(mod)

        ## print(parser.flag_mappings["ldiff_hyper3_polar__mod__density"])
        #if parser.include_diagnostics:
        #  parser.flag_mappings["ldiagnos__mod__cdata"] = ".true."
        parser.normalize_impossible_val()



        # for x in parser.flag_mappings:
        #     if "(" in parser.flag_mappings[x]:
        #         print("hmm",x,parser.flag_mappings[x])
    if config["test"]:

        parser.inline_all_function_calls(filename,subroutine_name) 
        new_lines = parser.func_info[subroutine_name]["inlined_lines"][filename]
        local_variables = {parameter:v for parameter,v in parser.get_variables(new_lines, {},filename,True).items() }
        writes = parser.get_writes(new_lines)
        #adding expand size in test
        new_lines = [parser.expand_size_in_line(line,local_variables,writes) for line in new_lines]
        new_lines= parser.eliminate_while(new_lines)
        new_lines = parser.unroll_constant_loops(new_lines,local_variables)
        global_loop_lines,iterators = parser.get_global_loop_lines(new_lines,local_variables)
        if len(global_loop_lines) > 0:
            new_lines= parser.remove_global_loops(new_lines,local_variables,global_loop_lines,iterators)
        # new_lines= parser.inline_1d_writes(new_lines,local_variables)
        print("\n\nDONE transform\n\n")

        out_file = open(f"{parser.directory}/inlined_bc.inc","w")
        for line in new_lines:
            res_line = line.replace(subroutine_name,f"{subroutine_name}_inlined")
            out_file.write(f"{res_line}\n")
        out_file.close()

        out_file = open(f"{parser.directory}/inlined_bc_declaration.inc","w")
        out_file.write("public:: " +  subroutine_name + "_inlined")
        out_file.close()

        #generate new boundcond.f90
        old_lines = open(f"{parser.directory}/boundcond.f90").readlines()
        res_lines = []
        for line_index,line in enumerate(old_lines):
            res_line = line.strip()
            #don't want to take line continuation into account
            if len(res_line) > 0 and res_line[0] != "!" and subroutine_name in res_line and "&" not in res_line:
                func_calls = [call for call in parser.get_function_calls_in_line(res_line,{}) if call["function_name"] == subroutine_name]
                if len(func_calls) == 1:
                    replace_val = f"test_bc({subroutine_name},{subroutine_name}_inlined"
                    for param in func_calls[0]["parameters"]:
                        replace_val += "," + param
                    replace_val += ")"
                    res_line = parser.replace_func_call(res_line,func_calls[0],replace_val)
            # if "endmodule" in res_line:
            #     res_lines.append("include 'inlined_bc'")
            res_lines.append(res_line)
        # out_file = open(f"{parser.directory}/boundcond.f90","w")
        # for line in res_lines:
        #     out_file.write(f"{line}\n")
        # out_file.close()
        pexit("done test setup")
    if config["offload"]:
        #get flag mappings from cparam.inc
        #for testing
        # parser.flag_mappings["topbot"] = "top"
        # parser.flag_mappings["lone_sided"] = ".true."
        #
        parser.ignored_subroutines.extend(["mpiwtime","random_number_wrapper"])
        parser.ignored_subroutines.extend(["timing"])
        parser.safe_subs_to_remove.extend(["timing"])
        parser.ignored_subroutines.extend(["sum_mn_name","max_mn_name","yzsum_mn_name_x","xzsum_mn_name_y","xysum_mn_name_z","zsum_mn_name_xy","ysum_mn_name_xz","phizsum_mn_name_r","phisum_mn_name_rz","integrate_mn_name","sum_lim_mn_name","save_name"])
        parser.safe_subs_to_remove.append("save_name")
        if not parser.include_diagnostics:
          parser.safe_subs_to_remove.extend(["sum_mn_name","max_mn_name","yzsum_mn_name_x","xzsum_mn_name_y","xysum_mn_name_z","zsum_mn_name_xy","ysum_mn_name_xz","phizsum_mn_name_r","phisum_mn_name_rz","integrate_mn_name","sum_lim_mn_name","save_name"])
          parser.ignored_subroutines.extend(["diagnostic_magnetic","xyaverages_magnetic","yzaverages_magnetic","xzaverages_magnetic"])
          parser.safe_subs_to_remove.extend(["diagnostic_magnetic","xyaverages_magnetic","yzaverages_magnetic","xzaverages_magnetic"])
          parser.ignored_subroutines.extend(["calc_diagnostics_density","calc_diagnostics_magnetic","calc_diagnostics_energy","calc_diagnostics_hydro"])
          parser.safe_subs_to_remove.extend(["calc_diagnostics_density","calc_diagnostics_magnetic","calc_diagnostics_energy","calc_diagnostics_hydro"])

        parser.ignored_subroutines.extend(["die_gracefully", "stop_it","stop_it_if_any"])
        parser.safe_subs_to_remove.extend(["die_gracefully","stop_it","stop_it_if_any"])
        parser.safe_subs_to_remove.extend(["open","close"])


        parser.ignored_subroutines.extend(["fatal_error","not_implemented","fatal_error_local","error","inevitably_fatal_error"])
        parser.safe_subs_to_remove.extend(["fatal_error","not_implemented","fatal_error_local","error","inevitably_fatal_error"])

        parser.ignored_subroutines.extend(["vecout","vecout_finalize","vecout_initialize"])
        parser.safe_subs_to_remove.extend(["vecout","vecout_finalize","vecout_initialize"])

        parser.ignored_subroutines.extend(["output_crash_files"])
        parser.safe_subs_to_remove.extend(["output_crash_files"])

        parser.safe_subs_to_remove.extend(["write"])
        parser.safe_subs_to_remove.extend(["initiate_isendrcv_bdry","finalize_isendrcv_bdry"])
        parser.ignored_subroutines.extend(["initiate_isendrcv_bdry","finalize_isendrcv_bdry"])

        parser.safe_subs_to_remove.extend(["boundconds_y","boundconds_z"])
        parser.safe_subs_to_remove.extend(["notanumber"])
        parser.safe_subs_to_remove.extend(["notanumber_0"])
        parser.safe_subs_to_remove.extend(["notanumber_1"])
        parser.safe_subs_to_remove.extend(["notanumber_2"])
        parser.safe_subs_to_remove.extend(["notanumber_3"])
        parser.safe_subs_to_remove.extend(["notanumber_4"])
        parser.safe_subs_to_remove.extend(["notanumber_5"])
        #special not used for the moment
        parser.safe_subs_to_remove.extend(["caller0","caller1","caller2","caller3","caller4",])
        parser.safe_subs_to_remove.extend(["deri_3d_inds"])
        parser.ignored_subroutines.extend(["boundconds_y","boundconds_z"])
        parser.ignored_subroutines.extend(["deri_3d_inds"])

        # parser.safe_subs_to_remove.extend(["calc_all_pencils"])
        # parser.ignored_subroutines.extend(["calc_all_pencils"])

        subs_not_to_inline = parser.ignored_subroutines.copy()
        #der and others are handled by the DSL
        subs_not_to_inline.extend(der_funcs)
        subs_not_to_inline.extend(parser.safe_subs_to_remove)
        orig_lines = parser.get_subroutine_lines(subroutine_name,filename)

        # #used to generate res-all-inlined.txt
        # parser.inline_all_function_calls(filename,subroutine_name,orig_lines,parser.ignored_subroutines) 
        # all_inlined_lines = parser.func_info[subroutine_name]["inlined_lines"][filename]
        # file = open("res-all-inlined.txt","w")
        # for line in all_inlined_lines:
        #     file.write(f"{line}\n")
        # file.close()
        # exit()

        # subroutine_name = "duu_dt"
        # filename = f"{parser.directory}/hydro.f90"




        # call get_shared_variable('ftop',ftop)
        # call get_shared_variable('fbotkbot',fbotkbot)
        # call get_shared_variable('ftopktop',ftopktop)
        # call get_shared_variable('chi',chi)
        # call get_shared_variable('lmultilayer',lmultilayer)
        # call get_shared_variable('lheatc_chiconst',lheatc_chiconst)
        # call get_shared_variable('lheatc_kramers',lheatc_kramers)
        # if(lheatc_kramers) then
        # call get_shared_variable('hcond0_kramers',hcond0_kramers)
        # call get_shared_variable('nkramers',nkramers)


        if not os.path.isfile("res-inlined.txt"):
            parser.inline_all_function_calls(filename,subroutine_name,orig_lines,subs_not_to_inline) 
            new_lines = parser.func_info[subroutine_name]["inlined_lines"][filename]
            file = open("res-inlined.txt","w")
            for line in new_lines:
                file.write(f"{line}\n")
            file.close()
        else:
            file = open("res-inlined.txt","r")
            new_lines = []
            for line in file.readlines():
                new_lines.append(line)
            file.close()
        print("\n\nDONE inlining\n\n")
        #file = open("res-without-mod-names.txt","w")
        #for line in new_lines:
        #  file.write(f"{remove_mod(line)}\n")
        #file.close()
        #exit(0)
        if parser.include_diagnostics:
          chosen_idiags = [f"idiag_{x}" for x in ["urms","umax","rhom","ssm","dtc","dtu","dtnu","dtchi"]]
          idiag_vars = parser.get_idiag_vars(new_lines)
          print("\nidiag vars\n")
          for var in idiag_vars:
            print(var)
          chosen_idiag_vars = [x for x in idiag_vars if remove_mod(x) in chosen_idiags]
          not_chosen_idiag_vars = [x for x in idiag_vars if x not in chosen_idiag_vars]

          remove_indexes = []
          for line_index,line in enumerate(new_lines):
            for var in chosen_idiag_vars:
              line = line.replace(f"{var}/=0",".true.")
            for var in not_chosen_idiag_vars:
              line = line.replace(f"{var}/=0",".false.")
            if len(line) >= 2 and line[:2] != "if":
              if any([x in line for x in not_chosen_idiag_vars]):
                remove_indexes.append(line_index)
            new_lines[line_index] = line
          new_lines = [x[1] for x in enumerate(new_lines) if x[0] not in remove_indexes]
        local_variables = {parameter:v for parameter,v in parser.get_variables(new_lines, {},filename,True).items() }
        variables = merge_dictionaries(local_variables,parser.static_variables)
        for line_index, _ in enumerate(new_lines):
            new_lines[line_index] = parser.transform_get_shared_variable_line(new_lines[line_index],local_variables,new_lines)

        new_lines = parser.elim_empty_branches(new_lines,local_variables)
        new_lines = parser.eliminate_while(new_lines)


        local_variables = {parameter:v for parameter,v in parser.get_variables(new_lines, {},filename,True).items() }
        new_lines = parser.transform_any_calls(new_lines,local_variables,merge_dictionaries(local_variables,parser.static_variables))

        file = open("res-without-mod-names.txt","w")
        for line in new_lines:
          file.write(f"{remove_mod(line)}\n")
        file.close()

        file = open("res-without-mod-names-after-elim-any_calls.txt","w")
        for line in new_lines:
          file.write(f"{remove_mod(line)}\n")
        file.close()

        new_lines = parser.eliminate_while(new_lines,True,True)
        file = open("res-without-mod-names.txt","w")
        for line in new_lines:
          file.write(f"{remove_mod(line)}\n")
        file.close()

        #TODO remove this quick test hack
        #new_lines = parser.set_optional_param_not_present(new_lines,"lone_sided")

        local_variables = {parameter:v for parameter,v in parser.get_variables(new_lines, {},filename,True).items() }
        variables = merge_dictionaries(variables,parser.static_variables)

        file = open("res-elim-where.txt","w")
        for line in new_lines:
          file.write(f"{line}\n")
        file.close()
        new_lines = parser.eliminate_while(new_lines)
        new_lines = parser.eliminate_while(new_lines)
        file = open("res-before-transform-lines.txt","w")
        for line in new_lines:
          file.write(f"{line}\n")
        file.close()
        new_lines = parser.eliminate_while(new_lines)
        orig_lines = new_lines.copy()
        variables = merge_dictionaries(variables,local_variables)
        for line_index, line in enumerate(new_lines):
                func_calls = parser.get_function_calls_in_line(line,local_variables)
                where_func_calls = [call for call in func_calls if call["function_name"] == "where"]
                where_call_segments = [(None, call["range"][0], call["range"][1]) for call in where_func_calls]
                where_map_vals = []
                for call in where_func_calls:
                  is_scalar_if = False
                  for seg in parser.get_array_segments_in_line(line,variables):
                    param_info = parser.get_param_info((line[seg[1]:seg[2]],False),local_variables,parser.static_variables)
                    if(len(param_info[3]) > 0):
                      range_len = parser.evaluate_integer(param_info[3][0])
                      assert(len(param_info[3]) <= 2)
                      range_len_2nd = None
                      if(len(param_info[3]) == 1):
                          is_scalar_if = is_scalar_if or range_len == global_subdomain_range_x
                      elif(len(param_info[3]) == 2):
                          range_len_2nd = parser.evaluate_integer(param_info[3][1])
                          is_scalar_if = is_scalar_if or (range_len == global_subdomain_range_x and range_len_2nd == "3")
                  for seg in parser.get_struct_segments_in_line(line,variables):
                    param_info = parser.get_param_info((line[seg[1]:seg[2]],False),local_variables,parser.static_variables)
                    assert(len(param_info[3]) <= 2)
                    range_len = parser.evaluate_integer(param_info[3][0])
                    range_len_2nd = None
                    if(len(param_info[3]) == 1):
                        is_scalar_if = is_scalar_if or range_len in [global_subdomain_range_x, global_subdomain_range_x_inner]
                    elif(len(param_info[3]) == 2):
                        range_len_2nd = parser.evaluate_integer(param_info[3][1])
                        is_scalar_if = is_scalar_if or (range_len in [global_subdomain_range_x, global_subdomain_range_x_inner] and (range_len_2nd == "3"))
                  for seg in parser.get_array_segments_in_line(line,variables):
                    param_info = parser.get_param_info((line[seg[1]:seg[2]],False),local_variables,parser.static_variables)
                    assert(len(param_info[3]) <= 2)
                    print("HMM: ",line)
                    range_len = parser.evaluate_integer(param_info[3][0])
                    range_len_2nd = None
                    if(len(param_info[3]) == 1):
                        is_scalar_if = is_scalar_if or range_len in [global_subdomain_range_x, global_subdomain_range_x_inner]
                    elif(len(param_info[3]) == 2):
                        range_len_2nd = parser.evaluate_integer(param_info[3][1])
                        is_scalar_if = is_scalar_if or (range_len in [global_subdomain_range_x, global_subdomain_range_x_inner] and (range_len_2nd == "3"))
                  if not is_scalar_if:
                    print("what to about where")
                    print(parser.get_array_segments_in_line(line,variables))
                    print(param_info)
                    range_len = parser.evaluate_integer(param_info[3][0])
                    if range_len_2nd == None:
                        print("SINGLE RANGE")
                    print("RANGE ",range_len == global_subdomain_range_x)
                    if(range_len_2nd): 
                        print("RANGE 2nd: ",range_len_2nd)
                    print(global_subdomain_range_x)
                    pexit(line)
                  else:
                    where_map_vals.append(line[call["range"][0]:call["range"][1]].replace("where","if",1) + " then")
                new_lines[line_index] = parser.replace_segments(where_call_segments, line, parser.map_val_func,local_variables, {"map_val": where_map_vals})
                if line == "endwhere":
                  new_lines[line_index] = "endif"
                elif line == "elsewhere":
                  new_lines[line_index] = "else"
        new_lines = parser.eliminate_while(new_lines)
        for line in new_lines:
          if "elsewhere" in line:
            pexit("WRONG\n");
        local_variables = {parameter:v for parameter,v in parser.get_variables(new_lines, {},filename,True).items() }
        #transform_func = parser.transform_line_boundcond if config["boundcond"] else parser.transform_line_stencil
        transform_func = parser.transform_line_boundcond_DSL if config["boundcond"] else parser.transform_line_stencil


        # all_inlined_lines = [line.replace("\n","") for line in open("res-all-inlined.txt","r").readlines()]
        # res = parser.transform_lines(new_lines,all_inlined_lines, local_variables,transform_func)
        if parser.offload_type == "stencil":
          output_filename = "mhdsolver-rhs.inc"
        elif parser.offload_type == "boundcond":
          output_filename = "res-boundcond.inc"

        if parser.offload_type == "boundcond":
            first_line = new_lines[0]
            func_calls = parser.get_function_calls_in_line(first_line,local_variables)
            assert(len(func_calls) == 1)
            #make one kernel for top and one for bot
            res_lines = parser.transform_lines(new_lines,new_lines,local_variables,transform_func)
            res_lines = [remove_mod(line) for line in res_lines]
            file = open(output_filename,"w")
            for line in res_lines:
                file.write(f"{line}")
            file.close()
            print_ranges(parser)
            exit()

        #file = open("save-new-lines.txt","w")
        #for line in new_lines:
        #    file.write(f"{line}\n")
        #file.close()
        #TP: to read from already inlined
        #file = open("save-new-lines.txt","r")
        #content = file.read()
        #file.close()
        #new_lines = content.split("\n")
        #print(new_lines)
        local_variables = {parameter:v for parameter,v in parser.get_variables(new_lines, {},filename,True).items() }
        
        res = parser.transform_lines(new_lines,new_lines, local_variables,transform_func)
        res = [remove_mod(line) for line in res]
        res = [normalize_reals(line).replace("(:,1)",".x").replace("(:,2)",".y").replace("(:,3)",".z") for line in res]
        file = open(output_filename,"w")
        for line in res:
            file.write(f"{line}\n") 
        file.close()
        print("DONE")

        exit()

    check_functions = []
    if config["communication"]:
        check_functions = parser.get_function_declarations(parser.get_lines(header), [], header)

    # parser.parse_subroutine_all_files(subroutine_name, "", check_functions, False, {})

    #slice buffers are safe, TODO add check to proof this
    parser.ignored_subroutines.extend(["store_slices","store_slices_scal","store_slices_vec"])
    # parser.parse_subroutine_in_file(f"{parser.directory}/chemistry.f90","calc_pencils_chemistry",[],False)
    parser.parse_subroutine_in_file(filename, subroutine_name, check_functions, config["offload"])
    print("DONE PARSING")
    print("STATIC WRITES") 
    file = open(f"changing_variables.h","w")
    changing_variables = []
    var_names = []
    for write in parser.static_writes:
        var = remove_mod(write["variable"]).strip()
        func = get_func_from_trace(write["call_trace"])
        if var not in var_names:
            var_names.append(var)
            changing_variables.append((var,func))
    for var in changing_variables:
         file.write(f"{var[0]}\t{var[1]}\n")
    file.close()
    exit()
    for module in parser.module_variables:
        print(module)
    parser.save_static_writes(parser.static_writes)

    writes = parser.static_writes
    variables = unique_list([x["variable"] for x in writes if not x["local"]])
    variables = unique_list(variables)

    critical_variables = ["num_of_diag_iter_done","lfinalized_diagnostics","lstarted_finalizing_diagnostics","lstarted_writing_diagnostics","lwritten_diagnostics","lout_save","l1dphiavg_safe","l1davgfirst_save","ldiagnos_save","l2davgfirst_save","lout_save","l1davg_save","l2davg_save","lout_sound_save","lwrite_slices_save","lchemistry_diag_save","it_save","p_fname","p_fname_keep","p_fnamer","p_fname_sound","p_ncountsz","p_fnamex","p_fnamey","p_fnamez","p_fnamexy","p_fnamerz"]

    public_variables = [variable for variable in variables if variable in parser.static_variables and parser.static_variables[variable]["public"]]
    threadprivate_declarations = parser.get_threadprivate_declarations_and_generate_threadpriv_modules(parser.used_files,variables,critical_variables)
    threadprivate_var = []

    all_file = open(f"{parser.directory}/omp_includes/omp.inc","w")

    for file in threadprivate_declarations:
        file_ending= file.rsplit('/',1)[-1].replace('.f90','')
        
    #     print("Creating ", include_filename)
    #     include_file = open(include_filename, "w")
    #     include_file.write(threadprivate_declarations[file])
    #     all_file.write(threadprivate_declarations[file])
    #     include_file.close()
    #     orig_file_lines = open(file,"r").readlines()
    #     if not any([x == "
    #       new_orig_file = open(file,"w")
    #       for line in orig_file_lines:
    #         if line.split(" ",1)[0].strip().lower() == "endmodule":
    #           new_orig_file.write("
    #           new_orig_file.write(f"
    #         new_orig_file.write(f"{line}")
    #       new_orig_file.close()
    #     orig_file_lines = open(file,"r").readlines()
    #     if not any([x == "\n" for x in orig_file_lines]):
    #       new_orig_file = open(file,"w")
    #       for line in orig_file_lines:
    #         if line.strip().lower() == "contains":
    #           new_orig_file.write("\n")
    #           new_orig_file.write(f"
    #         new_orig_file.write(f"{line}")
    #       new_orig_file.close()
    #     orig_file_lines = open(file,"r").readlines()
    #     if not any([x == "!Public declaration added by preprocessor\n" for x in orig_file_lines]):
    #       new_orig_file = open(file,"w")
    #       for line in orig_file_lines:
    #         if line.strip().lower() == "contains":
    #             new_orig_file.write("!Public declaration added by preprocessor\n")
    #             new_orig_file.write(f"_{parser.get_own_module(file)}\n")
    #         new_orig_file.write(f"{line}")
    #       new_orig_file.close()
    #       #Have to now for all other module files add the declaration for the copy in func
    #       for file in parser.find_module_files(parser.get_own_module(file)):
    #         orig_file_lines = open(file,"r").readlines()
    #         if not any([x == "!Public declaration added by preprocessor\n" for x in orig_file_lines]):
    #           new_orig_file = open(file,"w")
    #           for line in orig_file_lines:
    #             if line.strip().lower() == "contains":
    #               new_orig_file.write("!Public declaration added by preprocessor\n")
    #               new_orig_file.write(f"_{parser.get_own_module(file)}\n")
    #             new_orig_file.write(f"{line}")
    #           new_orig_file.close()
    # all_file.close()

    print("DONE")
    for x in checked_local_writes:
      print(x)

            

    

if __name__ == "__main__":
    main()
