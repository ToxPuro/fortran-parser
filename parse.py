from asyncore import write
import math
import re
import os
import csv
import argparse
import glob
import cProfile
import ctypes

global_loop_y = "m__mod__cdata"
global_loop_z = "n__mod__cdata"
global_subdomain_range_x = "nxgrid/nprocx"
global_subdomain_range_y = "nygrid/nprocy"
global_subdomain_range_z = "nzgrid/nprocz"
nghost_val = "3"
global_subdomain_range_with_halos_x = f"{global_subdomain_range_x}+2*{nghost_val}"
global_subdomain_range_with_halos_y = f"{global_subdomain_range_y}+2*{nghost_val}"
global_subdomain_range_with_halos_z = f"{global_subdomain_range_z}+2*{nghost_val}"
global_subdomain_range_x_upper = "l2__mod__cparam"
global_subdomain_range_x_lower= "4"

global_subdomain_range_y_upper = "m2__mod__cparam"
global_subdomain_range_y_lower= "m1__mod__cparam"

global_subdomain_range_z_upper = "n2__mod__cparam"
global_subdomain_range_z_lower= "n1__mod__cparam"

farray_register_funcs = ["farray_register_pde","farray_register_auxiliary"]

number_of_fields = "5+0+0+0"

checked_local_writes = []
der_funcs = ["der","der2","der3","der4","der5","der6","der4i2j","der2i2j2k","der5i1j","der3i3j","der3i2j1k","der4i1j1k","derij"]
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
del_funcs = ["del2v_etc"]

mk_param_lines = [
    "logical, parameter, dimension(npencils):: lpenc_required  = .false.",
    "logical,            dimension(npencils):: lpenc_diagnos   = .false.",
    "logical,            dimension(npencils):: lpenc_diagnos2d = .false.",
    "logical,            dimension(npencils):: lpenc_video     = .false.",
    "logical,            dimension(npencils):: lpenc_requested = .false.",
    "logical,            dimension(npencils):: lpencil         = .false."
]

c_helpers = ctypes.CDLL("./helpers.so") 
c_helpers.hi_from_c.argtypes = [ctypes.c_int]

#transforms .5 -> 0.5 and etc.
#needed since the Astaroth grammar does not support e.g. .5
def normalize_reals(line):
  add_indexes = []
  if line[0] == ".":
    line = "0" + line
  for i,char in enumerate(line):
    if i>0 and i<len(line)-1:
      if char in "." and line[i+1].isnumeric() and not line[i-1].isnumeric():
          add_indexes.append(i)
  res_line = ""
  print(add_indexes)
  for i,char in enumerate(line):
    if i in add_indexes:
      res_line += "0"
    res_line += char
  return res_line
def pexit(str):
  print("TP: DEBUG EXIT: ",str)
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
    return f"{x}__mod__{module}"
def get_mod_from_physics_name(x):
    if x in ["density","hydro","magnetic","viscosity","gravity","energy"]:
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
                print("huh")
                pexit(char)
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
    if "'" in param and ("," in param or "*" in param):
      return parse_input_string(param,"'")
    if '"' in param and ("," in param or "*" in param):
      return parse_input_string(param,'"')
    if "." in param and ("," in param or "*" in param):
      return parse_input_number(param)
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
        return index in [f"{global_subdomain_range_x_lower}:{global_subdomain_range_x_upper}"]
        global_subdomain_range_x_
    if i==1:
        return index in [global_loop_y]
    if i==2:
        return index in [global_loop_z]
def is_vector_stencil_index(index):
    if ":" not in index:
        return False
    lower,upper= [part.strip() for part in index.split(":")]
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
    ## VEC informs that it is a vector access 
    if ":" in index:
        lower,upper= [part.strip() for part in index.split(":")]
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
    print("base",base)
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
def translate_to_c(type):
    if type =="real":
        return "AcReal"
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
        # print("INDEX LINE",index_line)
        num_of_left_brackets = 0
        num_of_right_brackets = 0
        for char in index_line:
            if char == "(": 
                num_of_left_brackets = num_of_left_brackets + 1
            if char == ")":
                num_of_right_brackets = num_of_right_brackets + 1
            if char == "," and num_of_left_brackets == num_of_right_brackets:
                # indexes.append(index.split("(")[-1].split(")")[0].strip())
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
    print(line_elem)
    parts = line_elem.split("::")
    if "!" in parts[0]:
        return False
    return len(parts)>1
def merge_dictionaries(dict1, dict2):
    merged_dict = {}
    merged_dict.update(dict1)
    merged_dict.update(dict2)
    return merged_dict
def get_var_name_segments(line,variables):
  buffer = ""
  res = []
  start_index = 0
  num_of_single_quotes = 0
  num_of_double_quotes = 0
  num_of_left_brackets = 0
  num_of_right_brackets = 0
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
        if buffer and buffer in variables:
          #inside brackets
          if num_of_left_brackets != num_of_right_brackets:
            nsi = i
            while line[nsi] in " ":
              nsi += 1
            #a named param type so don't change it
            if line[nsi] not in "=":
              res.append((buffer,start_index,i))
            else:
              #if == then it is equaility check not named param type
              if line[nsi+1] in "=":
                res.append((buffer,start_index,i))
              #expections are do,if,where,forall
              elif len(line)>=3 and line[:3] == "do":
                res.append((buffer,start_index,i))
              elif len(line)>=len("forall") and line[:len("forall")] == "forall":
                res.append((buffer,start_index,i))
          else:
              res.append((buffer,start_index,i))
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


class Parser:

    def __init__(self, files,config):
        self.known_ints = {
            "iux__mod__cdata-iuu__mod__cdata": "0",
            "iuz__mod__cdata-iuu__mod__cdata": "2",
            "iuu__mod__cdata": "iux__mod__cdata",
            "iuu__mod__cdata+1": "iuy__mod__cdata",
            "iuu__mod__cdata+2": "iuz__mod__cdata",
            "iux__mod__cdata": "iux__mod__cdata",
            "iux__mod__cdata+1": "iuy__mod__cdata",
            "iux__mod__cdata+2": "iuz__mod__cdata",
            'iux__mod__cdata-iuu__mod__cdata+1': "1",
            'iuz__mod__cdata-iuu__mod__cdata+1': "3",
            "{global_subdomain_range_x_upper}-l1+1": global_subdomain_range_x,
            "n1": "1+nghost",
            "iux+1": "iuy",
        }
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
        self.flag_mappings= {
            "headtt":".false.",
            "ldebug":".false.",
        }
        self.default_mappings = {
            # #for now set off
            # "ltime_integrals__mod__cdata":".false.",
        }
        self.default_mappings["l1dphiavg__mod__cdata"] = ".false."
        self.default_mappings["lwrite_phiaverages__mod__cdata"] = ".false."
        self.safe_subs_to_remove = ["print","not_implemented","fatal_error","keep_compiler_quiet","warning"]
        # self.profile_mappings = {
        #     "sinth": "PROFILE_Y", 
        #     "cosph": "PROFILE_Z",
        #     "costh":"PROFILE_Y",
        #     "sinph":"PROFILE_Z",
        #     "x":"PROFILE_X",
        #     "y":"PROFILE_Y",
        #     "z": "PROFILE_Z",
        #     "dline_1_x": "PROFILE_X",
        #     "dline_1_y": "PROFILE_Y",
        #     "dline_1_z": "PROFILE_Z",
        #     "etatotal": "PROFILE_X",
        #     "cs2cool_x": "PROFILE_X",
        #     "cs2mxy": "PROFILE_XY",
        #     "cs2mx": "PROFILE_X",
        #     }
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
        self.rewritten_functions = {}
        self.directory = config["directory"]
        self.subroutine_modifies_param = {}
        self.struct_table = {}
        self.module_variables = {}
        ignored_files = []
        self.get_chosen_modules(config["Makefile"])
        self.ignored_modules = ["hdf5"]
        self.ignored_files = ["nodebug.f90","/boundcond_examples/","deriv_alt.f90","boundcond_alt.f90", "diagnostics_outlog.f90","pscalar.f90", "/cuda/", "/obsolete/", "/inactive/", "/astaroth/", "/pre_and_post_processing/", "/scripts/"]
        # self.ignored_files = ["nodebug.f90","/boundcond_examples/","deriv_alt.f90","boundcond_alt.f90", "diagnostics_outlog.f90","pscalar.f90", "/cuda/", "/obsolete/", "/inactive/", "/astaroth/", "/initial_condition/", "/pre_and_post_processing/", "/scripts/"]
        self.ignored_files.append("magnetic_ffreeMHDrel.f90")
        self.ignored_files.append("photoelectric_dust.f90")
        self.ignored_files.append("interstellar_old.f90")
        self.ignored_files.append("spiegel.f90")
        self.used_files = [file for file in self.used_files if not any([ignored_file in file for ignored_file in self.ignored_files])  and ".f90" in file]
        self.main_program = f"{self.directory}/run.f90"
        if self.offloading:
          if self.chosen_modules["energy"] != "noenergy":
            self.used_files = [x for x in self.used_files if "noenergy.f90" not in x]
          # for module in ["energy"]:
            
        self.not_chosen_files = []
        for file in self.used_files:
            self.get_lines(file)
        self.used_files = [file for file in self.used_files if not self.file_info[file]["is_program_file"] and file not in self.not_chosen_files]
        self.used_files.append(self.main_program)
        # self.used_files = list(filter(lambda x: x not in ignored_files and not self.is_program_file(x) and "/inactive/" not in x and "deriv_alt.f90" not in x and "diagnostics_outlog.f90" not in x "boundcond_alt.f90" and not re.search("cuda",x) and re.search("\.f90", x) and not re.search("astaroth",x)  and "/obsolete/" not in x,self.used_files))
        ##Intrinsic functions
        self.ignored_subroutines = ["alog10","count", "min1", "erf","aimag", "cmplx","len", "inquire", "floor", "matmul","ceiling", "achar", "adjustl", "index", "iabs","tiny","dble","float","nullify","associated","nint","open","close","epsilon","random_seed","modulo","nearest","xor","ishft","iand","ieor","ior","random_number","all","any","deallocate","cshift","allocated","allocate","case","real","int","complex","character","if","elseif","where","while","elsewhere","forall","maxval", "minval", "dot_product", "abs", "alog", "mod", "size",  "sqrt", "sum","isnan", "exp", "spread", "present", "trim", "sign","min","max","sin","cos","log","log10","tan","tanh","cosh","sinh","asin","acos","atan","atan2","write","read","char","merge"]
        ##Ask Matthias about these
        self.ignored_subroutines.extend(["DCONST","fatal_error", "terminal_highlight_fatal_error","warning","caller","caller2", "coeffsx","coeffsy","coeffsz","r1i","sth1i","yh_","not_implemented","die","deri_3d","u_dot_grad_mat"])
        
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
    def replace_variables_multi(self,line, new_vars):
        segments = get_var_name_segments(line,new_vars)
        return self.replace_segments(segments,line,self.map_val_func,{},{"map_val": [new_vars[x[0]] for x in segments]})
    def get_der_index(self,index):
      #this whole stuff is really unnecessary since the orig motivation 
      #was cause I used the Field() syntax wrongly :////
      if index in ["iux__mod__cdata","iuy__mod__cdata","iuz__mod__cdata","ilnrho__mod__cdata","iss__mod__cdata"]:
        return f"F_{remove_mod(index[1:]).upper()}"
      else:
        print("have to deduce der index")
        print(index)
        print(index in self.known_values)
        for val in self.known_values:
          if val in index:
            index = replace_variable(index,val,f"({self.known_values[val]})")
        index = self.evaluate_indexes(index)
        if index in self.known_ints:
          index = self.known_ints[index]
        if index in ["iuu__mod__cdata","iux__mod__cdata","iuy__mod__cdata","iuz__mod__cdata","ilnrho__mod__cdata","iss__mod__cdata"]:
          return f"F_{remove_mod(index[1:]).upper()}"
        #if not was able to deduce just keep it as
        return f"Field({index})"
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
                print("whole range in last f index?")
                pexit(line)
        lower, upper = [part.strip() for part in index.split(":")]
        if index == "idx.x":
            return "idx.x"
        if index == "idx.y":
            return "idx.y"
        if index == "idx.z":
            return "idx.z"

        if index == "1:mx":
            return "idx.x"
        if index == "1:nx":
            return "idx.x"

        if index == "1:my":
            return "idx.y"
        if index == "1:ny":
            return "idx.y"

        if index == "1:mz":
            return "idx.z"
        if index == "1:nz":
            return "idx.z"

        elif index == f"{global_subdomain_range_x_lower}:{global_subdomain_range_x_upper}":
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
        pexit("How to handle f index?: ",index)
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
        if "(" in value and ("/" in value or "*" in value):
            pexit("/ and * not supported with braces")

        if ("+" in value or "-" in value) and "*" not in value and "/" not in value:
            while "(" in value:
                start_index = -1
                end_index = -1
                for i,char in enumerate(value):
                    if char == "(" and end_index<0 and (value[i-1] in " */-+( " or i==0) :
                        start_index = i
                    if char == ")" and end_index<0 and start_index>-1:
                        end_index = i
                if start_index > -1 and end_index >-1:
                    inside = self.evaluate_integer(value[start_index+1:end_index])
                    res_value = value[:start_index]
                    res_value += inside
                    res_value += value[end_index+1:]
                    value = res_value
            res = ""
            sum = 0
            remove_indexes = []
            if value[0].isnumeric():
              remove_indexes.append(0)
              sum += eval(value[0])
            for i in range(1,len(value)):
                last_char = value[i-1]
                char = value[i]
                if last_char == "+" and char.isnumeric():
                    remove_indexes.extend([i-1,i])
                    sum += eval(char)
                if last_char == "-" and char.isnumeric():
                    remove_indexes.extend([i-1,i])
                    sum -= eval(char)
            res_value = ""
            for i, char in enumerate(value):
                if i not in remove_indexes:
                    res_value += char
            if sum == 0:
                return res_value
            #if neg then - is already there
            if sum < 0:
                return res_value + f"{sum}"
            #was able to full constant folding
            if res_value == "":
              return f"{sum}"
            return res_value + "+" + f"{sum}"
        if value in self.known_ints:
            value = self.known_ints[value]
        return value

    def evaluate_indexes(self,value):
        if value == ":":
            return ":"
        print(value)
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
            #not handling unsupported func calls
            if len(func_calls)>0:
                #   print("unsupported func call in evaluate_boolean",value)
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
                    res_value += "ldon't_know"
                res_value += value[end_index+1:]
                res = self.evaluate_boolean_helper(res_value,local_variables,func_calls)
                #if was not able to deduce abort
                # if not has_balanced_parens(value):
                #     print("not balanced")
                #     print(value)
                #     print("---->")
                #     print(res)
                if "ldon't_know" in res:
                    return value
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
        if ">" in value and len(value.split(">")) == 2:
            lhs, rhs = [part.strip() for part in value.split(">")]
            #integer and float comparison
            if lhs.replace(".","").replace("-","").isnumeric() and rhs.replace(".","").replace("-","").isnumeric():
                return ".true." if ((eval(lhs)-eval(rhs)) > 0) else ".false."
            else:
                return value
        if "<" in value and len(value.split("<")) == 2:
            return opposite(self.evaluate_boolean_helper(value.replace("<",">",1),local_variables,func_calls),value)
        if "==" in value and len(value.split("==")) == 2:
            lhs, rhs = [part.strip() for part in value.split("==")]
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
            elif lhs.replace(".","").replace("-","").isnumeric() and rhs.replace(".","").replace("-","").isnumeric():
                return ".true." if ((eval(lhs)-eval(rhs)) == 0) else ".false."
            #string comparison
            elif lhs[0] == "'" and lhs[-1] == "'" and rhs[0] == "'" and rhs[-1] == "'":
                if lhs == rhs:
                    return ".true."
                else:
                    return ".false."
            #TOP /= BOT
            elif lhs == "top" and rhs == "bot":
                return ".false."
            #no less than 3d runs
            elif lhs in ["nxgrid","nygrid","nzgrid","nwgrid__mod__cparam"] and rhs in ["1"]:
                return ".false."
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

    def get_chosen_modules(self,makefile):
        self.chosen_modules = {}
        if makefile:
            lines = [line.strip().lower() for line in open(makefile,'r').readlines() if  line.strip() != "" and line.strip()[0] != "#" and line.split("=")[0].strip() != "REAL_PRECISION"] 
            for line in lines:
                if len(line.split("=")) == 2:
                    variable = line.split("=")[0].strip()
                    value = line.split("=")[1].strip()
                    self.chosen_modules[variable] = value
                    if variable == "density":
                        self.chosen_modules["density_methods"] = f"{value}_methods"
                    if variable == "eos":
                        self.chosen_modules["equationofstate"] = f"{value}"
                    if variable == "entropy":
                        self.chosen_modules["energy"] = f"{value}"

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
              # print("local static write",write)
              # print("abort!")
              # print(self.static_writes[-1])
              # exit()
          else:
            checked_local_writes.append({"variable": variable, "line_num": line_num, "local": is_local, "filename": filename, "call_trace": call_trace, "line": line})


    def get_pars_lines(self,prefix, name, lines):
        res = []
        in_pars = False
        for line in lines:
            line = line.strip().lower()
            if in_pars and line == "/":
                in_pars = False
            if in_pars:
                res.append(line)
            if f"&{name}{prefix}_pars" in line:
                in_pars = True
                res.append(line.replace(f"&{name}{prefix}_pars","").replace("/","").strip())
        #/ is replaced since it represents the end of line
        return [line.replace("/","").strip() for line in res]
    def get_flags_from_lines(self,lines):
        for module in ["grav","density","magnetic","hydro","entropy","viscosity","eos",""]:
            if module:
                prefixes = ["_init","_run"]
            else:
                prefixes = ["init","run"]
            for prefix in prefixes:
                grav_lines = self.get_pars_lines(prefix, module,lines)
                res_lines = []
                for line in grav_lines:
                    res_line = line
                    if len(res_line) >0 and res_line[-1] == ",":
                      res_line = res_line[:-1]
                    res_lines.append(res_line)
                grav_lines = res_lines
                grav_writes = self.get_writes(grav_lines,False)
                if module:
                    mod = get_mod_from_physics_name(module)
                    for write in grav_writes:
                        if write["variable"] in self.rename_dict[mod]:
                            write["variable"] = self.rename_dict[mod][write["variable"]]
                        else:
                            pos_mod = self.get_module_where_declared(write["variable"],self.get_module_file(mod))
                            write["variable"] = self.rename_dict[pos_mod][write["variable"]]
                else:
                    for write in grav_writes:
                        pos_mod = self.get_module_where_declared(write["variable"],f"{self.directory}/param_io.f90")
                        write["variable"] = self.rename_dict[pos_mod][write["variable"]]
                for write in grav_writes:
                    if write["value"] == "t":
                        self.flag_mappings[write["variable"]] = ".true."
                    elif write["value"] == "f":
                        self.flag_mappings[write["variable"]] = ".false."
                    elif "'" in write["value"] and "," not in write["value"] and "*" not in write["value"]:
                        parsed_value = "'" + write["value"].replace("'","").strip() +"'"
                        self.flag_mappings[write["variable"]] = parsed_value
                    #impossible value
                    elif write["value"] == "3.908499939e+37":
                        self.flag_mappings[write["variable"]] = "impossible"
                    else:
                      self.flag_mappings[write["variable"]] = write["value"]
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
        writes = []
        if is_parameter or "=" in line:
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
        if is_public_declaration or in_public:
            if "public_variables" not in self.module_info[module]:
                self.module_info[module]["public_variables"] = []
            for name in variable_names:
                self.module_info[module]["public_variables"].append(name)
        if is_public_declaration:
          return
        #add internal names first for offloading
        if self.offloading:
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
            search = re.search(f"{variable_name}\(((.*?))\)",line) 
            ## check if line is only specifying intent(in) or intent(out)
            if search:
                dims = [index.strip() for index in search.group(1).split(",")]
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
                var_object = {"type": type, "dims": dims, "allocatable": allocatable, "origin": [filename], "public": public, "threadprivate": False, "saved_variable": (saved_variable or "=" in line_variables.split(",")[i]), "parameter": is_parameter, "on_target": False, "optional": is_optional, "line_num": line_num}
                if is_parameter or "=" in line:
                  var_writes = [x for x in writes if x["variable"] == variable_name and "kind" not in x["value"]]
                  #these have meaning to Astaroth
                  if len(var_writes) == 1 and dims==[]:
                    var_object["value"] = var_writes[0]["value"]
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
                                if module_name in ["special","density","energy","hydro","gravity","viscosity","poisson","weno_transport"] and self.chosen_modules[module_name] not in filepath:
                                  self.not_chosen_files.append(filepath)
                                  return
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
                                if(search and "function" in search_line):
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
            elif elem == "=" and line[index-1] not in "/<>=!" and line[index+1] not in "/<>=!" and num_of_single_quotes%2==0 and num_of_double_quotes%2==0 and num_of_left_brackets == num_of_right_brackets:
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
          #for getting idiag /= 0, but there is not that check for every idiag
          # if "if" in line and "(" in line:
          #   if_calls = self.get_function_calls_in_line(line,local_variables)
          #   if len(if_calls) == 1 and len(if_calls[0]["parameters"]) == 1:
          #     if_call = if_calls[0]
          #     possible_var = if_call["parameters"][0].split("/=",1)[0].strip()
          #     if "idiag_" in possible_var:
          #       idiag_vars.append(possible_var)
        return idiag_vars

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
        # index = 0
        # start_index = 0
        # num_of_single_quotes = 0
        # num_of_double_quotes = 0
        # num_of_left_brackets = 0
        # num_of_right_brackets = 0
        # while index<len(line)-1:
        #     index += 1
        #     elem = line[index]
        #     if elem == "'":
        #         num_of_single_quotes += 1
        #     elif e:lem == '"':
        #         num_of_double_quotes += 1
        #     elif elem == "(":
        #         num_of_left_brackets = num_of_left_brackets + 1
        #     elif elem == ")":
        #         num_of_right_brackets = num_of_right_brackets + 1
        #     elif elem == "=" and line[index-1] not in "<>=!" and line[index+1] not in "<>=!" and num_of_single_quotes%2==0 and num_of_double_quotes%2==0 and num_of_left_brackets == num_of_right_brackets:

        #             return parse_variable(line[start_index:index])
        # return None
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
        return [module_name for module_name in  self.get_used_modules(self.get_subroutine_lines(subroutine_name,filename)) if module_name in self.ignored_modules]

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
        #if there is a restriction on the module then it is not recursed
        modules_to_recurse = [x[1] for x in enumerate(modules_to_add) if x[0] in added_module_indexes and not to_add_restrictions[i]]
        for module in modules_to_recurse:
            for file in self.find_module_files(module):
                self.get_all_modules_in_file_scope(file, modules)
    # def get_all_modules_in_subroutine_scope(self,filename,subroutine_name):
    #     if filename not in self.modules_in_scope:
    #         self.modules_in_scope["filename"] = self.get_all_modules_in_file_scope(filename, [self.get_own_module(filename)])
    #     res = unique_list(self.modules_in_scope["filename"] + self.get_subroutine_modules(filename,subroutine_name))
    #     res = [module for module in res if module not in self.ignored_modules]
    #     return res

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
      for var in self.static_variables:
        for i, dim in enumerate(self.static_variables[var]["dims"]):
          file = self.static_variables[var]["origin"][0]
          if dim in ["nx","ny","nz","mx","my","mz"]:
            self.static_variables[var]["dims"][i] = self.rename_line_to_internal_names(dim, {},self.file_info[file]["used_modules"], self.get_own_module(file))
            dim = self.static_variables[var]["dims"][i]
            known_parameters_in_dim= [x for x in self.static_variables if self.static_variables[x]["parameter"] and "value" in self.static_variables[x] and x in dim]
            replace_dict = {}
            for x in known_parameters_in_dim:
              replace_dict[x] = self.static_variables[x]["value"]
            self.static_variables[var]["dims"][i] = self.replace_variables_multi(dim, replace_dict)
            # for var in self.static_variables
            # if dim in self.static_variables and self.static_variables[dim]["parameter"] and "value" in self.static_variables[dim]:
            #   self.static_variables[var]["dims"][i] = self.static_variables[dim]["value"]

      for struct in self.struct_table:
        for field in self.struct_table[struct]:
          file = self.struct_table[struct][field]["origin"][0]
          for i, dim in enumerate(self.struct_table[struct][field]["dims"]):
            if dim in ["nx","ny","nz","mx","my","mz","n_forcing_cont_max"]:
                self.struct_table[struct][field]["dims"][i] = self.rename_line_to_internal_names(dim, {},self.file_info[file]["used_modules"], self.get_own_module(file))
                dim = self.struct_table[struct][field]["dims"][i]
                known_parameters_in_dim= [x for x in self.static_variables if self.static_variables[x]["parameter"] and "value" in self.static_variables[x] and x in dim]
                replace_dict = {}
                for x in known_parameters_in_dim:
                  replace_dict[x] = self.static_variables[x]["value"]
                self.struct_table[struct][field]["dims"][i] = self.replace_variables_multi(dim, replace_dict)

    def update_static_var_values(self):
      for var in self.static_variables:
        if "value" in self.static_variables[var] and "mpi" not in self.static_variables[var]["value"]:
          file = self.static_variables[var]["origin"][0]
          self.static_variables[var]["value"] = self.rename_line_to_internal_names(self.static_variables[var]["value"], {},self.file_info[file]["used_modules"], self.get_own_module(file))
      
      #recurse until base known values
      for var in self.static_variables:
        if "value" in self.static_variables[var] and "mpi" not in self.static_variables[var]["value"]:
          value = self.static_variables[var]["value"]
          known_parameters_in_value= [x for x in self.static_variables if self.static_variables[x]["parameter"] and "value" in self.static_variables[x] and x in value]
          while(len(known_parameters_in_value)>0):
            replace_dict = {}
            for x in known_parameters_in_value:
              replace_dict[x] = self.static_variables[x]["value"]
            value = self.replace_variables_multi(value, replace_dict)
            known_parameters_in_value= [x for x in self.static_variables if self.static_variables[x]["parameter"] and "value" in self.static_variables[x] and x in value]
          self.static_variables[var]["value"] = value

    def update_static_vars(self):
      self.update_static_var_values()
      self.update_static_var_dims()
    
    def get_local_module_variables(self,filename,subroutine_name):
        res = {}
        #Variables in own file take precedence 
        self.load_static_variables(filename,res)

        for module in [x for x in self.get_all_modules_in_subroutine_scope(filename,subroutine_name) if x!=self.get_own_module(filename)]:
            if module not in self.module_variables and module in self.parsed_modules:
                pexit("module was parsed but not self.module_variables",module)
            for var in self.module_variables[module]:
                if var not in res:
                    res[var] = self.module_variables[module][var]
        return res

    def get_subroutine_lines(self, filename, subroutine_name):
        func_info = self.get_function_info(subroutine_name)
        if "lines" not in func_info:
            pexit("trying to get func lines without lines", subroutine_name, filename)
        return func_info["lines"][filename]

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
        # type =  res[2] if res[2] != "" else source[var]["type"]
        type = source[var]["type"] if "%" not in var else "pencil_case"
        return (var,is_static, type, dims)
    def get_param_info(self,parameter,local_variables,local_module_variables):
        if len(parameter[0][0]) == 0:
            pexit("INCORRECT PARAM",parameter)
        if parameter[0][0] == "(" and parameter[0][-1] == ")":
            return self.get_param_info((parameter[0][1:-1],parameter[1]),local_variables,local_module_variables)
        #is scientific number
        if "e" in parameter[0] and parameter[0].replace(".","").replace("-","").replace("e","").replace("+","").isnumeric():
            return (parameter[0],False,"real",[])
        if parameter[0] in local_variables:
            return (parameter[0],parameter[1],local_variables[parameter[0]]["type"],local_variables[parameter[0]]["dims"])
        if parameter[0] in local_module_variables:
            return (parameter[0],parameter[1],local_module_variables[parameter[0]]["type"],local_module_variables[parameter[0]]["dims"])
        if parameter[0] in self.static_variables:
            return (parameter[0],parameter[1],self.static_variables[parameter[0]]["type"],self.static_variables[parameter[0]]["dims"])
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
            elif char == "/" and not in_array and parameter[0][char_index-1] != "(" and parameter[0][char_index+1] != ")" and char_index != 0 and char_index != len(parameter[0])-1:
                if char_index < len(parameter[0])-1:
                    is_division = parameter[0][char_index+1] not in ")!"
                else:
                    is_division= True
                possible_var = ""
            else:
                possible_var = possible_var + char
        operations = (is_sum,is_difference,is_product,is_division)
        # print("PARAM",parameter,operations)
        #Inline array
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
            info = self.get_param_info((parameters[0],False),local_variables,local_module_variables)
            return (parameter[0],"False",info[2],[":"])
        func_calls = self.get_function_calls_in_line(parameter[0],local_variables)
        if len(func_calls)>0 and not any(operations):
            first_call = func_calls[0]
                #Functions that simply keep the type of their arguments
            if first_call["function_name"] in ["sqrt","alog","log","exp","sin","cos","log","abs"]:
                return self.get_param_info((first_call["parameters"][0],False),local_variables,local_module_variables)
            #Array Functions that return single value if single param else an array
            if first_call["function_name"] in ["sum"]:
                new_param = (first_call["parameters"][0],False)
                inside_info =  self.get_param_info(new_param,local_variables,local_module_variables)
                if len(first_call["parameters"]) == 1:
                    return (first_call["parameters"][0],False,inside_info[2],[])
                else:
                    return (first_call["parameters"][0],False,inside_info[2],[":"])
            #Array Functions that return scalar value, multiple params, return type is passed on first param
            if first_call["function_name"] in ["dot_product"]:
                
                inside_info =  self.get_param_info((first_call["parameters"][0],False),local_variables, local_module_variables)
                return (parameter[0],False,inside_info[2],[])
            #Array Functions that return the largest value in params, multiple params, return type is passed on first param
            if first_call["function_name"] in ["max","min"]:
                inside_info =  self.get_param_info((first_call["parameters"][0],False),local_variables, local_module_variables)
                return (parameter[0],False,inside_info[2],inside_info[3])
            
            if first_call["function_name"] in ["maxval","minval"]:
                inside_info =  self.get_param_info((first_call["parameters"][0],False),local_variables, local_module_variables)
                return (parameter[0],False,inside_info[2],inside_info[3][:-1])

            #SPREAD
            if first_call["function_name"] == "spread":
                first_param = parameter[0].split("(",1)[1].split(")")[0].split(",")[0].strip()
                inside_info =  self.get_param_info((first_param,False),local_variables, local_module_variables)
                spread_dims = [":" for x in inside_info[3]]
                #Result dim is source array dim+1
                spread_dims.append(":")
                return (parameter[0],False,inside_info[2],spread_dims)
            if first_call["function_name"] == "real":
                inside_info = self.get_param_info((first_call["parameters"][0],False),local_variables, local_module_variables)
                return (parameter[0],False,"real",inside_info[3])
            if first_call["function_name"] == "int":
                inside_info = self.get_param_info((first_call["parameters"][0],False),local_variables, local_module_variables)
                return (parameter[0],False,"integer",inside_info[3])
            if first_call["function_name"] == "trim":
                return (parameter[0],False,"character",[])
            if first_call["function_name"] == "len":
                return (parameter[0],False,"integer",[])
            #DCONST is Astaroth specific
            if first_call["function_name"] in ["merge","dconst"]:
                return self.get_param_info((first_call["parameters"][0],False),local_variables,local_module_variables)
            if first_call["function_name"] in self.func_info:
                file_path = self.find_subroutine_files(func_calls[0]["function_name"])[0]
                interfaced_call = self.get_interfaced_functions(file_path,func_calls[0]["function_name"])[0]
                _,type,param_dims = self.get_function_return_var_info(interfaced_call,self.get_subroutine_lines(interfaced_call, file_path), local_variables)
                return (parameter[0],False,type, param_dims)
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
                part_res = self.get_param_info((part,False),local_variables,local_module_variables)
                if parts_res[2] == "" or len(part_res[3])>len(parts_res[3]):
                    parts_res = (parts_res[0],False,part_res[2],part_res[3])
            return parts_res
        elif "%" in parameter[0] and "'" not in parameter[0] and '"' not in parameter[0]:
            var_name,field_name = [part.strip() for part in parameter[0].split("%")]
            ##var_name can be array access if array of structures
            var_name = var_name.split("(")[0]
            struct = ""
            if var_name in local_variables:
                struct = local_variables[var_name]["type"]
            else:
                struct = local_module_variables[var_name]["type"]
            field_name = field_name
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
        elif "(" in parameter[0] and ")" in parameter[0] and not any(operations) and ".and." not in parameter[0] and ".not." not in parameter[0]:
            var = parameter[0].split("(")[0].strip()
            if var in local_variables or var in local_module_variables:
                return self.get_var_info_from_array_access(parameter,local_variables,local_module_variables)
            ##Boolean intrinsic funcs
            elif var in ["present","isnan","associated","allocated","all","any"]:
                return (parameter[0],False,"logical",[])
            elif var in ["trim"]:
                return (parameter[0],False,"character",[])
            else:
                #check if function in source code
                print(parameter)
                pexit("how did I end up here?")
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
            print("subroutine in it's own interface")
            pexit(subroutine_name)
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
        suitable_functions = []
        for function in interfaced_functions:
            is_suitable = True
            subroutine_lines = self.get_subroutine_lines(function, file_path)
            parameters = self.get_parameters(subroutine_lines[0])
            is_elemental = "elemental" in subroutine_lines[0]
            local_variables = {parameter:v for parameter,v in self.get_variables(subroutine_lines, {},file_path, True).items() }
            mapping = self.get_parameter_mapping(parameters,parameter_list)
            ##For testing uncomment
            # print("FUNCTION",function, "IN", file_path)
            # print("PARAMS")
            # for param in parameters:
            #     print(param, local_variables[param])
            #if mapping is less than parameter list than some named optional paramaters are not present in sub parameters
            is_suitable  = len(mapping) == len(parameter_list) and len(parameters) >= len(parameter_list)
            ##check if type and length of dims match between passed parameter and function parameter.
            ## All other parameters need to be optional 
            if is_suitable:
                # print(len(parameter_list))
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
        if len(suitable_functions) > 1 or len(suitable_functions)<num_of_suitable_needed:
            print(f"There are {len(suitable_functions)} suitable functions for the interface call: ",subroutine_name, "in file", file_path)
            print("Params: ",parameter_list)
            print("Original candidates: ", interfaced_functions)
            print("Possibilities", suitable_functions)
            pexit(call)
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
        print("CHOOSING RIGHT MODULE")
        for i, path in enumerate(filepaths):
            for module in self.chosen_modules:
                if path.lower() == f"{self.directory}/{self.chosen_modules[module]}.f90":
                    return filepaths[i]
        if original_file in self.func_info[call["function_name"]]["lines"]:
          return original_file
        pexit("did not found module in files",filepaths)
        
    def parse_subroutine_all_files(self, sub_call, call_trace, check_functions, offload,local_variables,file_called_from, layer_depth=math.inf, parameter_list=[], only_static=True):
        subroutine_name = sub_call["function_name"]
        if layer_depth<0:
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
            print(f"There are no suitable function for the interface call: ",subroutine_name)
            print("Params: ",parameter_list)
            print("Original candidates: ", all_functions)
            pexit(sub_call)
        self.subroutine_modifies_param[subroutine_name][str(param_types)] = global_modified_list

    def replace_vars_in_lines(self,lines, new_names,exclude_variable_lines=False):
        if exclude_variable_lines:
            res_lines = []
            for line in lines:
                if is_body_line(line):
                    res_lines.append(self.replace_variables_multi(line,new_names))
                else:
                    res_lines.append(line)
            return res_lines
        return [self.replace_variables_multi(line, new_names) for line in lines]
    def replace_var_in_lines(self, lines, old_var, new_var):
        # return [add_splits(replace_variable(line,old_var,new_var)) for line in lines]
        return [replace_variable(line,old_var,new_var) for line in lines]
    def get_subroutine_lines(self,subroutine_name,filename):
        func_info = self.get_function_info(subroutine_name)
        print("sub name",subroutine_name)
        print("files", [file for file in func_info["lines"]])
        print("files", [file for file in func_info["files"]])
        print("lines", [file for file in func_info["files"]])
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
                print("Weird dim in size")
                print(dim)
                pexit(params)
        else:
            return "mx*my*mz"
    def get_dim_info(self,param,local_variables,variables_in_scope,writes):
        if param in local_variables:
            src = local_variables
        else:
            src = self.static_variables
        if not any([":" in dim or "size" in dim for dim in src[param]["dims"]]):
            print("here",param)
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
        print("returning",res_dims)
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
            print("size call", func_call)
            info = self.get_param_info((params[0],False),local_variables,self.static_variables)
            print("param info",info)
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
            print(subroutine_lines[0])
            return (return_var, type, return_dims)
    def get_replaced_body(self, filename, parameter_list, function_call_to_replace, variables_in_scope,global_init_lines,subs_not_to_inline,elim_lines):
        original_subroutine_name = function_call_to_replace["function_name"]
        ##in case is interfaced call get the correct subroutine
        print("GETTING REPLACED BODY FOR: ", function_call_to_replace)
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
        #subroutine_lines = self.get_subroutine_lines(subroutine_name,filename)

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

        #remove_lines that write to parameters that are not passed
        remove_indexes = []
        for i,line in enumerate(new_lines):
            writes = self.get_writes_from_line(line)
            if(len(writes) == 1):
                if writes[0]["variable"] in not_present_params:
                    remove_indexes.append(i)

        new_lines = [x[1] for x in enumerate(new_lines) if x[0] not in remove_indexes]

        for i,line in enumerate(new_lines):
            #replace present with false if not present and true if present
            #Done for speed optimization
            # if "present(" in line or "loptest(" in line or any([x in line for x in not_present_params]):
            if "present(" in line or "where(" in line:
                func_calls = self.get_function_calls_in_line(line,local_variables)
                present_func_calls = [func_call for func_call in func_calls if func_call["function_name"] == "present"]
                present_func_call_segments = [(None, call["range"][0], call["range"][1]) for call in present_func_calls]
                # present_vars = [call["parameters"][0] for call in present_func_calls]
                present_map_val = [line[call["range"][0]:call["range"][1]] if call["parameters"][0] in optional_present_params else ".false." if call["parameters"][0] in not_present_params else ".true." for call in present_func_calls]
                # for x in present_func_calls:
                #   if x["parameters"] in optional_present_params:
                #     present_map_val.append(line[x["range"][0]:x["range"][1]])
                #   elif x["para"]

                #replace loptest with false if not present and with the value if present
                # loptest_func_calls = [func_call for func_call in func_calls if func_call["function_name"] == "loptest" and len(func_call["parameters"]) == 1]

                # loptest_func_call_segments = [(None, call["range"][0], call["range"][1]) for call in loptest_func_calls]
                # loptest_vars = [call["parameters"][0] for call in loptest_func_calls]
                # loptest_map_val = [".false." if var in not_present_params else var for var in loptest_vars]


                #remove params from funcs if the param was not passed.
                #This enables correct deduction for present values
                # not_present_func_call = [x for x in func_calls if x["function_name"] not in ["present","loptest"] and any([param in x["parameters"] for param in not_present_params])]
                # not_present_map_val = []
                # not_present_func_call_segments = [(None, call["range"][0], call["range"][1]) for call in not_present_func_call]
                # for call in not_present_func_call:
                #   only_present_call_params = [x for x in call["parameters"] if x not in not_present_params]
                #   not_present_map_val.append(f"{call['function_name']}({','.join(only_present_call_params)})")
                # where_func_calls = [call for call in func_calls if call["function_name"] == "where"]
                # where_call_segments = [(None, call["range"][0], call["range"][1]) for call in where_func_calls]
                # where_map_vals = []
                # for call in where_func_calls:
                #   is_scalar_if = False
                #   for seg in self.get_array_segments_in_line(line,variables):
                #     param_info = self.get_param_info((line[seg[1]:seg[2]],False),local_variables,self.static_variables)
                #     print(param_info)
                #     is_scalar_if = is_scalar_if or (param_info[3] in [[global_subdomain_range_x,"3"],[global_subdomain_range_x]] )
                #   for seg in self.get_struct_segments_in_line(line,variables):
                #     param_info = self.get_param_info((line[seg[1]:seg[2]],False),local_variables,self.static_variables)
                #     print(param_info)
                #     is_scalar_if = is_scalar_if or (param_info[3] in  [[global_subdomain_range_x],[global_subdomain_range_x,"3"]])
                #   if not is_scalar_if:
                #     print("what to about where")
                #     pexit(line)
                #   else:
                #     where_map_vals.append(line[call["range"][0]:call["range"][1]].replace("where","if",1) + " then")
                func_call_segments = present_func_call_segments
                # func_call_segments.extend(where_call_segments)
                # func_call_segments.extend(loptest_func_call_segments)
                # func_call_segments.extend(not_present_func_call_segments)
                map_val = present_map_val
                # map_val.extend(where_map_vals)
                # map_val.extend(loptest_map_val)
                # map_val.extend(not_present_map_val)

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

        print("PARAMS:",subroutine_lines,params)
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
              



        init_variables= {parameter:v for parameter,v in self.get_variables(init_lines, {},filename,True).items() }
        global_init_lines.extend(init_lines)
        global_init_lines = unique_list(global_init_lines)

        # #check for saved local variables
        # local_variables = {parameter:v for parameter,v in self.get_variables([(line,0) for line in new_lines], {},filename).items() }
        # for var in local_variables:
        #   if local_variables[var]["saved_variable"] and not local_variables[var]["parameter"]:
        #     if subroutine_name not in ["set_from_slice_x", "set_from_slice_y", "set_from_slice_z","bc_aa_pot_field_extrapol","div","get_reaction_rate"] and self.get_own_module(filename) not in ["special","boundcond"]:
        #       print("saved variable",var)
        #       print(local_variables[var])
        #       print("in:",subroutine_name,filename)
        #       pexit("abort")

        lines = new_lines
        #Todo has to evaluate whether return line is hit or not
        has_return_line = False 
        for count,line in enumerate(lines):
            if line.strip() == "return":
                has_return_line = True
        # if has_return_line:
        #     print("\n\nlines before elimination\n\n")
        #     for (line,count) in lines:
        #         print(line)
            # lines = [(line,0) for line in self.eliminate_while([x[0] for x in lines])]
        #TODO: fix
        # if has_return_line:
        #     print("\n\nlines after elimination\n\n")
        #     for (line,count) in lines:
        #         print(line)
        #     if self.has_no_ifs(lines):
        #         remove_indexes = []
        #         has_return_line = False
        #         for line_index,(line,count) in enumerate(lines):
        #             has_return_line = has_return_line or line == "return"
        #             if has_return_line:
        #                 remove_indexes.append(line_index)
        #         lines = [x[1] for x in enumerate(lines) if x[0] not in remove_indexes]
        #     else:
        #         pexit("don't know what to do since return statement in a conditional branch")
        for count,line in enumerate(lines):
            if ".false.=.true." in line:
                print("don't want to inline this func")
                print(subroutine_name, original_subroutine_name)
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
                pexit("iux__mod__cdata" in self.static_variables)
        return ([line for line in lines if is_body_line(line)],is_function,type)
    def parse_subroutine_in_file(self, filename, subroutine_name, check_functions, offload, global_modified_list = [], layer_depth=math.inf, call_trace="", parameter_list=[], only_static=True):
        print("parse_in_file", subroutine_name, filename,parameter_list)
        if layer_depth < 0:
            return []
        if subroutine_name not in self.subroutine_modifies_param:
            self.subroutine_modifies_param[subroutine_name] = {}
        subroutine_lines = self.get_subroutine_lines(subroutine_name, filename)
        lines = subroutine_lines[1:]
        own_module = self.get_own_module(filename)
        #used to parse module don't think it is needed anymore
        local_variables = {parameter:v for parameter,v in self.get_variables(lines, {},filename, True).items() }
        for var in local_variables:
          if local_variables[var]["saved_variable"] and not local_variables[var]["parameter"] and subroutine_name not in ["div"]:
            #skipping special modules for now
            if subroutine_name not in ["set_from_slice_x", "set_from_slice_y", "set_from_slice_z","bc_aa_pot_field_extrapol"] and own_module not in ["special","boundcond"]:
              print("saved variable",var)
              print(local_variables[var])
              print("in:",subroutine_name,filename)
              pexit("abort")
        local_module_variables = self.get_local_module_variables(filename,subroutine_name)
        print("param_list",parameter_list)
        print(":(")
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
                print([var for var in  local_variables])
                print([var for var in  self.static_variables])
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
                print("adding normal write")
                print("from", subroutine_name, filename)
                self.add_write(write["variable"], write["line_num"], write["variable"] in local_variables and local_variables[write["variable"]]["saved_variable"], filename, call_trace, write["line"])
        for function_call in self.get_function_calls(lines, local_variables):
            function_name = function_call["function_name"].lower()
            parse = True
            if function_name in check_functions or function_name.lower().startswith("mpi"):
                self.found_function_calls.append((call_trace, function_name, parameter_list))
                parse = False
            parse = parse and function_name.lower().strip() not in self.ignored_subroutines
            if parse:
                print("FINDING PARAMS IN", subroutine_name, filename, "FOR", function_name)
                print(f"FUNC PARAMS for {function_name}",function_call["parameters"])
                new_param_list = self.get_static_passed_parameters(function_call["parameters"],local_variables,local_module_variables)
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
                                    print("add write after call")
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
                                    print("recursive func?")
                                    self.add_write(passed_parameter, 0, False, filename, call_trace, global_modified_list[i]["line"])
                else:
                    new_param_list = self.get_static_passed_parameters(function_call["parameters"],local_variables,local_module_variables)
                    for i in range(len(new_param_list)):
                        if self.subroutine_modifies_param[function_name][str(param_types)][i]["is modified"] and ((new_param_list[i][0] in self.static_variables and new_param_list[i][0] not in local_variables) or (new_param_list[i][0] in local_variables and local_variables[new_param_list[i][0]]["saved_variable"])): 
                            print("adding from skipped call from: ",subroutine_name, filename)
                            print("to call", function_call)
                            self.add_write(new_param_list[i][0], 0, new_param_list[i][0] in local_variables, self.subroutine_modifies_param[function_name][str(param_types)][i]["filename"], self.subroutine_modifies_param[function_name][str(param_types)][i]["call trace"], self.subroutine_modifies_param[function_name][str(param_types)][i]["line"])
    def evaluate_ifs(self,lines,local_variables):
        for line_index, line in enumerate(lines):
            # print("evaluating ifs",line)
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
                        print("wrong replacement_index")
                        print(info)
                        pexit(line)
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
    def rename_lines_to_internal_names(self,lines,local_variables,filename):
        modules = self.file_info[filename]["used_modules"]
        for i, line in enumerate(lines):
            line_before = lines[i]
            # print(line_before)
            # print("--->")
            lines[i] = self.rename_line_to_internal_names(line,local_variables,modules,self.get_own_module(filename))
            # print(lines[i])

        return lines

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
        return self.replace_segments(var_segments,line,self.rename_to_internal_module_name,local_variables,{"modules": modules,"own_module":own_module})

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
        if len(found_modules) == 0:
            if segment[0] in self.rename_dict[info["own_module"]]:
                found_modules.append(info["own_module"])
        # print("var:", segment[0])
        # print("seg:",line[segment[1]:segment[2]])
        # print("line",line)
        # print(found_modules)
        assert(len(found_modules) == 1)
        return line[segment[1]:segment[2]].replace(segment[0],self.rename_dict[found_modules[0]][segment[0]])

    def unroll_range(self,segment,segment_index,line,local_variables,info):
        variables = merge_dictionaries(local_variables,self.static_variables)
        sg_indexes = get_segment_indexes(segment,line,0)
        if variables[segment[0]]["dims"] in [[global_subdomain_range_x,"3"],[global_subdomain_range_x,"3","3"]] and len(sg_indexes) > 1:
            for i, index in enumerate(sg_indexes):
                if i==info["index_num"] and index==info["old_index"]:
                    sg_indexes[i] = info["new_index"]
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
        lines = self.replace_vars_in_lines(lines,profile_replacements)
        res_lines = []
        for line_index,line in enumerate(lines):
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
            for sg in arr_segs_in_line:
                indexes = get_segment_indexes(sg,line,0)
                unroll = unroll or variables[sg[0]]["dims"] in [[global_subdomain_range_x,"3"],[global_subdomain_range_x,"3","3"]] and len(indexes) > 1 and indexes[1] == "1:2"
            if unroll:
                info = {
                    "index_num": 1,
                    "old_index": "1:2",
                    "new_index": "1",
                }
                line_0 = self.replace_segments(arr_segs_in_line,line,self.unroll_range,local_variables,info)
                info["new_index"] = "2"
                line_1 = self.replace_segments(arr_segs_in_line,line,self.unroll_range,local_variables,info)
                res_lines.extend([line_0,line_1])
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
              val_info = self.get_param_info((write["value"],False),local_variables,local_variables)
              if val_info[3] == []:
                for dim in ["1","2","2"]:
                  if len(rhs_info[3]) == 2:
                    rhs_dims = [":",dim]
                  else:
                    rhs_dims = [dim]
                  res_lines.append(f"{build_new_access(rhs_segment[0],rhs_dims)} = {write['value']}") 
                  print(res_lines)
              else:
                res_lines.append(line)
            else:
              res_lines.append(line)
          else:
            res_lines.append(line)
        return res_lines
    def inline_0d_replacer(self,segment,segment_index,line,local_variables,info):
        if segment_index== 0 and len(self.get_writes_from_line(line)) > 0:
            return line[segment[1]:segment[2]]
        if segment[0] in info["possible_values"]:
            return replace_variable(line[segment[1]:segment[2]], segment[0], f"({info['possible_values'][segment[0]]})")
            
            # indexes = get_segment_indexes(segment,line,1)
            # is_safe = indexes[0] in [":",f"{global_subdomain_range_x_lower}:{global_subdomain_range_x_upper}"] or len(local_variables[var]["dims"]) == 0
            # if is_safe:
            #     res = "("  + replace_variable(line[segment[1]:segment[2]].split("(",1)[0].strip(), var, info["possible_values"][var]) + ")"
            #     return res
            #     # add_line =  replace_variable(line[segment[1]:segment[2]], var, possible_values[var])
            # else:
            #     print("not safe abort")
            #     print(line)
            #     print(line[segment[1]:segment[2]])
            #     exit()
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
            print("SAFE VARS TO INLINE",safe_vars_to_inline)
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
            print("SAFE VARS TO INLINE",safe_vars_to_inline)
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
      if also_local_variables:
        for var in local_variables:
            if local_variables[var]["parameter"] and local_variables[var]["dims"] == [] and "value" in local_variables[var]:
                inline_constants[var] = local_variables[var]["value"]
                #remove declarations for local parameters since they are not needed anymore
                remove_indexes.append(local_variables[var]["line_num"])
      for var in self.static_variables:
          if self.static_variables[var]["parameter"] and self.static_variables[var]["dims"] == [] and "value" in self.static_variables[var] and var not in local_variables:
              inline_constants[var] = self.static_variables[var]["value"]
      return [x[1] for x in enumerate(self.replace_vars_in_lines(lines,inline_constants)) if x[0] not in remove_indexes]
    def eliminate_while(self,lines,unroll=False,take_last_write_as_output=False):
        local_variables = {parameter:v for parameter,v in self.get_variables(lines, {},self.file, True).items() }
        lines = self.normalize_if_calls(lines, local_variables)
        lines = self.normalize_where_calls(lines, local_variables)
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
        #if there are some writes to flagged params then not safe to substitue
        removed_flags = {}
        writes = self.get_writes(lines,False)
        analyse_lines = self.get_analyse_lines(lines,local_variables)
        flag_mappings = [x for x in self.flag_mappings]
        for flag_mapping in flag_mappings:
                # if write is in if clauses can't be sure about value
                if len([write for write in writes if write["variable"] == flag_mapping]) > 0:
                    removed_flags[flag_mapping] = self.flag_mappings[flag_mapping]
                    del self.flag_mappings[flag_mapping]
        
        local_variables = {parameter:v for parameter,v in self.get_variables(lines, {},self.file, True).items() }
        lines = self.inline_known_parameters(lines,local_variables)
        if unroll:
          lines = self.unroll_constant_loops(lines,local_variables)
        for line in lines:
          if "if(*/=0.) then" in line:
            print("WRONG")
            pexit(line)

        for line_index in range(len(lines)):
            for x in [x for x in self.flag_mappings if "(" in x and ")" in x]:
                orig_line = lines[line_index]
                lines[line_index]= lines[line_index].replace(x,self.flag_mappings[x])
                if "if(*/=0.) then" in lines[line_index]:
                  print("before:", orig_line)
                  print("mapping",x)
                  print("mapping val",self.flag_mappings[x])
                  print("WRONG")
                  pexit(lines[line_index])
        #     for mapping in self.flag_mappings:
        #         if "(" in mapping:
        #             lines[line_index] = lines[line_index].replace(mapping,self.flag_mappings[mapping])
        for line_index in range(len(lines)):
            lines[line_index] = self.replace_variables_multi(lines[line_index],self.flag_mappings)
        for line in lines:
          if "if(*/=0.) then" in line:
            print("WRONG")
            pexit(line)
        # done twice since can have lflag_a -> lflag_b .or. lflag_c
        for line_index in range(len(lines)):
            lines[line_index] = self.replace_variables_multi(lines[line_index],self.flag_mappings)


        for line in lines:
          if "if(*/=0.) then" in line:
            print("WRONG")
            pexit(line)
        lines = self.transform_case(lines)
        ##Needed to remove size from variable dims
        for line in lines:
          if "if(*/=0.) then" in line:
            print("WRONG")
            pexit(line)
        local_variables = {parameter:v for parameter,v in self.get_variables(lines, {},self.file, True).items() }


        len_orig_removed_flags = len(removed_flags) + 1
        while(len_orig_removed_flags > len(removed_flags)):
          len_orig_removed_flags = len(removed_flags)
          lines = [self.expand_size_in_line(line,local_variables,writes) for line in lines]
          self.evaluate_ifs(lines,local_variables)
          orig_lines_len = len(lines)+1
          local_variables = {parameter:v for parameter,v in self.get_variables(lines, {},self.file, True).items() }
          while(orig_lines_len > len(lines)):
                  writes = self.get_writes(lines,False)
                  lines = self.eliminate_dead_branches(lines,local_variables)
                  self.try_to_deduce_if_params(lines,writes,local_variables,take_last_write_as_output)
                  self.evaluate_ifs(lines,local_variables)
                  orig_lines_len = len(lines)

          writes = self.get_writes(lines,False)
          analyse_lines = self.get_analyse_lines(lines,local_variables)
          func_calls = self.get_function_calls(lines,local_variables)
          self.try_to_deduce_if_params(lines,writes,local_variables,take_last_write_as_output)
          for flag in [x for x in removed_flags]:
            if flag in self.known_values:
              self.flag_mappings[flag] = self.known_values[flag]
              del removed_flags[flag]
          for flag in [x for x in removed_flags]:
            #if write in conditional branch not sure of value
            # if len([x for x in writes if x["variable"] == flag and analyse_lines[x["line_num"]][1] > 0]) == 0 and len([x for x in func_calls if flag in x["parameters"] and x["function_name"] not in ["if"]]) == 0:
            if [x for x in writes if x["variable"] == flag and analyse_lines[x["line_num"]][1] > 0] == []:
              if [x for x in func_calls if flag in x["parameters"] and x["function_name"] not in ["if","put_shared_variable","allocate"]] == []:
                self.flag_mappings[flag] = removed_flags[flag]
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
              self.flag_mappings[val] =self.known_values[val]
          for flag in [x for x in removed_flags]:
            if flag in self.known_values:
              self.flag_mappings[flag] = self.known_values[flag]
              del removed_flags[flag]
          for flag in [x for x in removed_flags]:
            #if write in conditional branch not sure of value
            # if len([x for x in writes if x["variable"] == flag and analyse_lines[x["line_num"]][1] > 0]) == 0 and len([x for x in func_calls if flag in x["parameters"] and x["function_name"] not in ["if"]]) == 0:
            if [x for x in writes if x["variable"] == flag and analyse_lines[x["line_num"]][1] > 0] == []:
              if [x for x in func_calls if flag in x["parameters"] and x["function_name"] not in ["if","put_shared_variable"]] == []:
                self.flag_mappings[flag] = removed_flags[flag]
                del removed_flags[flag]
          self.try_to_deduce_if_params(lines,writes,local_variables,take_last_write_as_output)
          for flag in [x for x in removed_flags]:
            if flag in self.known_values:
              self.flag_mappings[flag] = self.known_values[flag]
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

        for line in lines:
          if "if(*/=0.) then" in line:
            print("WRONG")
            pexit(line)
        return lines
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
    def transform_line_stencil(self,line,num_of_looped_dims, local_variables, array_segments_indexes,rhs_var,vectors_to_replace, writes):
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
        rhs_segment = get_variable_segments(line, [rhs_var])
        if len(rhs_segment) == 0:
            rhs_segment = self.get_struct_segments_in_line(line, [rhs_var])
        rhs_segment  = rhs_segment[0]
        rhs_info = self.get_param_info((line[rhs_segment[1]:rhs_segment[2]],False),local_variables,self.static_variables)
        rhs_dim = [self.evaluate_indexes(dim) for dim in rhs_info[3]]
        if (
            (rhs_var in local_variables and
                (
                    (num_of_looped_dims == 0 and len(rhs_dim) == 0)
                    or (num_of_looped_dims == 1 and rhs_dim in [[global_subdomain_range_x,"3"],[global_subdomain_range_x]])
                    or (num_of_looped_dims == 2 and rhs_dim in [[global_subdomain_range_x,"3"]])
                    or (num_of_looped_dims == 3 and rhs_dim in [[global_subdomain_range_x,"3","3"]])
                ))
                or (rhs_var in ["df"] or rhs_var in vectors_to_replace)
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
                        if is_vector_stencil_index(indexes[-1]):
                            make_vector_copies = True
                        else:
                            print("range in df index 3")
                            print(line[segment[1]:segment[2]])
                            print(indexes)
                            pexit(orig_indexes)
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
                          res = f"value({vtxbuf_name})"
                else:
                    var_dims = src[segment[0]]["dims"]
                    indexes = [self.evaluate_indexes(index) for index  in get_indexes(line[segment[1]:segment[2]],segment[0],0)]
                    is_profile = (
                        segment[0] in self.static_variables
                        and var_dims in [[global_subdomain_range_x],[global_subdomain_range_with_halos_x],[global_subdomain_range_y],[global_subdomain_range_with_halos_y],[global_subdomain_range_z],[global_subdomain_range_with_halos_z],[global_subdomain_range_x,global_subdomain_range_with_halos_y]]
                        and len([x for x in writes if x["variable"] == segment[0]])  == 0
                      )
                    if is_profile:
                        #PROFILE_X
                        if var_dims in [[global_subdomain_range_x],[global_subdomain_range_with_halos_x]]:
                            if indexes not in [[],[f"{global_subdomain_range_x_lower}:{global_subdomain_range_x_upper}"],[f"{global_subdomain_range_x_lower}+3:{global_subdomain_range_x_upper}+3"],[f"{global_subdomain_range_x_lower}-1:{global_subdomain_range_x_upper}-1"]]:
                                print(indexes)
                                print("WEIRD INDEX in profile_x")
                                print(indexes)
                                pexit(line[segment[1]:segment[2]])
                            profile_index = "0"
                            if indexes == [f"{global_subdomain_range_x_lower}+3:{global_subdomain_range_x_upper}+3"]:
                                profile_index = "3"
                            if indexes == [f"{global_subdomain_range_x_lower}-1:{global_subdomain_range_x_upper}-1"]:
                                profile_index = "-1"
                        #PROFILE_Y
                        elif var_dims in [[global_subdomain_range_y],[global_subdomain_range_with_halos_y]]:
                            if indexes not in  [[global_loop_y],[f"{global_loop_y}-1"]]:
                                print("WEIRD INDEX in profile_y")
                                print(indexes)
                                pexit(line[segment[1]:segment[2]])
                            profile_index = "0"
                            if indexes == [f"{global_loop_y}-1"]:
                                profile_index = "-1"
                        #PROFILE_Z
                        elif var_dims in [[global_subdomain_range_z],[global_subdomain_range_with_halos_z]]:
                            if indexes not in [[global_loop_z],[f"{global_loop_z}+3"],[f"{global_loop_z}-1"],[f"{global_loop_z}-3"]]:
                                print("WEIRD INDEX in profile_z")
                                print(indexes)
                                pexit(line[segment[1]:segment[2]])
                            profile_index = "0"
                            if indexes == [f"{global_loop_z}+3"]:
                                profile_index = "3"
                            if indexes == [f"{global_loop_z}-1"]:
                                profile_index = "-1"
                            if indexes == [f"{global_loop_z}-3"]:
                              profile_index = "-3"
                        #PROFILE_XY
                        elif var_dims in [[global_subdomain_range_x,global_subdomain_range_with_halos_y]]:
                            if indexes not in [[":",global_loop_y]]:
                                print("WEIRD INDEX in profile_xy")
                                print(indexes)
                                pexit(line[segment[1]:segment[2]])
                            profile_index = "0"
                        else:
                            pexit("add profile mapping to profile " + self.profile_mappings[segment[0]])
                        res = "AC_PROFILE_" + segment[0].upper() + "[" + profile_index + "]"
                    #assume that they are auxiliary variables that similar to pencils but not inside pencil case
                    elif segment[0] in self.static_variables and var_dims in [[global_subdomain_range_x],[global_subdomain_range_with_halos_x]]:
                      if indexes  in [[":"]]:
                        res = segment[0]
                      else:
                        res = line[segment[1]:segment[2]]
                    #these turn to scalar read/writes
                    elif segment[0] in local_variables and self.evaluate_indexes(src[segment[0]]["dims"][0]) == global_subdomain_range_x and indexes in [[],[":"]]:
                        res = segment[0]
                    elif segment[0] in local_variables and self.evaluate_indexes(src[segment[0]]["dims"][0]) == global_subdomain_range_with_halos_x and indexes in [f"{global_subdomain_range_x_lower}:{global_subdomain_range_x_upper}"]:
                        res = segment[0]
                    #global vec
                    elif segment[0] in self.static_variables and src[segment[0]]["dims"] == ["3"] and indexes in [["1"],["2"],["3"]]:
                        #do nothing
                        return line[segment[1]:segment[2]]
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
                          assert(False)
                    #constant local array
                    elif len(src[segment[0]]["dims"]) == 1 and src[segment[0]]["dims"][0].isnumeric() and len(indexes) == 1:
                        return line[segment[1]:segment[2]]
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
                    elif src[segment[0]]["dims"] == [global_subdomain_range_x,"3","3","3"]:
                        #read/write to tensor indexes:
                        if num_of_looped_dims == 1 and indexes[0] == ":":
                            res = f"{segment[0]}.data[{indexes[1]}][{indexes[2]}][{indexes[3]}]"
                        else:
                          print("unsupported tensor read/write")
                          print(line[segment[1]:segment[2]])
                    else:
                        print("what to do?")
                        print(line[segment[1]:segment[2]])
                        print(segment[0])
                        print("is static: ",segment[0] in self.static_variables)
                        print("is local: ",segment[0] in local_variables)
                        print("var dims",var_dims)
                        print(src[segment[0]])
                        pexit(line)
                res_line += line[last_index:segment[1]] + res
                last_index = segment[2]
            # res_line += line[last_index:] + ";"
            res_line += line[last_index:]
            # #For time being commented out
            # if "%" in res_line:
            #     res_line = res_line.replace(";","")
            #     res_final_line = ""
            #     last_index = 0
            #     struct_segs = self.get_struct_segments_in_line(res_line,variables)
            #     for seg in struct_segs:
            #         var_name,field = [part.strip() for part in seg[0].split("%",1)]
            #         if var_name in local_variables:
            #             src = local_variables
            #         elif var_name in self.static_variables:
            #             src = self.static_variables
            #         if src[var_name]["type"] != "pencil_case":
            #             print("what to do non pencil_case struct ?")
            #             print("struct seg", seg[0], res_line[seg[1]:seg[2]])
            #             exit()
            #         else:
            #             indexes = [self.evaluate_indexes(index) for index in get_segment_indexes(seg, res_line, 0)]
            #             #replace 1:n -> with : if an index dim is n
            #             for i,index in enumerate(indexes):
            #                 if ":" in index and index != ":":
            #                     lower,upper = [part.strip() for part in index.split(":")]
            #                     if lower == "1" and upper == self.struct_table[src[var_name]["type"]][field]["dims"][i]:
            #                         indexes[i] = ":"
            #             ##Pencil case will become a x dimensional profile.
            #             ##If vec make three x dimensional profiles
            #             if "(" not in res_line[seg[1]:seg[2]] or indexes == [":"] or (len(indexes) == 2 and indexes[0] == ":" and (indexes[1].isnumeric() or indexes[1] == ":")):
            #                 res = "AC_PROFILE_" + field.upper()
            #             else:
            #                 print("weird array access in pencil case")
            #                 print(res_line[seg[1]:seg[2]])
            #                 print(indexes)
            #                 print(line)
            #                 exit()
            #         res_final_line= res_final_line + res_line[last_index:seg[1]]
            #         res_final_line = res_final_line + res 
            #         last_index = seg[2]
            #     res_final_line= res_final_line + res_line[last_index:]
            #     res_line = res_final_line + ";"
            return res_line

        else:
            print("NO case for",line)
            print("RHS VAR",rhs_var)
            print(rhs_var in local_variables)
            pexit(local_variables[rhs_var]["dims"])
    def transform_line_boundcond(self,line,num_of_looped_dims, local_variables, array_segments_indexes,rhs_var,vectors_to_replace):
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
                            indexes[i] = f"{indexes[i]}-1"
                    res = f"{new_var}[DEVICE_VTXBUF_IDX({','.join(indexes)})]"
                ##Value local to kernel i.e. from the viewpoint of a thread a scalar
                elif segment[0] in local_variables and len(local_variables[segment[0]]["dims"]) == 2 and num_of_looped_dims == 2:
                    res = segment[0] 
                else:

                    indexes = [self.evaluate_indexes(index) for index in get_segment_indexes(segment,line,len(src[segment[0]]["dims"]))]
                    num_of_looping_dim = -1
                    for i,index in enumerate(indexes):
                        if ":" in index:
                            num_of_looping_dim += 1
                        if index == ":":
                            possible_index = src[segment[0]]["dims"][i]
                            if possible_index == ":":
                                possible_index = local_variables[rhs_var]["dims"][num_of_looping_dim]
                            indexes[i] = self.map_to_new_index("1:" + possible_index,local_variables)
                        elif ":" in index:
                            indexes[i] = self.map_to_new_index(indexes[i],local_variables)
                        else:
                            indexes[i] = index
                    res = segment[0]
                    for index in indexes:
                        res = res + f"[{index}]"
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
                    lower,upper = [self.evaluate_integer(part.strip()) for part in write["value"].split(",")]
                    # lower,upper= [part.strip() for part in write["value"].split(",")]
                    print("DO LINE",line)
                    print("PARTS:",lower,upper)
                    print(self.evaluate_integer("1+1"))
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
    def transform_line(self,i,lines,local_variables,loop_indexes,symbol_table,initialization_lines,orig_params,transform_func,vectors_to_replace,writes):
        line = lines[i]
        #we disregard some mn_loop setup lines
        if line in ["n__mod__cdata=nn__mod__cdata(imn__mod__cdata)","m__mod__cdata=mm__mod__cdata(imn__mod__cdata)","mn_loop: do imn=1,ny*nz","enddo mn_loop","headtt=.false.","lfirstpoint=.false."]:
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

                
            if len(vars_to_declare) > 0 and local_variables[vars_to_declare[0]]["type"] != "pencil_case":
                return translate_to_c(local_variables[vars_to_declare[0]]["type"]) + " " + ", ".join(vars_to_declare) + ";"
            else:
                return ""
        if line.strip()  == "exit":
            return "continue"
        if "else" in line and "if" in line:
            return "}\n" + line.replace("then","{").replace("elseif","else if")
        if "else" in line:
            return "}\nelse {"
        if "if" in line and "then" in line:
            params = self.get_function_calls_in_line(line.replace("then",""),local_variables)[0]["parameters"]
            if len(params) == 1:
                return line.replace("then","{")
            else:
                return line.replace("then","{")
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
            loop_indexes = loop_indexes[:-1]
            return "}\n"
        if "subroutine" in line and "end" in line:
            if self.test_to_c:
                return ""
            else:
                return "}\n"
        if "subroutine" in line:
            function_name = self.get_function_calls_in_line(line,local_variables)[0]["function_name"]
            if self.test_to_c:
                return ""
            elif self.offload_type == "boundcond":
                return f"static __global__ void\n {function_name}(const int3 dims, VertexBufferArray vba)\n"+"{\n"
            elif self.offload_type == "stencil":
              return "Kernel rhs(){"
        if is_use_line(line):
            return ""
        if "do" in line[:2]:
            loop_index = self.get_writes_from_line(line)[0]["variable"]
            lower,upper= [part.strip() for part in line.split("=")[1].split(",",1)]
            loop_indexes.append(loop_index)
            return f"for(int {loop_index} = {lower};{loop_index}<={upper};{loop_index}++)" +"{"
        if "endif" in line:
            return "}"
        original_line = line
        if new_is_variable_line(line):
            return ""

        func_calls = self.get_function_calls_in_line(line,local_variables)
        if len(func_calls) == 1 and func_calls[0]["function_name"] in der_funcs:
              file_path = self.find_subroutine_files(func_calls[0]["function_name"])[0]
              interfaced_call = self.get_interfaced_functions(file_path,func_calls[0]["function_name"])[0]
              #derij_main will do nothing if i==
              if interfaced_call == "derij_main" and func_calls[0]["parameters"][3] == func_calls[0]["parameters"][4]:
                return ""
              if interfaced_call not in implemented_der_funcs:
                print("implement der func:",interfaced_call, "in DSL")
                print(func_calls[0])
                exit()
              else:
                new_param_list = self.get_static_passed_parameters(func_calls[0]["parameters"],local_variables,self.static_variables)
                param_types = [(param[2],param[3]) for param in new_param_list]
                if str(param_types) not in implemented_der_funcs[interfaced_call]:
                  #for der4 if ignoredx is not on then it a normal ignoredx call
                  # if interfaced_call in ["der4","der6_main"] and len(func_calls[0]["parameters"]) == 5 and self.evaluate_boolean(func_calls[0]["parameters"][4],local_variables,[]) == ".false.":
                  #   pass
                  # else:
                  print(interfaced_call)
                  print("not implemented for these param types")
                  print(param_types)
                  print(func_calls[0])
                  exit()
              rest_params = func_calls[0]["parameters"][:2] + func_calls[0]["parameters"][3:]
              if interfaced_call in ["der_main","der2_main"]:
                res = f"{func_calls[0]['parameters'][2]} = {der_func_map[interfaced_call][func_calls[0]['parameters'][3]]}({self.get_der_index(func_calls[0]['parameters'][1])})"
              elif interfaced_call in ["der6_main"]:
                if len(new_param_list) == 4:
                  res = f"{func_calls[0]['parameters'][2]} = {der_func_map[interfaced_call][func_calls[0]['parameters'][3]]}({self.get_der_index(func_calls[0]['parameters'][1])})"
                else:
                  if new_param_list[4][-1] == "upwind":
                    res = f"{func_calls[0]['parameters'][2]} = {der_func_map[interfaced_call][func_calls[0]['parameters'][3]]}_upwd({self.get_der_index(func_calls[0]['parameters'][1])})"
                  else:
                    print("hmm is it ignoredx?")
                    print(new_param_list)
                    assert(False)
              else:
                print("no der case for ", interfaced_call)
                assert(False)
              return res

        rhs_segment = get_rhs_segment(line)
        if rhs_segment is None:
            print("rhs seg is None")
            print("line: ",line)
            print(func_calls)
            exit()
        rhs_var = self.get_rhs_variable(line)
        if rhs_var is None:
            print("rhs var is none")
            pexit(line)
        rhs_var = rhs_var.lower()
        if rhs_var not in local_variables:
            if rhs_var in [".false.",".true."]:
              return ""
            local_variables[rhs_var] = self.static_variables[rhs_var]
            # print("WHAT TO DO rhs not in variables",line)
            # print(rhs_var)

            # # #for the time being simply assume they are diagnostics writes so can simply remove them
            # # return ""
            
            # print(rhs_var in self.static_variables)
            # if rhs_var == global_loop_z:
            #     print("IS n_save in local_variables?","n_save" in local_variables)
            # exit()
        dim = len(variables[rhs_var]["dims"])
        indexes = get_indexes(get_rhs_segment(line),rhs_var,dim)
        dims, num_of_looped_dims = get_dims_from_indexes(indexes,rhs_var)

        #line = self.transform_spread(line,[f":" for i in range(num_of_looped_dims)],local_variables)
        array_segments_indexes = self.get_array_segments_in_line(line,variables)
        return transform_func(line,num_of_looped_dims, local_variables, array_segments_indexes,rhs_var,vectors_to_replace,writes)
    def transform_spreads(self,lines,local_variables,variables):
        res_lines = []
        for line_index, line in enumerate(lines):
            spread_calls= [call for call in self.get_function_calls_in_line(line,variables) if call["function_name"] == "spread"]
            if len(spread_calls) > 1:
                print("multiple spread calls",line)
                exit()
            elif len(spread_calls) == 1:
                if "*/=" in line:
                  print("wrong")
                  pexit(line)
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
                if rhs_indexes == [] and len(rhs_info[3]) == 1:
                    new_rhs = f"{var_name}"
                    res_line = new_rhs + res_line[rhs_segment[2]:]
                    res_lines.append(res_line)
                    # res_lines.append(f"do explicit_index = 1,{rhs_info[3][0]}")
                    # res_lines.append(res_line)
                    # res_lines.append("enddo")
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
                    if "*/=" in res_line:
                      print("wrong")
                      print(line)
                      print("--->")
                      pexit(res_line)

                #not really a spread
                elif call["parameters"][2] == "1":
                    res_lines.append(res_line)
                    if "*/=" in res_line:
                      print("wrong")
                      print(line)
                      print("--->")
                      pexit(res_line)
                else:
                    print("have to append it to")
                    print("rhs var",rhs_var)
                    print("indexes",rhs_indexes)
                    print(variables[rhs_var]["dims"])
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
            for line_index, line in enumerate(lines[:-2]):
                if "if" in line and "then" in line:
                    if_calls = [call for call in self.get_function_calls_in_line(line,local_variables) if call["function_name"] == "if"]
                    if len(if_calls) == 1:
                        if any([x == lines[line_index+1] for x in ["endif", "end if"]]):
                            remove_indexes.append(line_index)
                            remove_indexes.append(line_index+1)
                        elif lines[line_index+1] == "else" and any([x == lines[line_index+2] for x in ["endif", "end if"]]):
                            remove_indexes.extend([line_index,line_index+1,line_index+2])
            lines = [x[1] for x in enumerate(lines) if x[0] not in remove_indexes]
        return lines
    def elim_unnecessary_writes_and_calls(self,lines,local_variables,variables):
        orig_lines_len = len(lines)+1
        while(orig_lines_len>len(lines)):
            print(orig_lines_len,len(lines))
            orig_lines_len = len(lines)
            #elim empty if branches
            lines = self.elim_empty_branches(lines,local_variables)
            var_dict = {}
            writes = self.get_writes(lines,False)
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
                
            remove_indexes = []
            if "mass_per_proc" in var_dict:
                var_dict["mass_per_proc"]["is_used"] = False
            for var in var_dict:
                if not var_dict[var]["is_used"]:
                    if self.get_param_info((var,False),local_variables,variables)[2] != "integer":
                        remove_indexes.extend(var_dict[var]["indexes"])
            lines = [x[1] for x in enumerate(lines) if x[0] not in remove_indexes]
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
        for x in possible_modules:
            print(x, static_var in self.rename_dict[x])
            print(x, static_var in self.rename_dict[x])
        exit()
    def transform_lines(self,lines,all_inlined_lines,local_variables,transform_func):
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
            if "print*" in line or "write(*,*)" in line:
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
                            print("So can't generate cuda code for it")
                            exit()

        lines = [x[1] for x in enumerate(lines) if x[0] not in remove_indexes]
        for line in lines:
          if "if(*/=0.) then" in line:
            print("WRONG")
            pexit(line)

        file = open(f"res-eliminated.txt","w")
        for line in lines:
            file.write(f"{line}\n")
        file.close()
        print("check eliminated file")


        # global_loop_lines,iterators = self.get_global_loop_lines(lines,local_variables)
        # if len(global_loop_lines) > 0:
        #     lines = self.remove_global_loops(lines,local_variables,global_loop_lines,iterators)

        #unroll forall lines
        for line_index,line in enumerate(lines):
            search_line = line.replace("forall","call forall")
            func_calls = self.get_function_calls_in_line(search_line,local_variables)
            if len(func_calls) == 1 and func_calls[0]["function_name"] == "forall" and len(func_calls[0]["parameters"]) == 1:
                write = self.get_writes_from_line((func_calls[0]["parameters"])[0])[0]
                iterator = write["variable"]
                if ":" not in write["value"]:
                    print("should unroll for all")
                    print("don't know how to do it")
                    pexit(line)
                replacement_lower, replacement_upper = [part.strip() for part in write["value"].split(":")]

                res = self.replace_segments(self.get_array_segments_in_line(search_line,variables),search_line,self.global_loop_replacer,local_variables,{
                    "iterator": iterator, 
                    "replacement_lower": replacement_lower,
                    "replacement_upper": replacement_upper,
                    "replacement_index": 3,
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
                            print("struct seg", seg[0], res_line[seg[1]:seg[2]])
                            exit()
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

        variables = merge_dictionaries(self.static_variables, local_variables)
        for line in lines:
          if "if(*/=0.) then" in line:
            print("WRONG")
            pexit(line)
        print("Unrolling constant loops")
        lines = [line.strip() for line in lines]
        file = open("res-unroll.txt","w")
        for line in lines:
            file.write(f"{line}\n")
        file.close()
        for line in lines:
          if "if(*/=0.) then" in line:
            print("WRONG")
            pexit(line)
        print("Check unroll file")

        #get local variables back to get actual dims not size dims
        local_variables = {parameter:v for parameter,v in self.get_variables(lines, {},self.file, True).items() }

        lines = [line.strip() for line in lines if line.strip() != ""]
        file = open("res-inlined.txt","w")
        for line in lines:
            file.write(f"{line}\n")
        file.close()
        for line in lines:
          if "if(*/=0.) then" in line:
            print("WRONG")
            pexit(line)
        print("Check inlined file")
        #try to transform maxvals into max before transform 
        for line_index,line in enumerate(lines):
            maxval_calls = [call for call in self.get_function_calls_in_line(line,variables) if call["function_name"] == "maxval"]
            if len(maxval_calls) > 1:
                print("multiple maxval calls")
                pexit(line)
            elif len(maxval_calls) == 1:
                first_param_info = self.get_param_info((maxval_calls[0]["parameters"][0],False),local_variables,self.static_variables)
                #max of nx (possibly vector components), is safe to take max
                if ((len(maxval_calls[0]["parameters"]) == 1) or (len(maxval_calls[0]["parameters"]) == 2 and maxval_calls[0]["parameters"][1] in ["dim=2","2"])) and first_param_info[3] in [[global_subdomain_range_x], [global_subdomain_range_x,"3"]]:
                    lines[line_index] = self.replace_func_call(line,maxval_calls[0],f"max({maxval_calls[0]['parameters'][0]})")
                else:
                    print("first param info",first_param_info)
                    print("no case maxval")
                    pexit(line)
        file = open("res.txt","w")
        for line in lines:
          file.write(f"{line}\n")
        file.close()
        #replace all calls to tiny with predefined float 
        for line_index,line in enumerate(lines):
            for call in [x for x in self.get_function_calls_in_line(line,variables) if x["function_name"] == "tiny"]:
                type = self.get_param_info((call["parameters"][0],False), local_variables,self.static_variables)[2]
                if type == "real":
                    line = self.replace_func_call(line,call,"AC_tiny_val")
                    lines[line_index] = line
                else:
                    print("what to do with this tiny type?",type)
                    print(type)
                    exit()
        #alog -> log
        for line_index,line in enumerate(lines):
                alog_calls =  [x for x in self.get_function_calls_in_line(line,variables) if x["function_name"] == "alog"]
                while(len(alog_calls)>0):
                    call = alog_calls[0]
                    line = self.replace_func_call(line,call,f"log({','.join(call['parameters'])})")
                    lines[line_index] = line
                    alog_calls =  [x for x in self.get_function_calls_in_line(line,variables) if x["function_name"] == "alog"]
        #for calls [real] just return the param
        for line_index,line in enumerate(lines):
                real_calls = [x for x in self.get_function_calls_in_line(line,variables) if x["function_name"] == "real"]
                while(len(real_calls)>0):
                    call = real_calls[0]
                    if len(call["parameters"]) == 1:
                        line = self.replace_func_call(line,call,call['parameters'][0])
                        lines[line_index] = line
                        real_calls = [x for x in self.get_function_calls_in_line(line,variables) if x["function_name"] == "real"]
                    else:
                        print("multiple params in real")
                        pexit(line)



        #move written profiles to local_vars since we know their values
        lines = self.transform_pencils(lines,all_inlined_lines,local_variables)
        variables = merge_dictionaries(local_variables,self.static_variables)
        file = open("res-before.txt","w")
        for line in lines:
            file.write(f"{line}\n")
        file.close
        lines = self.elim_unnecessary_writes_and_calls(lines,local_variables,variables)
        #remove writes to fname,lfirstpoint and mass_per_proc
        remove_indexes = []
        writes = self.get_writes(lines,False)
        for x in [x for x in writes if x["variable"] in ["fname","mass_per_proc","lfirstpoint"]]:
            remove_indexes.append(x["line_num"])
        lines = [x[1] for x in enumerate(lines) if x[0] not in remove_indexes]
        lines = self.unroll_constant_loops(lines,local_variables)
        #transform spreads into do loops
        lines = self.transform_spreads(lines,local_variables,variables)
        self.known_values = {}
        lines = self.eliminate_while(lines)

        writes = self.get_writes(lines)
        self.try_to_deduce_if_params(lines,writes,local_variables)
        file = open("res-inlined-profiles.txt","w")
        for line in lines:
            file.write(f"{line}\n")
        file.close()
        print("replace 1d vecs with 3 1d arrays")
        vectors_to_try_to_replace = []
        for var in local_variables:
            vectors_to_try_to_replace.append(var)
        print("dline_1 is in self.static","dline_1" in self.static_variables)
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

                    

        #TODO: do this better
        lines = [line.replace("+(1-1)","+0").replace("+1-1","+0") for line in lines]
        self.known_values = {}
        lines = self.eliminate_while(lines)
        # lines = self.inline_0d_writes(lines,local_variables)
        #rewrite some ranges in AcVectors and AcMatrices
        variables = merge_dictionaries(local_variables,self.static_variables)
        lines = self.evaluate_leftover_pencils_as_true(lines,local_variables)
        #any -> normal if 
        variables = merge_dictionaries(local_variables,self.static_variables)
        for line_index,line in enumerate(lines):
                any_calls=  [x for x in self.get_function_calls_in_line(line,variables) if x["function_name"] == "any"]
                while(len(any_calls)>0):
                    call = any_calls[0]
                    assert(len(call["parameters"]) == 1)
                    if "/=" in call["parameters"][0]:
                      comp_symbol = "/="
                    elif "==" in call["parameters"][0]:
                      comp_symbol = "=="
                    elif "<" in call["parameters"][0]:
                      comp_symbol = "<"
                    elif ">" in call["parameters"][0]:
                      comp_symbol = ">"
                    lower,upper = [part.strip() for part in call["parameters"][0].split(comp_symbol)]
                    if lower in variables and variables[lower]["dims"] == ["3"]:
                      lines[line_index] = self.replace_func_call(line,call," .or. ".join([f"{lower}({index}) {comp_symbol} {upper}" for index in ["1","2","3"]]))
                    else:
                      pexit("what to do?")
                    any_calls=  [x for x in self.get_function_calls_in_line(lines[line_index],variables) if x["function_name"] == "any"]
        lines = self.eliminate_while(lines)
        lines = self.elim_empty_branches(lines,local_variables)
        lines = self.elim_empty_dos(lines,local_variables)
        local_variables = {parameter:v for parameter,v in self.get_variables(lines, {},"",True).items() }
        lines = self.unroll_constant_loops(lines,local_variables)
        lines = self.unroll_ranges(lines,local_variables)
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
        for i,line in enumerate(lines):
            res = self.transform_line(i,lines,local_variables,loop_indexes,symbol_table,initialization_lines,orig_params, transform_func,vectors_to_replace,writes)
            lines[i] = res
        file = open("res-transform.txt","w")
        for line in lines:
            file.write(f"{line}\n")
        file.close()
        lines = [line.replace("iuu__mod__cdata","F_UX") for line in lines]
        lines = [line.replace("F_UX","F_UU.x").replace("F_UY","F_UU.y").replace("F_UZ","F_UU.z") for line in lines]

        # for f_index in ["uu",""]

        lines = [line.replace(".false.","false") for line in lines]
        lines = [line.replace(".true.","true") for line in lines]
        lines = [line.replace(".and."," && ") for line in lines]
        lines = [line.replace(".or."," || ") for line in lines]
        lines = [line.replace(".not.","!") for line in lines]
        lines = [line.replace("loptest(false)","false") for line in lines]
        lines = [line.replace("loptest(true)","true") for line in lines]
        lines = [line.replace("/=","!=") for line in lines]

        lines = [replace_exp(line) for line in lines]
        file = open("res-replace_exp.txt","w")
        for line in lines:
            file.write(f"{line}\n")
        file.close()
        print("Check replace exp file")
 
        #for testing vtxbuf_ss -> vtxbuf_entropy
        lines = [line.replace("VTXBUF_SS","VTXBUF_ENTROPY") for line in lines]
        #close empty defaults
        for i,line in enumerate(lines):
            if "default:" in line and i<len(lines)-1 and "}" in lines[i+1]:
                lines[i] = line + ";"
        print("Check transform file")
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

        replace_dict = {
          global_subdomain_range_x:"AC_nx",
          global_subdomain_range_y:"AC_ny",
          global_subdomain_range_z:"AC_nz",
          global_subdomain_range_with_halos_x:"AC_nx+2*NGHOST",
          global_subdomain_range_with_halos_y:"AC_ny+2*NGHOST)",
          global_subdomain_range_with_halos_z:"AC_nz+2*NGHOST)",
          # global_subdomain_range_x_lower:"AC_nx_min",
          global_subdomain_range_x_upper:"AC_nx_max",
          global_subdomain_range_y_lower:"AC_ny_min",
          global_subdomain_range_y_upper:"AC_ny_max",
          global_subdomain_range_z_lower:"AC_nz_min",
          global_subdomain_range_z_upper:"AC_nz_max",
        }
        for x in replace_dict:
          lines = [line.replace(x,replace_dict[x]) for line in lines]





        orig_lines = lines.copy()
        for i,line in enumerate(lines):
            static_variables_in_line= unique_list([var for var in get_used_variables_from_line(line) if var.lower() in self.static_variables])
            res_line = line
            for var in static_variables_in_line:
                ##all uppercase means that it is a profile
                #pow is a function in this context not a parameter
                if var.lower() not in ["bot","top","nghost","pow"] and var.lower() not in local_variables and var.upper() != var:
                  if self.offload_type == "boundcond":
                    res_line = replace_variable(res_line, var, f"DCONST(AC_{var.lower()})")
                  elif self.offload_type == "stencil":
                    res_line = replace_variable(res_line, var, f"AC_{var.lower()}")
            lines[i] = res_line

        if self.test_to_c:
            idx_line = "const int3 idx = {i, j, k};"
            lines = [line.replace("vba.in","mesh_test.vertex_buffer") for line in lines]
            res_lines = [idx_line] + lines
            return res_lines
        static_vars = []
        rest_params = ""
        for param in orig_params:
            rest_params = rest_params + "," + translate_to_c(local_variables[param]["type"]) + " " + param
        lines[1] = lines[1].replace("[rest_params_here]",rest_params)
        lines = [line.replace("nghost","NGHOST") for line in lines]
        file = open("res-3.txt","w")
        for line in lines:
            file.write(f"{line}\n")
        file.close()
        for i,line in enumerate(lines):
            func_calls = self.get_function_calls_in_line(line,local_variables)
            if i>1:
                for func_call in func_calls:
                    #Acvector sums are okay
                    if func_call["function_name"].lower() not in ["derx","dery","derz","derxx","deryy","derzz","der6x","der6y","der6z","der6x_upwd","der6y_upwd","der6z_upwd","sum","&&", "||","all","col","row","sqrt","abs","sinh","cosh","tanh","min","max","der","der2","der3","der4","der5","der6","pow","DEVICE_VTXBUF_IDX".lower(),"DCONST".lower(),"value","vecvalue","exp","log","if","else","for","sin","cos"] and not "[" in func_call["function_name"]:
                        # if func_call["function_name"] == "sum" and self.get_param_info((func_call["parameters"][0],False),local_variables,self.static_variables)[3] == [global_subdomain_range_x,"3"]:
                        #     pass
                        if func_call["function_name"].lower() not in der_funcs:
                            print("STILL FUNC CALLS in line:",line,i)
                            print(func_call)
                            exit()
        vertexIdx_line = "const int3 vertexIdx = (int3){\nthreadIdx.x + blockIdx.x * blockDim.x,\nthreadIdx.y + blockIdx.y * blockDim.y,\nthreadIdx.z + blockIdx.z * blockDim.z,\n};\n"
        check_line = "if (vertexIdx.x >= dims.x || vertexIdx.y >= dims.y || vertexIdx.z >= dims.z) {\nreturn;\n}\n"
        idx_line = "const int3 idx = {vertexIdx.x, vertexIdx.y, vertexIdx.z};"
        dx_lines = [
          "real3 DF_UU",
          "DF_UU.x = 0.0",
          "DF_UU.y = 0.0",
          "DF_UU.z = 0.0",
          "DF_LNRHO = 0.0",
          "DF_SS = 0.0"
        ]
        #for now simply write DX out, normally would write out RHS3 substep
        write_fields_lines = [
          "write_vector(F_UU,DF_UU)",
          "write(F_LNRHO,DF_LNRHO)",
          "write(F_SS,DF_SS)"
        ]
        declarations_line = ""
        for var in unique_list(initialization_lines):
            declarations_line = declarations_line + translate_to_c(local_variables[var.lower()]["type"]) + " " + var + ";\n"
        if self.offload_type == "boundcond":
            res_lines = lines[:3] + [declarations_line, vertexIdx_line, check_line, idx_line] +lines[3:]
            lines = res_lines
        elif self.offload_type == "stencil":
            res_lines = lines[:1] + dx_lines + lines[1:-1] + write_fields_lines + [lines[-1]]
            lines = res_lines
        return lines
    def deduce_value(self,variable,writes,local_variables,analyse_lines,take_last_write_as_output=False):
        var_writes = [write for write in writes if write["variable"] == variable and write["line"].split(" ")[0].strip() != "do"]
        if variable in local_variables:
            src = local_variables
        else:
            src = self.static_variables
        if len(var_writes) > 1:
            if all([write["value"] == var_writes[0]["value"] for write in var_writes]):
                write = var_writes[0]
                if variable in local_variables:
                    local_variables[variable]["value"] = write["value"]
                    self.known_values[variable] = write["value"] 
                else:
                    if analyse_lines[write["line_num"]][1] == 0:
                        self.static_variables[variable]["value"] = write["value"]
                        self.known_values[variable] = write["value"] 
            #all writes are on the main branch then the value will be the last write after this subroutine
            elif take_last_write_as_output and all([analyse_lines[x["line_num"]][1] == 0 for x in var_writes]) and variable not in var_writes[-1]["value"]:
                src[variable]["value"] = var_writes[-1]["value"]
                self.known_values[variable] = var_writes[-1]["value"]



        elif len(var_writes) == 1:
            write = var_writes[0]
            if variable in local_variables:
                local_variables[variable]["value"] = write["value"]
                self.known_values[variable] = write["value"]
            else:
                if analyse_lines[write["line_num"]][1] == 0:
                    self.static_variables[variable]["value"] = write["value"]
                    self.known_values[variable] = write["value"] 
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
        print("offload type",self.offload_type)
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
                        print("Can't handle nested global loops")
                        exit()
                    in_global_loop = True
                    number_of_dos = 1
                    number_of_enddos = 0
                    lines_in_loop = []
                    lines_in_loop.append((line,i))
            elif self.offload_type == "boundcond" and "do" in line and "1,my" in line:
                write = self.get_writes_from_line(line,local_variables)[0]
                iterators.append((write["variable"],"1",global_subdomain_range_with_halos_y,1))
                if in_global_loop:
                    print("Can't handle nested global loops")
                    exit()
                in_global_loop = True
                number_of_dos = 1
                number_of_enddos = 0
                lines_in_loop = []
                lines_in_loop.append((line,i))
        return (global_loop_lines,iterators)
    def get_default_flags_from_file(self,filename,dst):
        module = self.get_own_module(filename)
        for x in self.file_info[filename]["variables"]:
            if x[0] == "l" and self.file_info[filename]["variables"][x]["type"] == "logical" and "value" in self.file_info[filename]["variables"][x]:
                val = self.file_info[filename]["variables"][x]["value"]
                if val in [".false.",".true."]:
                  if "__mod__" not in x:
                    print(filename)
                    print(module)
                    dst[self.rename_dict[module][x]] = val
                  else:
                    dst[x] = val


    def get_flags_from_initialization_func(self,subroutine_name, filename):
        print("Init func",subroutine_name)
        # print(self.flag_mappings["lgradu_as_aux__mod__hydro"])

        orig_lines = self.get_subroutine_lines(subroutine_name,filename)

        mod_default_mappings = {}
        self.get_default_flags_from_file(filename,mod_default_mappings)
        local_variables = {parameter:v for parameter,v in self.get_variables(orig_lines, {},filename,True).items() }
        orig_lines = self.rename_lines_to_internal_names(orig_lines,local_variables,filename)
        local_variables = {parameter:v for parameter,v in self.get_variables(orig_lines, {},filename,True).items() }

        mod = self.get_own_module(filename)
        if mod not in self.shared_flags_accessed:
          self.shared_flags_accessed[mod] = []
          self.shared_flags_given[mod] = []

        analyse_lines = self.get_analyse_lines(orig_lines,local_variables)
        for call in self.get_function_calls(orig_lines,local_variables,False):
          #check if in main branch
          if analyse_lines[call["line_num"]][1] == 0:
            if call["function_name"] == "get_shared_variable" and len(call["parameters"]) >= 2:
              flag = call["parameters"][1]
              if (flag[0] == "l" and flag in self.file_info[filename]["variables"] and self.file_info[filename]["variables"][flag]["type"] == "logical") or remove_mod(flag) in ["beta_glnrho_scaled"]:
                self.shared_flags_accessed[mod].append(flag)
            if call["function_name"] == "put_shared_variable" and len(call["parameters"]) >= 2:
              flag = call["parameters"][1]
              if (flag[0] == "l" and flag in self.file_info[filename]["variables"] and self.file_info[filename]["variables"][flag]["type"] == "logical") or remove_mod(flag) in ["beta_glnrho_scaled"]:
                self.shared_flags_given[mod].append(flag)
        new_lines = self.eliminate_while(orig_lines,True,True)
        new_lines = self.unroll_constant_loops(new_lines,{})
        self.known_values = {}
        new_lines = self.eliminate_while(new_lines,True,True)
        local_variables = {parameter:v for parameter,v in self.get_variables(new_lines, {},filename,True).items() }

        #variables that can be deduced after elimination
        self.known_values = {}
        self.try_to_deduce_if_params(new_lines,self.get_writes(new_lines,False), local_variables,True)
        for x in [x for x in self.known_values if x[0] == "l" and x in self.static_variables and self.static_variables[x]["type"] == "logical"]:
            self.flag_mappings[x] = self.known_values[x]
        new_lines = self.eliminate_while(new_lines,True,True)
        self.known_values = {}
        self.try_to_deduce_if_params(new_lines,self.get_writes(new_lines,False), local_variables,True)
        for x in [x for x in self.known_values if x[0] == "l" and x in self.static_variables and self.static_variables[x]["type"] == "logical"]:
            self.flag_mappings[x] = self.known_values[x]

        #if eliminated all writes then to flag than we take the default val
        # writes = self.get_writes(new_lines)
        # for var in [x for x in mod_default_mappings if x not in self.flag_mappings]:
        #     if len([x for x in writes if x["variable"] == var]) == 0:
        #         self.flag_mappings[var] = mod_default_mappings[var]
        local_variables = {parameter:v for parameter,v in self.get_variables(new_lines, {},filename,True).items() }
        analyse_lines = self.get_analyse_lines(new_lines,local_variables)
        for call in self.get_function_calls(new_lines,local_variables,False):
          #check if in main branch
          if analyse_lines[call["line_num"]][1] == 0:
            if call["function_name"] == "select_eos_variable":
              self.select_eos_variable_calls.append(call)
            if call["function_name"] in farray_register_funcs:
              self.farray_register_calls.append(call)
            if call["function_name"] == "get_shared_variable" and len(call["parameters"]) >= 2:
              flag = call["parameters"][1]
              if (flag[0] == "l" and flag in self.file_info[filename]["variables"] and self.file_info[filename]["variables"][flag]["type"] == "logical") or remove_mod(flag) in ["beta_glnrho_scaled"]:
                self.shared_flags_accessed[mod].append(flag)
            if call["function_name"] == "put_shared_variable" and len(call["parameters"]) >= 2:
              flag = call["parameters"][1]
              if (flag[0] == "l" and flag in self.file_info[filename]["variables"] and self.file_info[filename]["variables"][flag]["type"] == "logical") or remove_mod(flag) in ["beta_glnrho_scaled"]:
                self.shared_flags_given[mod].append(flag)

        # if subroutine_name == "initialize_density":
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
                print("missed this one",line)
                exit()
            if "if" in line or line=="else":
                case_num += 1
            analyse_lines.append((line,nest_num, if_num,line_index,case_num))
            if line in ["endif","end if"]:
                nest_num -= 1
                if_nums.pop()
                print(line_index)
                print(line)
                assert(len(if_nums)>0)
                if_num = if_nums[-1]
        assert(len(analyse_lines) == len(lines))
        return analyse_lines
    def has_no_ifs(self,lines):
        return all([x[1] == 0 for x in self.get_analyse_lines(lines,local_variables)])
    def eliminate_dead_branches_once(self,lines,local_variables):
        remove_indexes = []
        done = False

        # for line_index,line in enumerate(lines):
        #     if "if" in line and "then" in line and "elif" not in line and "else" not in line:
        #         if_num = max_if_num + 1
        #         if_nums.append(if_num)
        #         max_if_num = max(if_num, if_num)
        #         nest_num += 1
        #     analyse_lines.append((line,nest_num, if_num,line_index))
        #     if "endif" in line or "end if" in line:
        #         nest_num -= 1
        #         if_nums.pop()
        #         if_num = if_nums[-1]

        analyse_lines = self.get_analyse_lines(lines,local_variables)
        max_if_num = max([x[2] for x in analyse_lines])
        for if_num in range(1,max_if_num+1):
            if_lines = [line for line in analyse_lines if line[2] == if_num and line[1] > 0]
            choices = []
            for line in if_lines:
                if ("if" in line[0] and ("then" in line[0] or "end" in line[0] or "else" in line[0])) or line[0] == "else":
                    choices.append(line)
            # print("CHOICES")
            # print(choices)
            possibilities = []
            found_true = False
            res_index = 0
            for choice_index, choice in enumerate(choices):
                line = choice[0]
                if "if" in line and ".false." in line and ".true." not in line and ".and." not in line and ".or." not in line:
                    pass
                elif "if" in line and ".true." in line and ".false." not in line and ".and." not in line and ".or." not in line:
                    found_true = True
                    possibilities = [choice]
                    res_index = choice_index
                elif not found_true and "end" not in line:
                    possibilities.append(choice)
                    res_index = choice_index
            # print("POSSIBILITES")
            # print(possibilities)
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
                print(choices)
                keep_ending_index = choices[res_index+1][3]-1
                for index in range(starting_index, ending_index+1):
                    if not (index >= keep_starting_index and index<=keep_ending_index):
                        remove_indexes.append(index)
        return [x[1] for x in enumerate(lines) if x[0] not in remove_indexes]


        #choose correct single line ifs
        # remove_indexes = []
        # for line_index, line in enumerate(new_lines):
        #     if "if" in line and "else" not in line and "then" not in line and ".false." in line and ".true." not in line and ".and." not in line and ".or." not in line:
        #         remove_indexes.append(line_index)
        #     elif "if" in line and "else" not in line and "then" not in line and ".true." in line and ".else." not in line and ".and." not in line and ".or." not in line:
        #         new_lines[line_index] = new_lines[line_index].replace("if (.true.)","").replace("if(.true.)","")
        # lines = [x[1] for x in enumerate(new_lines) if x[0] not in remove_indexes]
    
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
        print("EXPAND FUNC CALL",call_to_expand["function_name"],"IN",subroutine_name)
        # local_variables = {parameter:v for parameter,v in self.get_variables(subroutine_lines, {},filename).items() }
        replaced_function_call_lines,is_function,res_type = self.expand_function_call(subroutine_lines, subroutine_name, filename, call_to_expand,local_variables,global_init_lines,subs_not_to_inline,elim_lines,local_variables)
        for line in replaced_function_call_lines:
          if ",," in line:
            print("wrong here",line)
            exit()
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
        print("len(sub lines) for ",call_to_expand["function_name"],len(subroutine_lines))
        for line_num in range(len(subroutine_lines)):
          if not has_replaced_call:
            line = subroutine_lines[line_num]
            # func_calls = [call for call in self.get_function_calls_in_line(subroutine_lines[line_num][0],local_variables) if call["function_name"] == call_to_expand["function_name"]]
            # if len(func_calls)>0:
            if line == call_to_expand["line"] or f"{call_to_expand['function_name']}(" in line or f"call {call_to_expand['function_name']}" in line and "print*" not in line:
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
                            self.struct_table["pencil_case"][field_name] = {"type": "real", "dims": field_dims, "origin": [file]}
        for field in [x for x in self.struct_table["pencil_case"]]:
          if "." in field:
            del self.struct_table["pencil_case"][field]

    def expand_function_call(self,lines,subroutine_name, filename, call_to_expand,variables_in_scope,global_init_lines,subs_not_to_inline,elim_lines,local_variables):

        function_to_expand = call_to_expand["function_name"]
        print(f"Expanding {function_to_expand} in {subroutine_name} in file {filename} inline_num {self.inline_num}")
        print("call:", call_to_expand)
        mpi_calls = ["mpi_send","mpi_barrier","mpi_finalize","mpi_wait"]
        if function_to_expand in mpi_calls:
            print("MPI call not safe :(")
            exit()
        file_paths = self.find_subroutine_files(function_to_expand)
        #if file_paths is [] then the function is only present in the current file and not public
        if file_paths == []:
            file_paths = [filename]
        modules = self.get_used_modules(lines)
        # self.parse_file_for_static_variables(filename)
        # local_variables = {parameter:v for parameter,v in self.get_variables(lines, {},filename).items() }
        # function_calls = self.get_function_calls(lines, local_variables)
        for line in lines:
            ##Old way to do it
            # function_calls = self.get_function_calls_in_line(line[0],local_variables)
            # for function_call in function_calls:

                if line == call_to_expand["line"] or f"{call_to_expand['function_name']}(" in line:
                    if not(line == call_to_expand["line"] or f"{call_to_expand['function_name']}(" in line):
                        print("WRONG!")
                        exit()
                    print(call_to_expand)
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
    def normalize_if_calls(self,lines,local_variables):
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

    def inline_all_function_calls(self,filename,subroutine_name,new_lines,subs_not_to_inline=None,elim_lines = True):
        self.known_values = {}
        if subs_not_to_inline == None:
          subs_not_to_inline = self.ignored_subroutines
        writes = self.get_writes(new_lines,False)
        local_variables = {parameter:v for parameter,v in self.get_variables(new_lines, {},filename,True).items() }
        new_lines = self.rename_lines_to_internal_names(new_lines,local_variables,filename)
        local_variables = {parameter:v for parameter,v in self.get_variables(new_lines, {},filename,True).items() }
        
        init_lines = [line for line in new_lines if is_init_line(line)]
        global_init_lines = init_lines
        global_init_lines = unique_list(global_init_lines)
        #normalize .eq. => == and .ne. => /=
        #.lt. => < and .gt. => >
        new_lines = [x.replace(".eq.","==").replace(".ne.","/=").replace(".lt.","<").replace(".gt.",">") for x in new_lines]
        if elim_lines:
            new_lines = self.eliminate_while(new_lines)
        for line in new_lines:
          print(line)
        local_variables = {parameter:v for parameter,v in self.get_variables(new_lines, {},filename,True).items() }
        for var in local_variables:
          if local_variables[var]["saved_variable"] and not local_variables[var]["parameter"]:
            if subroutine_name not in ["set_from_slice_x", "set_from_slice_y", "set_from_slice_z","bc_aa_pot_field_extrapol","div","get_reaction_rate"] and self.get_own_module(filename) not in ["special","boundcond"]:
              print("saved variable",var)
              print(local_variables[var])
              print("in:",subroutine_name,filename)
              print("abort")
              exit()
        func_calls_to_replace = [call for call in self.get_function_calls(new_lines,local_variables) if call["function_name"] != subroutine_name and call["function_name"] not in subs_not_to_inline]
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
        for line in new_lines:
          if "where" in line and "(" in line:
            where_lines.append(line)
          if line == "endwhere":
            endwhere_lines.append(line)
        if len(where_lines) != len(endwhere_lines): 
          print("WRONG")
          print(filename,subroutine_name)
          print(where_lines)
          print(endwhere_lines)
          for line in new_lines:
            print(line)
          assert(False)
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

def main():
    # line = 'forall( j = iux__mod__cdata:iuz__mod__cdata) df(1+3:l2__mod__cparam,m__mod__cdata,n__mod__cdata,j_92) = df(1+3:l2__mod__cparam,m__mod__cdata,n__mod__cdata,j_92) - p%uu(:,j_92-iuu__mod__cdata+1) * tmp_92'
    # print(get_var_name_segments(line,{"j":None,"j_92":None}))
    # exit()
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
    # print(parser.evaluate_indexes("1+3:l2__mod__cdata"))
    # exit()
    new_lines = ["if(idiag_umax/=0) then","max_mn_name(p%u)","endif"]
    # idiag_vars = parser.get_idiag_vars(new_lines)
    # print("idiag vars\n")
    # print(idiag_vars)
    # exit()
    # print(parser.include_diagnostics)
    local_variables = {
      "fvisc":{
        "type": "real",
        "dims": [global_subdomain_range_x,"3"],
        "saved_variable": False
      }
    }
    # lines = parser.unroll_ranges(lines,local_variables)
    # print("\nres\n")
    # print(lines)
    # exit()
    # lines = ["do i_8_53_54_55_83=1+1,3", 
    #   "ac_transformed_pencil_sij(:,i_8_53_54_55_83,1)=.5*(ac_transformed_pencil_uij(:,i_8_53_54_55_83,1)+ac_transformed_pencil_uij(:,1,i_8_53_54_55_83))",
    #   "ac_transformed_pencil_sij(:,1,i_8_53_54_55_83)=ac_transformed_pencil_sij(:,i_8_53_54_55_83,1)",
    #   "enddo"
    # ]
    # res = parser.unroll_constant_loops(lines,{})
    # print("res")
    # for line in res:
    #   print(line)
    # exit()
    # line = "if(lpencil__mod__cparam(i_sij2)) then"
    # print(parser.evaluate_leftover_pencils_as_true([line],{}))
    # exit()
    # if len(if_calls) == 1 and len(if_calls[0]["parameters"]) == 1 and "lpencil__mod__cparam" in if_calls[0]["parameters"][0]:


    # exit()
    # test_lines = ["if ( lfirst.and.ldt.and.(lcdt_tauf.or.ldiagnos.and.idiag_dtF/=0) .or ldiagnos.and.idiag_taufmin/=0 ) then", "where (abs(p%uu)>1)","uu1=1./p%uu","elsewhere","uu1=1.","endwhere","do j=1,3","ftot=abs(df(l1:l2,m,n,iux+j-1)*uu1(:,j))","if (ldt.and.lcdt_tauf) dt1_max=max(dt1_max,ftot/cdtf)","if (ldiagnos.and.(idiag_dtF/=0.or.idiag_taufmin/=0)) Fmax=max(Fmax,ftot)","enddo","endif"]
    # test_lines = parser.eliminate_while(test_lines)
    # print(test_lines)
    # exit()
    # line = "if(p%uu(:,j_21_22_79_83_86)>=0) then"
    # print(parser.get_function_calls([line],{}))
    # exit()
    #PENCIL SPECIFIC
    parser.populate_pencil_case()
    print("here." in parser.struct_table["pencil_case"])
    for symbol in ["nxgrid,nygrid,nzgrid","nprocx","nprocy","nprocz"]:
      parser.static_variables[symbol] = {
        "type": "integer",
        "saved_variable": False,
        "parameter": False,
        "dims": []
      }
    lines = parser.get_lines(f"{parser.sample_dir}/src/cparam.inc")
    #used to load sample specific values
    parser.get_variables(lines,parser.static_variables,f"{parser.directory}/cparam.f90",False,True)
    parser.update_used_modules()
    parser.update_static_vars()

    #for Pencil Code get variables added by scripts
    parser.get_variables(mk_param_lines,parser.static_variables,f"{parser.directory}/cparam.f90",False,True)




    variables = {}
    header = f"{parser.directory}/mpicomm.h"
    # cparam_pencils.inc might not be included
    parser.load_static_variables(f"{parser.directory}/cparam.f90")
    parser.struct_table["pencil_case"]["pnu"] = {
      "type": "real",
      "dims": [global_subdomain_range_x]
    }
    # parser.static_variables["lpencil"] = {"type": "logical", "dims": ["npencils"], "threadprivate": False,"parameter": False}
    # parser.static_variables["nghost"] = {"type": "integer", "dims": [],"threadprivate": False,"parameter": True, "value": "3"}

    if config["test"] or config["offload"]:
        #get flag mappings from cparam.inc
        lines = parser.get_lines(f"{parser.sample_dir}/src/cparam.inc")
        writes = parser.get_writes(lines,False)
        for write in writes:
            if write["variable"][0] == "l" and (write["value"] == ".true." or write["value"] == ".false."):
                parser.flag_mappings[write["variable"]] = write["value"]
            elif write["variable"] == "nghost":
                parser.flag_mappings[write["variable"]] = write["value"]

        #currently only done for npscalar
        lines = parser.get_lines(f"{parser.sample_dir}/src/cparam.inc")
        res_lines = []
        for x in lines:
            res_lines.extend([part.strip() for part in x.split(",")])
        lines = res_lines
        writes = parser.get_writes(lines,False)
        for write in writes:
            if write["variable"] not in parser.default_mappings and write["variable"] in ["npscalar"]:
                parser.default_mappings[write["variable"]] = write["value"]


        #get flags from params.log
        parser.get_flags_from_lines(parser.get_lines(f"{parser.sample_dir}/data/params.log"))
        print("\nMappings\n")
        for x in parser.flag_mappings:
            print(x)
        # parser.get_flags_from_lines(parser.get_lines(f"{parser.sample_dir}/run.in"))

        for map_param in parser.default_mappings:
            if map_param not in parser.flag_mappings:
                parser.flag_mappings[map_param] = parser.default_mappings[map_param]
        parser.ignored_subroutines.append("get_shared_variable")
        parser.ignored_subroutines.append("write_xprof")
        parser.ignored_subroutines.append("write_yprof")
        parser.ignored_subroutines.append("write_zprof")
        parser.ignored_subroutines.append("select_eos_variable")
        parser.ignored_subroutines.append("find_by_name")
        parser.ignored_subroutines.append("farray_index_append")
        parser.ignored_subroutines.append("save_analysis_info")
        parser.ignored_subroutines.append("information")
        #for testing
        # parser.flag_mappings["topbot"] = "top"
        # parser.flag_mappings["lone_sided"] = ".false."
        # parser.flag_mappings["loptest(lone_sided)"] = ".false."
        for x in [y for y in parser.flag_mappings]:
            parser.flag_mappings[x] = parser.flag_mappings[x].strip()
            #remove arrays and in place but each val
            if any([char in parser.flag_mappings[x] for char in "*,"]):
                if "'" in parser.flag_mappings[x] or '"':
                    arr = parse_input_param(parser.flag_mappings[x])
                    for i in [x[0] for x in enumerate(arr)]:
                        parser.flag_mappings[f"{x}({i+1})"] = arr[i]
                del parser.flag_mappings[x]
        for index in ["1","2","3"]:
          print(parser.flag_mappings[f"grads0_imposed__mod__energy({index})"])
          if eval(parser.flag_mappings[f"beta_glnrho_global__mod__density({index})"]) == 0:
            parser.flag_mappings[f"beta_glnrho_scaled__mod__density({index})"] = "0."
          print(parser.flag_mappings[f"beta_glnrho_scaled__mod__density({index})"])
        for index in ["1","2","3"]:
          if parser.flag_mappings[f"grid_func__mod__cdata({index})"] == '"linear"':
           parser.flag_mappings[f"lequidist__mod__cdata({index})"] = ".true."
        parser.get_default_flags_from_file(f"{parser.directory}/cdata.f90",parser.default_mappings)
        for mod in ["eos","viscosity","hydro","gravity","density","energy"]:
          parser.get_default_flags_from_file(f"{parser.directory}/{parser.chosen_modules[get_mod_from_physics_name(mod)]}.f90",parser.default_mappings)
        if parser.chosen_modules["forcing"] == "noforcing":
          parser.flag_mappings["lforcing_cont__mod__cdata"] = ".false."
        #TODO: have to check which indexes are registered with farray_registed_pde
        #For now we simply know what ioo is 0
        parser.flag_mappings["ioo__mod__cdata"] = "0"
        parser.get_flags_from_initialization_func("set_coorsys_dimmask",f"{parser.directory}/grid.f90")
        for mod in ["eos","viscosity","hydro","gravity","density","energy"]:
          parser.get_flags_from_initialization_func(f"register_{mod}",f"{parser.directory}/{parser.chosen_modules[get_mod_from_physics_name(mod)]}.f90")
        for index in ["iphiuu__mod__cdata"]:
          if len([call for call in parser.farray_register_calls if call["parameters"][1] == index]) == 0:
            parser.flag_mappings[index] = "0"
        for mod in ["eos","gravity","density","hydro","energy","viscosity"]:
          parser.get_flags_from_initialization_func(f"initialize_{mod}",f"{parser.directory}/{parser.chosen_modules[get_mod_from_physics_name(mod)]}.f90")

        for mod in parser.shared_flags_accessed:
          for flag in parser.shared_flags_accessed[mod]:
            cleaned_flag = remove_mod(flag)
            possible_modules = []
            for mod_second in parser.shared_flags_given:
              if get_mod_name(cleaned_flag,mod_second) in parser.shared_flags_given[mod_second]:
                possible_modules.append(mod_second)
            assert(len(possible_modules) == 1)
            if get_mod_name(cleaned_flag,possible_modules[0]) in parser.flag_mappings:
              parser.flag_mappings[get_mod_name(cleaned_flag,mod)] = parser.flag_mappings[get_mod_name(cleaned_flag,possible_modules[0])]
            for i in ["1","2","3"]:
              if f"{get_mod_name(cleaned_flag,possible_modules[0])}({i})" in parser.flag_mappings:
                parser.flag_mappings[f"{get_mod_name(cleaned_flag,mod)}({i})"] = parser.flag_mappings[f"{get_mod_name(cleaned_flag,possible_modules[0])}({i})"]
        print(parser.shared_flags_accessed["density"])
        print(parser.shared_flags_given["hydro"])
        print(parser.flag_mappings["lconservative__mod__hydro"])
        print(parser.flag_mappings["lconservative__mod__density"])
        print(parser.flag_mappings["lheatc_kprof__mod__energy"])
        # exit()
        for i in ["1","2","3"]:
          print(parser.flag_mappings[f"beta_glnrho_scaled__mod__energy({i})"])
        print(parser.flag_mappings[f"lweno_transport__mod__cdata"])
        if parser.chosen_modules["density"] == "density":
          parser.flag_mappings["lupdate_mass_source__mod__density"] =  "lmass_source__mod__density .and. t>=tstart_mass_source .and. (tstop_mass_source==-1.0 .or. t<=tstop_mass_source)"
        assert(len(parser.select_eos_variable_calls) == 2)
        lnrho_val = 2**0
        rho_val = 2**1
        ss_val = 2**2
        lntt_val = 2**3
        tt_val = 2**4
        cs2_val = 2**5
        pp_val = 2**6
        eth_val = 2**7
        select_val = 0
        for i, call in enumerate(parser.select_eos_variable_calls):
          if call["parameters"][0] == "'ss'":
            select_val += ss_val
          elif call["parameters"][0] == "'lnrho'":
            select_val += lnrho_val
          else:
            pexit(f"add eos var value: {call['parameters'][0]}")
          parser.flag_mappings[f"ieosvar{i+1}__mod__equationofstate"] = call["parameters"][1]
        if select_val == lnrho_val + ss_val:
          parser.flag_mappings["ieosvars__mod__equationofstate"] = parser.static_variables["ilnrho_ss__mod__equationofstate"]["value"]
        elif select_val == rho_val + ss_val:
          parser.flag_mappings["ieosvars__mod__equationofstate"] = parser.static_variables["irho_ss__mod__equationofstate"]["value"]
        elif select_val == lnrho_val + lntt_val:
          parser.flag_mappings["ieosvars__mod__equationofstate"] = parser.static_variables["ilnrho_lntt__mod__equationofstate"]["value"]
        else:
            pexit(f"add eos select val mapping: {parser.select_eos_variable_calls}")

        print("ieos1",parser.flag_mappings["ieosvar1__mod__equationofstate"])
        print("ieos2",parser.flag_mappings["ieosvar2__mod__equationofstate"])
        print("\n flag mappings: \n")
        for flag in parser.flag_mappings:
            print(flag,parser.flag_mappings[flag])
        # print("lread",parser.flag_mappings["lread_hcond__mod__energy"])
        if parser.flag_mappings["lgravz__mod__cdata"] == ".true.":
          parser.static_variables["hcond_prof__mod__energy"]["dims"] = [global_subdomain_range_z]
          parser.static_variables["dlnhcond_prof__mod__energy"]["dims"] = [global_subdomain_range_z]
        print("lgravz",parser.flag_mappings["lgravz__mod__cdata"])
        # print(parser.flag_mappings["omega__mod__cdata"])
        # exit()
        # print(parser.flag_mappings["lvisc_hyper3_cmu_const_strt_otf__mod__viscosity"])
        #TODO: this should be possible with looking into the shared variables module but don't want to do it know
        if "lffree__mod__density" in parser.flag_mappings:
          parser.flag_mappings["lffree__mod__hydro"] = parser.flag_mappings["lffree__mod__density"]
        print(parser.flag_mappings["lffree__mod__density"])
        print(parser.flag_mappings["lffree__mod__hydro"])
        for flag in parser.default_mappings:
          if flag not in parser.flag_mappings:
            parser.flag_mappings[flag] = parser.default_mappings[flag]
        print(parser.flag_mappings["ldiff_hyper3_polar__mod__density"])
        if parser.include_diagnostics:
          parser.flag_mappings["ldiagnos__mod__cdata"] = ".true."
        print(parser.flag_mappings["ldiagnos__mod__cdata"])
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
        print("LEN GLOBAL LOOP LINE",len(global_loop_lines))

        out_file = open(f"{parser.directory}/inlined_bc.inc","w")
        for line in new_lines:
            res_line = line.replace(subroutine_name,f"{subroutine_name}_inlined")
            print(res_line)
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
        print("done test setup")
        exit()
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
        parser.ignored_subroutines.extend(["boundconds_y","boundconds_z"])

        # parser.safe_subs_to_remove.extend(["calc_all_pencils"])
        # parser.ignored_subroutines.extend(["calc_all_pencils"])

        subs_not_to_inline = parser.ignored_subroutines.copy()
        #der and others are handled by the DSL
        subs_not_to_inline.extend(der_funcs)
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
        parser.inline_all_function_calls(filename,subroutine_name,orig_lines,subs_not_to_inline) 
        new_lines = parser.func_info[subroutine_name]["inlined_lines"][filename]
        print("\n\nDONE inlining\n\n")
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
            if "write(*,*)" in line:
              remove_indexes.append(line_index)
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
        new_lines = parser.elim_empty_branches(new_lines,local_variables)
        new_lines = parser.eliminate_while(new_lines)

        file = open("res-after-inlining.txt","w")
        for line in new_lines:
          file.write(f"{line}\n")
        file.close()
        file = open("res-without-mod-names.txt","w")
        for line in new_lines:
          file.write(f"{remove_mod(line)}\n")
        file.close()
        # exit()


        local_variables = {parameter:v for parameter,v in parser.get_variables(new_lines, {},filename,True).items() }
        variables = merge_dictionaries(variables,parser.static_variables)
        new_lines = parser.eliminate_while(new_lines)
        new_lines = parser.eliminate_while(new_lines)
        orig_lines = new_lines.copy()
        for line_index, line in enumerate(new_lines):
                func_calls = parser.get_function_calls_in_line(line,local_variables)
                where_func_calls = [call for call in func_calls if call["function_name"] == "where"]
                where_call_segments = [(None, call["range"][0], call["range"][1]) for call in where_func_calls]
                where_map_vals = []
                for call in where_func_calls:
                  is_scalar_if = False
                  for seg in parser.get_array_segments_in_line(line,variables):
                    param_info = parser.get_param_info((line[seg[1]:seg[2]],False),local_variables,parser.static_variables)
                    print(param_info)
                    is_scalar_if = is_scalar_if or (param_info[3] in [[global_subdomain_range_x,"3"],[global_subdomain_range_x]] )
                  for seg in parser.get_struct_segments_in_line(line,variables):
                    param_info = parser.get_param_info((line[seg[1]:seg[2]],False),local_variables,parser.static_variables)
                    print(param_info)
                    is_scalar_if = is_scalar_if or (param_info[3] in  [[global_subdomain_range_x],[global_subdomain_range_x,"3"]])
                  if not is_scalar_if:
                    print("what to about where")
                    print(param_info)
                    pexit(line)
                  else:
                    where_map_vals.append(line[call["range"][0]:call["range"][1]].replace("where","if",1) + " then")
                new_lines[line_index] = parser.replace_segments(where_call_segments, line, parser.map_val_func,local_variables, {"map_val": where_map_vals})
                if line == "endwhere":
                  new_lines[line_index] = "endif"
                elif line == "elsewhere":
                  new_lines[line_index] = "else"
        for line_index, line in enumerate(orig_lines):
          if orig_lines[line_index] != new_lines[line_index]:
            print(orig_lines[line_index],new_lines[line_index])
        file = open("res-elim-where.txt","w")
        for line in new_lines:
          file.write(f"{line}\n")
        file.close()
        new_lines = parser.eliminate_while(new_lines)
        # for line_index, line in enumerate(new_lines):
        #   if line.strip() == "elsewhere":
        #     new_lines[line_index] = "else"
        #   elif line.strip() == "endwhere":
        #     new_lines[line_index] = "endif"
        file = open("res-after-inlining.txt","w")
        for line in new_lines:
          file.write(f"{line}\n")
        file.close()
        # new_lines = parser.inline_known_parameters(new_lines,{},True)
        # file = open("res-after-inlining-2.txt","w")
        # for line in new_lines:
        #   file.write(f"{line}\n")
        # file.close()
        local_variables = {parameter:v for parameter,v in parser.get_variables(new_lines, {},filename,True).items() }
        transform_func = parser.transform_line_boundcond if config["boundcond"] else parser.transform_line_stencil


        # all_inlined_lines = [line.replace("\n","") for line in open("res-all-inlined.txt","r").readlines()]
        # res = parser.transform_lines(new_lines,all_inlined_lines, local_variables,transform_func)

        res = parser.transform_lines(new_lines,new_lines, local_variables,transform_func)
        res = [remove_mod(line) for line in res]
        res = [normalize_reals(line).replace("(:,1)",".x").replace("(:,2)",".y").replace("(:,3)",".z") for line in res]
        if parser.offload_type == "stencil":
          output_filename = "mhdsolver-rhs.inc"
        elif self.offload_type == "boundcond":
          output_filename = "res-boundcond.inc"
        file = open(output_filename,"w")
        for line in res:
            file.write(f"{line}\n") 
        file.close()
        print("DONE")

        print("Deduce dims")
        print("Ranges:")
        has_x_range = False
        has_y_range = False
        has_z_range = False
        for range,index,line in parser.ranges:
            if range not in [":","1:mx","1:my","1:mz"]:
                print(range)
                print("NOT full range check what to do?")
                exit()
            has_x_range = has_x_range or (index == 0)
            has_y_range = has_y_range or (index == 1)
            has_z_range = has_z_range or (index == 2)
        x_dim= global_subdomain_range_with_halos_x if has_x_range else "1"
        y_dim= global_subdomain_range_with_halos_y if has_y_range else "1"
        z_dim= global_subdomain_range_with_halos_z if has_z_range else "1"
        print(f"{x_dim},{y_dim},{z_dim}")
        print(config)
        print(parser.test_to_c)
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
    print("modules")
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
    exit()

            

    

if __name__ == "__main__":
    main()
