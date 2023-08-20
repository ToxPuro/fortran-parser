from asyncore import write
import math
import re
import os
import csv
import argparse

from pexpect import split_command_line
def new_is_variable_line(line):
    parts = line.split("::")
    if "!" in parts[0]:
        return False
    return len(parts)>1
def is_variable_line(line_elem):
    parts = line_elem[0].split("::")
    if "!" in parts[0]:
        return False
    return len(parts)>1
def merge_dictionaries(dict1, dict2):
    merged_dict = dict1.copy()
    merged_dict.update(dict2)
    return merged_dict
def get_variable_segments(line, variables):
    check_string = ""
    accumulating = False
    start_index = 0
    var = ""
    res = []
    i = 0
    inside_indexes = False
    while i < len(line):
        char = line[i]
        if inside_indexes and char == ")":
            inside_indexes = False
        if char in " /*+-<>=()" and not inside_indexes:
            if accumulating:
                accumulating = False
                if char == ")":
                    res.append((var,start_index,i+1))
                    i = i +1
                else:
                    res.append((var,start_index,i))
            check_string = ""
            start_index = i + 1
        else:
            check_string = check_string + char
        if check_string in variables :
            if i == len(line)-1:
                res.append((check_string,start_index,i+1))
                print("Variable segments: ", res)
                for sg in res:
                    print("ARR_SEG: ",line[sg[1]:sg[2]])
                return res
            elif line[i+1] in " /*+-<>=()":
                var = check_string
                if line[i+1] != "(":
                    res.append((var,start_index,i+1))
                else:
                    inside_indexes = True
                    accumulating = True
                    i = i + 1
        i = i + 1
    print("Variable segments: ", res)
    for sg in res:
        print("ARR_SEG: ",line[sg[1]:sg[2]])
    return res

def get_array_in_spread(line,rhs_dim):
    regex = "spread\((.*)\)"
    search = re.search(regex,line)
    if search is None:
        return None
    inside_spread= search.group(1)   
    regex = "(.*\(.*\))"
    array = re.search(regex,inside_spread)
    if array is None:
        array = inside_spread.split(",")[0] + "("
        for i in range(rhs_dim-1):
            array = array + ":"
            if i < rhs_dim-2:
                array = array + ","
        array = array + ")"
    else:
        array = array.group(1)
    return array

def get_spread_dim(line):
    regex = "spread\((.*)\)"
    search = re.search(regex,line)
    if search is None:
        return None
    inside_spread= search.group(1)
    regex = "(.*\(.*\))"
    array = re.search(regex,inside_spread)
    if array is None:
        return inside_spread.split(",")[1]
    else:
        after =  inside_spread.split(")")[1]
        return after.split(",")[1]

def transform_spread(line,explicit_indexes):
    rhs_dim = len(explicit_indexes)
    spread_array = get_array_in_spread(line,rhs_dim)
    if spread_array is None:
        return line
    spread_dim = int(get_spread_dim(line))
    index = [index.strip() for index in re.search("\((.*)\)",spread_array).group(1).split(",")]
    var = re.search("([^\(]*)(\(|)", spread_array).group(1)
    res = var + "("
    j = 0
    for i in range(len(index)):
        if j == (spread_dim-1):
            j = j +1
        if index[i] == ":":
            new_index = explicit_indexes[j]
        elif ":" in index[i]:
            lower, upper = [part.strip() for part in index[i].split(":")]
            new_index = f"lower + {explicit_indexes[i]}"
        else:
            new_index = index[i]
        res = res + new_index
        if i < len(index)-1:
            res = res + ","
    res = res + ")"
    regex = "spread\((.*)\)"
    index_pair = [(m.start(0),m.end(0)) for m in re.finditer(regex,line)][0]
    res_line = line[:index_pair[0]]
    res_line = res_line + res
    res_line = res_line + line[index_pair[1]:]
    return res_line

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

def get_used_variables_from_line(line):
    characters_to_space= ["/", ":", ",","+","-","*","(",")","=","<","!",">"]
    for character in characters_to_space:
        line = line.replace(character, " ")
    parts = [part.strip() for part in line.split(" ")]
    return [parse_variable(part) for part in parts if part]
def get_used_variables(lines):
    res = []
    for line in lines:
        res.extend(get_used_variables_from_line(line))
    return res
def get_array_segments_in_line(line,variables):
    array_vars = get_arrays_in_line(line,variables)
    res = get_variable_segments(line, array_vars)
    return sorted(res, key = second_val)
def get_arrays_in_line(line,variables):
    variables_in_line= get_used_variables_from_line(line)
    return list(set([var for var in variables_in_line if (var in variables and len(variables[var]["dims"])) > 0 ]))
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
        if index == ":":
            dims.append(f"size({rhs_var},dim={i+1})")
            num_of_looped_dims = num_of_looped_dims + 1
        elif ":" in index:
            lower, upper = [var.strip() for var in index.split(":")]
            dims.append(f"{upper}-{lower}+1")
            num_of_looped_dims = num_of_looped_dims + 1
    return (dims, num_of_looped_dims)
def get_rhs_variable(line):
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
        elif elem == "(":
            num_of_left_brackets = num_of_left_brackets + 1
        elif elem == ")":
            num_of_right_brackets = num_of_right_brackets + 1
        elif elem == "=" and line[index-1] not in "<>=!" and line[index+1] not in "<>=!" and num_of_single_quotes%2==0 and num_of_double_quotes%2==0 and num_of_left_brackets == num_of_right_brackets:

                return parse_variable(line[start_index:index])
    return None

def get_indexes(segment,var,dim):
    index_search = re.search(f"{var}\((.*)\)",segment)
    indexes = [":" for i in range(dim)]
    if index_search:
        indexes = [index.strip() for index in index_search.group(1).split(",")]
    return indexes
def transform_array_segment(segment,var,dim):
    print("SEG_IN: ",segment,var,dim)
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
    res = f"{var}("
    for i,index in enumerate(new_indexes):
        res = res + index
        if i < len(new_indexes)-1:
            res = res + ","
    res = res + ")"
        
    return res
def transform_implicit_loop(line,variables):
    if new_is_variable_line(line):
        return line
    rhs_segment = get_rhs_segment(line)
    if rhs_segment is None:
        return line
    rhs_var = get_rhs_variable(line)
    if rhs_var is None:
        return line
    if rhs_var not in variables:
        return line
    dim = len(variables[rhs_var]["dims"])
    indexes = get_indexes(get_rhs_segment(line),rhs_var,dim)
    dims, num_of_looped_dims = get_dims_from_indexes(indexes,rhs_var)
    if num_of_looped_dims == 0:
        return line
    line = transform_spread(line,[f"explicit_index_{i}" for i in range(num_of_looped_dims)])
    array_segments_indexes = get_array_segments_in_line(line,variables)
    last_index = 0
    res_line = "!$omp teams distribute parallel do\n"
    for i in range(num_of_looped_dims):
        res_line = res_line + f"do explicit_index_{i}=1,{dims[i]}\n"
    for segment in array_segments_indexes:
        res_line = res_line + line[last_index:segment[1]]
        res_line = res_line + transform_array_segment(line[segment[1]:segment[2]],segment[0],num_of_looped_dims)
        last_index = segment[2]
    res_line = res_line + line[last_index:] + "\n"
    for i in range(num_of_looped_dims):
        res_line = res_line + f"end do\n"
    res_line = res_line + "!$omp end teams distribute parallel do\n"
    return res_line

class Parser:

    def __init__(self, files,parse_only_once):
        self.static_variables = {}
        self.lines = {}
        self.parsed_files_for_static_variables = []
        self.parsed_modules = []
        self.parsed_subroutines = []
        self.loaded_files = []
        self.subroutine_order = 0
        self.used_files = []
        self.found_function_calls = []
        self.parse_only_once = parse_only_once
        self.used_files = files 
        self.used_static_variables = []
        self.functions_in_file = {}
        self.static_writes = []
        self.rewritten_functions = {}
        ignored_files = ["./pencil-code/src/boundcond.f90","./pencil-code/src/nompicomm.f90","./pencil-code/src/mpicomm.f90","./pencil-code/src/mpicomm_double.f90","./pencil-code/src/ghostfold_mpicomm_double.f90"]
        self.used_files = list(filter(lambda x: x not in ignored_files and not re.search("cuda",x) and re.search("\.f90", x) and not re.search("astaroth",x) ,self.used_files))
    def find_module_files(self, module_name, rootdir):
        res = [file for file in self.used_files if self.contains_module(self.get_lines(file), module_name)]
        self.loaded_files.extend(res)
        self.loaded_files = list(set(self.loaded_files))
        return res

    def find_subroutine_files(self, subroutine_name, rootdir):
        return [x for x in self.used_files if self.contains_subroutine(self.get_lines(x), subroutine_name)]

    def parse_module(self, module_name, rootdir):
        if module_name not in self.parsed_modules:
            self.parsed_modules.append(module_name)
            file_paths = self.find_module_files(module_name, rootdir)
            for file_path in file_paths:
                self.parse_file_for_static_variables(file_path)

    def parse_line(self, line):
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
        return line.strip()

    def get_lines(self, filepath, start=0, end=math.inf, include_comments=False):
        if filepath not in self.lines.keys():
            lines = []
            read_lines = open(filepath, 'r').readlines()
            index = 0
            while index<len(read_lines):
                line = read_lines[index].strip()
                if len(line)>0:
                    if line[0] != "!":
                        line = self.parse_line(line)
                    if len(line)>0:
                        #take care the case that the line continues after end of the line
                        if line[-1] == "&" and line[0] != "!":
                            while line[-1] == "&":
                                index += 1
                                next_line = read_lines[index].strip()
                                if len(next_line)>0 and next_line[0] != "!":
                                    line = (line[:-1] + " " + self.parse_line(next_line)).strip()
                        # split multilines i.e. fsdfaf;fasdfsdf; into their own lines for easier parsing
                        for line in self.split_line(line):
                            lines.append((line, index))
                index += 1
            self.lines[filepath] = lines
        res = [x for x in self.lines[filepath] if x[1] >= start and x[1] <= end]
        if not include_comments:
            res = [x for x in res if x[0][0] != "!"] 
        return res

    def contains_module(self, lines, module_name):
        for (line, count) in lines:
            if line.strip() == f"module {module_name}":
                return True
        return False

    def contains_subroutine(self, lines, function_name):
        for (line, count) in lines:
            if "(" in function_name or ")" in function_name:
                return False
            if re.search(f"\s?subroutine {function_name}[\s(]",line) or re.search(f"\s?function {function_name}[\s(]",line) or re.search(f"interface {function_name}\s",line) or line==f"subroutine {function_name}" or line==f"function {function_name}":
                return True
        return False

    def get_used_modules(self,lines):
        modules = []
        for (line, count) in lines:
            if line.strip().split(" ")[0].strip() == "use":
                modules.append(line.strip().split(" ")[1].strip().replace(",",""))
        return modules

    def get_mem_access_or_function_calls(self,lines):
        res = []
        for(line, count ) in filter(lambda x: not is_variable_line(x),lines):
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


    def get_writes_from_line(self,line,count):
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
                write = self.parse_variable(line[start_index:index])
                res.append({"variable": write, "line_num": count, "line": line, "is_static": write in self.static_variables})
                start_index = index
            elif elem == "(":
                num_of_left_brackets += 1
            elif elem == ")":
                num_of_right_brackets += 1
        return res

    def get_writes(self,lines):
        res = []
        for(line, count) in filter(lambda x: not is_variable_line(x),lines):
            res.extend(self.get_writes_from_line(line,count))
        return res
    def get_used_variables_from_line(self,line):
        characters_to_space= ["/",":", ",","+","-","*","(",")","=","<","!",">"]
        for character in characters_to_space:
            line = line.replace(character, " ")
        parts = [part.strip() for part in line.split(" ")]
        return [self.parse_variable(part) for part in parts if part]

    def get_used_variables(self,lines):
        res = []
        for(line, count) in filter(lambda x: not is_variable_line(x),lines):
            res.extend(self.get_used_variables_from_line(line))
        return res


    def get_function_name(self,line):
        iter_index = len(line)-1
        while(iter_index > 0):
            if line[iter_index] == "%":
                return line[0:iter_index]
            iter_index -= 1
        return line

    def get_function_calls(self,lines, local_variables):
        function_calls = []
        for(line, count ) in filter(lambda x: not is_variable_line(x),lines):
            function_call_indexes = []
            num_of_single_quotes = 0
            num_of_double_quotes = 0

            #get normal function calls i.e. with parameters and () brackets
            for i in range(len(line)):
                if line[i] == "'":
                    num_of_single_quotes += 1
                if line[i] == '"':
                    num_of_double_quotes += 1
                if line[i] == "(" and(i==0 or line[i-1] not in "^'=' '\/+-.*\(\)<>^") and num_of_double_quotes %2 == 0 and num_of_single_quotes %2 == 0:
                    function_call_indexes.append(i)
            for index in function_call_indexes:
                current_index = index-1
                while(line[current_index] not in "^'=' '\/+-.*\(\)<>^,"):
                    current_index -= 1
                current_index +=1
                function_name = self.get_function_name(line[current_index:index])
                current_index = index
                if function_name not in self.static_variables and function_name not in local_variables:
                    #step through (
                    current_index +=1
                    number_of_right_brackets = 1
                    number_of_left_brackets = 0
                    num_of_single_quotes = 0
                    num_of_double_quotes = 0
                    parameter_list_start_index = current_index
                    while(number_of_right_brackets>number_of_left_brackets):
                        if current_index >= len(line):
                            print(line, function_name)
                        if line[current_index] == "'" and num_of_double_quotes %2 == 0:
                            num_of_single_quotes += 1
                        if line[current_index] == '"' and num_of_single_quotes %2 == 0:
                            num_of_double_quotes += 1
                        if line[current_index] == "(" and num_of_double_quotes %2 == 0 and num_of_single_quotes %2 == 0:
                            number_of_right_brackets += 1
                        elif line[current_index] == ")" and num_of_double_quotes %2 == 0 and num_of_single_quotes %2 == 0: 
                            number_of_left_brackets += 1
                        
                        # if line[current_index] == ")":
                        #     print("Should be at least 1", number_of_left_brackets, number_of_right_brackets)
                        current_index += 1
                        
                    parameter_list = line[parameter_list_start_index:current_index-1]
                    parameters = [x.strip() for x in parameter_list.split(",")]
                    function_name = function_name.strip()
                    if len(function_name) >0:
                        function_calls.append({"function_name": function_name, "parameters": parameters})
            
            #get function calls with call function i.e. call infront and no brackets
            for index in [m.end(0) for m in re.finditer("call ", line)]:
                #skip empty spaces
                while line[index] == " ":
                    index= index+1
                if line[index] == "=":
                    break
                start_index = index
                while index<len(line) and line[index] not in " (":
                    index = index+1
                function_name = line[start_index:index]
                if("=" in function_name):
                    print(line)
                    assert("=" not in function_name)
                #don't add duplicates, no need since no parameters
                if function_name not in [function_call["function_name"] for function_call in function_calls]:
                    function_name = function_name.strip()
                    if len(function_name) > 0:
                        function_calls.append({"function_name": function_name, "parameters": []})
            
        
        return [function_call for function_call in function_calls if not function_call["function_name"].isspace()]



    def get_subroutine_line_start_and_end(self,filename, function_name, lines=None):
        subroutine_line_start = -1
        subroutine_line_end = -1
        if lines is None:
            lines = self.get_lines(filename)
        for (line, count) in lines:
            if line ==  "subroutine fatal_error(location,message,force)":
                print("COPY")
            if  subroutine_line_start<0 and (re.search(f"\s?subroutine\s?{function_name}[\s(]",line) or re.search(f"\s?function\s?{function_name}[\s(]",line) or line==f"subroutine {function_name}" or line==f"function {function_name}"):
                subroutine_line_start = count
            # line.strip() == f"endsubroutine {function_name}" or line.strip() == f"end subroutine {function_name}"    
            if  (re.match(f"end.+{function_name}",line) or (line == "end" or line == "endsubroutine" or line=="endfunction" or line=="end subroutine" or line=="end function")) and subroutine_line_start>-1:
                subroutine_line_end = count
                return (subroutine_line_start, subroutine_line_end)
        print("Didn't find function start and end", filename, function_name,subroutine_line_start,subroutine_line_end)
        return None

    def get_contains_line_num(self, filename):
        lines = self.get_lines(filename)
        for (line, count) in lines:
            if line.strip() == "contains":
                return count
            elif re.match("function\s+.+\(.+\)",line) or re.match("subroutine\s+.+\(.+\)",line):
                return count-1
        #whole file
        return lines[-1][1]

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
        while parse_still:
            current_elem = line_variables[current_index]
            if current_elem == "=":
            
                res.append(line_variables[start_index:current_index])
                parse_until_next_variable = True
                current_index +=1
                num_of_left_brackets = 0
                num_of_right_brackets = 0
                while parse_until_next_variable:
                    current_elem = line_variables[current_index]
                    not_inside_brackets = num_of_left_brackets == num_of_right_brackets
                    if current_elem == "!":
                        parse_until_next_variable = False
                        parse_still = False
                    if current_elem == "," and not_inside_brackets:
                        start_index = current_index+1
                        parse_until_next_variable=False
                    if current_elem == "(":
                        num_of_left_brackets += 1
                    if current_elem == ")":
                        num_of_right_brackets += 1
                    if parse_until_next_variable:
                        current_index +=1
                        if current_index >= len(line_variables):
                            start_index = current_index+1
                            parse_until_next_variable = False
                            parse_still = False
            elif current_elem == ",":
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

    def get_variables(self, lines, variables, filename):
        in_type = False
        for (line, count) in lines:
            parts = line.split("::")
            is_variable = True
            start =  parts[0].strip()
            type = start.split(",")[0].strip()
            if type == "public":
                is_variable = False
            if line.split(" ")[0] == "type" and len(line.split("::")) == 1:
                in_type = True
            if line.split(" ")[0] == "endtype":
                in_type = False
            if is_variable and not in_type and len(parts)>1:
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
                    dims = dimension
                    if "(" in variable_name:
                        variable_name = variable_name.split("(")[0].strip()
                        search = re.search(f"{variable_name}\(((.*?))\)",line) 
                        if search:
                            dims = [index.strip() for index in search.group(1)]
                    variables[variable_name] = {"type": type, "dims": dims, "allocatable": allocatable, "origin": filename, "public": public, "threadprivate": False, "saved_variable": (saved_variable or "=" in line_variables.split(",")[i])}
        return variables




    def parse_file_for_static_variables(self, filepath):
        if filepath not in self.parsed_files_for_static_variables:
            self.parsed_files_for_static_variables.append(filepath)
            modules = self.get_always_used_modules(filepath)
            for module in modules:
                self.parse_module(module, "./pencil-code/src")
            self.load_static_variables(filepath)

    def parse_subroutine_for_static_variables(self,subroutine_name, filepath):
        self.parse_file_for_static_variables(filepath)
        for module in self.get_subroutine_modules(filepath, subroutine_name):
            self.parse_module(module, "./pencil-code/src")

    def add_threadprivate_declarations_in_file(self,filename):
        res = []
        lines = self.get_lines(filename, include_comments=True)
        index = 0
        while index<len(lines):
            line, _ = lines[index]
            if "!$omp" in line and "threadprivate" in line.lower():
                if line[-1] == "&":
                    if line[-1] == "&":
                        while line[-1] == "&":
                            index += 1
                            next_line = lines[index][0].strip().replace("!$omp","")
                            if len(next_line)>0:
                                line = (line[:-1] + " " + next_line).strip()
                variable_names = [variable.strip() for variable in re.search("(threadprivate|THREADPRIVATE)\((.*)\)",line).group(2).split(",")]
                for variable in variable_names:
                    res.append(variable)
                    if variable in self.static_variables:
                        self.static_variables[variable]["threadprivate"] = True
            index = index+1
        return res

    def add_public_declarations_in_file(self,filename,lines):
        in_public_block = False
        for (line,count) in lines:
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


    def load_static_variables(self, filename):
        static_variables_end = self.get_contains_line_num(filename)
        self.get_variables(self.get_lines(filename, 1, static_variables_end), self.static_variables, filename)
        self.add_public_declarations_in_file(filename,self.get_lines(filename, 1, static_variables_end))
        self.add_threadprivate_declarations_in_file(filename)

    def get_subroutine_variables(self, filename, subroutine_name):
        subroutine_line_start,subroutine_line_end = self.get_subroutine_line_start_and_end(filename, subroutine_name)
        return self.get_variables(get_lines(filename, subroutine_line_start,subroutine_line_end),filename)

    def get_always_used_modules(self, filename):
        module_declaration_end = self.get_contains_line_num(filename)
        return self.get_used_modules(self.get_lines(filename, 1, module_declaration_end))

    def get_subroutine_modules(self, filename, subroutine_name):
        subroutine_line_start, subroutine_line_end = self.get_subroutine_line_start_and_end(filename, subroutine_name)
        return self.get_used_modules(self.get_lines(filename, subroutine_line_start, subroutine_line_end))

    def get_subroutine_lines(self, filename, subroutine_name):
        subroutine_line_start, subroutine_line_end = self.get_subroutine_line_start_and_end(filename, subroutine_name)
        return self.get_lines(filename, subroutine_line_start, subroutine_line_end)

    def get_static_parameters(self,line,parameter_list):
        check_subroutine= re.search("\s?subroutine.+\((.+)\)",line)
        if check_subroutine:
            parameters = [parameter.split("=")[-1].split("(")[0].strip() for parameter in check_subroutine.group(1).split(",")]
        else:
            param_search = re.search(".?function.+\((.+)\)",line)
            if not param_search:
                return []
            parameters = [parameter.split("=")[-1].split("(")[0].strip() for parameter in param_search.group(1).split(",")]
        return list(zip(parameters,parameter_list))
        
    def get_static_passed_parameters(self,parameters,local_variables):
        parameters = [parameter.split("=")[-1].split("(")[0] for parameter in parameters]
        if "added_mass_beta" in parameters:
            print(parameters)
        return list(zip(parameters,list(map(lambda x: (x in self.static_variables and x not in local_variables) or (x in local_variables and local_variables[x]["saved_variable"]),parameters))))

    def get_interfaced_functions(self,file_path,subroutine_name):
        res = []
        lines = self.get_lines(file_path)
        for i,(line, count) in enumerate(lines):
            if line.split(" ")[0] == "interface" and len(line.split(" "))>1 and line.split(" ")[1].strip() == subroutine_name:
               
                cur_index = i+1
                cur_line = lines[cur_index][0]
                while not re.match("endinterface",cur_line):
                    cur_index += 1
                    res.append(cur_line.split("module procedure ")[1].strip())
                    cur_line = lines[cur_index][0]
                break
        if len(res) == 0:
            return [subroutine_name]
        return res

        

    



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

    def write_to_files(self,files,threadprivate_variables,critical_variables):
        threadprivate_declarations = {}
        for file in files:
 
            already_threadprivate = self.add_threadprivate_declarations_in_file(file)
            threadprivate_variables_to_add = []
            static_variables_end = self.get_contains_line_num(file)
            vars = list(self.get_variables(self.get_lines(file, 1, static_variables_end), {}, file).keys())
            vars_to_make_private = [variable for variable in vars if variable in threadprivate_variables and variable not in already_threadprivate and variable != ""]
            if(len(vars_to_make_private) > 0):
                threadprivate_declarations[file] = f"!omp threadprivate({','.join(vars_to_make_private)})"
        return threadprivate_declarations

                    
    def add_atomic_declarations(self,variables):
        files = self.used_files
        for file in files:
            res_contents = []
            contents = self.get_lines(file,include_comments=True)
            for i,(line,count) in enumerate(contents):
                res_contents.append(line)
                if line[0] != "!":
                    writes = self.get_writes_from_line(line,count)

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
        files = list(set([self.static_variables[variable]["origin"] for variable in variables ]))
        for file in files:
            res_contents = []
            contents = self.get_lines(file,include_comments=True)
            in_module_declaration = True
            in_type = False
            for (line,count) in contents:
                res_contents.append(line)
                if line.split(" ")[0] == "type":
                    in_type = True
                if line.split(" ")[0] == "endtype":
                    in_type = False
                if in_module_declaration and not in_type:
                    if re.match("function\s+.+\(.+\)",line) or re.match("subroutine\s+.+\(.+\)",line) or line.strip() == "contains":
                        in_module_declaration = False
                    parts = line.split("::")
                    if len(parts) > 1:
                        line_variables = parts[1].strip()
                        variable_names = self.get_variable_names_from_line(line_variables)
                        res_contents.extend([f"public :: {variable}" for variable in variable_names if variable in variables and self.static_variables[variable]["origin"] == file and not self.static_variables[variable]["public"]])

            with open(f"./out/{file}","w") as f:
                f.write("\n".join(res_contents))

        
        
    def generate_commands(self, filename, save_variables, in_dir="./pencil-code/src/",outdir="./out/pencil-code/src/"):
        res_contents = []
        with open(f"{in_dir}{filename}.f90", "r") as f:
            contents = f.readlines()
        for index in range(len(contents)):
            res_contents.append(contents[index])
            if re.match("!\$parser-command:save-global-state.?", contents[index]):
                save_string = "".join([self.generate_save_array_store(static_variable) for static_variable in save_variables])
                res_contents.append(save_string)
            elif re.match("!\$parser-command:load-global-state.?", contents[index]):
                save_string = "".join([self.generate_read_from_save_array(static_variable) for static_variable in save_variables])
                res_contents.append(save_string)
            elif re.match("!\$parser-command:allocate-global-state-arrays.?", contents[index]):
                save_string = "".join([self.generate_allocation_for_save_array(static_variable) for static_variable in save_variables])
                res_contents.append(save_string)
            elif re.search("!\$parser-command:generate-firstprivate-pragma-for-static-variables.?",contents[index]):
                #remove the parser command for the omp pragma to be valid
                res_contents[-1] = res_contents[-1].replace("!$parser-command:generate-firstprivate-pragma-for-static-variables","firstprivate(&")
                save_string = ",&\n".join([f"!$omp {variable}" for variable in save_variables])
                save_string = f"{save_string})\n"
                res_contents.append(save_string)
            elif re.search(".+!\$parser-command:generate-copyin",contents[index]):
                res_contents[-1] = res_contents[-1].replace("!$parser-command:generate-copyin","copyin(&")
                save_string = ",&\n".join([f"!$omp {variable}" for variable in save_variables])
                save_string = f"{save_string})\n"
                res_contents.append(save_string)
                


        with open(f"{outdir}{filename}.f90", "w") as f:
            contents = "".join(res_contents)
            f.write(contents)
    def parse_subroutine_all_files(self, subroutine_name, rootdir, call_trace, check_functions, offload, layer_depth=math.inf, parameter_list=[], only_static=True):
        if layer_depth<0:
            return []
        self.subroutine_order += 1
        parse = True
        if not self.parse_only_once:
            parse = (subroutine_name, parameter_list) not in self.parsed_subroutines and not (all([not parameter[1] for parameter in parameter_list]) and subroutine_name in [subroutine[0] for subroutine in self.parsed_subroutines])
        else:
            parse = subroutine_name not in [subroutine[0] for subroutine in self.parsed_subroutines]
        if parse:
            file_paths = self.find_subroutine_files(subroutine_name, rootdir)
            for file_path in file_paths:
                interfaced_functions = self.get_interfaced_functions(file_path,subroutine_name)
                for function in interfaced_functions:
                    self.parse_subroutine_in_file(file_path, function, check_functions,offload,layer_depth, call_trace,parameter_list,only_static)

    def parse_subroutine_in_file(self, filename, subroutine_name, check_functions, offload, layer_depth=math.inf, call_trace="", parameter_list=[], only_static=True):
        self.parsed_subroutines.append((subroutine_name, parameter_list))
        print(subroutine_name, parameter_list)
        if subroutine_name == "mult_matrix":
            print(parameter_list)
        if layer_depth < 0:
            return []
        subroutine_line_start, subroutine_line_end = self.get_subroutine_line_start_and_end(filename, subroutine_name)
        lines = self.get_lines(filename, subroutine_line_start, subroutine_line_end)
        modules = self.get_used_modules(lines)
        self.parse_file_for_static_variables(filename)
        for module in self.get_subroutine_modules(filename, subroutine_name):
            self.parse_module(module, "./pencil-code/src")
        local_variables = {parameter:v for parameter,v in self.get_variables(lines, {},filename).items() }
        parameters = self.get_static_parameters(self.get_lines(filename,start=subroutine_line_start,end=subroutine_line_start)[0][0], parameter_list)
        if self.subroutine_order == 0:
            call_trace = subroutine_name
        else:
            call_trace = f"{call_trace} -> {subroutine_name}"
        writes = self.get_writes(lines)
        for write in writes:
            write["is_static"] = (write["variable"] in self.static_variables and write["variable"] not in local_variables) or (write["variable"] in local_variables and local_variables[write["variable"]]["saved_variable"])
            for (parameter,(passed_parameter,is_static)) in parameters:
                if write["variable"] == parameter:
                    write["variable"] = passed_parameter
                    write["is_static"] = is_static
        if only_static:
              self.static_writes.extend([{"variable": write["variable"],"line_num": write["line_num"], "filename": filename, "call_trace":call_trace, "line": write["line"]} for write in writes if write["is_static"]])
        else:
            self.static_writes.extend([{"variable": x[0],"line_num": x[1], "filename": filename, "call_trace":call_trace, "line": x[2]} for x in self.get_writes(lines) if x[0]])
        if offload:
            self.used_static_variables.extend([var for var in self.get_used_variables(lines) if var in self.static_variables])
            self.used_static_variables = list(set(self.used_static_variables))
            if filename not in self.functions_in_file:
                self.functions_in_file[filename] = []
            self.functions_in_file[filename].append(subroutine_name)
            #Add transformed function to written out later
            if filename not in self.rewritten_functions:
                self.rewritten_functions[filename] = {}
            combined_variables = merge_dictionaries(self.static_variables, local_variables)
            print(f"ADDED PARSED FUNCTION: {subroutine_name} TO FILE {filename}") 
            assert(subroutine_name not in self.rewritten_functions[filename])
            new_line = ""
            for line in lines:
                new_line = new_line + f"{transform_implicit_loop(line[0],combined_variables)}\n"
            ##new_line = f"{lines[0][0]}\nMamma Mia!\n{lines[-1][0]}"
            self.rewritten_functions[filename][subroutine_name] = (new_line,0)

        for function_call in self.get_function_calls(lines, local_variables):
            if function_call["function_name"] in check_functions or function_call["function_name"].lower().startswith("mpi"):
                self.found_function_calls.append((call_trace, function_call["function_name"], parameter_list))
            self.parse_subroutine_all_files(function_call["function_name"], "./pencil-code/src", call_trace, check_functions, offload, layer_depth-1, self.get_static_passed_parameters(function_call["parameters"],local_variables),only_static)
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
        in_type = False
        for (line, count) in lines:
            parts = line.split("::")
            is_variable = True
            start =  parts[0].strip()
            type = start.split(",")[0].strip()
            if type == "public":
                is_variable = False
            if line.split(" ")[0] == "type" and len(line.split("::")) == 1:
                in_type = True
            if line.split(" ")[0] == "endtype":
                in_type = False
            if not is_variable and not in_type and len(parts)>1:
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

def get_used_files(make_output):
    files = []
    if make_output is not None:
        with open(make_output, mode="r") as file:
                lines = file.readlines()
                for line in lines:
                    search = re.search("([^\s]+\.f90)", line)
                    if search is not None:
                        files.append(f"./pencil-code/src/{search.group(1)}")
        return files

    for root, _, files in os.walk("./pencil-code/src"):
        for name in files:
            files.append(os.path.join(root,name))
    return files



def main():
    argparser = argparse.ArgumentParser(description="Tool to find static writes in Fortran code",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument("-f", "--function", help="function to be parsed", required=True)
    argparser.add_argument("-F", "--file", help="File where to parse the function", required=True)
    argparser.add_argument("-m", "--use-make-output",type=str, help="Pass a file path if want to only analyze the files used in build process. Takes in the print output of make. If not given traverses the file structure to find all .f90 files in /src")
    argparser.add_argument("-c", "--communication",default=True,action=argparse.BooleanOptionalAction, help="Whether to check for mpi calls, can be used to check if code to be multithreaded is safe")
    argparser.add_argument("-o", "--offload",default=False,action=argparse.BooleanOptionalAction, help="Whether to offload the current code. Cannot multithread and offload at the same time")
    
    args = argparser.parse_args()
    config = vars(args)

    files = get_used_files(config["use_make_output"])
    parser = Parser(files,config["offload"])
    #variables = {"tmp_xy": {"dims": [2,2]}, "f": {"dims":[2,2,2,2]}, "dz2_bound": {"dims":[1]}}
    #line ="f(:,:,n1-i,iss)=f(:,:,n1+i,iss)-dz2_bound(-i)*tmp_xy"
    #print(transform_implicit_loop(line,variables))
    #return
    variables = {}

    header = "./pencil-code/src/mpicomm.h"
    filename = config["file"]
    subroutine_name = config["function"]
    check_functions = []
    if config["communication"]:
        check_functions = parser.get_function_declarations(parser.get_lines(header), [], header)
    parser.parse_subroutine_in_file(filename, subroutine_name, check_functions, config["offload"])
    #parser.save_static_writes(parse.static_writes)
    #parser.save_static_variables()

    #writes = parser.read_static_writes()
    #parser.read_static_variables()
    writes = parser.static_writes
    variables = list(set([x["variable"] for x in writes]))
    variables = list(set(variables) - set(["f","df","p","itype_name","fp","dfp","r_int_border","r_ext_border"]))
    #print("Num of variables: ", len(variables))
    public_variables = [variable for variable in variables if variable in parser.static_variables and parser.static_variables[variable]["public"]]
    threadprivate_declarations = parser.write_to_files(parser.used_files,variables,["fname","fname_keep","fnamer","fname_sound","fnamex","fnamey","fnamez","fnamexy","fnamexz","fnamerz","dt1_max","r_int_border","r_ext_border"])
    threadprivate_var = []
    ##print(threadprivate_declarations)
    # for file in threadprivate_declarations:
    #     parser.add_threadprivate_declarations_to_file(file, threadprivate_declarations[file])
    for file in parser.used_files:
        threadprivate_var.extend(parser.add_threadprivate_declarations_in_file(file))
    threadprivate_var = list(set(threadprivate_var))
    public_threadprivate_vars = [var for var in threadprivate_var if var in public_variables]
    
    
    variables.sort()
    threadprivate_var.sort()
    public_variables.sort()
    public_threadprivate_vars.sort()
    #print("Num of variables: ", len(variables))
    #print("Num of threadprivate variables: ", len(threadprivate_var))
    non_threadprivate_vars = [var for var in variables if var not in threadprivate_var]
    filenames = [parser.static_variables[var]["origin"] for var in non_threadprivate_vars]
    for file in filenames:
        inc_filename = f"{file}_omp_inc.h"
        print("Creating include file: ",inc_filename)
        f = open(inc_filename, "w")
        vars_to_include = [var for var in non_threadprivate_vars if parser.static_variables[var]["origin"] == file]
        for var in vars_to_include:
            f.write(f"!$omp threadprivate({vars_to_include})\n")
        f.close()
    #print(non_threadprivate_vars)
    #print(",".join(public_threadprivate_vars))
    #print("Static variables: \n\n")
    #for var in parser.static_variables:
    #    print(f"{var}: ",parser.static_variables[var])
    #print("\n\n")
    #print("Written static variables: \n\n")
    #for var in variables:
    #    print(var)
    #print("\n\n")
    #print("Used static variables: \n\n")
    #for var in parsernused_static_variablen:
    #    print(var)
    #print("\n\n")
    #print("Communication calls:\n\n")
    #for function_call in parser.found_function_calls:
    #    print(function_call)
    #print("\n\n")
    parsed_functions = []
    for filename in parser.functions_in_file:
        parsed_functions.extend(parser.functions_in_file[filename])
    parsed_functions = list(set(parsed_functions))
    #print("Parsed_functions:\n\n")
    #for function in parsed_functions:
        ##print(function)
    #print("\n\n")
    for filename in parser.rewritten_functions:
        output = ""
        lines = parser.get_lines(filename,include_comments=True)
        for function in parser.rewritten_functions[filename]:
            subroutine_line_start, subroutine_line_end = parser.get_subroutine_line_start_and_end(filename, function,lines)
            ##count to indexes
            start_index = 0
            end_index = 0
            for i, line in enumerate(lines):
                if line[1] == subroutine_line_start:
                    start_index = i
                if line[1] == subroutine_line_end:
                    end_index = i
            res_lines = lines[:start_index]
            res_lines = res_lines + [parser.rewritten_functions[filename][function]]
            res_lines = res_lines + lines[1+end_index:]
            lines = res_lines
        for line in lines:
            output = output + f"{line[0]}\n"
        if filename == config["file"]:
            f = open("new_file.txt", "w")
            f.write(output)
            f.close()
    ##print("bc_ss_flux loops rolled: \n")
    ##print(parser.rewritten_functions[config["file"]][subroutine_name][0])
    print("Preprocessing successful")
    

    

if __name__ == "__main__":
    main()
