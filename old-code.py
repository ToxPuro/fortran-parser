  #remove unnecessary writes
        # cProfile.runctx('self.elim_unnecessary_writes(lines,local_variables,variables)',globals(),locals())
        # exit()

        # #inline some profile meanings
        # remove_indexes = []
        # known_profiles = {}
        # for line_index,line in enumerate(lines):
        #     #only consider lines with profiles
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
        #             prof_val = write["value"]
        #             is_safe_to_inline = True
        #             #do only for profiles
        #             if "p%" in var_name:
        #                 for x, y in enumerate(lines):
        #                     if x != line_index:
        #                         # if write is not in an assignment in the line than it can be removed
        #                         if var_name in y:
        #                             y_writes = self.get_writes_from_line(y,local_variables)
        #                             if len(y_writes) > 0 :
        #                                 write = y_writes[0]
        #                                 rhs_segment = get_variable_segments(y, [write["variable"]])
        #                                 if len(rhs_segment) == 0:
        #                                     rhs_segment = self.get_struct_segments_in_line(y, [write["variable"]])
        #                                 rhs_segment  = rhs_segment[0]
        #                                 y_var_name= y[rhs_segment[1]:rhs_segment[2]].split("::",1)[-1].split("(",1)[0].strip()
        #                                 if y_var_name == var_name:
        #                                     is_safe_to_inline = False
        #                             var_segments = get_var_segments_in_line(line,[var_name])    
        #                             for seg in var_segments:
        #                                 indexes = self.get_segment_indexes(seg,line,0)
        #                                 if indexes not in [[],[":"]]:
        #                                     is_safe_to_inline = False
        #                 if is_safe_to_inline:
        #                     remove_indexes.append(line_index)
        #                     known_profiles[var_name] = prof_val 

        # lines = [x[1] for x in enumerate(lines) if x[0] not in remove_indexes]

        # for line_index,line in enumerate(lines):
        #     for profile in known_profiles:
        #         lines[line_index] = replace_variable(lines[line_index],profile,known_profiles[profile])










            # local_variables = {parameter:v for parameter,v in parser.get_variables([(line,0) for line in test_lines], {},parser.file).items() }
    # test_lines = parser.inline_0d_writes(test_lines,local_variables)
    # global_loop_lines,iterators = parser.get_global_loop_lines(test_lines,local_variables)
    # print(global_loop_lines)
    # if len(global_loop_lines) > 0:
    #     test_lines = parser.remove_global_loops(test_lines,local_variables,global_loop_lines,iterators)
    # for line in test_lines:
    #     print(line)
    # exit()


    # test_lines = parser.unroll_constant_loops(test_lines,{})
    # for line in test_lines:
    #     print(line)
    # exit()
    # line = "l1+(a1_6-1)"
    # print("?",parser.evaluate_integer(line))
    # exit()
    # line = "f(l1+(a1_6-1):l2-(a2_6-1),m,n,iux+(2-1))"
    # variables = {"f": {"dims":[":",":",":",":"], type: "real"}}
    # array_segments_indexes = parser.get_array_segments_in_line(line,variables)
    # seg = array_segments_indexes[0]
    # indexes = get_segment_indexes(seg,line,len(variables[seg[0]]["dims"]))
    # print(len(indexes))
    # print(indexes)
    # exit()

    # condition = "(i==1.and.j==2)"
    # print("?",parser.evaluate_boolean(condition,{}))
    # exit()
    # lines = [line.replace("\n","").strip() for line in open("res.txt").readlines()]
    # lines = parser.unroll_constant_loops(lines,{})
    # print("\n\nres\n\n")
    # for line in lines:
    #     print(line)
    # exit()
    # lines = parser.eliminate_while(lines)
    # for line in lines:
    #     print(line)
    # exit()
    # lines = parser.get_lines(file)
    # for line in lines:
    #     print(line)
    # print(parser.func_info["calc_pencils_magn_mf"]["files"])
    # exit()
    # line = "(.true. .or. .false.) .and. nwgrid/=1"
    # print(line)
    # print("-->")
    # print(self.evaluate_boolean(line))
    # exit()

    # lines = [line.replace("\n","") for line in open("./res.txt","r").readlines()]
    # lines = parser.transform_case(lines)
    # lines = parser.replace_var_in_lines(lines,"mass_source_profile","'exponential'")
    # local_variables = {}
    # parser.evaluate_ifs(lines,local_variables)
    # lines= parser.eliminate_while(lines)
    # print("res lines")
    # for line in lines:
    #     print(line)
    # exit()

        # def parse_file_for_struct(self,file_path,struct):
    #     lines = self.get_lines(file_path)
    #     in_struct = False
    #     for count,line in enumerate(lines):
    #         line = line.lower().strip()
    #         if line == "contains":
    #             in_struct =False
    #             break
    #         if f"type {struct}" in line and "(" not in line and ")" not in line and "end" not in line:
    #             in_struct = True
    #             if struct not in self.struct_table:
    #                 self.struct_table[struct] = {}
    #         elif in_struct and ("endtype" in line or "end type" in line):
    #             in_struct = False
    #             break
    #         elif in_struct:
    #             self.get_variables_from_line(line, self.struct_table[struct], file_path, self.get_own_module(file_path))

#used in transform_line_stencil for structs
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