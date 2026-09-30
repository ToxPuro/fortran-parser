"""
Adds DSL declarations for f-array fields that are registered in the Fortran
sources but not yet known to the handwritten Astaroth DSL files.

Registrations are found from calls to farray_register_pde, farray_register_auxiliary,
farray_register_global and register_report_aux. A field counts as missing if neither its
index variable nor its DSL name appears in any DSL file. For each missing field the
declarations are inserted above the MARKER line of (the marker is added if needed):

  DSL/fieldecs.h       -- every field
  DSL/solve.ac         -- PDE variables only (rk_final update)
  DSL/handwritten_end.h -- PDE variables only (rk_intermediate update)
  DSL/df_declares.h    -- PDE variables only (DF_ variable)

The generated names follow the conventions of parse.py:get_vtxbuf_name_from_index_base:
  f(:,:,:,iuub)          -> F_UUB           (index name without the leading i)
  f(:,:,:,iuubx:iuubz)   -> F_UUBVEC        (first index of range without leading i and trailing char + VEC)
  df(:,:,:,iaae+j)       -> DF_IAAE__MOD__DISP_CURRENT[j]
For vectors the DF variable is DF_<NAME> and the other names are #defined to it.

Can also be run standalone for checking:
  python3 field_declarations.py --dsl-dir $PENCIL_HOME/src/astaroth/DSL [--apply] file1.f90 file2.f90 ...
"""
import os
import re
import sys
import argparse

MARKER = "// @auto-field-declarations: fortran-parser adds missing fields above this line"
DSL_FILES = ["fieldecs.h", "solve.ac", "handwritten_end.h", "df_declares.h"]
#If a file has no MARKER yet it is put before (False) or after (True) these lines.
#None means the end of the file.
MARKER_ANCHORS = {
    "fieldecs.h": ('#include "$AC_HOME/acc-runtime/stdlib/map.h"', False),
    #last field update in twopass_solve_final
    "solve.ac": ("if(AC_iss_run_aver__mod__cdata != 0) write(F_SS_RUN_AVER,rk_final(F_SS_RUN_AVER,step_num,dt))", True),
    #last field update in the non-rkf branch
    "handwritten_end.h": ("if(AC_iss_run_aver__mod__cdata != 0) write(F_SS_RUN_AVER,rk_intermediate(F_SS_RUN_AVER,DF_SS_RUN_AVER,step_num,AC_dt__mod__cdata))", True),
    "df_declares.h": (None, True),
}

register_funcs = ["farray_register_pde", "farray_register_auxiliary", "farray_register_global", "register_report_aux"]
call_regex = re.compile(r"^\s*(?:if\s*\(.*\)\s*)?call\s+(" + "|".join(register_funcs) + r")\s*\((.*)\)\s*$", re.IGNORECASE)
identifier_regex = re.compile(r"^[a-z_][a-z0-9_]*$", re.IGNORECASE)

#Module name -> macro in PC_moduleflags.h if it is not simply the uppercased module name
module_flag_names = {
    "energy": "ENTROPY",
    "equationofstate": "EOS",
}


def strip_comment(line):
    in_string = None
    for i, char in enumerate(line):
        if in_string:
            if char == in_string:
                in_string = None
        elif char in "'\"":
            in_string = char
        elif char == "!":
            return line[:i]
    return line


def get_logical_lines(filename):
    """Lines without comments, continuation lines joined."""
    with open(filename, errors="replace") as file:
        raw_lines = file.readlines()
    res = []
    buffer = ""
    for line in raw_lines:
        line = strip_comment(line).strip()
        if buffer and line.startswith("&"):
            line = line[1:]
        if line.endswith("&"):
            buffer += line[:-1]
            continue
        buffer += line
        if buffer.strip():
            res.append(buffer.strip())
        buffer = ""
    return res


def split_args(args):
    res = []
    depth = 0
    in_string = None
    current = ""
    for char in args:
        if in_string:
            if char == in_string:
                in_string = None
        elif char in "'\"":
            in_string = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "," and depth == 0:
            res.append(current.strip())
            current = ""
            continue
        current += char
    if current.strip():
        res.append(current.strip())
    return res


def get_module_info(filename):
    """Returns (module name, module level variable names) of a Fortran file."""
    module = None
    variables = set()
    for line in get_logical_lines(filename):
        lower = line.lower()
        if module is None:
            match = re.match(r"^module\s+([a-z0-9_]+)\s*$", lower)
            if match:
                module = match.group(1)
            continue
        if lower == "contains":
            break
        if "::" in lower:
            for decl in split_args(lower.split("::", 1)[1]):
                name = decl.split("=")[0].split("(")[0].strip()
                if identifier_regex.match(name):
                    variables.add(name)
    #special modules are renamed to their file name during GPU builds
    if module == "special":
        module = os.path.splitext(os.path.basename(filename))[0].lower()
    return module, variables


class Registration:
    def __init__(self, func, index, module, ncomps, components, filename):
        self.func = func
        self.index = index
        self.module = module
        self.ncomps = ncomps
        #explicit component index variables (e.g. iuubx,iuuby,iuubz) if there are some
        self.components = components
        self.filename = filename

    @property
    def is_pde(self):
        return self.func == "farray_register_pde"

    def ac_name(self, var=None):
        return f"AC_{var or self.index}__mod__{self.module}"

    @property
    def name(self):
        return f"F_{self.index[1:].upper()}"

    @property
    def component_names(self):
        comps = self.components or [f"{self.index}{dim}" for dim in "xyz"]
        return [f"F_{comp[1:].upper()}" for comp in comps]

    @property
    def vec_name(self):
        first = self.components[0] if self.components else self.index
        return f"F_{first[1:-1].upper()}VEC"

    @property
    def guard(self):
        if self.module == "cdata":
            return None
        return f"L{module_flag_names.get(self.module, self.module.upper())}"


def get_registrations(files, cdata_variables):
    """
    Finds field registrations in the given files.
    The index variable has to be declared in cdata or at the module level of the registering file.
    """
    res = {}
    for filename in files:
        if not filename.endswith(".f90") or not os.path.isfile(filename):
            continue
        with open(filename, errors="replace") as file:
            if not any(func in file.read().lower() for func in register_funcs):
                continue
        own_module, own_variables = get_module_info(filename)
        for line in get_logical_lines(filename):
            match = call_regex.match(line)
            if not match:
                continue
            func = match.group(1).lower()
            args = split_args(match.group(2))
            if len(args) < 2:
                continue
            positional = [arg.lower() for arg in args if not re.match(r"^[a-z_]+\s*=", arg, re.IGNORECASE)]
            keywords = {arg.split("=")[0].strip().lower(): arg.split("=", 1)[1].strip().lower() for arg in args if re.match(r"^[a-z_]+\s*=", arg, re.IGNORECASE)}
            index = positional[1] if len(positional) > 1 else keywords.get("ind", keywords.get("index"))
            if index is None or not identifier_regex.match(index):
                continue

            components = None
            if func == "register_report_aux":
                aux = positional[2:5]
                if len(aux) == 0:
                    ncomps = 1
                elif len(aux) == 3 and all(identifier_regex.match(x) for x in aux):
                    ncomps = 3
                    components = aux
                else:
                    ncomps = len(aux)
            else:
                if "array" in keywords:
                    ncomps = -1
                elif "vector" in keywords:
                    ncomps = int(keywords["vector"]) if keywords["vector"].isnumeric() else -1
                else:
                    ncomps = 1

            if index in cdata_variables:
                module = "cdata"
            elif index in own_variables:
                module = own_module
            else:
                #e.g. dummy arguments of wrapper routines like register_report_aux
                module = None
            if module is None:
                continue
            if ncomps == 3 and components is None:
                #e.g. farray_register_pde('uu',iuu,vector=3) followed by iux = iuu etc.
                candidates = [f"{index}{dim}" for dim in "xyz"]
                if all(x in cdata_variables or x in own_variables for x in candidates):
                    components = candidates
            if ncomps not in [1, 3]:
                print(f"field_declarations: skipping {index} registered in {filename} with {ncomps} components (only scalars and vectors are supported)")
                continue
            if index not in res:
                res[index] = Registration(func, index, module, ncomps, components, filename)
            elif func == "farray_register_pde":
                #same index can be registered differently in different branches; pde takes precedence
                res[index].func = func
    return list(res.values())


def contains_word(text, word):
    return re.search(r"(?<![A-Za-z0-9_])" + re.escape(word) + r"(?![A-Za-z0-9_])", text) is not None


def declared_names(text):
    """Names of declared Fields and Field3s."""
    res = set()
    for line in text.split("\n"):
        line = line.split("//")[0]
        match = re.search(r"\bField3?\s+(.*?)(=|$)", line)
        if match:
            for name in match.group(1).split(","):
                name = name.split("[")[0].strip()
                if identifier_regex.match(name):
                    res.add(name)
    return res


def is_known(reg, text):
    """Whether the field already appears in text either through its index variable or its DSL names."""
    index_vars = [reg.index]
    if reg.components:
        index_vars.extend(reg.components)
    if reg.ncomps == 3:
        #e.g. iuu -> iux, iaa -> iax
        index_vars.extend([f"{reg.index[:-1]}{dim}" for dim in "xyz"])
    #a single letter suffix covers component indices like iuu_sphr of iuu_sph
    if any(re.search(r"\bAC_" + re.escape(var) + r"[a-z]?__mod__", text) for var in index_vars):
        return True
    names = [reg.name]
    if reg.ncomps == 3:
        names.append(reg.vec_name)
    return any(contains_word(text, name) for name in names)


def is_referenced(reg, code, sources):
    """
    Whether the generated DSL code uses the field through names that have to be declared.
    Accesses like Field(AC_iww1__mod__cdata-1) need no declarations. Writes to DF_ names count only for
    pushed PDEs, since otherwise the time update cannot be added.
    """
    names = [reg.name]
    if reg.ncomps == 3:
        names.extend([reg.vec_name] + reg.component_names)
    if any(contains_word(code, name) for name in names):
        return True
    df_names = [f"D{name}" for name in names] + [f"DF_{reg.index.upper()}__MOD__{reg.module.upper()}"]
    return reg.is_pde and is_pushed(reg, code, sources) and any(contains_word(code, name) for name in df_names)


def is_pushed(reg, code, sources):
    """
    Whether AC_<index>__mod__<module> will exist on the GPU side: it is pushed if it is used in the
    generated code (MODIFY_SOURCE_CODE adds it to the pushpars) or if it is already in the pushpars.
    """
    return re.search(r"\bAC_" + re.escape(reg.index) + r"__mod__", code) is not None \
        or re.search(r"\bcopy_addr\s*\(\s*" + re.escape(reg.index) + r"\s*,", sources, re.IGNORECASE) is not None


def guarded(reg, lines):
    if reg.guard is None:
        return lines
    return [f"#if {reg.guard}"] + lines + ["#endif"]


def gen_fieldecs(reg, declared):
    ac = reg.ac_name()
    if reg.ncomps == 1:
        return guarded(reg, [f"field_order({ac}-1) Field {reg.name}"])
    comps = reg.component_names
    lines = [f"field_order({ac} != 0 ? {ac}+{i}-1 : -1) Field {comp}" for i, comp in enumerate(comps)]
    #F_<NAME> is only needed for the time update of PDEs in solve.ac and handwritten_end.h
    vec_names = [reg.name, reg.vec_name] if reg.is_pde else [reg.vec_name]
    for name in unique(vec_names):
        if name not in declared:
            lines.append(f"const Field3 {name} = {{{', '.join(comps)}}}")
    return guarded(reg, lines)


def df_name(reg):
    return f"D{reg.name}"


def gen_solve(reg):
    name = reg.name
    return guarded(reg, [f"\t\tif({reg.ac_name()} != 0) write({name}, rk_final({name},step_num,dt))"])


def gen_handwritten_end(reg):
    name = reg.name
    return guarded(reg, [f"          if({reg.ac_name()} != 0) write({name}, rk_intermediate({name},{df_name(reg)},step_num,AC_dt__mod__cdata))"])


def gen_df_declares(reg, used_df_names):
    df = df_name(reg)
    used_df_names.add(df)
    if reg.ncomps == 1:
        return guarded(reg, [f"real {df} = rk_intermediate_split_first({reg.name},step_num)"])
    lines = [f"real3 {df} = rk_intermediate_split_first({reg.name},step_num)"]
    #the other names the transpiler can generate for df accesses of this field:
    #df(:,:,:,iaae+j), df(:,:,:,iuubx:iuubz) and df(:,:,:,iuubx)
    aliases = [(f"DF_{reg.index.upper()}__MOD__{reg.module.upper()}", df), (f"D{reg.vec_name}", df)]
    aliases.extend([(f"D{comp}", f"{df}.{dim}") for comp, dim in zip(reg.component_names, "xyz")])
    for alias, value in aliases:
        #e.g. iww1 and iww2 would both give DF_WWVEC
        if alias not in used_df_names:
            used_df_names.add(alias)
            lines.append(f"#define {alias} {value}")
    return guarded(reg, lines)


def unique(lst):
    res = []
    for x in lst:
        if x not in res:
            res.append(x)
    return res


def add_marker(text, file):
    anchor, after = MARKER_ANCHORS[file]
    if anchor is None:
        return text.rstrip("\n") + "\n" + MARKER + "\n"
    if text.count(anchor) != 1:
        return None
    return text.replace(anchor, f"{anchor}\n{MARKER}" if after else f"{MARKER}\n{anchor}")


def insert_before_marker(filename, lines, apply):
    with open(filename) as file:
        text = file.read()
    if MARKER not in text:
        text = add_marker(text, os.path.basename(filename))
        if text is None:
            print(f"field_declarations: WARNING: could not find where to add fields in {filename}, add the line\n{MARKER}\nwhere the declarations should go")
            return False
    if not apply:
        return True
    new_text = text.replace(MARKER, "\n".join(lines) + "\n" + MARKER, 1)
    with open(filename, "w") as file:
        file.write(new_text)
    return True


def read_all_dsl_files(dsl_dir):
    """Fields are declared also outside of fieldecs.h e.g. in module specific headers."""
    texts = []
    for root, dirs, files in os.walk(dsl_dir):
        #local contains the generated per sample files
        dirs[:] = [d for d in dirs if d != "local"]
        for file in sorted(files):
            if file.endswith((".h", ".ac")):
                with open(os.path.join(root, file), errors="replace") as f:
                    texts.append(f.read())
    return "\n".join(texts)


def add_missing_field_declarations(files, dsl_dir, code=None, apply=True):
    """
    Adds DSL declarations for the fields registered in files that are missing from the DSL files in dsl_dir.
    If code (the generated DSL code) is given only the fields it uses are added: the DSL files are shared
    by all samples so everything added has to compile also for the others.
    Returns the list of added registrations.
    """
    files = [file for file in files if file.endswith(".f90") and os.path.isfile(file)]
    cdata_file = next((file for file in files if os.path.basename(file) == "cdata.f90"), None)
    cdata_variables = get_module_info(cdata_file)[1] if cdata_file else set()
    registrations = get_registrations(files, cdata_variables)
    sources = ""
    for file in files:
        with open(file, errors="replace") as f:
            sources += f.read()
    if code is not None:
        registrations = [reg for reg in registrations if is_referenced(reg, code, sources)]

    texts = {}
    for file in DSL_FILES:
        with open(os.path.join(dsl_dir, file)) as f:
            texts[file] = f.read()
    all_dsl = read_all_dsl_files(dsl_dir)
    declared = declared_names(all_dsl)

    new_lines = {file: [] for file in DSL_FILES}
    used_df_names = set(re.findall(r"\bDF_[A-Za-z0-9_]+", texts["df_declares.h"]))
    added = []
    for reg in registrations:
        #Fields already known to the DSL are left alone: some PDEs are deliberately not updated in
        #solve.ac/handwritten_end.h since other kernels take care of them (e.g. gravitational waves)
        if is_known(reg, all_dsl):
            continue
        #names that are declared for something else would clash
        clashes = [name for name in ([reg.name] + (reg.component_names if reg.ncomps == 3 else [])) if name in declared]
        if clashes:
            print(f"field_declarations: WARNING: cannot declare {reg.index} ({reg.filename}) since {clashes} already declared")
            continue
        new_lines["fieldecs.h"].extend(gen_fieldecs(reg, declared))
        declared.update([reg.name, reg.vec_name] + reg.component_names)
        if reg.is_pde and code is not None and not is_pushed(reg, code, sources):
            print(f"field_declarations: WARNING: {reg.index} is not pushed to the GPU so cannot add its time update to solve.ac and handwritten_end.h, add it to the pushpars")
            if not is_known(reg, texts["df_declares.h"]):
                new_lines["df_declares.h"].extend(gen_df_declares(reg, used_df_names))
        elif reg.is_pde:
            for file, gen in [("solve.ac", gen_solve), ("handwritten_end.h", gen_handwritten_end), ("df_declares.h", lambda reg: gen_df_declares(reg, used_df_names))]:
                if not is_known(reg, texts[file]) and not contains_word(texts[file], df_name(reg)):
                    new_lines[file].extend(gen(reg))
        added.append(reg)

    for file in DSL_FILES:
        if new_lines[file]:
            print(f"field_declarations: {'adding' if apply else 'would add'} to {os.path.join(dsl_dir, file)}:")
            for line in new_lines[file]:
                print(f"    {line}")
            insert_before_marker(os.path.join(dsl_dir, file), new_lines[file], apply)
    return added


def default_dsl_dir():
    pencil_home = os.getenv("PENCIL_HOME")
    if pencil_home is None:
        pencil_home = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(pencil_home, "src", "astaroth", "DSL")


def main():
    argparser = argparse.ArgumentParser(description="Add missing field declarations to the Astaroth DSL files",
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--dsl-dir", default=default_dsl_dir(), help="Directory of fieldecs.h, solve.ac etc.")
    argparser.add_argument("--apply", default=False, action="store_true", help="Write the declarations (default is dry run)")
    argparser.add_argument("--code", help="Generated DSL code (e.g. DSL/local/rhs.h), only fields used in it are added")
    argparser.add_argument("files", nargs="+", help="Fortran files to search for field registrations")
    args = argparser.parse_args()
    code = None
    if args.code:
        with open(args.code) as file:
            code = file.read()
    add_missing_field_declarations(args.files, args.dsl_dir, code, apply=args.apply)


if __name__ == "__main__":
    sys.exit(main())
