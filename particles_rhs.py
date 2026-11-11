"""Back end of the transpiler for the right-hand sides of the particle equations.

The front end of parse.py inlines dxxp_dt/dvvp_dt and resolves every name to name__mod__module.
Here the resulting loops over the particles become the body of one DSL kernel over the particles:
fp(k,i) -> AC_fp_*[k], dfp(k,i) -> local right-hand sides, stored to AC_par_rhs at the end.
Interpolation is a primitive implemented by hand in DSL/particles.h since it depends on the data layout.
"""
import os
import re
import subprocess
import sys

# Calls that are kept as primitives by the inliner
PRIMITIVES = ["interpolate_linear", "interpolate_linear_range", "interpolate_linear_scalar", "interpolate_quadratic_spline",
              "interpolate_quadratic", "get_frictiontime", "get_rhopswarm", "calc_diagnostics_particles", "particle_gravity",
              "calc_diagnostics_from_particles_module", "calc_diagnostics_particles_selg", "calc_relative_velocity",
              "diffuse_backreaction", "dcaustics_dt", "dtetrad_dt", "dvvp_dt_potential", "indirect_inertial_particles", "count_particles",
              "max_name", "sum_name", "save_name", "max_par_name", "sum_par_name", "integrate_par_name", "sum_mn_name", "max_mn_name"]

FP_INDEX = {"ixp": ("xp", 0), "iyp": ("yp", 1), "izp": ("zp", 2), "ivpx": ("vpx", 3), "ivpy": ("vpy", 4), "ivpz": ("vpz", 5)}
RHS_NAMES = ["rhs_xp", "rhs_yp", "rhs_zp", "rhs_vpx", "rhs_vpy", "rhs_vpz"]
COMP = {"1": "x", "2": "y", "3": "z"}
# Fields that interpolation primitives can return, as (first index, DSL function suffix)
INTERP_RANGES = {"iux": "gas_u", "igpotselfx": "gpot"}


# Known in every kernel: headers and diagnostics are done on the host, dt1 is only used when the host asks for it
DEFAULT_ASSUME = {"headtt__mod__cdata": False, "lupdate_courant_dt__mod__cdata": True, "ldiagnos__mod__cdata": False,
                  "lfirstcall": False, "lheader": False, "lpointmasses__mod__cparam": False, "lparticle_gravity__mod__particles": False,
                  "lparticles_blocks__mod__cparam": False, "lselfgravity_particles__mod__particles_selfgravity": True,
                  "lcorotational_frame__mod__cdata": False, "lcorotational_frame__mod__particles_cdata": False,
                  "lnocollapse_xdir_onecell__mod__particles_cdata": False, "lnocollapse_ydir_onecell__mod__particles_cdata": False,
                  "lnocollapse_zdir_onecell__mod__particles_cdata": False}
# Error checks: the same conditions are rejected at startup by check_gpu_support
# Followed by the optional modules that check_gpu_support rejects, or gas-only work
ERROR_CALLS = ["fatal_error", "fatal_error_local", "not_implemented", "warning", "diffuse_backreaction", "dcaustics_dt", "dtetrad_dt",
               "dvvp_dt_potential", "calc_relative_velocity", "count_particles"]


ACTIVE_ASSUME = {}


class Unsupported(Exception):
    pass


def base(name):
    return name.split("__mod__")[0]


def find_close(s, i):
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "(":
            depth += 1
        elif s[j] == ")":
            depth -= 1
            if depth == 0:
                return j
    raise Unsupported(f"unbalanced parentheses: {s}")


def split_top(s):
    parts, depth, cur = [], 0, ""
    for c in s:
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        if c == "," and depth == 0:
            parts.append(cur.strip())
            cur = ""
        else:
            cur += c
    parts.append(cur.strip())
    return parts


def fp_ref(args, array):
    """Translates the argument list of fp(...) or dfp(...)"""
    k, idx = split_top(args)
    if k not in ["k", "1:npar_loc__mod__particles_cdata", "1:npar_loc"]:
        raise Unsupported(f"particle index {k} of {array}")
    if ":" in idx:
        first, last = [base(x) for x in idx.split(":")]
        if array == "fp" and (first, last) in [("ixp", "izp"), ("ivpx", "ivpz")]:
            names = ["AC_fp_%s[k]" % FP_INDEX[n][0] for n in (["ixp", "iyp", "izp"] if first == "ixp" else ["ivpx", "ivpy", "ivpz"])]
            return "real3(%s)" % ",".join(names)
        if array == "dfp" and (first, last) in [("ixp", "izp"), ("ivpx", "ivpz")]:
            start = FP_INDEX[first][1]
            return "real3(%s)" % ",".join(RHS_NAMES[start:start+3])
        raise Unsupported(f"section {idx} of {array}")
    b = base(idx)
    if array == "fp" and b == "iaps":
        return "AC_fp_aps[k]"
    if b not in FP_INDEX:
        raise Unsupported(f"index {idx} of {array}")
    return ("AC_fp_%s[k]" % FP_INDEX[b][0]) if array == "fp" else RHS_NAMES[FP_INDEX[b][1]]


def replace_refs(e):
    for array in ["dfp", "fp"]:
        pat = re.compile(r"\b%s\(" % array)
        pos = 0
        while True:
            m = pat.search(e, pos)
            if not m:
                break
            close = find_close(e, m.end()-1)
            new = fp_ref(e[m.end():close], array)
            e = e[:m.start()] + new + e[close+1:]
            pos = m.start() + len(new)
    return e


def powers(e):
    # x**n for an integer n, with x an identifier, an array element or a parenthesized expression
    while "**" in e:
        i = e.index("**")
        m = re.match(r"\s*(\d+)(?:\.0*)?(?![\w.])", e[i+2:])
        j = i
        if e[j-1] in ")]":
            open_c = "(" if e[j-1] == ")" else "["
            close_c = e[j-1]
            depth, j = 0, j-1
            while j >= 0:
                if e[j] == close_c:
                    depth += 1
                elif e[j] == open_c:
                    depth -= 1
                    if depth == 0:
                        break
                j -= 1
            while j > 0 and re.match(r"[\w.]", e[j-1]):
                j -= 1
        else:
            while j > 0 and re.match(r"[\w.]", e[j-1]):
                j -= 1
        x = e[j:i]
        if m:
            e = e[:j] + "(" + "*".join([x]*int(m.group(1))) + ")" + e[i+2+m.end():]
            continue
        # general exponent: a parenthesized expression or a name
        rest = e[i+2:].lstrip()
        offset = len(e) - len(rest)
        end = find_close(e, offset) + 1 if rest.startswith("(") else offset + re.match(r"[\w.]+", rest).end()
        e = e[:j] + f"pow({x},{e[offset:end]})" + e[end:]
    return e


def expr(e, vectors):
    e = replace_refs(e)
    for name, val in ACTIVE_ASSUME.items():
        e = re.sub(r"(?<![\w.])%s\b" % re.escape(name), "true" if val else "false", e)
    e = e.replace("1:npar_loc__mod__particles_cdata", "k").replace("1:npar_loc", "k")
    e = re.sub(r"\bipar__mod__particles_cdata\(k\)", "AC_fp_ipar[k]", e)
    e = re.sub(r"\bineargrid\(k,([123])\)", lambda m: f"particle_ineargrid_{COMP[m.group(1)]}(k)", e)
    e = re.sub(r"\b(?:ineargrid_part|ineargrid_point|ing_comp)\(([123])\)", lambda m: f"particle_ineargrid_{COMP[m.group(1)]}(k)", e)
    e = re.sub(r"\bt__mod__cdata\b", "t_now", e)
    for v in vectors:
        e = re.sub(r"\b%s\(([123])\)" % v, lambda m: f"{v}.{COMP[m.group(1)]}", e)
    e = re.sub(r"\.and\.", " && ", e)
    e = re.sub(r"\.or\.", " || ", e)
    e = re.sub(r"\.not\.", " !", e)
    e = e.replace(".true.", "true").replace(".false.", "false")
    e = e.replace("/=", "!=")
    e = re.sub(r"\b(\d+\.\d*|\.\d+|\d+)[dD]([+-]?\d+)", r"\1e\2", e)
    e = powers(e)
    e = re.sub(r"(?<![\w.])([a-z_]\w*__mod__\w+)\b", r"AC_\1", e)
    # 1-based host grid arrays, the ineargrid primitives return 1-based indices
    # compile-time parameters are plain names in the DSL, as in rhs.h
    e = re.sub(r"\bAC_(\w+?)__mod__cparam\b", r"\1", e)
    return grid_arrays(e)


def grid_arrays(e):
    # the block grids are the same grid as the host one on the device
    e = re.sub(r"\bAC_(dx|dy|dz)1b__mod__\w+\(([^,()]*),[^()]*\)", lambda m: f"AC_{m.group(1)}_1__mod__cdata[({m.group(2)})-1]", e)
    e = re.sub(r"\bAC_([xyz])b__mod__\w+\(([^,()]*),[^()]*\)", lambda m: f"AC_{m.group(1)}__mod__cdata[({m.group(2)})-1]", e)
    pat = re.compile(r"\bAC_(x|y|z|dx_1|dy_1|dz_1)__mod__cdata\(")
    while True:
        m = pat.search(e)
        if not m:
            return e
        close = find_close(e, m.end()-1)
        e = e[:m.start()] + f"AC_{m.group(1)}__mod__cdata[({e[m.end():close]})-1]" + e[close+1:]


def interp_call(args, kind, vectors):
    a = split_top(args)
    ivar1, ivar2, gp = base(a[1]), base(a[2]), a[4]
    if ivar1 not in INTERP_RANGES:
        raise Unsupported(f"interpolation of {a[1]}")
    if gp not in vectors:
        raise Unsupported(f"interpolation of one component into {gp}")
    return f"{gp} = particle_{kind}_{INTERP_RANGES[ivar1]}(k)"


def is_dropped(line):
    l = line.strip()
    if re.match(r"call\s+(%s)\b" % "|".join(ERROR_CALLS), l):
        return True
    return (l.startswith("print") or "keep_compiler_quiet" in l or re.match(r"(call\s+)?calc_diagnostics", l) is not None
            or l.startswith("intent") or l.startswith("use ") or re.match(r"[\w,\s()]*::", l) is not None or l == "return")


def eval_cond(c, assume):
    """True/False if the condition is decided by the assumed flags, otherwise None"""
    c = c.strip()
    while c.startswith("(") and find_close(c, 0) == len(c)-1:
        c = c[1:-1].strip()
    #inside the kernel over the particles there is always a particle
    if re.match(r"npar_(imn|iblock|loc)\w*(\(.*\))?\s*(/=|>)\s*0$", c):
        return True
    if ".or." in c or ".and." in c:
        # only conjunctions and disjunctions without nested parentheses are folded
        if "(" in c and ".and." in c and ".or." in c:
            return None
    parts = split_op(c, ".and.")
    if len(parts) > 1:
        vals = [eval_cond(p, assume) for p in parts]
        if any(v is False for v in vals):
            return False
        return True if all(v is True for v in vals) else None
    parts = split_op(c, ".or.")
    if len(parts) > 1:
        vals = [eval_cond(p, assume) for p in parts]
        if any(v is True for v in vals):
            return True
        return False if all(v is False for v in vals) else None
    if c.startswith(".not."):
        v = eval_cond(c[5:], assume)
        return None if v is None else (not v)
    m = re.match(r"(\w+)\s*==\s*enum_(\w*)_string(?:__mod__cparam)?$", c)
    if m and isinstance(assume.get(m.group(1)), str):
        return assume[m.group(1)] == m.group(2)
    if c == ".true.":
        return True
    if c == ".false.":
        return False
    return assume.get(c)


def split_op(c, op):
    parts, depth, cur, i = [], 0, "", 0
    while i < len(c):
        if c[i] == "(":
            depth += 1
        elif c[i] == ")":
            depth -= 1
        if depth == 0 and c.startswith(op, i):
            parts.append(cur)
            cur = ""
            i += len(op)
            continue
        cur += c[i]
        i += 1
    parts.append(cur)
    return parts


def split_if(l):
    m = re.match(r"(?:else\s*)?if\s*\(", l)
    close = find_close(l, l.index("("))
    return l[l.index("(")+1:close], l[close+1:].strip()


def fold_branches(lines, assume):
    """Removes branches decided by the assumed flags and the if blocks that are left empty"""
    out, i = [], 0
    while i < len(lines):
        l = lines[i].strip()
        if not re.match(r"if\s*\(.*\)\s*then$", l):
            if re.match(r"if\s*\(", l):
                c, stmt = split_if(l)
                v = eval_cond(c, assume)
                if v is True:
                    out.append(stmt)
                elif v is None:
                    out.append(l)
            else:
                out.append(l)
            i += 1
            continue
        # block if: collect branches at depth 0
        depth, j, branches, start, conds = 1, i+1, [], i+1, [split_if(l)[0]]
        while depth > 0:
            t = lines[j].strip()
            if re.match(r"if\s*\(.*\)\s*then$", t):
                depth += 1
            elif re.match(r"end\s*if$", t):
                depth -= 1
            if depth == 1 and (re.match(r"else\s*if\s*\(", t) or t == "else"):
                branches.append(lines[start:j])
                conds.append(split_if(t)[0] if t != "else" else None)
                start = j+1
            j += 1
        branches.append(lines[start:j-1])
        kept = []
        for c, b in zip(conds, branches):
            v = True if c is None else eval_cond(c, assume)
            if v is False:
                continue
            b = fold_branches(b, assume)
            kept.append((c if v is None else None, b))
            if v is True:
                break
        while kept and kept[-1][1] == [] :
            kept.pop()
        if kept:
            if kept[0][0] is None:
                out.extend(kept[0][1])
            else:
                for n, (c, b) in enumerate(kept):
                    if n == 0:
                        out.append(f"if ({c}) then")
                    elif c is None:
                        out.append("else")
                    else:
                        out.append(f"elseif ({c}) then")
                    out.extend(b)
                out.append("endif")
        i = j
    return out


def translate(lines, assume):
    ACTIVE_ASSUME.clear()
    ACTIVE_ASSUME.update({**DEFAULT_ASSUME, **assume})
    vectors, scalars = [], []
    body = []
    for raw in lines:
        l = raw.strip()
        if not l or l.startswith("subroutine") or l.startswith("endsubroutine") or l.startswith("end subroutine"):
            continue
        m = re.match(r"(real|integer|logical)\s*(,\s*dimension\s*\((.*?)\))?\s*(,[^:]*)?::\s*(.*)$", l)
        if m:
            kind, dims = m.group(1), (m.group(3) or "").replace(" ", "")
            for name in split_top(m.group(5)):
                name = name.split("=")[0].strip()
                if name in ("f", "df", "fp", "dfp", "ineargrid", "p", "k") or name.startswith("ac_"):
                    continue
                if dims == "3":
                    vectors.append(name)
                elif dims == "":
                    scalars.append((name, kind))
            continue
        if l.startswith("complex") or l.startswith("type") or l.startswith("character"):
            continue
        if not is_dropped(l):
            body.append(l)
    body = fold_branches(body, {**DEFAULT_ASSUME, **assume})
    out, kept_enddo = [], []
    for l in body:
        m = re.match(r"do\s+(\w+)\s*=\s*(.*)$", l)
        if m:
            if m.group(1) in ("k", "iblock"):
                kept_enddo.append(False)
                continue
            a, b = split_top(m.group(2))[:2]
            kept_enddo.append(True)
            out.append(f"for {m.group(1)} in {expr(a, vectors)}:({expr(b, vectors)})+1 {{")
            continue
        if re.match(r"end\s*do$", l):
            if kept_enddo.pop():
                out.append("}")
            continue
        if re.match(r"if\s*\(.*\)\s*then$", l):
            out.append(f"if ({expr(split_if(l)[0], vectors)}) {{")
        elif re.match(r"else\s*if\s*\(.*\)\s*then$", l):
            out.append(f"}} else if ({expr(split_if(l)[0], vectors)}) {{")
        elif l == "else":
            out.append("} else {")
        elif re.match(r"end\s*if$", l):
            out.append("}")
        elif re.match(r"if\s*\(", l):
            c, stmt = split_if(l)
            translated = statement(stmt, vectors)
            if translated:
                out.append(f"if ({expr(c, vectors)}) {translated}")
        else:
            s = statement(l, vectors)
            if s:
                out.append(s)
    # locals that are only assigned are not needed (e.g. the coordinates passed to the interpolation calls)
    for v in vectors + [n for n, _ in scalars]:
        if not any(re.search(r"\b%s\b" % v, o.split("=", 1)[1]) if "=" in o else re.search(r"\b%s\b" % v, o) for o in out):
            out = [o for o in out if not re.match(r"%s\s*=" % v, o)]
    used = "\n".join(out)
    decls = [f"real3 {v} = real3(0.0,0.0,0.0)" for v in vectors if re.search(r"\b%s\b" % v, used)]
    for name, kind in scalars:
        if re.search(r"\b%s\b" % name, used):
            decls.append({"real": f"real {name} = 0.0", "integer": f"int {name} = 0", "logical": f"bool {name} = false"}[kind])
    return decls + out


def statement(l, vectors):
    m = re.match(r"call\s+(\w+)\s*\((.*)\)$", l)
    if m:
        name = m.group(1)
        if name.startswith("interpolate_linear"):
            return interp_call(m.group(2), "linear", vectors)
        if name == "interpolate_quadratic_spline":
            return interp_call(m.group(2), "tsc", vectors)
        if name == "get_frictiontime":
            a = split_top(m.group(2))
            return f"{a[4]} = particle_tausp1(particle_species(AC_fp_ipar[k]))\nAC_par_tausp1_k[k] = {a[4]}"
        raise Unsupported(f"call of {name}")
    m = re.match(r"([^=]*?)\s*=(?!=)\s*(.*)$", l)
    if not m:
        raise Unsupported(l)
    lhs, rhs = m.group(1).strip(), m.group(2)
    if lhs in ACTIVE_ASSUME:
        return ""
    if re.match(r"dt1_max__mod__cdata\(", lhs):
        start = rhs.index("(")
        a = split_top(rhs[start+1:find_close(rhs, start)])
        return f"par_dt1 = max(par_dt1,{expr(a[1], vectors)})"
    # whole-vector copies of the nearest grid point are replaced by the index primitives in expr
    if lhs in ("ineargrid_point", "ineargrid_part", "ing_comp") or re.match(r"ineargrid_(point|part)\s*$", lhs):
        return ""
    if re.match(r"dfp\(", lhs) and ":" in lhs.split(",", 1)[1]:
        target = replace_refs(lhs)[len("real3("):-1].split(",")
        tmp = f"dfp_vec_{statement.counter}"
        statement.counter += 1
        return "\n".join([f"real3 {tmp} = {expr(rhs, vectors)}"] + [f"{t} = {tmp}.{c}" for t, c in zip(target, "xyz")])
    result = f"{expr(lhs, vectors)} = {expr(rhs, vectors)}"
    if lhs == "dragforce":   # for the back-reaction on the gas, hand-written in the gather kernels
        result += "".join(f"\nAC_par_drag_{c}[k] = dragforce.{c}" for c in "xyz")
    return result


statement.counter = 0


def compile_time_flags(cparam_file="../cparam.inc"):
    """Logical parameters fixed at compile time, e.g. lshear in cparam.inc"""
    res = {}
    with open(cparam_file) as f:
        for line in f:
            if re.match(r"\s*logical\b.*\bparameter\b", line, re.I):
                for name, val in re.findall(r"(\w+)\s*=\s*\.(true|false)\.", line, re.I):
                    res[f"{name.lower()}__mod__cparam"] = val.lower() == "true"
    return res


def write_kernel_body(filename, lines, assume):
    body = translate(lines, {**compile_time_flags(), **assume})
    with open(filename, "w") as f:
        f.write("//Generated by fortran-parser/particles_rhs.py, do not edit\nif (k >= 0)\n{\n")
        for l in body:
            f.write(f"\t{l}\n")
        f.write("}\n")


def routines(flags_file="PC_moduleflags.h"):
    """Fortran routines making up the particle right-hand side as (routine, module, flags known at compile time),
    with the flags fixed by check_gpu_support. Empty without particles."""
    flags = open(flags_file).read() if os.path.exists(flags_file) else ""
    if "LPARTICLES_TRACERS 1" in flags:
        assume = "ltrace_dust__mod__particles=F,lparticlemesh_tsc__mod__particles_cdata=F"
        return [("dxxp_dt_pencil", "particles_tracers", assume)]
    if "LPARTICLES_NBODY 1" in flags:
        assume = "lparticlemesh_tsc__mod__particles_cdata=F,lparticlemesh_cic__mod__particles_cdata=T"
        return [("dxxp_dt", "particles_dust", assume), ("dxxp_dt_pencil", "particles_dust", assume),
                ("dvvp_dt", "particles_dust", assume), ("dvvp_dt_selfgrav", "particles_selfgravity", assume)]
    if "LPARTICLES_DUST 1" in flags:
        assume = "lparticlemesh_tsc__mod__particles_cdata=T,lparticlemesh_cic__mod__particles_cdata=F,linterpolate_spline__mod__particles_cdata=T," \
                 "gravx_profile__mod__particles=S:,gravz_profile__mod__particles=S:linear,gravr_profile__mod__particles=S:,lshearadvection_as_shift__mod__shear=F"
        # The back-reaction on the gas and the friction time step are computed on the grid in DSL/particles.h
        drag = ",ldragforce_gas_par__mod__particles=F,lupdate_courant_dt__mod__cdata=F"
        return [("dxxp_dt", "particles_dust_blocks", assume), ("dxxp_dt_blocks", "particles_dust_blocks", assume),
                ("dvvp_dt", "particles_dust_blocks", assume), ("dvvp_dt_blocks", "particles_dust_blocks", assume + drag)]
    return []


def generate_kernel(sample_dir, src_dir, out="particles_rhs.h"):
    """Writes the particles_rhs kernel to out, an empty file if the sample does not use particles.
    Each routine is transpiled by a separate run of parse.py with --particles-rhs so that they do not share parser state."""
    parts = routines()
    if not parts:
        open(out, "w").close()
        return False
    parse_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parse.py")
    res = ["//Generated at compilation time from the Fortran right-hand sides, see fortran-parser/particles_rhs.py",
           "Kernel particles_rhs(real t_now)", "{", "\tsuppress_unused_warning(t_now)", "\tconst int k = vertexIdx.x", "\treal par_dt1 = 0.0"]
    res += [f"\treal {name} = 0.0" for name in RHS_NAMES]
    res += ["#if LPARTICLES_DUST", "#if LPARTICLES_NBODY", "#else", "\tAC_par_drag_x[k] = 0.0", "\tAC_par_drag_y[k] = 0.0",
            "\tAC_par_drag_z[k] = 0.0", "\tAC_par_tausp1_k[k] = 0.0", "#endif", "#endif"]
    for routine, module, assume in parts:
        part = f"particles_rhs_{routine}.h"
        subprocess.run([sys.executable, parse_py, "-f", routine, "-F", f"{src_dir}/{module}.f90", "-o", "-s",
                        "--sample-dir", sample_dir, "--dir", src_dir, "--particles-rhs", part, "--particles-assume", assume], check=True)
        with open(part) as f:
            res += f.read().split("\n")[1:]
        os.remove(part)
    res += [f"\tAC_par_rhs[{i}][k] = {name}" for i, name in enumerate(RHS_NAMES)]
    res += ["#if LPARTICLES_DUST", "\tAC_par_dt1[k] = par_dt1", "#else", "\tsuppress_unused_warning(par_dt1)", "#endif", "}"]
    with open(out, "w") as f:
        f.write("\n".join(res) + "\n")
    return True
