#include <vector>
#include <stdlib.h>
#include <unordered_map>
#include <string>
#include <algorithm>

typedef enum OffloadType{
  BOUNDCOND,
  STENCIL,
};
typedef struct Var{

};
typedef struct FuncCall{

};
typedef struct FileInfo{
  bool is_program_file;
};

typedef struct ModuleInfo{

};

typedef struct FuncInfo{

};


typedef struct Write{

};
typedef struct Config{
  bool offload;
  bool stencil;
  bool boundcond;
  bool diagnostics;
  std::string sample_dir;
  std::string directory;
  bool to_c;
};
std::unordered_map<std::string, std::string>
get_chosen_modules(const std::string filename)
{
  return {};
}
template <typename T>
bool
contains(std::vector<T> vec, T x)
{
  return std::find(vec.begin(), vec.end(), x) != vec.end();
}
class Parser{
  public:
    std::unordered_map<std::string,std::string> pde_names{};
    std::vector<FuncCall> select_eos_variable_calls{};
    std::vector<FuncCall> farray_register_calls{};
    std::unordered_map<std::string, std::vector<std::string>> shared_flags_accessed{};
    std::unordered_map<std::string, std::vector<std::string>> shared_flags_given{};
    bool offloading;
    bool include_diagnostics;
    bool test_to_c;
    std::unordered_map<std::string,std::string> known_values{};
    int inline_num = 0;
    std::string sample_dir;
    OffloadType offload_type;
    std::vector<std::string> ranges{};
    std::unordered_map<std::string,std::string> flag_mappings{};
    std::unordered_map<std::string,std::string> default_mappings{};
    std::unordered_map<std::string,FuncInfo> func_info{};
    std::unordered_map<std::string,FileInfo> file_info{};
    std::unordered_map<std::string,ModuleInfo> module_info{};
    std::unordered_map<std::string,Var> static_variables{};
    std::vector<std::string> safe_subs_to_remove{};
    std::unordered_map<std::string,std::string> rename_dict{};
    std::unordered_map<std::string,std::vector<std::string>> lines{};
    std::vector<std::string> parsed_files_for_static_variables{};
    std::vector<std::string> parsed_modules{};
    std::vector<std::string> parsed_subroutines{};
    std::vector<std::string> loaded_files{};
    int subroutine_order = 0;
    std::vector<std::string> used_files{};
    std::vector<FuncCall> found_func_calls{};
    std::vector<std::string> used_static_variables{};
    std::unordered_map<std::string,std::string> functions_in_file{};
    std::vector<Write> static_writes{};
    std::string directory;
    std::unordered_map<std::string,std::string> subroutine_modifies_param{};
    std::unordered_map<std::string, std::unordered_map<std::string,Var>> struct_table{};
    std::unordered_map<std::string,Var> module_variables{};
    std::vector<std::string> ignored_files{};
    std::vector<std::string> ignored_modules{};
    std::vector<std::string> ignored_subroutines{};
    std::unordered_map<std::string,std::string> chosen_modules;
    std::string main_program;
    std::vector<std::string> not_chosen_files;

    void
    get_lines(const std::string filename)
    {
      return;
    }
    Parser(std::vector<std::string> files, Config config){
      offloading = config.offload;
      include_diagnostics = config.diagnostics;
      test_to_c = config.to_c;
      sample_dir = config.sample_dir;
      if(config.stencil)
        offload_type = STENCIL;
      if(config.boundcond)
        offload_type = BOUNDCOND;
      default_mappings["headtt__mod__cdata"] = ".false.";
      default_mappings["ldebug__mod__cdata"] = ".false.";
      default_mappings["l1dphiavg__mod__cdata"] = ".false.";
      default_mappings["lwrite_phiaverages__mod__cdata"] = ".false.";
      default_mappings["ltime_integrals_phiaverages__mod__cdata"] = ".false.";
      used_files = files;
      directory = config.directory;
      std::unordered_map<std::string,std::string> default_modules = get_chosen_modules(sample_dir + "/src/Makefile.src");
      chosen_modules = get_chosen_modules(sample_dir + "/src/Makefile.src");
      for (auto &it : default_modules)
      {
        if (chosen_modules.find(it.first) == chosen_modules.end())
          chosen_modules[it.first] = it.second;
      }
      ignored_modules = {"hdf5"};
      ignored_files =  {"nodebug.f90","/boundcond_examples/","deriv_alt.f90","boundcond_alt.f90", "diagnostics_outlog.f90","pscalar.f90", "/cuda/", "/obsolete/", "/inactive/", "/astaroth/", "/pre_and_post_processing/", "/scripts/"};
      ignored_files.push_back("magnetic_ffreeMHDrel.f90");
      ignored_files.push_back("photoelectric_dust.f90");
      ignored_files.push_back("interstellar_old.f90");
      ignored_files.push_back("spiegel.f90");
      //used_files = 
      //self.used_files = [file for file in self.used_files if not any([ignored_file in file for ignored_file in self.ignored_files])  and ".f90" in file]
      main_program = directory + "/run.f90";
      for(auto file : used_files){
        get_lines(file);
      }
      used_files.erase(std::remove_if(
        used_files.begin(), used_files.end(),
        [](const std::string& filename) { 
            return file_info[filename].is_program_file or contains(not_chosen_files, x);
        }), used_files.end());
      used_files.push_back(main_program);
      ignored_subroutines = {
          "alog10","count", "min1", "erf","aimag", "cmplx","len", "inquire", "floor", "matmul","ceiling", "achar", "adjustl", "index", "iabs","tiny","dble","float","nullify","associated","nint","open","close","epsilon","random_seed","modulo","nearest","xor","ishft","iand","ieor","ior","random_number","all","any","deallocate","cshift","allocated","allocate","case","real","int","complex","character","if","elseif","where","while","elsewhere","forall","maxval", "minval", "dot_product", "abs", "alog", "mod", "size",  "sqrt", "sum","isnan", "exp", "spread", "present", "trim", "sign","min","max","sin","cos","log","log10","tan","tanh","cosh","sinh","asin","acos","atan","atan2","write","read","char","merge",
          "DCONST","fatal_error", "terminal_highlight_fatal_error","warning","caller","caller2", "coeffsx","coeffsy","coeffsz","r1i","sth1i","yh_","not_implemented","die","deri_3d","u_dot_grad_mat",
          'keep_compiler_quiet','keep_compiler_quiet_r', 'keep_compiler_quiet_r1d', 'keep_compiler_quiet_r2d', 'keep_compiler_quiet_r3d', 'keep_compiler_quiet_r4d', 'keep_compiler_quiet_p', 'keep_compiler_quiet_bc', 'keep_compiler_quiet_sl', 'keep_compiler_quiet_i', 'keep_compiler_quiet_i1d', 'keep_compiler_quiet_i81d', 'keep_compiler_quiet_i2d', 'keep_compiler_quiet_i3d', 'keep_compiler_quiet_l', 'keep_compiler_quiet_l1d', 'keep_compiler_quiet_c', 'keep_compiler_quiet_c1d',
          "helflux", "curflux_ds",
          "cffti","cffti1","cfftf","cfftb","kx_fft","ky_fft",
          "bc_aa_pot2",
          "caller", "caller0", "caller1", "caller2", "caller3", "caller4", "caller5", "caller5_str5",
          "output_penciled_scal_c","output_penciled_vect_c","output_pencil_scal","output_pencil_vect","output_pencil"
      };




    }

};
/***
class Parser:

    def __init__():

        ##Intrinsic functions
        ##Ask Matthias about these
        self.ignored_subroutines.extend([])
        

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
    def replace_variables_multi_array_indexing(self,line,new_vars):
        #TODO: change if is a local variable instead of global one
        vars = [x.split("(")[0].strip() for x in new_vars]
        segments = self.get_array_segments_in_line(line,{var: self.static_variables[var] for var in vars})
        return self.replace_segments(segments,line,self.map_val_func,{},{"map_val": [new_vars[line[x[1]:x[2]]] for x in segments]})

    #TODO: do not use this anymore
    # def get_der_index(self,index):
    #   #this whole stuff is really unnecessary since the orig motivation 
    #   #was cause I used the Field() syntax wrongly :////
    #   if index in ["iux__mod__cdata","iuy__mod__cdata","iuz__mod__cdata","ilnrho__mod__cdata","iss__mod__cdata"]:
    #     return f"F_{remove_mod(index[1:]).upper()}"
    #   else:
    #     print("have to deduce der index")
    #     print(index)
    #     print(index in self.known_values)
    #     for val in self.known_values:
    #       if val in index:
    #         index = replace_variable(index,val,f"({self.known_values[val]})")
    #     index = self.evaluate_indexes(index)
    #     if index in self.known_ints:
    #       index = self.known_ints[index]
    #     if index in ["iuu__mod__cdata","iux__mod__cdata","iuy__mod__cdata","iuz__mod__cdata","ilnrho__mod__cdata","iss__mod__cdata"]:
    #       return f"F_{remove_mod(index[1:]).upper()}"
    #     #if not was able to deduce just keep it as
    #     return f"Field({index})"
***/
