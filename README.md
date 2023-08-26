# fortran-parser
Fortran preprocessor designed for multithreading and GPU offloading through OpenMP

Features:
  * Automatic privatization of modified global variables in multithreaded code
  * Fortran Array operation unrolling to DO-loops with automatic declaratives to spread work either along CPU or GPU threads
  * Automatic declarations for global variables needed in target regions
  * Function inlining which helps OpenMP offloading 

Parameters:
  * -f --function Which function to parse
  * -F --file In which file the function exists
  * -o --offload --no-offload Whether to offload the given function
  * -c --communication --no-communication Whether to warn about communication calls inside the given subroutine
  * -d --directory From which directory to look for file
  * -M --Makefile. Pencil Code specific. Filename to Makefile.local to know which modules are included (might change in the future) 

  Example call to modify a given function to be suittable for offloading (Work in progress)
  * python parse.py -f example_function -F ./dir/file_where_function_is -o

  Example call to make needed global variables threadprivate:
  * python parse.py -f example_function -F ./dir/file_where_function_is

OpenMP declarations are in files {original_file_path}_omp_incl.h

  
  
