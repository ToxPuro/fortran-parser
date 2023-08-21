# fortran-parser
Fortran preprocessor designed for multithreading and GPU offloading through OpenMP

Features: \\
  * Automatic privatization of modified global variables in multithreaded code \\
  Fortran Array operation unrolling to DO-loops with automatic declaratives to spread work either along CPU or GPU threads\\
  Automatic declarations for global variables needed in target regions \\
  Function inlining which helps OpenMP offloading \\

Parameters: \\
  -f --function Which function to parse \\
  -F --file In which file the function exists \\
  -o --offload --no-offload Whether to offload the given function \\
  -c --communication --no-communication Whether to warn about communication calls inside the given subroutine \\
  
  
