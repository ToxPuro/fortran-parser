import os
def main():
    funcs = [
            #"bc_ss_flux"
            #"bc_ss_flux_turb",
            #"bc_ss_flux_turb_x",
            #"bc_ss_flux_condturb_x",
            #"bc_ss_flux_condturb_z",
            #"bc_ss_temp_old",
            #"bc_ss_temp_x",
            #"bc_ss_temp_y",
            "bc_ss_temp_z",
            #"bc_lnrho_temp_z",
            #"bc_lnrho_pressure_z",
            #"bc_ss_temp2_z",
            #"bc_ss_temp3_z",
            #"bc_ss_stemp_x",
            #"bc_ss_stemp_y",
            #"bc_ss_stemp_z",
            #"bc_ss_a2stemp_x",
            #"bc_ss_a2stemp_y", 
            #"bc_ss_a2stemp_z", 
            #"bc_ss_energy",
            #"bc_lnrho_cfb_r_iso",
            #"bc_lnrho_hds_z_iso",
            #"bc_ism",
        ]
    for func in funcs:
        command = f"python3 parse.py -f {func} -F ~/my_pc_copy/samples/gputest/src/eos_idealgas.f90 -o -b --sample-dir ~/my_pc_copy/samples/gputest --dir ~/my_pc_copy/samples/gputest/src "
        os.system("rm -f res-inlined.txt")
        os.system(command)

if __name__ == "__main__":
    main()
