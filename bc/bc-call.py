import os
def main():
    funcs = [
            "bc_ss_flux"
            #"bc_ss_flux_turb",
            #"bc_ss_flux_turb_x",
            #"bc_ss_flux_condturb_x",
            #"bc_ss_flux_condturb_z",
            #"bc_ss_temp_old",
            #"bc_ss_temp_x",
            #"bc_ss_temp_y",
            #"bc_ss_temp_z",
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
            #"rhs_cpu"
            #"rhs_cpu"
            #"dchemistry_dt"
            #get_reaction_rate"
            #"rampup_secondary_mass"
            #"calc_pencils_gravity"
            #"secondary_body_gravity"
            #"dlnrho_dt"
            #"daa_dt"
            #"my_test"
        ]
    #sample = "MRI-turbulence_hyper"
    sample = "gputest"
    #sample = "turbulent_flame"
    #sample = "geodynamo"
    #sample = "cartesian-convection-kramers"
    #file = "chemistry.f90"
    #file = "equ.f90"
    file = "entropy_bcs.f90"
    #file = "magnetic.f90"
    #file = "gravity_r.f90"
    for func in funcs:
        command = f"python3 ../parse.py -f {func} -F ~/pencil-code/samples/{sample}/src/{file} -o -b --sample-dir ~/pencil-code/samples/{sample} --dir ~/pencil-code/samples/{sample}/src"
        os.system("rm -f res-inlined.txt")
        os.system(command)

if __name__ == "__main__":
    main()
