import os
import shutil
import sys
from random import randint
import tempfile
import re
import multiprocessing
from itertools import repeat
from shared.simulations import run_parallel, init_process
import numpy as np
from pyemd import emd
from scipy.interpolate import interp1d

def sed_inplace(filename, pattern, repl):
    '''
    Perform the pure-Python equivalent of in-place `sed` substitution: e.g.,
    `sed -i -e 's/'${pattern}'/'${repl}' "${filename}"`.
    '''
    # For efficiency, precompile the passed regular expression.
    pattern_compiled = re.compile(pattern)

    # For portability, NamedTemporaryFile() defaults to mode "w+b" (i.e., binary
    # writing with updating). This is usually a good thing. In this case,
    # however, binary writing imposes non-trivial encoding constraints trivially
    # resolved by switching to text writing. Let's do that.
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_file:
        with open(filename) as src_file:
            for line in src_file:
                tmp_file.write(pattern_compiled.sub(repl, line))

    # Overwrite the original file with the munged temporary file in a
    # manner preserving file attributes (e.g., permissions).
    shutil.copystat(filename, tmp_file.name)
    shutil.move(tmp_file.name, filename)

def calculate_score_for_particle(args):
    """
    Calculate the score attributed to a given particle of the swarm, according to the results of potentially multiple
    parallel simulations.

        Parameters:
            args (dict): Contains the arguments passed via the config file and everything else required for processing.

        Returns:
            score (float): Score for this given particle of the swarm.

    """

    # TODO: Calculate a real score

    score = randint(0, 10)
    return score


def parallel_eval_function(parameters_for_swarm, ns):
    """
    This function receives a list of lists, which corresponds, in order, to the parameters suggested by FST-PSO
    for each particle of the swarm. It has to return a list of floats which correspond to the score for each particle
    in this swarm iteration.

        Parameters:
            parameters_for_swarm (list of lists of floats): Inner lists are the parameters per particle of the swarm.
            args (dict): Contains the arguments passed via the config file and everything else required for processing.

        Returns:
            scores_for_swarm (list of floats): Score for each particle of the swarm.

    """
    p_evaluation_number = []
    p_job_exec_dir = []
    p_temperature = []

    ns.current_swarm_iteration += 1
    print(f"Starting with swarm iteration {ns.current_swarm_iteration}")

    ############################################
    # prepare files to run the swarm iteration #
    ############################################
    for particle_number, particle_parameters in enumerate(parameters_for_swarm):
        evaluation_number = (ns.current_swarm_iteration -1) * ns.particles_in_swarm + particle_number +1
        # create directories and copy input files
        os.makedirs(f"{ns.args['exec_folder']}/eval_{evaluation_number}", exist_ok=True)
        for temperature in ns.args['temperatures']:
            os.makedirs(f"{ns.args['exec_folder']}/eval_{evaluation_number}/{temperature}K", exist_ok=True)
            for file in ['1024_wat.gro', 'topol.top']:
                shutil.copy(f"simulation_inputs/{file}",
                            f"{ns.args['exec_folder']}/eval_{evaluation_number}/{temperature}K")
            for stage in ['minimization', 'equilibration', 'run']:
                os.makedirs(f"{ns.args['exec_folder']}/eval_{evaluation_number}/{temperature}K/{stage}", exist_ok=True)
                shutil.copy(f"simulation_inputs/{stage}.mdp",
                            f"{ns.args['exec_folder']}/eval_{evaluation_number}/{temperature}K/{stage}")

            # store folder name for parallel simulation executions
            p_job_exec_dir.append(f"{ns.args['exec_folder']}/eval_{evaluation_number}/{temperature}K")
            p_evaluation_number.append(evaluation_number)
            p_temperature.append(temperature)
            # change topology
            ns.patterns = ['ph_ow_sig', 'ph_ow_eps','ph_ow_ch', 'ph_doh', 'ph_dhh']
            for particle_parameter, pattern in zip(particle_parameters, ns.patterns):
                sed_inplace(f"{ns.args['exec_folder']}/eval_{evaluation_number}/{temperature}K/topol.top", pattern, "{:.8f}".format(particle_parameter))
            # now change hydrogen too
            hydrogen_charge = - (particle_parameters[2]/2)
            sed_inplace(f"{ns.args['exec_folder']}/eval_{evaluation_number}/{temperature}K/topol.top", 'ph_hw_ch',
                        "{:.8f}".format(hydrogen_charge))
            # change mdps
            for stage in ['equilibration', 'run']:
                sed_inplace(f"{ns.args['exec_folder']}/eval_{evaluation_number}/{temperature}K/{stage}/{stage}.mdp", 'ph_ref_t',
                            str(temperature))

    ###########################################
    # run fking simulations XDDDDDDDDDDDDDDDDD #
    ###########################################
    slots_states = multiprocessing.Array('i', ns.nb_slots, lock=True)
    for j in range(ns.nb_slots):  # multiprocessing.Array does NOT like list comprehension
        slots_states[j] = 1  # mark slot as available

    with multiprocessing.Pool(processes=ns.nb_slots, initializer=init_process, initargs=(slots_states,)) as pool:
        p_args = zip(repeat(ns), p_job_exec_dir, p_evaluation_number, p_temperature)
        p_res = pool.starmap(run_parallel, p_args)
        p_time_start_str, p_time_end_str, p_time_elapsed_str, esplosa = list(map(list, zip(*p_res)))

    ########################
    # SCORE EACH PARTICLE  # <---for the current swarm iteration
    ########################
    # create score arrays to store score of every SIMULATION ( not particle)
    rdfs = ['goo','goh','ghh']
    density_score = np.zeros((len(p_job_exec_dir)))
    epsilon_score = np.zeros((len(p_job_exec_dir)))
    emd_score = np.zeros((len(p_job_exec_dir), len(rdfs)))
    for evaluation_number, (exec_dir, temperature) in enumerate(zip(p_job_exec_dir, p_temperature)):
        relative_path = f"{exec_dir}/run"
        if not esplosa[evaluation_number]:
            simulated_density = np.loadtxt(f'{relative_path}/density.dat')
            simulated_epsilon = np.loadtxt(f'{relative_path}/eps.dat')
            density_score[evaluation_number] = (30/35)*abs((simulated_density[0] - ns.experimental_density[temperature])/ns.experimental_density[temperature])*100
            epsilon_score[evaluation_number] = (20/800)*abs((simulated_epsilon[0] - ns.experimental_epsilon[temperature])/ns.experimental_epsilon[temperature])*100
            #emd
            for rdf_column, rdf in enumerate(rdfs):
                simulated_rdf = np.loadtxt(f"{relative_path}/{rdf}.dat")
                simulated_rdf = np.ascontiguousarray(simulated_rdf[:, 1])
                # n_contacts_simulated = simulated_rdf * (ns.rdf_distance ** 2) * 4 * np.pi * simulated_density[0] * ns.bw_rdfs
                # n_contacts_experimental = ns.experimental_rdfs[temperature][rdf] * (ns.rdf_distance ** 2) * 4 * np.pi * ns.experimental_density[temperature] * ns.bw_rdfs
                # # normalize them to 100
                # n_contacts_simulated = n_contacts_simulated * (100 / np.sum(n_contacts_simulated))
                # n_contacts_experimental = n_contacts_experimental * (100 / np.sum(n_contacts_experimental))
                emd_score[evaluation_number,rdf_column] = (16.6667/200)*emd(simulated_rdf * ns.rdf_distance, ns.experimental_rdfs[temperature][rdf] * ns.rdf_distance, ns.m_euclidean)
        else:
            density_score[evaluation_number] = 30
            epsilon_score[evaluation_number] = 20
            emd_score[evaluation_number, :] = [50/3]*3
    aggregated_score = emd_score.sum(axis=1) + density_score  + epsilon_score
    #store the score of every simulation
    for i in range(0,len(p_job_exec_dir)):
        s_goo = emd_score[i,0]
        s_goh = emd_score[i, 1]
        s_ghh = emd_score[i, 2]
        np.savetxt(f"{p_job_exec_dir[i]}/simulation.log", np.column_stack([s_goo, s_goh, s_ghh, density_score[i], epsilon_score[i], aggregated_score[i]]), fmt='%f', delimiter='\t')

    # obtain score at PARTILE LEVEL and no more at simulation/job level #
    density_score_particle = np.zeros((ns.particles_in_swarm))
    epsilon_score_particle = np.zeros((ns.particles_in_swarm))
    emd_score_particle = np.zeros((ns.particles_in_swarm, len(rdfs)))
    s_goo_part = np.zeros((ns.particles_in_swarm))
    s_goh_part = np.zeros((ns.particles_in_swarm))
    s_ghh_part = np.zeros((ns.particles_in_swarm))
    sims_per_particle = len(ns.args['temperatures'])
    for x in range(0,ns.particles_in_swarm):
        if True not in esplosa[x*sims_per_particle: x*sims_per_particle+sims_per_particle]:
            density_score_particle[x] = np.mean(density_score[x*sims_per_particle: x*sims_per_particle+sims_per_particle])
            epsilon_score_particle[x] = np.mean(epsilon_score[x * sims_per_particle: x * sims_per_particle + sims_per_particle])
            emd_score_particle[x,:] = np.mean(emd_score[x  * sims_per_particle: x * sims_per_particle + sims_per_particle, :])
            s_goo_part[x] = np.mean(emd_score[x  * sims_per_particle: x * sims_per_particle + sims_per_particle, 0])
            s_goh_part[x] = np.mean(emd_score[x * sims_per_particle: x * sims_per_particle + sims_per_particle, 1])
            s_ghh_part[x] = np.mean(emd_score[x * sims_per_particle: x * sims_per_particle + sims_per_particle, 2])
        else:
            density_score_particle[x] = 30
            epsilon_score_particle[x] = 20
            emd_score_particle[x,:] = [50/3]*3
            s_goo_part[x] = 50/3
            s_goh_part[x] = 50/3
            s_ghh_part[x] = 50/3
    aggregated_score_particle = s_goo_part + s_goh_part + s_ghh_part + density_score_particle + epsilon_score_particle

    np.savetxt(f"{ns.args['exec_folder']}/swarm_{ns.current_swarm_iteration}_detailed.log", np.column_stack([s_goo_part, s_goh_part, s_ghh_part , density_score_particle , epsilon_score_particle , aggregated_score_particle]),fmt='%f', delimiter='\t')
    np.savetxt(f"{ns.args['exec_folder']}/swarm_{ns.current_swarm_iteration}.log", np.column_stack([parameters_for_swarm, aggregated_score_particle]), fmt='%f', delimiter='\t')
    print('swarm iteration ended')

    for particle_number, score in enumerate(aggregated_score_particle):
        if score < ns.args['best_solution']['score']:
            ns.args['best_solution']['score'] = score
            ns.args['best_solution']['parameters'] = parameters_for_swarm[particle_number]
            ns.args['best_solution']['swarm_iter'] = ns.current_swarm_iteration
            ns.args['best_solution']['particle_iter'] = particle_number+1
    print(f"best solution at particle {ns.args['best_solution']['particle_iter']} of swarm {ns.args['best_solution']['swarm_iter']}, score = {ns.args['best_solution']['score']}")
    return aggregated_score_particle  # this goes to FST-PSO for choosing the next sets of parameters supposed to improve the score

