#!/usr/bin/env python3

import os, time, copy, sys
from fstpso import FuzzyPSO
import yaml
from shared.fitness_function import parallel_eval_function
from argparse import Namespace
import numpy as np
from scipy.interpolate import interp1d
from shared.fs import import_experimental_data, check_slot_consistency
import pickle

ns = Namespace()

with open("input.yaml", 'r') as stream:
    ns.args = yaml.safe_load(stream)
check_slot_consistency(ns)

ns.process_alive_time_sleep = 20  # nb of seconds between process alive check cycles
ns.process_alive_nb_cycles_dead = int(ns.args['sim_kill_delay'] / ns.process_alive_time_sleep)

# OUTPUT FOLDER - checkpoints
#by default we assume that there are no checkpoints
if ns.args['exec_folder'] == '':
    # ns.args['exec_folder'] = time.strftime('EAU_OPTI_STARTED_%d-%m-%Y_%Hh%Mm%Ss')
    ns.args['exec_folder'] = sys.argv[1]
if ns.args['continue_from_checkpoint']:
    ns.fstpso_checkpoint = f"{ns.args['exec_folder']}/fstpso_checkpoint.obj"
    # look for last swarm iteration
    g = open(ns.fstpso_checkpoint, "rb")
    obj = pickle.load(g)
    ns.current_swarm_iteration = obj._Iteration + 2
else:
    ns.fstpso_checkpoint = None
    ns.current_swarm_iteration = 0
os.makedirs(ns.args['exec_folder'], exist_ok=True)

ns.tuned_parameters = ['oxygen_sigma', 'oxygen_epsilon', 'oxygen_charge', 'doh', 'dhh']
#placeholder strings for topology
ns.patterns = ['ph_ow_sig', 'ph_ow_eps', 'ph_ow_ch', 'ph_doh', 'ph_dhh']

water_models = np.array([
    [0.31742704, 0.6836907, -0.89517, 0.0978882, 0.1598507],  # OPC3
    [0.316557, 0.650629, -0.82, 0.1, 0.1633],  # SPC
    [0.316557, 0.650629, -0.8476, 0.1, 0.1633],  # SPCE
    [0.31657195, 0.6497752, -0.8476, 0.101, 0.16493],  # SPCEb
    [0.315061, 0.636386, -0.834, 0.09572, 0.15139],  # TIP3P
    [0.31779646, 0.65214334, -0.848448, 0.1011811, 0.1638684]  # TIP3P-FB
])

# compute mean and std of AA forcefields
ns.search_space = np.zeros((len(ns.tuned_parameters), 2))
# each row of the search space is a variable
for index, row in enumerate(ns.search_space):
    #start from opc3 +/- 5sigma
    row[0] = water_models[0][index] - water_models.std(axis=0)[index] * 5
    row[1] = water_models[0][index] + water_models.std(axis=0)[index] * 5
    # row[0] = water_models.mean(axis=0)[index] - water_models.std(axis=0)[index] * 3
    # row[1] = water_models.mean(axis=0)[index] + water_models.std(axis=0)[index] * 3

# number of particles of the swarm
ns.particles_in_swarm = ns.args['particles_in_swarm']


# ###########################
# #set rdf cutoff and nbins # --> the vector of distances is EQUAL for experimental rdf & simulated (mda.bins)
###########################
ns.bw_rdfs = 0.02  # ANGSTROM, codio !!
ns.rdf_cutoff = 10  # ANGSTROM, codio !!
ns.nbins = round(ns.rdf_cutoff / ns.bw_rdfs)
ns.rdf_distance = np.arange(ns.bw_rdfs, ns.rdf_cutoff + ns.bw_rdfs, ns.bw_rdfs)

#######################################################################
import_experimental_data(ns)
with open('exp_data.pkl', "wb") as f:
    pickle.dump([ns.experimental_rdfs, ns.experimental_density, ns.experimental_epsilon], f)

########################################
# define wasserstein distance matrices #
########################################
vol_shell = 4 * np.pi * np.power(ns.rdf_distance, 2) * ns.bw_rdfs
ns.m_euclidean = np.empty([len(vol_shell), len(vol_shell)], dtype=float)
ns.m_shell_vol = np.empty([len(vol_shell), len(vol_shell)], dtype=float)
ns.m_shell_vol_ratio = np.empty([len(vol_shell), len(vol_shell)], dtype=float)
for i in range(len(vol_shell)):
    for j in range(len(vol_shell)):
        if i == j:
            ns.m_euclidean[i, j] = 0
            ns.m_shell_vol[i, j] = 0
            ns.m_shell_vol_ratio[i, j] = 0
        else:
            if j > i:
                ns.m_shell_vol[i, j] = vol_shell[j] - vol_shell[i]
                ns.m_shell_vol_ratio[i, j] = vol_shell[j] / vol_shell[i]
                ns.m_euclidean[i, j] = ns.rdf_distance[j] - ns.rdf_distance[i]
            else:
                ns.m_shell_vol[i, j] = vol_shell[i] - vol_shell[j]
                ns.m_euclidean[i, j] = ns.rdf_distance[i] - ns.rdf_distance[j]
                ns.m_shell_vol_ratio[i, j] = vol_shell[i] / vol_shell[j]

ns.args['best_solution'] = {
    'score': np.inf,
    'parameters': [],
    'swarm_iter': 0,
    'particle_iter': 0
}

FP = FuzzyPSO()
FP.set_search_space(ns.search_space)
FP.set_swarm_size(ns.particles_in_swarm)  # if not set, then the choice of nb of particles is automatic
FP.set_parallel_fitness(fitness=parallel_eval_function, arguments=ns, skip_test=True)
result = FP.solve_with_fstpso(max_iter=ns.args['max_swarm_iterations'],
                              initial_guess_list=None,
                              max_iter_without_new_global_best=ns.args['max_swarm_iterations_without_improvement'],
                              restart_from_checkpoint=ns.fstpso_checkpoint,
                              save_checkpoint=f"{ns.args['exec_folder']}/fstpso_checkpoint.obj",
                              verbose=True
                              )
# print(f"Best solution found in swarm iter {ns.args['best_solution']['swarm_iter']} at particle "
#        f"{ns.args['best_solution']['particle_iter']} with score {ns.args['best_solution']['score']}")
#  print(f"Best solution: {ns.args['best_solution']['parameters']}")
