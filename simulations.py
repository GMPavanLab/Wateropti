import os
import sys
import signal
import subprocess
import time
from datetime import datetime
from shared.context_managers import working_dir
import MDAnalysis as mda
from MDAnalysis.analysis import rdf
from scipy.interpolate import interp1d
import numpy as np


# build gromacs command with arguments
def gmx_args(gmx_cmd, gpu_id):
    if gpu_id != '':
        gmx_cmd += f" -gpu_id {gpu_id}"
    return gmx_cmd


def run_sims(ns, gpu_id, temperature):
    gmx_start = datetime.now().timestamp()
    esplosa = False
    # grompp -- MINI
    os.chdir('minimization')
    gmx_cmd = f"{ns.args['gmx_path']} grompp -f minimization.mdp -c ../1024_wat.gro -p ../topol.top -o minimization.tpr"
    gmx_process = subprocess.Popen([gmx_cmd], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    gmx_out = gmx_process.communicate()[1].decode()

    if os.path.isfile('minimization.tpr'):
        minimization_attempt = 0
        while (minimization_attempt < 3):
            gmx_cmd = f"{ns.args['gmx_path']} mdrun -deffnm minimization -nsteps -1 -nt {ns.args['nt']}"
            gmx_cmd = gmx_args(gmx_cmd, gpu_id)
            gmx_process = subprocess.Popen([gmx_cmd], shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                           preexec_fn=os.setsid)  # create a process group for the MINI run
            # check if MINI run is stuck because of instabilities
            cycles_check = 0
            last_log_file_size = 0
            while gmx_process.poll() is None:  # while process is alive
                time.sleep(5)
                if os.path.isfile('minimization.log'):
                    log_file_size = os.path.getsize('minimization.log')  # get size of .log file in bytes, as a mean of detecting the MINI run is stuck
                else:
                    log_file_size = last_log_file_size  # MINI is stuck if the process was not able to create log file at start

                if log_file_size == last_log_file_size:  # MINI is stuck if the process is not writing to log file anymore
                    esplosa = True
                    os.killpg(os.getpgid(gmx_process.pid), signal.SIGKILL)  # kill all processes of process group
                    # sim_status = 'Minimization run failed (stuck simulation was killed)'
                else:
                    last_log_file_size = log_file_size

            # check if  minimization.gro has -nan
            if os.path.isfile('minimization.gro'):
                command = "grep nan minimization.gro >/dev/null; echo $?"
                p = subprocess.Popen([command], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if p.communicate()[0].decode().strip() == '1':  # nan not present
                    # break the while
                    minimization_attempt = 5
                else:
                    minimization_attempt += 1
                    if minimization_attempt == 3:
                        esplosa = True
            else:
                esplosa = True #segfault
                minimization_attempt = 5
    else:
        # pass
        esplosa = True
        print(
            'Gmx grompp failed at the MINIMIZATION step, see gmx error message above\nPlease check the parameters of the provided MDP file\n')

    os.chdir('../equilibration')
    # if MINI finished properly, we just check for the .gro file printed in the end
    if os.path.isfile('../minimization/minimization.gro') and (esplosa is False):
        # grompp -- EQUI
        gmx_cmd = f"{ns.args['gmx_path']} grompp -f equilibration.mdp -c ../minimization/minimization.gro -p ../topol.top -o equilibration.tpr"
        gmx_process = subprocess.Popen([gmx_cmd], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        gmx_out = gmx_process.communicate()[1].decode()

        if os.path.isfile('equilibration.tpr'):
            # mdrun -- EQUI
            equilibration_tstart = datetime.now().timestamp()
            gmx_cmd = f"{ns.args['gmx_path']} mdrun -deffnm equilibration -nt {ns.args['nt']}"
            gmx_cmd = gmx_args(gmx_cmd, gpu_id)
            gmx_process = subprocess.Popen([gmx_cmd], shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                           start_new_session=True)  # create a process group for the EQUI run
            # print(f"launched equi at {os.getcwd()}")
            # check if EQUI run is stuck because of instabilities
            cycles_check = 0
            last_log_file_size = 0
            while gmx_process.poll() is None:  # while process is alive
                time.sleep(ns.process_alive_time_sleep)
                # print(f"doing check on equilibration.log at time {time.strftime('%H:%M:%S')} in {os.getcwd()}")
                if ((datetime.now().timestamp() - equilibration_tstart)/3600) > ns.args['walltime_equilibration']:
                    esplosa = True
                    os.killpg(os.getpgid(gmx_process.pid), signal.SIGKILL)
                if os.path.isfile('equilibration.log'):
                    log_file_size = os.path.getsize('equilibration.log')
                    # print(f"at time {time.strftime('%H:%M:%S')} the size is {log_file_size} ")
                    if last_log_file_size == log_file_size:
                        os.killpg(os.getpgid(gmx_process.pid), signal.SIGKILL)
                        # print(f"killing, last size = {last_log_file_size}, size = {log_file_size}")
                    else:
                        last_log_file_size = log_file_size
        else:

            esplosa = True
            print(
                'Gmx grompp failed at the EQUILIBRATION step, see gmx error message above\nPlease check the parameters of the provided MDP file\n')

    os.chdir('../run')
    # if EQUI finished properly, we just check for the .gro file printed in the end
    if os.path.isfile('../equilibration/equilibration.gro') and (esplosa is False):
        # grompp -- PROD
        gmx_cmd = f"{ns.args['gmx_path']} grompp -f run.mdp  -c ../equilibration/equilibration.gro  -p ../topol.top -o run.tpr"
        gmx_process = subprocess.Popen([gmx_cmd], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        gmx_out = gmx_process.communicate()[1].decode()

        if os.path.isfile('run.tpr'):
            # mdrun -- PROD
            run_tstart = datetime.now().timestamp()
            gmx_cmd = f"{ns.args['gmx_path']} mdrun -deffnm run -nt {ns.args['nt']}"
            gmx_cmd = gmx_args(gmx_cmd, gpu_id)
            gmx_process = subprocess.Popen([gmx_cmd], shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                           start_new_session=True)  # create a process group for the MD run

            # check if PROD run is stuck because of instabilities
            last_log_file_size = 0
            while gmx_process.poll() is None:  # while process is alive
                time.sleep(ns.process_alive_time_sleep)
                if ((datetime.now().timestamp() - run_tstart)/3600) > ns.args['walltime_run']:
                    esplosa = True
                    os.killpg(os.getpgid(gmx_process.pid), signal.SIGKILL)
                if os.path.isfile('run.log'):
                    log_file_size = os.path.getsize('run.log')
                    if last_log_file_size == log_file_size:
                        os.killpg(os.getpgid(gmx_process.pid), signal.SIGKILL)
                    else:
                        last_log_file_size = log_file_size
        else:
            esplosa = True
            # sim_status = 'MD run failed (simulation crashed)'
            print(
                'Gmx grompp failed at the PRODUCTION step, see gmx error message above\nPlease check the parameters of the provided MDP file\n')

    if os.path.isfile('run.gro'):
        tpr_file_path = "run.tpr"
        traj_file_path = "run.xtc"
        u = mda.Universe(tpr_file_path, traj_file_path)
        ###########################
        # dielectric constant  PD #
        ###########################
        start_times = [0, 2500, 5000, 7500]
        end_times = [2500, 5000, 7500, 10000]
        for start_time, end_time in zip(start_times, end_times):
            gmx_command = f"echo 0 | {ns.args['gmx_path']} dipoles -s run.tpr -f run.xtc -temp {temperature} -b {start_time} -e {end_time} > log_epsilon  2>&1"
            p = subprocess.Popen([gmx_command], shell=True)
            p.communicate()
            gmx_command = "cat log_epsilon | grep 'Epsilon = ' | awk '{print $3}' >> eps_partial.dat "
            p = subprocess.Popen([gmx_command], shell=True)
            p.communicate()
        # read eps_dat.dat
        eps_partial = np.loadtxt('eps_partial.dat')
        np.savetxt('eps.dat', np.column_stack((np.mean(eps_partial), np.std(eps_partial))), header='#mean std',
                   fmt=['%6.4f', '%6.4f'])

        # rdf
        # select atoms
        hydrogens = u.select_atoms('type HW')
        oxygens = u.select_atoms('type OW')
        # do rdf computacciòn
        goo = rdf.InterRDF(oxygens, oxygens, range=(ns.bw_rdfs / 2, ns.rdf_cutoff + (ns.bw_rdfs / 2)), nbins=ns.nbins)
        goo.run()
        goh = rdf.InterRDF(oxygens, hydrogens, exclusion_block=(1, 2),
                           range=(ns.bw_rdfs / 2, ns.rdf_cutoff + (ns.bw_rdfs / 2)), nbins=ns.nbins)
        goh.run()
        ghh = rdf.InterRDF(hydrogens, hydrogens, exclusion_block=(2, 2),
                           range=(ns.bw_rdfs / 2, ns.rdf_cutoff + (ns.bw_rdfs / 2)), nbins=ns.nbins)
        ghh.run()
        # save results
        np.savetxt(f"goo.dat", np.column_stack((goo.results.bins, goo.results.rdf)), fmt=['%8.6f', '%8.6f'])
        np.savetxt(f"goh.dat", np.column_stack((goh.results.bins, goh.results.rdf)), fmt=['%8.6f', '%8.6f'])
        np.savetxt(f"ghh.dat", np.column_stack((ghh.results.bins, ghh.results.rdf)), fmt=['%8.6f', '%8.6f'])
        ###################
        # compute density #
        ###################
        density_mean = []
        density_std = []
        d = []
        for ts in u.trajectory:
            d.append(1024 * 18.01528 / 6.02214e23 / (ts.volume * 1e-27))
        np.asarray(d)
        density_mean = np.mean(d)
        density_std = np.std(d)
        # save file
        np.savetxt(f"density.dat", np.column_stack((density_mean, density_std)), header='#mean std',
                   fmt=['%8.6f', '%8.6f'])
    else:
        esplosa = True

    gmx_time = datetime.now().timestamp() - gmx_start

    return gmx_time, esplosa


# for making a shared multiprocessing.Array() to handle slots states when running simulations in LOCAL (= NOT HPC)
def init_process(arg):
    global g_slots_states
    g_slots_states = arg


def run_parallel(ns, job_exec_dir, evaluation_number, temperature):
    while True:
        time.sleep(1)
        for i in range(len(g_slots_states)):
            if g_slots_states[i] == 1:  # if slot is available
                # print(f'Starting simulation for evaluation {evaluation_number}')
                g_slots_states[i] = 0  # mark slot as busy
                gpu_id = ns.gpu_ids[i]
                with working_dir(job_exec_dir):
                    gmx_time, esplosa = run_sims(ns, gpu_id, temperature)
                g_slots_states[i] = 1  # mark slot as available
                # print(f'Finished simulation for particle {nb_eval_particle} with {lipid_code} {temp} on slot {i + 1}')

                time_start_str, time_end_str = '', ''  # NOTE: this is NOT displayed anywhere atm & we don't care much
                time_elapsed_str = time.strftime('%H:%M:%S', time.gmtime(round(gmx_time)))

                return time_start_str, time_end_str, time_elapsed_str, esplosa


