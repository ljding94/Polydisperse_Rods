import hoomd
import rowan
import numpy as np
from IQ import *


def create_file_label(system_params):
    """
    Create a unique label for the simulation folder based on system parameters and run number.
    """
    run_type = system_params["run_type"]
    run_num = system_params["run_num"]
    pd_type = system_params["pd_type"]
    N = system_params["N"]
    if run_type == "prec":
        phi = system_params["phi"]
        meanL = system_params["meanL"]
        sigmaL = system_params["sigmaL"]
        sigmaD = system_params["sigmaD"]
        label = f"{run_type}_run{run_num:d}_{pd_type}_N{N:d}_phi{phi:.2f}_meanL{meanL:.2f}_sigmaL{sigmaL:.2f}_sigmaD{sigmaD:.2f}"
    elif run_type == "rand":
        label = f"{run_type}_run{run_num:d}_{pd_type}_N{N:d}"
        # set random parameters
    else:
        raise ValueError(f"Unknown run type: {run_type}")
    return label


# Custom action for dumping to TXT file
class DumpTXT(hoomd.custom.Action):
    def __init__(self, subfolder, type_Ls, type_Ds, type_Vs, label):
        self.type_Ls = type_Ls  # Per-particle length
        self.type_Ds = type_Ds  # Per-particle diameter
        self.type_Vs = type_Vs

        # it's a list type_length[typeid] = L/d
        self.subfolder = subfolder
        self.label = label
        self.first_write = True

    def act(self, timestep):
        filename = f"{self.subfolder}/{self.label}_dump_{timestep}.txt"
        snap = self._state.get_snapshot()
        with open(filename, "w") as f:
            f.write("ITEM: TIMESTEP\n")
            f.write(f"{timestep}\n")
            f.write("ITEM: NUMBER OF ATOMS\n")
            f.write(f"{snap.particles.N}\n")
            f.write("ITEM: BOX BOUNDS pp pp pp\n")
            box = snap.configuration.box
            f.write(f"{-box[0]/2} {box[0]/2}\n")
            f.write(f"{-box[1]/2} {box[1]/2}\n")
            f.write(f"{-box[2]/2} {box[2]/2}\n")
            f.write("ITEM: ATOMS id type x y z quatw quati quatj quatk shapex shapey shapez volume\n")
            for i in range(snap.particles.N):
                pos = snap.particles.position[i]
                orient = snap.particles.orientation[i]  # [w, i, j, k]
                typeid = snap.particles.typeid[i]
                shapex = 1.0 * self.type_Ds[typeid]
                shapey = 1.0 * self.type_Ds[typeid]
                shapez = 2.0 * self.type_Ls[typeid]
                # reference: https://www.ovito.org/manual/advanced_topics/aspherical_particles.html
                # for cylinder, shapex, shapey are the radius in x and y, and shapez is the length along z, but a factor of 0.5 is multiplied when reading
                volume = self.type_Vs[typeid]
                f.write(f"{i} {typeid} {pos[0]} {pos[1]} {pos[2]} {orient[0]} {orient[1]} {orient[2]} {orient[3]} {shapex} {shapey} {shapez} {volume} \n")


class MeasureLiquidCrystalOrder(hoomd.custom.Action):
    """Compute nematic order parameter S, smectic order parameter τ, and optimal layer spacing d."""

    def __init__(self, subfolder, system_params, label, num_d=100):
        self.filename = f"{subfolder}/{label}_LC_order.csv"
        self.meanL = system_params["meanL"]  # For d range; D=1 fixed
        self.num_d = num_d  # Resolution for d optimization
        data = np.load(f"{subfolder}/particle_data.npz")
        self.total_volume = data["total_V"]
        self._header_written = False
        self.nematicS = []
        self.smecticTau = []
        self.opt_d = []
        self.phi = []

        # other simulation info
        self.system_params = system_params  # e.g., {'pd_type': "uniform", 'N': 100, 'phi': 0.5, 'mean_ld': 2.0, 'sigma': 0.1}

    def act(self, timestep):
        snap = self._state.get_snapshot()
        if snap.communicator.rank != 0:
            return

        # Rod directors (body-frame axis along z)
        local_axis = np.array([0.0, 0.0, 1.0])
        directors = rowan.rotate(snap.particles.orientation, local_axis)

        # Nematic tensor Q
        N = directors.shape[0]
        Q = (3.0 / (2.0 * N)) * directors.T @ directors - 0.5 * np.eye(3)

        # Nematic order S (largest eigenvalue)
        S = np.linalg.eigvalsh(Q).max()

        # Global director n (largest eigenvector)
        eigvals, eigvecs = np.linalg.eigh(Q)
        n = eigvecs[:, np.argmax(eigvals)]

        # Project positions along n for smectic order
        positions = snap.particles.position
        s = positions @ n

        # Optimize d over plausible range (around mean total length ± margin)
        d_min = 0.5 * self.meanL + 1
        d_max = 1.5 * self.meanL + 1
        d_values = np.linspace(d_min, d_max, self.num_d)

        tau_max = 0.0
        d_opt = 0.0
        for d in d_values:
            phase = np.exp(1j * 2 * np.pi * s / d)
            tau = np.abs(np.mean(phase))
            if tau > tau_max:
                tau_max = tau
                d_opt = d

        # Compute current phi
        box = snap.configuration.box
        box_volume = box[0] * box[1] * box[2]
        phi = self.total_volume / box_volume

        # Append to file
        mode = "w" if not self._header_written else "a"
        with open(f"{self.filename}", mode) as f:
            if not self._header_written:
                f.write("pd_type," + f"{self.system_params['pd_type']}\n")
                f.write("N," + f"{self.system_params['N']}\n")
                f.write("phi," + f"{self.system_params['phi']}\n")
                f.write("meanL," + f"{self.system_params['meanL']}\n")
                f.write("sigmaL," + f"{self.system_params['sigmaL']}\n")
                f.write("sigmaD," + f"{self.system_params['sigmaD']}\n")
                f.write("step,S,tau,d,phi\n")
                self._header_written = True
            f.write(f"{timestep},{S},{tau_max},{d_opt},{phi}\n")

        # add to list
        self.nematicS.append(S)
        self.smecticTau.append(tau_max)
        self.opt_d.append(d_opt)
        self.phi.append(phi)

    def act_end(self):
        # append statistics, mean, std, to file
        if self.nematicS and self.smecticTau:
            mean_S = np.mean(self.nematicS)
            std_S = np.std(self.nematicS)
            mean_tau = np.mean(self.smecticTau)
            std_tau = np.std(self.smecticTau)
            mean_d = np.mean(self.opt_d)
            std_d = np.std(self.opt_d)

            with open(f"{self.filename}", "a") as f:
                f.write(f"Mean, {mean_S}, {mean_tau}, {mean_d}, {self.phi[-1]} \n")
                f.write(f"Std, {std_S}, {std_tau}, {std_d}, 0 \n")


class MeasureScattering(hoomd.custom.Action):
    r"""Compute scattering function I(Q) and append to a CSV file."""

    def __init__(self, folder, subfolder, q_values, particle_types, type_Ls, type_Ds, type_Vs, system_params, label):
        self.system_params = system_params
        self.filename = f"{subfolder}/{label}_Iq.csv"
        self.stats_filename = f"{folder}/stats_{label}_Iq_{create_file_label(system_params)}.csv"
        self.q_values = q_values  # Array of q-vectors magnitute to evaluate
        self.particle_types = particle_types  # Per-particle type index
        self.type_Ls = type_Ls  # Per-type length
        self.type_Ds = type_Ds  # Per-type diameter
        self.type_Vs = type_Vs  # Per-type volume

        self.Iq = []
        self.alpha_values = np.linspace(0, np.pi, 100)  # Angle between rod axis and q
        print(f"q {len(self.q_values)}, alpha {len(self.alpha_values)}; total grid point {len(self.alpha_values)* len(self.q_values)} * numer of interpolator {len(self.type_Ls)}")
        print("Building FQ interpolator")
        # Fq_interp should be a list per type, and only interpolate on q and alpha
        type_Fq_interps, type_Fq_meshs = build_FQalpha_interpolator(self.q_values, self.alpha_values, self.type_Ls, self.type_Ds)

        print("done building FQ interpolator")
        self.type_Fq_interps = type_Fq_interps  # list of interpolators, one per type

        self._header_written = False

    def act(self, timestep):
        snap = self._state.get_snapshot()
        if snap.communicator.rank != 0:  # avoid every rank writing
            return

        positions = snap.particles.position
        # Rod directors (body-frame axis along z)
        local_axis = np.array([0.0, 0.0, 1.0])
        directors = rowan.rotate(snap.particles.orientation, local_axis)

        Iq = calculate_IQ(positions, directors, self.particle_types, self.type_Vs, self.q_values, self.type_Fq_interps)

        self.Iq.append(Iq)

        # Append to file
        mode = "w" if not self._header_written else "a"
        with open(f"{self.filename}", mode) as f:
            if not self._header_written:
                f.write("pd_type," + f"{self.system_params['pd_type']}\n")
                f.write("N," + f"{self.system_params['N']}\n")
                f.write("phi," + f"{self.system_params['phi']}\n")
                f.write("meanL," + f"{self.system_params['meanL']}\n")
                f.write("sigmaL," + f"{self.system_params['sigmaL']}\n")
                f.write("sigmaD," + f"{self.system_params['sigmaD']}\n")
                header = "step,q," + ",".join([f"{self.q_values[i]}" for i in range(len(self.q_values))]) + "\n"
                f.write(header)
                self._header_written = True
            line = f"{timestep},Iq," + ",".join([f"{Iq[i]}" for i in range(len(Iq))]) + "\n"
            f.write(line)

        Iq_mean = np.mean(self.Iq, axis=0)
        Iq_std = np.std(self.Iq, axis=0)
        sqrtM = np.sqrt(len(self.Iq))
        dIq = Iq_std / sqrtM  # Standard error of the mean

        # write tosummary file
        summary_file = f"{self.stats_filename}"
        with open(summary_file, "w") as f:
            # Always write the metadata/header (summary file is opened with "w" each call,
            # so we must rewrite the header every time to keep it consistent)
            f.write("pd_type," + f"{self.system_params['pd_type']}\n")
            f.write("N," + f"{self.system_params['N']}\n")
            f.write("phi," + f"{self.system_params['phi']}\n")
            f.write("meanL," + f"{self.system_params['meanL']}\n")
            f.write("sigmaL," + f"{self.system_params['sigmaL']}\n")
            f.write("sigmaD," + f"{self.system_params['sigmaD']}\n")
            f.write("q,Iq,dIq\n")
            # write the current averaged Iq and its error (will be overwritten on subsequent calls)
            for i in range(len(self.q_values)):
                f.write(f"{self.q_values[i]},{Iq_mean[i]},{dIq[i]}\n")

    '''
    def act_end(self):
        # Append statistics, mean, std, to file
        Iq_mean = np.mean(self.Iq, axis=0)
        Iq_std = np.std(self.Iq, axis=0)

        sqrtM = np.sqrt(len(self.Iq))
        dIq = Iq_std / sqrtM  # Standard error of the mean
        with open(f"{self.filename}", "a") as f:
            f.write("Mean,Iq," + ",".join([f"{Iq_mean[i]}" for i in range(len(Iq_mean))]) + "\n")
            f.write("Std/sqrt(M),Iq," + ",".join([f"{dIq[i]}" for i in range(len(dIq))]) + "\n")

        # write to a new summary file
        summary_file = f"{self.stats_filename}"
        with open(summary_file, "w") as f:
            f.write("pd_type," + f"{self.system_params['pd_type']}\n")
            f.write("N," + f"{self.system_params['N']}\n")
            f.write("phi," + f"{self.system_params['phi']}\n")
            f.write("mean_ld," + f"{self.system_params['mean_ld']}\n")
            f.write("sigma," + f"{self.system_params['sigma']}\n")
            f.write("q,Iq,dIq\n")
            for i in range(len(self.q_values)):
                f.write(f"{self.q_values[i]},{Iq_mean[i]},{dIq[i]}\n")
    '''
