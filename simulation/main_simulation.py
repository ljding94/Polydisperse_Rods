import os
import argparse
import time
from rods_init import *
from rods_visual import *
from rods_sample import *


def main(args):

    system_params = {
        "folder": args.folder,
        "run_type": args.run_type,
        "run_num": args.run_num,
        "pd_type": args.pd_type,
        "N": args.N,
    }
    if args.run_type == "prec":

        system_params["phi"] = args.phi
        system_params["meanL"] = args.meanL
        # meanD = 1.0 as unit of length
        system_params["sigmaL"] = args.sigmaL
        system_params["sigmaD"] = args.sigmaD

    elif args.run_type == "rand":
        system_params["phi"] = np.random.uniform(0.01, 0.3)
        system_params["meanL"] = np.random.uniform(0.0, 6.0)
        # system_params["sigmaL"] = np.random.uniform(0.00, 0.30)
        system_params["sigmaL"] = 0.0  # sigmaL is minor effect
        if args.pd_type == "uniform":
            system_params["sigmaD"] = np.random.uniform(0.00, 0.30)
        elif args.pd_type == "normal":
            system_params["sigmaD"] = np.random.uniform(0.00, 0.15)
        elif args.pd_type == "lognormal":
            system_params["sigmaD"] = np.random.uniform(0.00, 0.15)

    label = create_file_label(system_params)
    folder = f"{args.folder}/{label}"
    if not os.path.exists(folder):
        os.makedirs(folder)

    '''
    run_initialization(
        save_dump_detail=args.save_dump_detail,
        system_params=system_params,
        randomization_steps=5000,
        max_compression_steps=50000,
        num_compression_stage=2,
    )
    '''
    run_initialization(
        save_dump_detail=args.save_dump_detail,
        system_params=system_params,
        randomization_steps=2000,
        max_compression_steps=20000,
        num_compression_stage=2,
    )

    # render(folder, "compressed.gsd", frame=-1) # no need, confirm OVITO works
    q_values = np.linspace(3.0, 13, 100)
    # q_values = np.logspace(np.log10(0.1), np.log10(20), 200)
    run_sampling(
        save_dump_detail=args.save_dump_detail,
        system_params=system_params,
        q_values=q_values,
        measurement_steps=1000,
        N_measurement=100,
    )


if __name__ == "__main__":

    start_time = time.time()

    parser = argparse.ArgumentParser(description="Run polydisperse rods simulation.")

    parser.add_argument("--folder", type=str, required=True, help="folder to save data.")
    parser.add_argument("--run_type", type=str, choices=["prec", "rand"], required=True, default="prec", help="random parameter or precise parameter.")
    parser.add_argument("--run_num", type=int, default=0, help="run number to keep track of the run.")
    parser.add_argument("--pd_type", type=str, choices=["uniform", "normal", "lognormal"], required=True, help="Polydispersity type.")
    parser.add_argument("--N", type=int, required=True, help="Number of particles.")

    arg_required = False
    parser.add_argument("--phi", type=float, required=arg_required, help="Packing fraction.")
    parser.add_argument("--meanL", type=float, required=arg_required, help="Mean aspect ratio (L/d).")
    parser.add_argument("--sigmaL", type=float, required=arg_required, help="Polydispersity (standard deviation).")
    parser.add_argument("--sigmaD", type=float, required=arg_required, help="Diameter polydispersity (standard deviation).")

    parser.add_argument("--save_dump_detail", type=bool, default=False, help="save lots of dump file or not.")

    args = parser.parse_args()

    print(args)

    main(args)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Simulation completed in {elapsed_time:.2f} seconds.")

    # time referece: 10000 random, 100000 compression
    # N50 phi0.3: 52s
    # N500 phi0.3: 168s

# sample usage:
# conda activate hoomd-env
# python3 main_simulation.py --run_type prec --run_num 0 --pd_type uniform --N 100 --phi 0.10 --meanL 4.00 --sigmaL 0.10 --sigmaD 0.10 --folder /Users/ldq/Work/Polydisperse_Rods/data_local/data_pool --save_dump_detail 1


# python3 main_simulation.py --run_type rand --run_num 0 --pd_type uniform --N 100 --folder /Users/ldq/Work/Polydisperse_Rods/data_local/data_pool --save_dump_detail 1