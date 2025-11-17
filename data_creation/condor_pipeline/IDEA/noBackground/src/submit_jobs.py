#!/usr/bin/env python
import os, sys
import glob
import argparse
from pathlib import Path

# ____________________________________________________________________________________________________________
def absoluteFilePaths(directory):
    files = []
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            files.append(os.path.abspath(os.path.join(dirpath, f)))
    return files

# _____________________________________________________________________________________________________________
def main(base_path):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        help = "output directory",
        default = "",
    )

    parser.add_argument("--njobs", help = "max number of jobs", default = 2)
    parser.add_argument("--type", help = "simulation type", default = "Pythia")
    parser.add_argument("--config", help = "Pythia configuration card name", default="")
    parser.add_argument("--detectorVersion", help = "Detector Version", default = 3)
    parser.add_argument("--detectorOption", help = "Detector Option", default = 1)
    parser.add_argument("--train_or_val", help = "Dataset type", default = "train")
    parser.add_argument("--use_lr", help = "Use left-right positions?", default = "True")
    
    parser.add_argument(
        "--queue",
        help="queue for condor",
        choices=[
            "espresso",
            "microcentury",
            "longlunch",
            "workday",
            "tomorrow",
            "testmatch",
            "nextweek",
        ],
        default="longlunch",
    )

    args = parser.parse_args()
    
    queue = args.queue
    outdir = os.path.abspath(args.outdir)
    
    njobs = int(args.njobs)
    sim_type = args.type
    config = args.config
    detectorVersion = int(args.detectorVersion)
    detectorOption = int(args.detectorOption)
    train_or_val = args.train_or_val
    use_lr = args.use_lr

    os.makedirs(f"{outdir}/{sim_type}/{config}", exist_ok=True)
    storage_path = f"{outdir}/{sim_type}/{config}"

    list_of_outfiles = glob.glob(f"{storage_path}/*.root")

    if sim_type == "Pythia":
        script = "src/run_sequence_global.sh"
    elif sim_type == "gun":
        # TO-DO
        sys.exit(0)
    else:
        print(f"Unknown simulation type: {sim_type}")
        sys.exit(1)

    arguments_list = []
    jobCount = 0

    discard_events = -1
    for job in range(njobs):
        if job > discard_events:
            seed = str(job + 1)
            basename = f"{config}_graphs_{seed}.root"
            outputFile = f"{storage_path}/{basename}"
            if outputFile not in list_of_outfiles:
                print(f"{outputFile} : missing output file")
                argts = f"{outdir} {sim_type} {config} {detectorVersion} {detectorOption} {seed} {train_or_val} {use_lr} {base_path}"
                arguments_list.append(argts)
                jobCount += 1
                if jobCount == 1:
                    print("")
                    print(f"rm -rf job*; ./{script} {argts}")

    gun_name = f"gun/{sim_type}_{config}.sub"
    with open(gun_name, "w") as f:
        f.write(f"""executable    = {script}
            output        = std/condor.$(ClusterId).$(ProcId).out
            error         = std/condor.$(ClusterId).$(ProcId).err
            log           = std/condor.$(ClusterId).log

            +AccountingGroup = "group_u_FCC.local_gen"
            +JobFlavour      = "{queue}"
            
            RequestCpus = 3
            arguments = $(ARGS)
            queue ARGS from (
            """)
        
        for args in arguments_list:
            f.write(f"{args}\n")
        f.write(")\n")

    if jobCount > 0:
        print("")
        print(f"[Submitting {jobCount} jobs] ... ")
        os.system(f"condor_submit {gun_name}")

# _______________________________________________________________________________________
if __name__ == "__main__":

    script_dir = Path(__file__).resolve().parent
    work_dir = script_dir
    
    while not (work_dir / "data_creation").is_dir() and work_dir != work_dir.parent:
        work_dir = work_dir.parent
        
    if not (work_dir / "data_creation").is_dir():
        raise RuntimeError("Could not find WORK_DIR containing data_creation")

    print("Project root (WORK_DIR):", work_dir)
    
    main(work_dir)
