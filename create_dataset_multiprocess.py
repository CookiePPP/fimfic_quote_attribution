import argparse
import subprocess

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_rank', type=int, default=16)
    args = parser.parse_args()
    
    # run n_rank copies of create_dataset.py
    # (and kill them all when this script is killed)
    processes = []
    for rank in range(args.n_rank):
        processes.append(subprocess.Popen(["python", "create_dataset.py", "--rank", str(rank), "--n_rank", str(args.n_rank)]))
    
    try:
        for p in processes:
            p.wait()
    except KeyboardInterrupt:
        for p in processes:
            p.kill()
        for p in processes:
            p.wait()