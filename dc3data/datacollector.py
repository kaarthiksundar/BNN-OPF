import subprocess

filenames = ['random_nonconvex_dataset_var70_ineq50_eq20_ex5000.npz',
             'random_nonconvex_dataset_var20_ineq5_eq10_ex5000.npz']
seeds = [0, 1, 2, 3, 4]  # 5 trials

for filename in filenames:
    for seed in seeds:
        subprocess.run([
            'python', 'train_sandwich.py',
            '--data', filename,
            '--seed', str(seed),
        ])
