import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
import subprocess
import csv
import sys



python_path = sys.executable
script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'main_cold.py')
work_dir = os.path.dirname(script_path)
os.makedirs(os.path.join(work_dir, 'result'), exist_ok=True)



Deng ={'dropout': 0.5,      'rels': 65,      'n_epochs': 400,     'batch_size': 2048,
       'graph_alpha': 0.6,  'min_sim': 0.6,  'temperature': 0.2,  'cl_alpha': 0.7,
       'w_cl': 0.1,         'lr': 0.002,     'dataset': 'Deng'}

Lin ={'dropout': 0.3,      'rels': 100,      'n_epochs': 400,     'batch_size': 8192,
       'graph_alpha': 0.7,  'min_sim': 0.7,  'temperature': 0.05,  'cl_alpha': 0.7,
       'w_cl': 0.1,         'lr': 0.001,     'dataset': 'Lin'}




gpu = 0
folds = [0,1,2,3,4]
datasets = [Deng,Lin]




for dataset in datasets:
        for fold in folds:
                cmd = [python_path, script_path, '--dropout',str(dataset['dropout']), '--rels', str(dataset['rels']),
                       '--graph_alpha', str(dataset['graph_alpha']), '--min_sim', str(dataset['min_sim']),'--batch_size',str(dataset['batch_size']),
                       '--w_cl', str(dataset['w_cl']), '--lr', str(dataset['lr']), '--dataset', dataset['dataset'],
                        '--temperature',str(dataset['temperature']),'--n_epochs', str(dataset['n_epochs']), '--model', str('MMQF_cold'),
                       '--gpu_id', str(gpu), '--fold', str(fold), '--filename', f"result/{gpu}_{dataset['dataset']}_cold.csv", ]
                subprocess.run(cmd, cwd=work_dir, check=True)




