import torch
import multiprocessing
import os
import argparse
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from batchgenerators.utilities.file_and_folder_operations import load_json, join
from training.trainer.mamba_trainer import MambaTrainer
from paths import nnUNet_preprocessed, nnUNet_results

def run_training():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke-test', action='store_true', help='Run a quick 10-iteration test')
    args = parser.parse_args()

    # 1. Setup paths for Dataset007 (Pancreas)
    dataset_name = 'Dataset007_Pancreas'
    plans_file = join(nnUNet_preprocessed, dataset_name, 'nnUNetPlans.json')
    dataset_json_file = join(nnUNet_preprocessed, dataset_name, 'dataset.json')
    
    print(f"Loading plans from: {plans_file}")
    plans = load_json(plans_file)
    dataset_json = load_json(dataset_json_file)

    # 2. Instantiate MambaTrainer
    # fold=0, configuration='2d'
    trainer = MambaTrainer(
        plans=plans,
        configuration='2d',
        fold=0,
        dataset_json=dataset_json,
        device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    )

    # 3. Setup Smoke Test params
    if args.smoke_test:
        print("\n!!! SMOKE TEST MODE ACTIVE !!!")
        trainer.num_epochs = 1000
        trainer.initial_lr = 1e-4
        trainer.num_iterations_per_epoch = 100
        trainer.num_val_iterations_per_epoch = 20
        trainer.save_every = 50
    else:
        # Standard training hparams
        trainer.num_epochs = 1000
        trainer.initial_lr = 1e-4  # Mamba often likes slightly lower LR than CNNs
        trainer.num_iterations_per_epoch = 250
        trainer.num_val_iterations_per_epoch = 50
        trainer.save_every = 50

    # 4. Initialize and Run
    trainer.initialize()
    print("\nStarting training loop...")
    # 2. RESUME FROM EPOCH 6
    import os
    checkpoint_path = os.path.join(trainer.output_folder, "checkpoint_final.pth")
    if os.path.exists(checkpoint_path):
        print(f"Resuming training from {checkpoint_path}...")
        trainer.load_checkpoint(checkpoint_path)

    # 3. Start the run! (It will instantly jump to Epoch 7)
    
    trainer.run_training()

if __name__ == '__main__':
    # This protects the background workers from accidentally running the main script!
    multiprocessing.freeze_support() # Optional, but good practice on Windows
    run_training()
