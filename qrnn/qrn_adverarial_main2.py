import os
import shutil
from typing import List, Tuple, Dict, Optional
import numpy as np
import jax
import jax.numpy as jnp
import optax
import pennylane as qml
import time
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from jax import config
from data_loader import load_mnist, load_cifar2, load_old_cifar2
from qrnn_utils2 import (
    binary_cross_entropy, accuracy, make_forward_pass, make_train_step, make_evaluate,
    create_optimizer, create_batches, FGSM, PGD, APGD, MIM,
    save_to_csv, save_to_json, plot_confusion_matrix, plot_multi_barplot
)

config.update("jax_enable_x64", False)

class QRNN:
    def __init__(self, anc_q: int, n_qub_enc: int, seq_num: int, D: int, encoding_type: str = 'angle'):
        self.anc_q = anc_q
        self.n_qub_enc = n_qub_enc
        self.seq_num = seq_num
        self.D = D
        self.encoding_type = encoding_type
        self.circuit = self._create_circuit()
        self.params = self._init_params()

    def _create_circuit(self) -> callable:
        num_ansatz_q = self.anc_q + self.n_qub_enc
        num_q = self.n_qub_enc * self.seq_num + self.anc_q
        dev = qml.device("default.qubit", wires=num_q)

        @qml.qnode(dev, interface="jax")
        def circuit(inputs: jnp.ndarray, weights: jnp.ndarray) -> float:
            for i in range(self.seq_num):
                start = i * self.n_qub_enc
                if self.encoding_type == 'angle':
                    for j in range(self.n_qub_enc):
                        qml.RY(inputs[start + j], j + self.anc_q)
                else:
                    raise ValueError(f"Unknown encoding type: {self.encoding_type}")
                
                num_para_per_bloc = num_ansatz_q * (3 * self.D + 2)
                block_weights = weights[i * num_para_per_bloc:(i + 1) * num_para_per_bloc]
                idx = 0
                for j in range(num_ansatz_q):
                    qml.RX(block_weights[idx], wires=j)
                    qml.RZ(block_weights[idx + 1], wires=j)
                    qml.RX(block_weights[idx + 2], wires=j)
                    idx += 3
                for d in range(self.D):
                    for j in range(num_ansatz_q):
                        qml.IsingZZ(block_weights[idx], wires=[j, (j + 1) % num_ansatz_q])
                        idx += 1
                    for j in range(num_ansatz_q):
                        qml.RY(block_weights[idx], wires=j)
                        idx += 1
                if i != self.seq_num - 1:
                    for j in range(self.n_qub_enc):
                        qml.SWAP(wires=[j + self.anc_q, (i + 1) * self.n_qub_enc + j + self.anc_q])
            return qml.expval(qml.sum(*[qml.PauliZ(i) for i in range(num_ansatz_q)]))
        
        return circuit

    def _init_params(self) -> jnp.ndarray:
        key = jax.random.PRNGKey(0)
        num_ansatz_q = self.anc_q + self.n_qub_enc
        num_para_per_bloc = num_ansatz_q * (3 * self.D + 2)
        total_params = num_para_per_bloc * self.seq_num
        return jax.random.uniform(key, (total_params,), minval=-jnp.pi, maxval=jnp.pi, dtype=jnp.float32)

    def update_params(self, new_params: jnp.ndarray) -> None:
        self.params = new_params

def clear_dir() -> None:
    """Manually clear all contents of QRNN_JAX directories."""
    main_folder = 'QRNN_JAX'
    subdirs = ['adversarial_results', 'plots', 'confusion_matrix']
    for subdir in subdirs:
        path = os.path.join(main_folder, subdir)
        if os.path.exists(path):
            shutil.rmtree(path)
            print(f"Cleared directory: {path}")
        os.makedirs(path, exist_ok=True)
        print(f"Recreated directory: {path}")

def get_batch_size(n_train: int) -> int:
    """Determine batch size based on training size."""
    if n_train <= 1000:
        return 64
    elif n_train <= 5000:
        return 128
    elif n_train <= 10000:
        return 256
    else:
        return 512

def train_and_evaluate_adversarial(
    n_train: int = 100,
    n_test: int = 100,
    n_epochs: int = 10,
    dataset: str = 'mnist',
    classify_choice: List[int] = [1, 7],
    feature_size: str = '2x2',
    desc: str = '',
    mode: str = 'a',
    log: bool = True,
    clear_dir_auto: bool = False,
    use_batching: bool = True
) -> Optional[Dict]:
    main_folder = 'QRNN_JAX'
    header = mode == 'w'
    
    if clear_dir_auto and log and header:
        clear_dir()

    # Load dataset
    if dataset == 'mnist':
        x_train, y_train, x_test, y_test = load_mnist(n_train, n_test, classify_choice=classify_choice, show_size=False)
        seq_num = 5
    elif dataset == 'cifar2':
        seq_len = 4 if feature_size == '2x2' else 9
        x_train, y_train, x_test, y_test = load_cifar2(n_train, n_test, seq_len=4, classify_choice=classify_choice)
        seq_num = 4
    else:
        raise ValueError("Available datasets: 'mnist', 'cifar2'")

    # Initialize model and training components
    model = QRNN(anc_q=2, n_qub_enc=1, seq_num=seq_num, D=1)
    optimizer = create_optimizer()
    opt_state = optimizer.init(model.params)
    train_step = make_train_step(model.circuit, optimizer)
    evaluate = make_evaluate(model.circuit)
    forward_pass = make_forward_pass(model.circuit)
    batch_size = get_batch_size(n_train)

    # Initialize metrics storage
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []
    metrics_per_epoch = {
        'clean_accuracy': [],
        'FGSM': {'adv_accuracy': [], 'asr': [], 'robustness_gap': [], 'fidelity': []},
        'PGD': {'adv_accuracy': [], 'asr': [], 'robustness_gap': [], 'fidelity': []},
        'APGD': {'adv_accuracy': [], 'asr': [], 'robustness_gap': [], 'fidelity': []},
        'MIM': {'adv_accuracy': [], 'asr': [], 'robustness_gap': [], 'fidelity': []}
    }
    total_time = 0
    print(f"Training QRNN on {dataset} ({len(x_train)} train, {len(x_test)} test, {n_epochs} epochs, batch_size={batch_size})...")
    
    attack_configs = [
        ('FGSM', {'eps': 2/255}, lambda x, y, p: FGSM(forward_pass, model.params, x, y, **p, batch_size=batch_size)),
        ('PGD', {'eps': 2/255, 'steps': 100}, lambda x, y, p: PGD(forward_pass, model.params, x, y, **p, alpha=2/(255*100), batch_size=batch_size)),
        ('APGD', {'eps': 2/255, 'steps': 100}, lambda x, y, p: APGD(forward_pass, model.params, x, y, **p, alpha=2/(255*100), batch_size=batch_size)),
        ('MIM', {'eps': 2/255, 'steps': 100}, lambda x, y, p: MIM(forward_pass, model.params, x, y, **p, alpha=2/(255*100), batch_size=batch_size))
    ]
    print('Attacks parameters initialized.')
    # Training loop with loss, accuracy, and adversarial metrics
    for epoch in range(n_epochs):
        start_epoch = time.time()
        key = jax.random.PRNGKey(epoch)
        train_loss_epoch, num_batches = 0.0, 0
        
        for x_batch, y_batch in create_batches(x_train, y_train, batch_size, key):
            model.params, opt_state, batch_loss = train_step(model.params, opt_state, x_batch, y_batch)
            train_loss_epoch += batch_loss
            num_batches += 1
        
        train_loss_epoch /= num_batches
        train_loss_val, train_acc_val = evaluate(model.params, x_train, y_train)
        test_loss_val, test_acc_val = evaluate(model.params, x_test, y_test)
        
        train_losses.append(float(train_loss_epoch))
        test_losses.append(float(test_loss_val))
        train_accs.append(float(train_acc_val))
        test_accs.append(float(test_acc_val))
        metrics_per_epoch['clean_accuracy'].append(float(test_acc_val))
        print(f'startepoch{epoch}.')
        # Compute adversarial metrics for the epoch
        if use_batching:
            num_samples = x_test.shape[0]
            num_batches = (num_samples + batch_size - 1) // batch_size
            clean_loss, clean_acc = evaluate(model.params, x_test, y_test)
            print(f'Epoch {epoch} Clean acc : {clean_acc}\nClean loss : {clean_loss}')
            for attack_name, attack_params, attack_fn in attack_configs:
                x_adv_full = []
                for i in range(num_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, num_samples)
                    x_batch = x_test[start_idx:end_idx]
                    y_batch = y_test[start_idx:end_idx]
                    x_adv_full.append(attack_fn(x_batch, y_batch, attack_params))
                x_adv_full = jnp.concatenate(x_adv_full, axis=0)

                adv_loss, adv_acc = evaluate(model.params, x_adv_full, y_test)
                asr = 1.0 - adv_acc
                robustness_gap = clean_acc - adv_acc
                print(f'Adv_loss : {adv_loss}, Adv_acc : {adv_acc}, Asr : {asr}, robustness_gap: {robustness_gap}')
                
                # Fix fidelity computation: use same samples for comparison
                fidel_clean = forward_pass(model.params, x_test)  # Clean predictions on test set
                fidel_adv = forward_pass(model.params, x_adv_full)  # Adversarial predictions on same samples
                
                # Proper fidelity calculation using L2 distance between prediction vectors
                # Fidelity = 1 - normalized_distance (higher is better)
                pred_diff = jnp.linalg.norm(fidel_clean - fidel_adv, axis=1)  # L2 distance per sample
                max_possible_diff = jnp.sqrt(2.0)  # Maximum possible L2 distance between [0,1] predictions
                fidel_val = jnp.mean(1.0 - pred_diff / max_possible_diff)  # Normalize to [0,1]
                
                print(f'Fidelity: {fidel_val}')
                metrics_per_epoch[attack_name]['adv_accuracy'].append(float(adv_acc))
                metrics_per_epoch[attack_name]['asr'].append(float(asr))
                metrics_per_epoch[attack_name]['robustness_gap'].append(float(robustness_gap))
                metrics_per_epoch[attack_name]['fidelity'].append(float(fidel_val))
        total_time += time.time() - start_epoch
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}: Train Loss: {train_loss_val:.4f}, Acc: {train_acc_val:.4f}")
            print(f"Test Loss: {test_loss_val:.4f}, Acc: {test_acc_val:.4f}")
            print(f"Approx 10 epochs time: {10*(time.time() - start_epoch):.2f} seconds")

    # Compute final metrics
    y_pred = (forward_pass(model.params, x_test) > 0.5).astype(int).flatten()
    precision, recall, f1 = map(float, (precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1_score(y_test, y_pred)))

    print(f"Total training time: {total_time:.2f} seconds")

    # Save or display confusion matrix
    if log:
        plot_confusion_matrix(y_test, y_pred, dataset, classify_choice, main_folder, desc, n_train, log)

    # Save metrics to JSON
    if log:
        filename = f"{dataset}_{desc}_metrics.json"
        save_path = f'{main_folder}/adversarial_results/{filename}'
        save_to_json(metrics_per_epoch, save_path)
        print(f"\nAdversarial metrics for all epochs saved to {save_path}")

    return {
        'dataset_name': f'{dataset}_ntrain{n_train}_{desc}',
        'clean_acc': float(test_accs[-1]) if test_accs else None,
        'adv_metrics': [
            {
                'Attack': attack_name,
                'Adversarial_Acc': metrics_per_epoch[attack_name]['adv_accuracy'][-1],
                'Attack_Success_Rate': metrics_per_epoch[attack_name]['asr'][-1],
                'Robustness_Gap': metrics_per_epoch[attack_name]['robustness_gap'][-1],
                'Fidelity': metrics_per_epoch[attack_name]['fidelity'][-1]
            } for attack_name, _, _ in attack_configs
        ],
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    } if log else None

def run_experiments(experiments: List[Dict], train_sizes: List[int],
                    test_sizes: List[int], n_reps: int = 5, desc: str = '',
                    clear_dir_auto: bool = False, log: bool = True) -> None:
    all_results = []
    main_folder = 'QRNN_JAX'
    
    for exp in experiments:
        for idx, (n_train, n_test) in enumerate(zip(train_sizes, test_sizes)):
            metrics_per_epoch_all = {
                'clean_accuracy': [],
                'FGSM': {'adv_accuracy': [], 'asr': [], 'robustness_gap': [], 'fidelity': []},
                'PGD': {'adv_accuracy': [], 'asr': [], 'robustness_gap': [], 'fidelity': []},
                'APGD': {'adv_accuracy': [], 'asr': [], 'robustness_gap': [], 'fidelity': []},
                'MIM': {'adv_accuracy': [], 'asr': [], 'robustness_gap': [], 'fidelity': []}
            }
            train_loss_lsts_size = []
            test_loss_lsts_size = []
            train_acc_lsts_size = []
            test_acc_lsts_size = []
            clean_accs = []
            
            # Run repetitions
            for rep in range(n_reps):
                print(f"Training for {exp['dataset']} with {n_train} train, {n_test} test, rep {rep + 1}/{n_reps}")
                mode = exp['mode'] if idx == 0 and rep == 0 else 'a'
                result = train_and_evaluate_adversarial(
                    n_train=n_train,
                    n_test=n_test,
                    n_epochs=100,
                    dataset=exp['dataset'],
                    classify_choice=exp['classify_choice'],
                    desc=f"{desc + exp['desc']}",
                    mode=mode,
                    log=log,  # Don't log individual runs to JSON
                    clear_dir_auto=clear_dir_auto and mode == 'w',
                    use_batching=True
                )
                if result:
                    train_loss_lsts_size.append(result['train_losses'])
                    test_loss_lsts_size.append(result['test_losses'])
                    train_acc_lsts_size.append(result['train_accs'])
                    test_acc_lsts_size.append(result['test_accs'])
                    if result['clean_acc'] is not None:
                        clean_accs.append(result['clean_acc'])
                    # Aggregate epoch-wise metrics
                    for attack in ['FGSM', 'PGD', 'APGD', 'MIM']:
                        for metric in ['adv_accuracy', 'asr', 'robustness_gap', 'fidelity']:
                            # Find the dictionary in adv_metrics where 'Attack' matches the current attack
                            adv_metric = next(item for item in result['adv_metrics'] if item['Attack'] == attack)
                            metrics_per_epoch_all[attack][metric].append(adv_metric[metric])

            # Compute mean metrics across repetitions
            if train_loss_lsts_size:
                mean_train_l = np.mean(train_loss_lsts_size, axis=0)
                mean_test_l = np.mean(test_loss_lsts_size, axis=0)
                mean_train_acc = np.mean(train_acc_lsts_size, axis=0)
                mean_test_acc = np.mean(test_acc_lsts_size, axis=0)
                mean_clean_acc = float(np.mean(clean_accs)) if clean_accs else None
                
                # Compute mean adversarial metrics per epoch
                mean_metrics_per_epoch = {
                    'clean_accuracy': np.mean(metrics_per_epoch_all['clean_accuracy'], axis=0).tolist(),
                    'FGSM': {k: np.mean(v, axis=0).tolist() for k, v in metrics_per_epoch_all['FGSM'].items()},
                    'PGD': {k: np.mean(v, axis=0).tolist() for k, v in metrics_per_epoch_all['PGD'].items()},
                    'APGD': {k: np.mean(v, axis=0).tolist() for k, v in metrics_per_epoch_all['APGD'].items()},
                    'MIM': {k: np.mean(v, axis=0).tolist() for k, v in metrics_per_epoch_all['MIM'].items()}
                }
                
                # Save mean metrics to JSON
                if log:
                    filename = f"{exp['dataset']}_{desc + exp['desc']}_metrics.json"
                    save_path = f'{main_folder}/adversarial_results/{filename}'
                    save_to_json(mean_metrics_per_epoch, save_path)
                
                all_results.append({
                    'dataset_name': f"{exp['dataset']}_ntrain{n_train}_{desc + exp['desc']}",
                    'clean_acc': mean_clean_acc,
                    'adv_metrics': [
                        {
                            'Attack': attack,
                            'Adversarial_Acc': mean_metrics_per_epoch[attack]['adv_accuracy'][-1],
                            'Attack_Success_Rate': mean_metrics_per_epoch[attack]['asr'][-1],
                            'Robustness_Gap': mean_metrics_per_epoch[attack]['robustness_gap'][-1],
                            'Fidelity': mean_metrics_per_epoch[attack]['fidelity'][-1]
                        } for attack in ['FGSM', 'PGD', 'APGD', 'MIM']
                    ],
                    'train_losses': mean_train_l.tolist(),
                    'test_losses': mean_test_l.tolist(),
                    'train_accs': mean_train_acc.tolist(),
                    'test_accs': mean_test_acc.tolist()
                })
                
                print(f"Mean metrics for {exp['dataset']} with {n_train} train: "
                      f"Train Loss: {mean_train_l[-1]:.4f}, Test Loss: {mean_test_l[-1]:.4f}, "
                      f"Train Acc: {mean_train_acc[-1]:.4f}, Test Acc: {mean_test_acc[-1]:.4f}")

    # Plot final comparison plots
    if log:
        plot_multi_barplot(all_results, 'Accuracy', 'Clean Accuracy Across Datasets', 'Accuracy', main_folder, 'multi_barplot_clean_accuracy.png')
        plot_multi_barplot(all_results, 'Adversarial_Acc', 'Adversarial Accuracy Across Datasets', 'Adversarial Accuracy', main_folder, 'multi_barplot_adversarial_accuracy.png')
        plot_multi_barplot(all_results, 'Attack_Success_Rate', 'Attack Success Rate Across Datasets', 'Attack Success Rate', main_folder, 'multi_barplot_attack_success_rate.png')
        plot_multi_barplot(all_results, 'Fidelity', 'Fidelity Across Datasets', 'Fidelity', main_folder, 'multi_barplot_fidelity.png')

if __name__ == "__main__":
    train_sizes = [10000]
    test_sizes = [i // 5 for i in train_sizes]
    n_reps = 1
    experiments = [
        # {'dataset': 'mnist', 'classify_choice': [1, 7], 'desc': '1_7', 'mode': 'a'},
        {'dataset': 'mnist', 'classify_choice': [6, 9], 'desc': '6_9', 'mode': 'a'},
        {'dataset': 'mnist', 'classify_choice': [0, 8], 'desc': '0_8', 'mode': 'a'},
        {'dataset': 'cifar2', 'classify_choice': [0, 1], 'desc': 'plane_automobile_cnn', 'mode': 'a'},
        {'dataset': 'cifar2', 'classify_choice': [3, 5], 'desc': 'dog_cat_cnn', 'mode': 'a'},
    ]
    run_experiments(experiments, train_sizes, test_sizes, n_reps=n_reps, clear_dir_auto=False, log=True)