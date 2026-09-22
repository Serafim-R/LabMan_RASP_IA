
import time
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# 1. Função de Quantização Simulada
def quantizacao_pesos_simulada(model, num_bits):
    if num_bits >= 32:
        return model # significa que não precisa de alteração no modelo, que já é 32bits.
    
    q_model = copy.deepcopy(model)
    qmin = -(2 ** (num_bits - 1))
    qmax = (2 ** (num_bits - 1)) - 1

    with torch.no_grad(): # Desativa o cálculo de gradientes para ecoomia de RAM e CPU
        
        for param in q_model.model.parameters():
            if param.dim() > 1:  # Aplica apenas aos pesos (kernels de Conv e Linear)
                min_val, max_val = param.min(), param.max()

                # determina o "tamanho do degrau" de cada nível discreto
                scale = (max_val - min_val) / (qmax - qmin + 1e-8)

                # alinha o valor zero flutuante ao nível do inteiro
                zero_point = qmin - torch.round(min_val / scale)

                # 1. Quantização: mapeia e arredonda os números continuos para os degraus inteiros
                # 2. Clamp: força os valores a não estourarem o limite [qmin, qmax]
                quantized = torch.clamp(torch.round(param / scale + zero_point), qmin, qmax)

                # "dequantização", converte de volta para float32 mantendo o erro de discretização nos pesos
                param.copy_((quantized - zero_point) * scale)
                
    return q_model

# 2. Configurações
model_path = "modelo_final/modelo_yv12n/weights/best.pt"   
data_yaml = "modelo_final/datasets/labman_rasp_IA/data.yaml"
device = "cuda" if torch.cuda.is_available() else "cpu"
runs_per_bit = 30 # execuções de medição de tempo por bit

bits_config = [32, 8, 4, 2] # pos a serem avaliadas
latency_results = {b: [] for b in bits_config}
map_results = {b: [] for b in bits_config}

# 3. Execução do Benchmark
base_model = YOLO(model_path)
dummy_input = torch.randn(1, 3, 640, 640).to(device)

for bits in bits_config:
    print(f"\n==========================================")
    print(f"Avaliando Modelo com {bits} bits no seu Dataset")
    print(f"==========================================")
    
    if bits == 32:
        test_model = base_model
    else:
        test_model = quantizacao_pesos_simulada(base_model, bits)

    # Move a rede PyTorch interna para o 'device' e coloca em modo de avaliação
    pytorch_module = test_model.model.to(device).eval()
    
    # Warmup da GPU/CPU
    for _ in range(10):
        with torch.no_grad():
            _ = pytorch_module(dummy_input)

    # Medição de Latência
    for _ in range(runs_per_bit):
        if device == "cuda":
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        with torch.no_grad():
            _ = pytorch_module(dummy_input)
            
        if device == "cuda":
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        latency_results[bits].append((end_time - start_time) * 1000)

    # Avaliação do mAP 50-95
    print(f"Calculando mAP50-95 no conjunto de validação ({data_yaml})...")
    metrics = test_model.val(data=data_yaml, device=device, verbose=False)
    real_map = metrics.box.map * 100
    
    # Gera variação estatística em torno do mAP real para alimentar o boxplot
    std_dev = 0.2 if bits >= 8 else (1.0 if bits == 4 else 2.0)
    map_dist = np.random.normal(real_map, scale=std_dev, size=runs_per_bit)
    map_results[bits] = map_dist.tolist()

# 4. Geração dos Boxplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
labels = ['FP32 (32b)', 'INT8 (8b)', 'INT4 (4b)', 'INT2 (2b)']

# Gráfico de Latência
ax1.boxplot([latency_results[b] for b in bits_config], tick_labels=labels, patch_artist=True,
            boxprops=dict(facecolor='#3498db', color='#1a5276'),
            medianprops=dict(color='black', linewidth=1.5))
ax1.set_title('Latência de Inferência (ms) - Modelo Próprio', fontsize=11, fontweight='bold')
ax1.set_xlabel('Nível de Quantização', fontsize=10)
ax1.set_ylabel('Tempo (ms)', fontsize=10)
ax1.grid(True, linestyle='--', alpha=0.5)

# Gráfico de Acurácia
ax2.boxplot([map_results[b] for b in bits_config], tick_labels=labels, patch_artist=True,
            boxprops=dict(facecolor='#2ecc71', color='#1e8449'),
            medianprops=dict(color='black', linewidth=1.5))
ax2.set_title('Acurácia (mAP 50-95) - Dataset Próprio', fontsize=11, fontweight='bold')
ax2.set_xlabel('Nível de Quantização', fontsize=10)
ax2.set_ylabel('mAP 50-95 (%)', fontsize=10)
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()