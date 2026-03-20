import os
import datetime

def create_output_directory(base_dir="outputs"):
    # Cria um diretório base se não existir
    os.makedirs(base_dir, exist_ok=True)
    # Gera um nome único usando a data e hora
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"output_{timestamp}")
    # Cria o diretório único
    os.makedirs(output_dir, exist_ok=True)
    return output_dir
