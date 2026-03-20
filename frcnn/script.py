import glob
import os

images_path = "~/FRCNN/data/labman_pascal_voc/archive/train/images"
file_type = "*.jpg"
files = glob.glob(os.path.join(images_path, file_type))
print(f"Arquivos encontrados: {files}\nI_P: {images_path}")

images_path = "home/labinfo8/FRCNN/data/labman_pascal_voc/archive/train/images"
images_path = os.getcwd()+"/data/labman_pascal_voc/archive/train/images"
print(images_path)
# files = [f for f in os.listdir(images_path) if f.endswith('.jpg')]
# print(f"Arquivos encontrados: {files}")
