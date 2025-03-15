import os

def contar_dicoms(base_dir):
    images_dir = os.path.join(base_dir, "images")
    total_dicoms = 0
    total_exames = 0
    
    if not os.path.exists(images_dir):
        print(f"Erro: O diretório {images_dir} não existe.")
        return total_exames, total_dicoms
    
    for exame in os.listdir(images_dir):
        caminho_exame = os.path.join(images_dir, exame)
        if os.path.isdir(caminho_exame):
            total_exames += 1
            dicom_files = [f for f in os.listdir(caminho_exame) if f.lower().endswith(".dicom")]
            total_dicoms += len(dicom_files)
        else:
            print(f"Aviso: {caminho_exame} não é um diretório.")
    
    return total_exames, total_dicoms

if __name__ == "__main__":
    base_dir = "/media/felipe-matsuoka/FelipeSSD/datasets/physionet.org/files/vindr-mammo/1.0.0"
    
    expected_exames = 5000
    expected_dicoms = expected_exames * 4
    
    exames, dicoms = contar_dicoms(base_dir)
    print(f"Exames encontrados: {exames} (esperado: {expected_exames})")
    print(f"Arquivos DICOM encontrados: {dicoms} (esperado: {expected_dicoms})")
    
    if dicoms < expected_dicoms:
        faltam = expected_dicoms - dicoms
        print(f"Faltam {faltam} arquivos DICOM para completar o download.")
    else:
        print("Todos os arquivos DICOM estão presentes.")