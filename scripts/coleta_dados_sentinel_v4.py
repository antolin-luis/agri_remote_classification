import sys
sys.path.append('..')

import ee
from pathlib import Path
from src.data.gee_collector import SentinelCollector

# Configurar diretórios
data_dir = Path('../data')
kml_dir = data_dir / 'kmls'  # Diretório para todos os KMLs

# Criar diretório se não existir
kml_dir.mkdir(parents=True, exist_ok=True)

# Inicializar o coletor
collector = SentinelCollector(train_val_dir=str(kml_dir))

# Configurar parâmetros de exportação
export_params = {
    'scale': 10,  # Resolução de 10 metros
    'bands': ['B8', 'B4', 'B11'],  # Bandas relevantes para vegetação
    'folder': 'sentinel2_series'  # Pasta no Google Drive
}

print("\nIniciando processamento dos KMLs...")
print("Cada KML será processado independentemente para otimizar o uso de memória.")

# Processar cada KML independentemente
train_val_data = collector.dataset_manager.load_train_val_data()

for geometries, kml_name, kml_path in train_val_data:
    print(f"\nProcessando KML: {kml_name}")
    print(f"Número de geometrias: {len(geometries)}")
    
    # Criar collection específica para este KML
    collection = collector.get_sentinel_collection(
        regions=geometries,
        years_back=2,  # Últimos 2 anos de dados
        max_cloud_cover=20  # Máximo de 20% de cobertura de nuvens
    )
    
    # Exportar imagens para este KML
    collector.export_to_drive(
        collection=collection,
        regions=geometries,
        output_prefix=f"sentinel2_{kml_name}",
        **export_params
    )
    
    print(f"Exportação iniciada para {kml_name}")

print("\nImportante:")
print("1. As imagens serão exportadas para a pasta 'sentinel2_series' no seu Google Drive")
print("2. Cada imagem terá o nome no formato: sentinel2_<nome_kml>_YYYY-MM-DD_region_N")
print("3. As imagens serão exportadas no formato Cloud Optimized GeoTIFF")
print("4. Cada KML é processado independentemente para evitar sobrecarga de memória")
print("\nVocê pode monitorar o progresso em: https://code.earthengine.google.com/tasks")
