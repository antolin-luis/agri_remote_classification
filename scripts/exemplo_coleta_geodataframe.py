"""
Exemplo de uso do SentinelCollector com GeoDataFrame
"""

import geopandas as gpd
from src.data.gee_collector import SentinelCollector
import ee

def main():
    # Inicializar Earth Engine
    ee.Initialize()
    
    # Criar SentinelCollector
    collector = SentinelCollector(
        train_val_dir='data/train_val_kmls',
        prediction_dir='data/prediction_kmls'
    )
    
    # Carregar seu GeoDataFrame
    # Exemplo: pode ser de um arquivo shapefile, GeoJSON, etc.
    gdf = gpd.read_file('caminho/para/seu/arquivo.shp')  # Ajuste o caminho
    
    # Definir parâmetros da coleta
    start_date = '2023-01-01'
    end_date = '2023-12-31'
    export_folder = 'sentinel_exports_gdf'
    
    # Coletar dados
    collector.collect_from_geodataframe(
        gdf=gdf,
        start_date=start_date,
        end_date=end_date,
        export_folder=export_folder,
        patch_size=64,
        scale=10,
        max_cloud_cover=20.0
    )

if __name__ == "__main__":
    main()
