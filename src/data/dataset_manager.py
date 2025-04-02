from pathlib import Path
from typing import List, Dict, Tuple
import geopandas as gpd
import ee
from .geometry_handler import GeometryHandler

class DatasetManager:
    def __init__(self, train_val_dir: str, prediction_dir: str = None):
        """
        Initialize the dataset manager
        
        Args:
            train_val_dir: Directory containing training and validation KML files
            prediction_dir: Optional directory containing KML files for prediction
        """
        self.train_val_dir = Path(train_val_dir)
        self.prediction_dir = Path(prediction_dir) if prediction_dir else None
        self.geometry_handler = GeometryHandler()
        
    def load_train_val_data(self) -> List[Tuple[List[ee.Geometry], str, Path]]:
        """
        Load all training and validation KML files independently
        
        Returns:
            List of tuples containing:
                - List of Earth Engine geometries for each KML
                - KML filename without extension
                - Full path to the KML file
        """
        return self.geometry_handler.load_kml_directory(self.train_val_dir)
    
    def load_prediction_data(self) -> List[Tuple[List[ee.Geometry], str, Path]]:
        """
        Load all prediction KML files independently
        
        Returns:
            List of tuples containing:
                - List of Earth Engine geometries for each KML
                - KML filename without extension
                - Full path to the KML file
        """
        if not self.prediction_dir:
            raise ValueError("Prediction directory not specified")
            
        return self.geometry_handler.load_kml_directory(self.prediction_dir)
