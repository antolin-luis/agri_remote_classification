import geopandas as gpd
import ee
from pathlib import Path
from typing import Union, List, Tuple

class GeometryHandler:
    def __init__(self):
        """Initialize the geometry handler"""
        try:
            ee.Initialize()
        except Exception as e:
            print("Error initializing Earth Engine. Make sure you're authenticated.")
            raise e

    def load_kml(self, kml_path: Union[str, Path]) -> List[ee.Geometry]:
        """
        Load KML file and convert geometries to Earth Engine format
        
        Args:
            kml_path: Path to the KML file
            
        Returns:
            List of ee.Geometry objects
        """
        # Read KML using geopandas
        gdf = gpd.read_file(kml_path, driver='KML')
        return self._gdf_to_ee_geometries(gdf)

    def load_geodataframe(self, gdf: gpd.GeoDataFrame) -> List[ee.Geometry]:
        """
        Convert GeoDataFrame geometries to Earth Engine format
        
        Args:
            gdf: Input GeoDataFrame
            
        Returns:
            List of ee.Geometry objects
        """
        return self._gdf_to_ee_geometries(gdf)

    def load_kml_directory(self, directory: Union[str, Path]) -> List[Tuple[List[ee.Geometry], str, Path]]:
        """
        Load all KML files from a directory and convert them to Earth Engine geometries.
        Process each KML file independently to avoid memory overload.
        
        Args:
            directory (Union[str, Path]): Path to directory containing KML files
            
        Returns:
            List[Tuple[List[ee.Geometry], str, Path]]: List of tuples containing:
                - List of Earth Engine geometries for each KML
                - KML filename without extension
                - Full path to the KML file
            
        Raises:
            ValueError: If no KML files are found in the directory
        """
        # Convert to Path object if string
        directory = Path(directory)
        if not directory.exists():
            raise ValueError(f"Directory {directory} does not exist")
        
        # Find all KML files
        kml_files = list(directory.glob('*.kml'))

        if not kml_files:
            raise ValueError(f"No KML files found in {directory}")
        
        # Process each KML file independently
        kml_data = []
        
        for kml_file in kml_files:
            # Read KML file
            gdf = gpd.read_file(kml_file, driver='KML')
            
            # Convert geometries for this KML
            geometries = []
            for _, row in gdf.iterrows():
                geometry = row['geometry']
                ee_geometry = self._convert_to_ee_geometry(geometry)
                geometries.append(ee_geometry)
            
            print(f"Loaded {len(geometries)} geometries from {kml_file.name}")
            
            # Store the geometries with the KML info
            kml_data.append((geometries, kml_file.stem, kml_file))
        
        print(f"\nProcessed {len(kml_files)} KML files")
        return kml_data

    def _gdf_to_ee_geometries(self, gdf: gpd.GeoDataFrame) -> List[ee.Geometry]:
        """
        Convert GeoDataFrame geometries to Earth Engine format
        
        Args:
            gdf: Input GeoDataFrame
            
        Returns:
            List of ee.Geometry objects
        """
        ee_geometries = []
        
        for idx, row in gdf.iterrows():
            geom = row.geometry
            
            # Convert to GeoJSON
            geojson = gpd.GeoSeries([geom]).__geo_interface__
            
            # Create EE geometry
            ee_geometry = ee.Geometry(geojson['features'][0]['geometry'])
            ee_geometries.append(ee_geometry)
        
        return ee_geometries

    def _convert_to_ee_geometry(self, geometry):
        geojson = gpd.GeoSeries([geometry]).__geo_interface__
        return ee.Geometry(geojson['features'][0]['geometry'])

    @staticmethod
    def geojson_to_ee_geometry(geojson: dict) -> ee.Geometry:
        """
        Convert a GeoJSON dictionary to Earth Engine geometry
        
        Args:
            geojson: GeoJSON dictionary representation of a geometry
            
        Returns:
            ee.Geometry: Earth Engine geometry object
        """
        # Convert GeoJSON to Earth Engine Geometry
        try:
            ee_geometry = ee.Geometry(geojson)
            return ee_geometry
        except Exception as e:
            print(f"Error converting GeoJSON to Earth Engine geometry: {str(e)}")
            raise
