import ee
import numpy as np
from datetime import datetime, timedelta
import os
from typing import List, Union, Tuple, Dict
from pathlib import Path
from dateutil.relativedelta import relativedelta
import geopandas as gpd
from .geometry_handler import GeometryHandler
from .dataset_manager import DatasetManager

class SentinelCollector:
    def __init__(self, train_val_dir: str = None, prediction_dir: str = None):
        """
        Initialize the Earth Engine API and dataset manager
        
        Args:
            train_val_dir: Directory containing training and validation KML files
            prediction_dir: Optional directory containing KML files for prediction
        """
        try:
            ee.Initialize()
        except Exception as e:
            print("Error initializing Earth Engine. Make sure you're authenticated.")
            raise e
        
        self.dataset_manager = DatasetManager(train_val_dir, prediction_dir)
        self.geometry_handler = GeometryHandler()

    def _get_date_range(self, years_back: int = 5) -> Tuple[str, str]:
        """
        Calculate the date range from current date to N years back
        
        Args:
            years_back: Number of years to look back
            
        Returns:
            Tuple of (start_date, end_date) in 'YYYY-MM-DD' format
        """
        end_date = datetime.now()
        start_date = end_date - relativedelta(years=years_back)
        
        return (
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )

    def load_regions(self, source: Union[str, Path, gpd.GeoDataFrame]) -> List[ee.Geometry]:
        """
        Load regions from KML file or GeoDataFrame
        
        Args:
            source: KML file path or GeoDataFrame
            
        Returns:
            List of ee.Geometry objects
        """
        if isinstance(source, (str, Path)):
            return self.geometry_handler.load_kml(source)
        elif isinstance(source, gpd.GeoDataFrame):
            return self.geometry_handler.load_geodataframe(source)
        else:
            raise ValueError("Source must be a path to KML file or a GeoDataFrame")

    def get_sentinel_collection(self, regions: List[ee.Geometry], 
                              years_back: int = 5,
                              start_date: str = None,
                              end_date: str = None,
                              max_cloud_cover: float = 20):
        """
        Get Sentinel-2 image collection for specified parameters, selecting the best image
        (lowest cloud cover) for each month in the time period.
        
        Args:
            regions: List of ee.Geometry objects defining regions of interest
            years_back: Number of years to look back (ignored if start_date and end_date are provided)
            start_date: Optional, start date in 'YYYY-MM-DD' format
            end_date: Optional, end date in 'YYYY-MM-DD' format
            max_cloud_cover: Maximum cloud cover percentage
        
        Returns:
            ee.ImageCollection: Filtered Sentinel-2 collection with one best image per month
        """
        # Combine regions into a FeatureCollection
        region = ee.FeatureCollection(regions).geometry()
        
        # Get date range if not provided
        if start_date is None or end_date is None:
            start_date, end_date = self._get_date_range(years_back)
        
        # Get initial collection filtered by region, date and cloud cover
        collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                     .filterBounds(region)
                     .filterDate(start_date, end_date)
                     .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', max_cloud_cover))
                     .select(['B8', 'B4', 'B11']))
        
        # Function to get the best image for a specific year and month
        def get_best_monthly_image(year, month):
            # Create date range for the month
            start = ee.Date.fromYMD(year, month, 1)
            end = start.advance(1, 'month')
            
            # Filter collection to the month and sort by cloud cover
            monthly = (collection
                      .filterDate(start, end)
                      .sort('CLOUDY_PIXEL_PERCENTAGE'))
            
            # Return the first image (lowest cloud cover) or null if no images
            return ee.Image(monthly.first())
        
        # Generate list of all year-month combinations
        start_date = ee.Date(start_date)
        end_date = ee.Date(end_date)
        months = end_date.difference(start_date, 'month').round()
        
        # Create a list to store monthly images
        monthly_images = []
        
        # Iterate through each month in the range
        for i in range(months.getInfo()):
            current_date = start_date.advance(i, 'month')
            year = current_date.get('year').getInfo()
            month = current_date.get('month').getInfo()
            
            best_image = get_best_monthly_image(year, month)
            if best_image is not None:
                monthly_images.append(best_image)
        
        # Create new image collection from monthly images
        return ee.ImageCollection.fromImages(monthly_images)

    def preprocess_image(self, image):
        """
        Preprocess Sentinel-2 image
        
        Args:
            image: ee.Image, input image
            
        Returns:
            ee.Image: Preprocessed image
        """
        return image.divide(10000)  # Convert to reflectance values

    def export_image_patches(self, collection, region, scale=10, patch_size=64, 
                           drive_folder='sentinel2_patches', prefix='patch'):
        """
        Export image patches for the selected bands to Google Drive
        
        Args:
            collection: ee.ImageCollection, input image collection
            region: ee.Geometry, region of interest
            scale: int, pixel resolution in meters
            patch_size: int, size of image patches
            drive_folder: str, name of the Google Drive folder to export to
            prefix: str, prefix for exported file names
            
        Returns:
            list: List of task IDs for monitoring export progress
        """
        # Mosaic the collection
        mosaic = collection.mosaic()
        
        # Calculate the number of patches in x and y directions
        bounds = region.bounds().getInfo()['coordinates'][0]
        min_x, min_y = bounds[0][0], bounds[0][1]
        max_x, max_y = bounds[2][0], bounds[2][1]
        
        x_steps = int((max_x - min_x) * 111319 / (scale * patch_size))  # 111319 meters per degree
        y_steps = int((max_y - min_y) * 111319 / (scale * patch_size))
        
        export_tasks = []
        
        for i in range(x_steps):
            for j in range(y_steps):
                # Calculate patch bounds
                patch_min_x = min_x + (i * patch_size * scale / 111319)
                patch_max_x = min_x + ((i + 1) * patch_size * scale / 111319)
                patch_min_y = min_y + (j * patch_size * scale / 111319)
                patch_max_y = min_y + ((j + 1) * patch_size * scale / 111319)
                
                patch_region = ee.Geometry.Rectangle([
                    patch_min_x, patch_min_y,
                    patch_max_x, patch_max_y
                ])
                
                # Clip image to patch region
                patch = mosaic.clip(patch_region)
                
                # Export to Drive
                task = ee.batch.Export.image.toDrive(
                    image=patch,
                    description=f'{prefix}_{i}_{j}',
                    folder=drive_folder,
                    scale=scale,
                    region=patch_region,
                    maxPixels=1e9,
                    fileFormat='GeoTIFF'
                )
                
                task.start()
                export_tasks.append(task.id)
        
        return export_tasks

    def check_export_status(self, task_ids):
        """
        Check the status of export tasks
        
        Args:
            task_ids: list of task IDs to check
            
        Returns:
            dict: Dictionary mapping task IDs to their current status
        """
        statuses = {}
        for task_id in task_ids:
            task = ee.batch.Task.list()[task_id]
            statuses[task_id] = task.status()
        return statuses

    def export_to_drive(self, collection: ee.ImageCollection, regions: List[ee.Geometry], 
                       output_prefix: str, folder: str = 'sentinel2_data', 
                       scale: int = 10, bands: List[str] = ['B8', 'B4', 'B11']):
        """
        Export images from the collection to Google Drive.
        
        Args:
            collection (ee.ImageCollection): Collection of images to export
            regions (List[ee.Geometry]): List of regions to clip images to
            output_prefix (str): Prefix for output filenames
            folder (str): Google Drive folder to export to
            scale (int): Resolution in meters
            bands (List[str]): List of bands to export
        """
        # Get list of images and collection size
        image_list = collection.toList(collection.size())
        collection_size = collection.size().getInfo()
        
        print(f"\nPreparing to export {collection_size} images for {output_prefix}...")
        
        # Export each image
        for j in range(collection_size):
            image = ee.Image(image_list.get(j))
            
            # Get image date for filename
            date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
            
            # Process each region
            for i, region in enumerate(regions):
                try:
                    # Clip image to region
                    clipped = image.clip(region).select(bands)
                    
                    # Create description for the task
                    description = f"{output_prefix}_{date}_region_{i}"
                    
                    # Create export task
                    task = ee.batch.Export.image.toDrive(
                        image=clipped,
                        description=description,
                        folder=folder,
                        fileNamePrefix=description,
                        scale=scale,
                        region=region.bounds(),
                        fileFormat='GeoTIFF',
                        formatOptions={'cloudOptimized': True}
                    )
                    
                    # Start the task
                    task.start()
                    print(f"Started export task for {description}")
                    
                except Exception as e:
                    print(f"Error exporting image {j} for region {i}: {str(e)}")
                    continue
        
        print("\nAll export tasks have been started.")
        print("You can monitor their progress at: https://code.earthengine.google.com/tasks")

    def collect_train_val_data(self, years_back: int = 5, val_split: float = 0.2,
                             drive_folder: str = 'sentinel2_dataset'):
        """
        Collect and export training and validation data, processing one KML at a time.
        
        Args:
            years_back: Number of years to look back
            val_split: Fraction of regions to use for validation
            drive_folder: Base folder in Google Drive for export
            
        Returns:
            Dictionary with task IDs for training and validation exports
        """
        # Load all KML data
        train_val_data = self.dataset_manager.load_train_val_data()
        
        # Process each KML file independently
        for geometries, kml_name, kml_path in train_val_data:
            print(f"\nProcessing KML: {kml_name}")
            
            # Get image collection for this KML's geometries
            collection = self.get_sentinel_collection(geometries, years_back=years_back)
            
            # Export the collection for this KML
            self.export_to_drive(
                collection=collection,
                regions=geometries,
                output_prefix=f"train_{kml_name}",
                folder=drive_folder,
            )
            
            print(f"Completed processing for {kml_name}")
        
        return {"status": "All KMLs processed independently"}

    def collect_prediction_data(self, years_back: int = 5,
                              drive_folder: str = 'sentinel2_predictions'):
        """
        Collect and export data for prediction, processing one KML at a time
        
        Args:
            years_back: Number of years to look back
            drive_folder: Folder in Google Drive for export
            
        Returns:
            Dictionary with status of the export process
        """
        # Load all prediction KML data
        prediction_data = self.dataset_manager.load_prediction_data()
        
        print(f"\nIniciando processamento de {len(prediction_data)} KMLs para predição...")
        
        # Process each KML file independently
        for geometries, kml_name, kml_path in prediction_data:
            print(f"\nProcessando KML: {kml_name}")
            print(f"Número de geometrias: {len(geometries)}")
            
            # Get image collection for this KML's geometries
            collection = self.get_sentinel_collection(
                regions=geometries,
                years_back=years_back,
                max_cloud_cover=20
            )
            
            # Export the collection for this KML
            self.export_to_drive(
                collection=collection,
                regions=geometries,
                output_prefix=f"predict_{kml_name}",
                folder=drive_folder
            )
            
            print(f"Exportação iniciada para {kml_name}")
        
        return {"status": "All prediction KMLs processed independently"}

    def collect_from_geodataframe(self, gdf: gpd.GeoDataFrame, 
                              start_date: str, end_date: str,
                              export_folder: str = 'sentinel_exports',
                              patch_size: int = 64,
                              scale: int = 10,
                              max_cloud_cover: float = 20.0) -> None:
        """
        Collect Sentinel-2 data for areas defined in a GeoDataFrame
        
        Args:
            gdf: GeoDataFrame containing geometries to collect data for
            start_date: Start date in format 'YYYY-MM-DD'
            end_date: End date in format 'YYYY-MM-DD'
            export_folder: Folder name in Google Drive to export images to
            patch_size: Size of image patches in pixels
            scale: Scale in meters for the exported images
            max_cloud_cover: Maximum cloud cover percentage allowed
        """
        print(f"Starting collection from GeoDataFrame with {len(gdf)} geometries...")
        
        # Create geometry handler if not exists
        if not hasattr(self, 'geometry_handler'):
            self.geometry_handler = GeometryHandler()
        
        # Convert GeoDataFrame geometries to Earth Engine format
        for idx, row in gdf.iterrows():
            try:
                # Convert geometry to Earth Engine format using existing method
                ee_geometry = self.geometry_handler._convert_to_ee_geometry(row.geometry)
                
                # Get name for the export (using index if no name column)
                name = str(row.get('name', f'geometry_{idx}'))
                
                print(f"Processing geometry: {name}")
                
                # Get Sentinel collection
                collection = self.get_sentinel_collection(
                    regions=[ee_geometry],
                    start_date=start_date,
                    end_date=end_date,
                    max_cloud_cover=max_cloud_cover
                )
                
                # Export images
                self.export_to_drive(
                    collection=collection,
                    regions=[ee_geometry],
                    output_prefix=name,
                    folder=export_folder,
                    scale=scale,
                    bands=['B8', 'B4', 'B11']
                )
                
            except Exception as e:
                print(f"Error processing geometry {idx}: {str(e)}")
                continue
        
        print("Collection process initiated. Check your Google Drive for the exported files.")

if __name__ == "__main__":
    # Example usage
    collector = SentinelCollector(
        train_val_dir='path/to/train_val_kmls',
        prediction_dir='path/to/prediction_kmls'
    )
    
    # Collect training and validation data
    task_ids = collector.collect_train_val_data(
        years_back=5,
        val_split=0.2,
        drive_folder='meu_projeto_sentinel'
    )
    
    # Monitor export progress
    train_status = collector.check_export_status(task_ids['train_tasks'])
    val_status = collector.check_export_status(task_ids['val_tasks'])
    
    print("Training export status:", train_status)
    print("Validation export status:", val_status)
    
    # Later, when you want to make predictions...
    pred_task_ids = collector.collect_prediction_data(
        years_back=5,
        drive_folder='meu_projeto_sentinel/predictions'
    )
    
    # Monitor prediction export progress
    pred_status = collector.check_export_status(pred_task_ids)
    print("Prediction export status:", pred_status)
