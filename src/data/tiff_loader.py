import rasterio
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from sklearn.model_selection import train_test_split
import tensorflow as tf

class TiffLoader:
    def __init__(self, 
                 train_val_dir: Optional[str] = None,
                 prediction_dir: Optional[str] = None,
                 patch_size: int = 64):
        """
        Initialize the TiffLoader
        
        Args:
            train_val_dir: Directory containing training/validation GeoTIFFs
            prediction_dir: Directory containing GeoTIFFs for prediction
            patch_size: Size of image patches to extract
        """
        self.train_val_dir = Path(train_val_dir) if train_val_dir else None
        self.prediction_dir = Path(prediction_dir) if prediction_dir else None
        self.patch_size = patch_size

    def _load_tiff(self, tiff_path: Path) -> np.ndarray:
        """
        Load a GeoTIFF file and normalize values
        
        Args:
            tiff_path: Path to GeoTIFF file
            
        Returns:
            Normalized numpy array with shape (height, width, channels)
        """
        with rasterio.open(tiff_path) as src:
            # Read all bands
            image = src.read()
            
            # Transpose to (height, width, channels)
            image = np.transpose(image, (1, 2, 0))
            
            # Normalize to [0, 1]
            image = image.astype(np.float32)
            image = np.clip(image / 10000.0, 0, 1)  # Assuming Sentinel-2 reflectance values
            
            return image

    def _extract_patches(self, image: np.ndarray) -> np.ndarray:
        """
        Extract patches from an image
        
        Args:
            image: Input image of shape (height, width, channels)
            
        Returns:
            Array of patches with shape (n_patches, patch_size, patch_size, channels)
        """
        height, width = image.shape[:2]
        
        patches = []
        for y in range(0, height - self.patch_size + 1, self.patch_size):
            for x in range(0, width - self.patch_size + 1, self.patch_size):
                patch = image[y:y + self.patch_size, x:x + self.patch_size]
                if patch.shape[:2] == (self.patch_size, self.patch_size):
                    patches.append(patch)
        
        return np.array(patches)

    def load_train_val_data(self, val_split: float = 0.2) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Load and split training/validation data
        
        Args:
            val_split: Fraction of data to use for validation
            
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        if not self.train_val_dir:
            raise ValueError("Training/validation directory not specified")
        
        # Load all TIFFs
        all_patches = []
        for tiff_file in self.train_val_dir.glob('*.tif*'):
            image = self._load_tiff(tiff_file)
            patches = self._extract_patches(image)
            if patches.ndim == 1:
                continue
            all_patches.append(patches)
        
        # Combine all patches
        all_patches = np.concatenate(all_patches, axis=0)
        
        # Split into train and validation
        train_patches, val_patches = train_test_split(
            all_patches, 
            test_size=val_split,
            random_state=42
        )
        
        # Create TensorFlow datasets
        train_dataset = tf.data.Dataset.from_tensor_slices(train_patches)
        val_dataset = tf.data.Dataset.from_tensor_slices(val_patches)
        
        return train_dataset, val_dataset

    def load_prediction_data(self) -> tf.data.Dataset:
        """
        Load data for prediction
        
        Returns:
            TensorFlow dataset containing patches for prediction
        """
        if not self.prediction_dir:
            raise ValueError("Prediction directory not specified")
        
        # Load all TIFFs
        all_patches = []
        tiff_metadata = []  # Store metadata for reconstruction
        
        for tiff_file in self.prediction_dir.glob('*.tif*'):
            image = self._load_tiff(tiff_file)
            patches = self._extract_patches(image)
            
            # Store metadata for each patch
            n_patches = len(patches)
            tiff_metadata.extend([(tiff_file.name, i) for i in range(n_patches)])
            
            all_patches.append(patches)
        
        # Combine all patches
        all_patches = np.concatenate(all_patches, axis=0)
        
        # Create TensorFlow dataset
        pred_dataset = tf.data.Dataset.from_tensor_slices(all_patches)
        
        return pred_dataset, tiff_metadata

    def prepare_dataset(self, dataset: tf.data.Dataset, 
                       batch_size: int = 32,
                       shuffle: bool = True,
                       augment: bool = False) -> tf.data.Dataset:
        """
        Prepare dataset for training/validation/prediction
        
        Args:
            dataset: Input TensorFlow dataset
            batch_size: Batch size
            shuffle: Whether to shuffle the data
            augment: Whether to apply data augmentation
            
        Returns:
            Prepared dataset
        """
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)
        
        if augment:
            dataset = dataset.map(self._augment_data, 
                                num_parallel_calls=tf.data.AUTOTUNE)
        
        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    def _augment_data(self, image):
        """
        Apply data augmentation
        
        Args:
            image: Input image
            
        Returns:
            Augmented image
        """
        # Random flip left/right
        image = tf.image.random_flip_left_right(image)
        
        # Random flip up/down
        image = tf.image.random_flip_up_down(image)
        
        # Random rotation
        image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
        
        return image
