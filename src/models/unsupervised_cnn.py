import tensorflow as tf
import numpy as np
from sklearn.cluster import KMeans
import mlflow
import mlflow.keras
from typing import Optional, Dict, Any

class UnsupervisedCNN:
    def __init__(self, input_shape=(64, 64, 3), n_clusters=5, latent_dim=128,
                 experiment_name: Optional[str] = "unsupervised_sentinel_classification"):
        """
        Initialize the Unsupervised CNN model
        
        Args:
            input_shape: tuple, shape of input images (height, width, channels)
            n_clusters: int, number of clusters for classification
            latent_dim: int, dimension of the latent space
            experiment_name: str, name of the MLflow experiment
        """
        self.input_shape = input_shape
        self.n_clusters = n_clusters
        self.latent_dim = latent_dim
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        self.autoencoder = self._build_autoencoder()
        self.kmeans = KMeans(n_clusters=n_clusters)
        
        # Set up MLflow
        mlflow.set_experiment(experiment_name)

    def _build_encoder(self):
        """Build the encoder part of the network"""
        inputs = tf.keras.Input(shape=self.input_shape)
        
        x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        
        x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(self.latent_dim, activation='relu')(x)
        
        return tf.keras.Model(inputs, x, name='encoder')

    def _build_decoder(self):
        """Build the decoder part of the network"""
        latent_inputs = tf.keras.Input(shape=(self.latent_dim,))
        
        # Calculate the shape after encoding
        h = self.input_shape[0] // 8
        w = self.input_shape[1] // 8
        
        x = tf.keras.layers.Dense(h * w * 128, activation='relu')(latent_inputs)
        x = tf.keras.layers.Reshape((h, w, 128))(x)
        
        x = tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=2, activation='relu', padding='same')(x)
        x = tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same')(x)
        x = tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same')(x)
        
        outputs = tf.keras.layers.Conv2D(self.input_shape[-1], (3, 3), activation='sigmoid', padding='same')(x)
        
        return tf.keras.Model(latent_inputs, outputs, name='decoder')

    def _build_autoencoder(self):
        """Combine encoder and decoder into an autoencoder"""
        inputs = tf.keras.Input(shape=self.input_shape)
        latent = self.encoder(inputs)
        outputs = self.decoder(latent)
        return tf.keras.Model(inputs, outputs, name='autoencoder')

    def train(self, X, epochs=100, batch_size=32, validation_split=0.2,
             model_params: Optional[Dict[str, Any]] = None):
        """
        Train the model
        
        Args:
            X: numpy array, training data
            epochs: int, number of training epochs
            batch_size: int, batch size for training
            validation_split: float, fraction of data to use for validation
            model_params: dict, additional parameters to log in MLflow
        """
        with mlflow.start_run():
            # Log parameters
            params = {
                'input_shape': self.input_shape,
                'n_clusters': self.n_clusters,
                'latent_dim': self.latent_dim,
                'epochs': epochs,
                'batch_size': batch_size,
                'validation_split': validation_split
            }
            if model_params:
                params.update(model_params)
            mlflow.log_params(params)
            
            # Compile model
            self.autoencoder.compile(optimizer='adam', loss='mse')
            
            # Train model with validation split
            history = self.autoencoder.fit(
                X, X,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=[
                    tf.keras.callbacks.ModelCheckpoint(
                        'best_model.h5',
                        save_best_only=True,
                        monitor='val_loss'
                    )
                ]
            )
            
            # Log metrics
            for epoch in range(len(history.history['loss'])):
                mlflow.log_metrics({
                    'loss': history.history['loss'][epoch],
                    'val_loss': history.history['val_loss'][epoch]
                }, step=epoch)
            
            # Extract features and perform clustering
            latent_features = self.encoder.predict(X)
            self.kmeans.fit(latent_features)
            
            # Log models
            mlflow.keras.log_model(self.autoencoder, "autoencoder")
            mlflow.sklearn.log_model(self.kmeans, "kmeans")
            
            # Calculate and log clustering metrics
            if validation_split > 0:
                val_size = int(len(X) * validation_split)
                X_val = X[-val_size:]
                val_features = self.encoder.predict(X_val)
                val_clusters = self.kmeans.predict(val_features)
                
                # Log silhouette score
                from sklearn.metrics import silhouette_score
                silhouette_avg = silhouette_score(val_features, val_clusters)
                mlflow.log_metric('silhouette_score', silhouette_avg)

    def predict(self, X):
        """
        Predict clusters for new data
        
        Args:
            X: numpy array, input data
            
        Returns:
            numpy array: cluster assignments
        """
        latent_features = self.encoder.predict(X)
        return self.kmeans.predict(latent_features)

    def reconstruct(self, X):
        """
        Reconstruct input images
        
        Args:
            X: numpy array, input data
            
        Returns:
            numpy array: reconstructed images
        """
        return self.autoencoder.predict(X)
