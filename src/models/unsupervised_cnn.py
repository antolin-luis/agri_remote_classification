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
        
        # Set up MLflow with proper tracking URI
        try:
            if mlflow.get_tracking_uri() is None:
                mlflow.set_tracking_uri("file:./mlruns")
            mlflow.set_experiment(experiment_name)
        except Exception as e:
            print(f"Warning: Could not set up MLflow experiment: {str(e)}")
            print("Training will continue but metrics may not be logged")

    def _build_encoder(self):
        """Build the encoder model"""
        inputs = tf.keras.layers.Input(shape=self.input_shape)
        
        # Normalização dos dados de entrada
        x = tf.keras.layers.Normalization()(inputs)
        
        # Primeira camada convolucional com batch normalization
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
        
        # Segunda camada com skip connection
        skip = x
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Add()([x, skip])  # Skip connection
        x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
        
        # Terceira camada
        x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
        
        # Flatten e Dense com regularização
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(128, activation='relu',
                                kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        
        # Latent space
        latent = tf.keras.layers.Dense(self.latent_dim, name='latent_space',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        
        return tf.keras.Model(inputs, latent, name='encoder')

    def _build_decoder(self):
        """Build the decoder model"""
        latent_inputs = tf.keras.layers.Input(shape=(self.latent_dim,))
        
        # Dense layers com regularização
        x = tf.keras.layers.Dense(128, activation='relu',
                                kernel_regularizer=tf.keras.regularizers.l2(0.01))(latent_inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        
        # Calcula as dimensões corretas para o reshape
        # Se a entrada é 64x64, após 3 MaxPooling2D (2,2) fica 8x8
        h = self.input_shape[0] // 8
        w = self.input_shape[1] // 8
        
        # Reshape para começar deconvolução
        x = tf.keras.layers.Dense(h * w * 128, activation='relu')(x)
        x = tf.keras.layers.Reshape((h, w, 128))(x)
        
        # Camadas deconvolucionais com batch normalization
        # Primeira upsampling: 8x8 -> 16x16
        x = tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        
        # Segunda upsampling: 16x16 -> 32x32
        x = tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        
        # Terceira upsampling: 32x32 -> 64x64
        x = tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        
        # Última camada com sigmoid para normalizar saída
        outputs = tf.keras.layers.Conv2D(self.input_shape[-1], (3, 3), 
                                       activation='sigmoid', padding='same')(x)
        
        return tf.keras.Model(latent_inputs, outputs, name='decoder')

    def _build_autoencoder(self):
        """Combine encoder and decoder into an autoencoder"""
        inputs = tf.keras.Input(shape=self.input_shape)
        latent = self.encoder(inputs)
        outputs = self.decoder(latent)
        return tf.keras.Model(inputs, outputs, name='autoencoder')

    def train(self, X, epochs=100, batch_size=32, validation_split=0.2,
             model_params: Optional[Dict[str, Any]] = None,
             learning_rate=0.0005,  # Learning rate reduzido
             dropout_rate=0.3,
             use_data_augmentation=True):
        """
        Train the model with improved parameters and techniques
        """
        def _train_model():
            # Normalize input data
            X_norm = (X - X.mean()) / (X.std() + 1e-8)
            
            # Data Augmentation com menos transformações
            if use_data_augmentation:
                data_augmentation = tf.keras.Sequential([
                    tf.keras.layers.RandomFlip("horizontal"),
                    tf.keras.layers.RandomRotation(0.1)
                ])
                X_augmented = np.concatenate([X_norm, data_augmentation(X_norm).numpy()], axis=0)
            else:
                X_augmented = X_norm
            
            # Optimizer com clipping mais conservador
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=learning_rate,
                clipnorm=0.5,  # Clipping mais conservador
                beta_1=0.9,    # Momentum padrão
                beta_2=0.999   # RMSprop padrão
            )
            
            # Loss function customizada
            def custom_loss(y_true, y_pred):
                # MSE para reconstrução
                mse = tf.reduce_mean(tf.square(y_true - y_pred))
                # L2 regularization nos pesos
                reg_loss = tf.reduce_sum(self.autoencoder.losses)
                return mse + 0.01 * reg_loss
            
            # Compile com a nova loss
            self.autoencoder.compile(
                optimizer=optimizer,
                loss=custom_loss,
                metrics=['accuracy', 'mae']
            )
            
            # Callbacks ajustados
            callbacks = [
                tf.keras.callbacks.ModelCheckpoint(
                    'best_model.h5',
                    save_best_only=True,
                    monitor='val_loss'
                ),
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True,
                    min_delta=1e-4
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.2,     # Redução mais gradual
                    patience=5,
                    min_lr=1e-6
                ),
                tf.keras.callbacks.TensorBoard(
                    log_dir='./logs',
                    histogram_freq=1
                )
            ]
            
            # Train with improved settings
            history = self.autoencoder.fit(
                X_augmented, X_augmented,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=callbacks,
                shuffle=True,
                verbose=1
            )
            
            # Extract features and perform clustering with better initialization
            print("Extracting features for clustering...")
            latent_features = self.encoder.predict(X, batch_size=batch_size)
            
            print("Performing clustering with improved initialization...")
            # Use better initialization for K-means
            self.kmeans = KMeans(
                n_clusters=self.n_clusters,
                n_init=10,
                max_iter=300,
                init='k-means++'
            )
            self.kmeans.fit(latent_features)
            
            return history, latent_features

        # MLflow tracking
        try:
            if mlflow.get_tracking_uri() is None:
                mlflow.set_tracking_uri("file:./mlruns")
                print("Set MLflow tracking URI to: file:./mlruns")

            active_run = mlflow.active_run()
            if active_run:
                print("Using existing MLflow run")
                run = active_run
            else:
                print("Starting new MLflow run")
                run = mlflow.start_run()

            with run:
                # Log enhanced parameters
                params = {
                    'input_shape': self.input_shape,
                    'n_clusters': self.n_clusters,
                    'latent_dim': self.latent_dim,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'validation_split': validation_split,
                    'learning_rate': learning_rate,
                    'dropout_rate': dropout_rate,
                    'use_data_augmentation': use_data_augmentation
                }
                if model_params:
                    params.update(model_params)
                
                try:
                    mlflow.log_params(params)
                except Exception as e:
                    print(f"Warning: Could not log parameters: {str(e)}")

                # Train the model
                history, latent_features = _train_model()
                
                # Log metrics with error handling
                if history and hasattr(history, 'history'):
                    for epoch in range(len(history.history.get('loss', []))):
                        try:
                            metrics_dict = {
                                'loss': history.history['loss'][epoch],
                                'val_loss': history.history.get('val_loss', [0])[epoch],
                                'accuracy': history.history.get('accuracy', [0])[epoch],
                                'val_accuracy': history.history.get('val_accuracy', [0])[epoch],
                                'mae': history.history.get('mae', [0])[epoch],
                                'val_mae': history.history.get('val_mae', [0])[epoch]
                            }
                            mlflow.log_metrics(metrics_dict, step=epoch)
                        except Exception as e:
                            print(f"Warning: Could not log metrics for epoch {epoch}: {str(e)}")
                
                # Log models and additional metrics
                try:
                    mlflow.keras.log_model(self.autoencoder, "autoencoder")
                    mlflow.sklearn.log_model(self.kmeans, "kmeans")
                    
                    # Calculate and log clustering metrics
                    if validation_split > 0:
                        val_size = int(len(X) * validation_split)
                        X_val = X[-val_size:]
                        val_features = self.encoder.predict(X_val, batch_size=batch_size)
                        val_clusters = self.kmeans.predict(val_features)
                        
                        # Log silhouette score
                        from sklearn.metrics import silhouette_score, calinski_harabasz_score
                        silhouette_avg = silhouette_score(val_features, val_clusters)
                        calinski_score = calinski_harabasz_score(val_features, val_clusters)
                        
                        mlflow.log_metrics({
                            'silhouette_score': silhouette_avg,
                            'calinski_harabasz_score': calinski_score
                        })
                except Exception as e:
                    print(f"Warning: Could not log models or clustering metrics: {str(e)}")
        
        except Exception as e:
            print(f"Warning: MLflow tracking disabled due to error: {str(e)}")
            print("Continuing training without MLflow...")
            # Train without MLflow
            _train_model()

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
