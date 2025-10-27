"""
CNN model architecture using transfer learning.
Implements MobileNetV2 with custom classification head.
"""

import tensorflow as tf
import json
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Tuple, Union

# Import Keras components (TensorFlow 2.x style)
from tensorflow.keras import layers, Model, Input  # type: ignore
from tensorflow.keras.applications import MobileNetV2, ResNet50  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.callbacks import (  # type: ignore
    EarlyStopping, 
    ReduceLROnPlateau, 
    ModelCheckpoint,
    TensorBoard
)
from tensorflow.keras.utils import Sequence  # type: ignore


class WasteClassificationModel:
    """Builds and manages CNN model for waste classification."""
    
    def __init__(self, num_classes: int, input_shape: Tuple[int, int, int] = (224, 224, 3), backbone: str = 'mobilenetv2'):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.backbone = backbone
        self.model: Optional[Model] = None
        self.loss: Optional[str] = None
        
    def build_model(self, freeze_backbone=True, dropout_rate=0.5):
        """
        Build transfer learning model with custom classification head.
        
        Args:
            freeze_backbone: If True, freeze pretrained layers initially
            dropout_rate: Dropout probability for regularization
        """
        # Load pretrained backbone
        if self.backbone == 'mobilenetv2':
            base_model = MobileNetV2(
                input_shape=self.input_shape,
                include_top=False,
                weights='imagenet',
                pooling='avg'
            )
        elif self.backbone == 'resnet50':
            base_model = ResNet50(
                input_shape=self.input_shape,
                include_top=False,
                weights='imagenet',
                pooling='avg'
            )
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone}")
        
        # Freeze backbone if specified
        base_model.trainable = not freeze_backbone
        
        # Build custom head
        inputs = Input(shape=self.input_shape)
        
        # Preprocessing (already done, but explicit for model tracing)
        x = inputs
        
        # Base model
        x = base_model(x, training=False)
        
        # Classification head
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(256, activation='relu', name='fc1')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate * 0.5)(x)
        x = layers.Dense(128, activation='relu', name='fc2')(x)
        x = layers.BatchNormalization()(x)
        
        # Output layer
        if self.num_classes == 2:
            outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
            loss = 'binary_crossentropy'
        else:
            outputs = layers.Dense(self.num_classes, activation='softmax', name='output')(x)
            loss = 'sparse_categorical_crossentropy'
        
        self.model = Model(inputs=inputs, outputs=outputs, name=f'{self.backbone}_waste_classifier')
        self.loss = loss
        
        # Type assertion for linter
        assert self.model is not None
        
        print(f"\n Model built: {self.backbone}")
        print(f"Total params: {self.model.count_params():,}")
        print(f"Backbone frozen: {freeze_backbone}")
        
        return self.model
    
    def compile_model(self, learning_rate: float = 1e-3) -> None:
        """Compile model with optimizer and metrics."""
        if self.model is None:
            raise ValueError("Model must be built before compilation")
            
        optimizer = Adam(learning_rate=learning_rate)
        
        metrics = ['accuracy']
        if self.num_classes > 2:
            metrics.append('sparse_categorical_accuracy')
        
        self.model.compile(
            optimizer=optimizer,
            loss=self.loss,
            metrics=metrics
        )
        
        print(f" Model compiled with lr={learning_rate}")
    
    def unfreeze_backbone(self, num_layers: Optional[int] = None) -> None:
        """
        Unfreeze backbone layers for fine-tuning.
        
        Args:
            num_layers: Number of layers to unfreeze from the top. 
                       If None, unfreeze all.
        """
        if self.model is None:
            raise ValueError("Model must be built before unfreezing")
            
        if num_layers is None:
            # Unfreeze all
            for layer in self.model.layers:
                layer.trainable = True
            print(" All layers unfrozen")
        else:
            # Unfreeze top N layers
            total_layers = len(self.model.layers)
            for layer in self.model.layers[-num_layers:]:
                layer.trainable = True
            print(f" Unfrozen top {num_layers}/{total_layers} layers")
    
    def get_callbacks(self, checkpoint_dir='models/checkpoints', log_dir='logs'):
        """Create training callbacks."""
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        
        callbacks = [
            # Early stopping
            EarlyStopping(
                monitor='val_loss',
                patience=7,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Learning rate reduction
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
            
            # Model checkpoint
            ModelCheckpoint(
                filepath=f'{checkpoint_dir}/best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            
            # TensorBoard
            TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
                write_graph=True
            )
        ]
        
        return callbacks
    
    def summary(self):
        """Print model summary."""
        if self.model:
            self.model.summary()
        else:
            print("Model not built yet. Call build_model() first.")
    
    def save_model(self, save_path: str = 'api/model/waste_classifier.h5') -> None:
        """Save trained model."""
        if self.model is None:
            raise ValueError("Model must be built before saving")
            
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(save_path)
        print(f" Model saved to {save_path}")
    
    def load_model(self, model_path: str) -> None:
        """Load saved model."""
        from tensorflow.keras.models import load_model  # type: ignore
        self.model = load_model(model_path)
        print(f" Model loaded from {model_path}")


class DataGenerator(Sequence):
    """Custom data generator for efficient batch loading."""
    
    def __init__(self, dataframe, batch_size=32, img_size=(224, 224), 
                 augment=False, shuffle=True):
        self.df = dataframe.reset_index(drop=True)
        self.batch_size = batch_size
        self.img_size = img_size
        self.augment = augment
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.df))
        
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))
    
    def __getitem__(self, index):
        # Get batch indexes
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_df = self.df.iloc[batch_indexes]
        
        # Load batch
        X, y = self._load_batch(batch_df)
        
        return X, y
    
    def _load_batch(self, batch_df):
        import cv2
        
        X = np.zeros((len(batch_df), *self.img_size, 3), dtype=np.float32)
        y = np.zeros(len(batch_df), dtype=np.int32)
        
        for i, (_, row) in enumerate(batch_df.iterrows()):
            # Load image
            img = cv2.imread(row['path'])
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, self.img_size)
                img = img.astype(np.float32) / 255.0
                
                # Augmentation
                if self.augment:
                    img = self._augment_image(img)
                
                X[i] = img
                y[i] = row['label']
        
        return X, y
    
    def _augment_image(self, img):
        """Apply data augmentation."""
        # Horizontal flip
        if np.random.rand() > 0.5:
            img = cv2.flip(img, 1)
        
        # Brightness
        if np.random.rand() > 0.5:
            factor = np.random.uniform(0.7, 1.3)
            img = np.clip(img * factor, 0, 1)
        
        # Rotation
        if np.random.rand() > 0.5:
            angle = np.random.uniform(-20, 20)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h))
        
        return img
    
    def on_epoch_end(self):
        """Shuffle indexes after each epoch."""
        if self.shuffle:
            np.random.shuffle(self.indexes)


# Import numpy for generator
import numpy as np


def create_model_pipeline(num_classes, backbone='mobilenetv2'):
    """
    Factory function to create complete model pipeline.
    
    Returns:
        model: WasteClassificationModel instance
    """
    model = WasteClassificationModel(
        num_classes=num_classes,
        backbone=backbone
    )
    
    return model


def load_data_splits(data_dir='data/processed'):
    """
    Load train/val/test splits from CSV files generated by data_loader.py
    
    Args:
        data_dir: Directory containing train.csv, val.csv, test.csv
        
    Returns:
        tuple: (train_df, val_df, test_df, class_mapping)
    """
    import pandas as pd
    import json
    from pathlib import Path
    
    data_path = Path(data_dir)
    
    # Load DataFrames
    train_df = pd.read_csv(data_path / 'train.csv')
    val_df = pd.read_csv(data_path / 'val.csv')
    test_df = pd.read_csv(data_path / 'test.csv')
    
    # Load class mapping
    with open(data_path / 'class_mapping.json', 'r') as f:
        class_mapping = json.load(f)
    
    print(f"Loaded data splits:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val:   {len(val_df)} samples")
    print(f"  Test:  {len(test_df)} samples")
    print(f"  Classes: {class_mapping['class_to_idx']}")
    
    return train_df, val_df, test_df, class_mapping


if __name__ == '__main__':
    # Example usage for waste bin sorter (O=Organic, R=Recyclable)
    print("Building waste classification model...")
    
    # Load class mapping to get correct number of classes
    import json
    from pathlib import Path
    
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    mapping_path = project_root / 'data' / 'processed' / 'class_mapping.json'
    
    if mapping_path.exists():
        with open(mapping_path, 'r') as f:
            class_mapping = json.load(f)
        num_classes = len(class_mapping['class_to_idx'])
        print(f"Detected {num_classes} classes: {class_mapping['class_to_idx']}")
    else:
        # Default for binary classification (O=Organic, R=Recyclable)
        num_classes = 2
        print("Using default binary classification (Organic vs Recyclable)")
    
    model = create_model_pipeline(num_classes=num_classes)
    model.build_model(freeze_backbone=True)
    model.compile_model(learning_rate=1e-3)
    model.summary()