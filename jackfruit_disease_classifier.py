# jackfruit_disease_classifier.py
# Complete Image Classification Pipeline for Jackfruit Diseases with Transfer Learning

# CORRECT IMPORTS FOR TENSORFLOW 2.20.0 (Keras 3.x)
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import load_img, img_to_array
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Print TensorFlow version
print(f"TensorFlow Version: {tf.__version__}")
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# ======================
# CONFIGURATION
# ======================
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.0001

# Paths (UPDATE THESE PATHS)
PRE_TRAINED_MODEL_PATH = 'your_existing_model.h5'  # Path to your existing .h5 model
NEW_TRAIN_DATA_DIR = './data/train'    # Path to new training data
NEW_VALID_DATA_DIR = './data/validation'    # Path to new validation data
NEW_TEST_DATA_DIR = './data/test'      # Path to new test data
MODEL_SAVE_PATH = 'jackfruit_disease_updated.h5'

# ======================
# LOAD PRE-TRAINED MODEL
# ======================
print("\n" + "="*50)
print("LOADING PRE-TRAINED MODEL")
print("="*50)

try:
    # Load your existing model
    model = load_model(PRE_TRAINED_MODEL_PATH)
    print(f"✓ Pre-trained model loaded successfully from {PRE_TRAINED_MODEL_PATH}")
    
    # Get the original model's input shape and number of classes
    original_input_shape = model.input_shape
    original_num_classes = model.output_shape[-1]
    print(f"✓ Original model input shape: {original_input_shape}")
    print(f"✓ Original model output classes: {original_num_classes}")
    
except Exception as e:
    print(f"✗ Error loading pre-trained model: {e}")
    print("Please check the path to your .h5 file")
    exit()

# ======================
# DATA PREPARATION FOR NEW DATA
# ======================
print("\n" + "="*50)
print("PREPARING NEW DATA GENERATORS")
print("="*50)

# Data Augmentation for training data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
    brightness_range=[0.8, 1.2]
)

# Only rescaling for validation and test data
valid_test_datagen = ImageDataGenerator(rescale=1./255)

# Training data generator for new data
train_generator = train_datagen.flow_from_directory(
    NEW_TRAIN_DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    seed=42,
    color_mode='rgb'
)

# Validation data generator for new data
validation_generator = valid_test_datagen.flow_from_directory(
    NEW_VALID_DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    seed=42,
    color_mode='rgb'
)

# Test data generator for new data
test_generator = valid_test_datagen.flow_from_directory(
    NEW_TEST_DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
    color_mode='rgb'
)

# Get new class names and indices
new_class_names = list(train_generator.class_indices.keys())
new_num_classes = len(new_class_names)

print(f"\nNew Classes found: {new_class_names}")
print(f"New Class indices: {train_generator.class_indices}")
print(f"New Training samples: {train_generator.samples}")
print(f"New Validation samples: {validation_generator.samples}")
print(f"New Test samples: {test_generator.samples}")

# Verify that all generators have the same classes
train_classes = set(train_generator.class_indices.keys())
valid_classes = set(validation_generator.class_indices.keys())
test_classes = set(test_generator.class_indices.keys())

if train_classes == valid_classes == test_classes:
    print("✓ All datasets have the same classes")
else:
    print("⚠ Warning: Class mismatch between datasets!")
    print(f"Train classes: {train_classes}")
    print(f"Valid classes: {valid_classes}")
    print(f"Test classes: {test_classes}")

# ======================
# MODEL MODIFICATION FOR NEW CLASSES
# ======================
print("\n" + "="*50)
print("MODIFYING MODEL FOR NEW CLASSES")
print("="*50)

# Strategy 1: Replace only the last layer (if architecture allows)
if new_num_classes != original_num_classes:
    print(f"Updating model from {original_num_classes} to {new_num_classes} classes")
    
    # Strategy 1: Replace the output layer
    try:
        # Remove the last layer
        model.pop()
        
        # Add new output layer with correct number of classes
        model.add(Dense(new_num_classes, activation='softmax', name='new_output'))
        
        # Recompile the model with lower learning rate for fine-tuning
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE/10),  # Lower LR for fine-tuning
            loss='categorical_crossentropy',
            metrics=['accuracy', 
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall')]
        )
        print("✓ Output layer successfully replaced")
        
    except Exception as e:
        print(f"✗ Error modifying model: {e}")
        print("Trying alternative approach...")
        
        # Strategy 2: Create a new model with transferred features
        try:
            # Create a new model with the same architecture but new output
            new_model = Sequential()
            
            # Add all layers except the last one from the original model
            for layer in model.layers[:-1]:
                new_model.add(layer)
            
            # Freeze early layers for transfer learning
            for layer in new_model.layers[:-3]:  # Freeze all but last 3 layers
                layer.trainable = False
                print(f"✓ Frozen layer: {layer.name}")
            
            # Add new output layer
            new_model.add(Dense(new_num_classes, activation='softmax'))
            
            # Compile the new model
            new_model.compile(
                optimizer=Adam(learning_rate=LEARNING_RATE/10),
                loss='categorical_crossentropy',
                metrics=['accuracy', 
                        tf.keras.metrics.Precision(name='precision'),
                        tf.keras.metrics.Recall(name='recall')]
            )
            
            model = new_model
            print("✓ New model created with transferred features")
            
        except Exception as e2:
            print(f"✗ Alternative approach failed: {e2}")
            print("Creating a completely new model...")
            
            # Strategy 3: Create a fresh model (fallback)
            model = Sequential([
                Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
                MaxPooling2D(2, 2),
                Dropout(0.25),
                
                Conv2D(64, (3, 3), activation='relu'),
                MaxPooling2D(2, 2),
                Dropout(0.25),
                
                Conv2D(128, (3, 3), activation='relu'),
                MaxPooling2D(2, 2),
                Dropout(0.25),
                
                Conv2D(256, (3, 3), activation='relu'),
                MaxPooling2D(2, 2),
                Dropout(0.25),
                
                Flatten(),
                Dense(512, activation='relu'),
                Dropout(0.5),
                Dense(256, activation='relu'),
                Dropout(0.3),
                Dense(new_num_classes, activation='softmax')
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=LEARNING_RATE),
                loss='categorical_crossentropy',
                metrics=['accuracy', 
                        tf.keras.metrics.Precision(name='precision'),
                        tf.keras.metrics.Recall(name='recall')]
            )
            print("✓ New model created from scratch")
else:
    print("✓ Number of classes matches original model")
    # Fine-tune with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE/10),
        loss='categorical_crossentropy',
        metrics=['accuracy', 
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')]
    )

# Display model summary
print("\nModel Summary:")
model.summary()

# ======================
# CALLBACKS
# ======================
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

# Model checkpoint to save best model
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'best_model_updated.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

callbacks = [early_stop, reduce_lr, checkpoint]

# ======================
# MODEL TRAINING WITH NEW DATA
# ======================
print("\n" + "="*50)
print("TRAINING MODEL WITH NEW DATA")
print("="*50)

# Calculate steps per epoch
STEP_SIZE_TRAIN = train_generator.samples // train_generator.batch_size
STEP_SIZE_VALID = validation_generator.samples // validation_generator.batch_size

print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {validation_generator.samples}")
print(f"Steps per epoch (Training): {STEP_SIZE_TRAIN}")
print(f"Steps per epoch (Validation): {STEP_SIZE_VALID}")

# Train the model with new data
history = model.fit(
    train_generator,
    steps_per_epoch=STEP_SIZE_TRAIN,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=STEP_SIZE_VALID,
    callbacks=callbacks,
    verbose=1
)

# Save the final updated model
model.save(MODEL_SAVE_PATH)
print(f"\n✓ Final updated model saved as {MODEL_SAVE_PATH}")

# Load the best model from checkpoint
try:
    model = load_model('best_model_updated.h5')
    print("✓ Best model loaded from checkpoint")
except:
    print("⚠ Using final model instead of best checkpoint")

# ======================
# MODEL EVALUATION
# ======================
print("\n" + "="*50)
print("EVALUATING UPDATED MODEL")
print("="*50)

# Evaluate on validation set first
print("\n1. Evaluation on Validation Set:")
valid_results = model.evaluate(validation_generator, verbose=1)
print(f"Validation Loss: {valid_results[0]:.4f}")
print(f"Validation Accuracy: {valid_results[1]*100:.2f}%")
print(f"Validation Precision: {valid_results[2]*100:.2f}%")
print(f"Validation Recall: {valid_results[3]*100:.2f}%")

# Evaluate on test set
print("\n2. Final Evaluation on Test Set:")
test_results = model.evaluate(test_generator, verbose=1)
print(f"Test Loss: {test_results[0]:.4f}")
print(f"Test Accuracy: {test_results[1]*100:.2f}%")
print(f"Test Precision: {test_results[2]*100:.2f}%")
print(f"Test Recall: {test_results[3]*100:.2f}%")

# Generate predictions
print("\n3. Generating predictions...")
test_generator.reset()
y_pred = model.predict(test_generator, verbose=1)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes

# Classification report
print("\n4. Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=new_class_names))

# Confusion matrix
print("5. Confusion Matrix:")
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=new_class_names, yticklabels=new_class_names)
plt.title('Confusion Matrix - Updated Model')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.tight_layout()
plt.savefig('confusion_matrix_updated.png', dpi=300, bbox_inches='tight')
plt.show()
# ======================
# TRAINING HISTORY PLOTS
# ======================
print("\n6. Generating training history plots...")

# Create comprehensive plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# Plot accuracy
ax1.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
ax1.set_title('Model Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True)

# Plot loss
ax2.plot(history.history['loss'], label='Training Loss', marker='o')
ax2.plot(history.history['val_loss'], label='Validation Loss', marker='o')
ax2.set_title('Model Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True)

# Plot learning rate
if 'lr' in history.history:
    ax3.plot(history.history['lr'], marker='o', color='red')
    ax3.set_title('Learning Rate')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.grid(True)
    ax3.set_yscale('log')

# Plot precision and recall
ax4.plot(history.history['precision'], label='Training Precision', marker='o')
ax4.plot(history.history['val_precision'], label='Validation Precision', marker='o')
ax4.plot(history.history['recall'], label='Training Recall', marker='o')
ax4.plot(history.history['val_recall'], label='Validation Recall', marker='o')
ax4.set_title('Precision and Recall')
ax4.set_xlabel('Epoch')
ax4.set_ylabel('Score')
ax4.legend()
ax4.grid(True)

plt.tight_layout()
plt.savefig('training_history_comprehensive.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*50)
print("TRANSFER LEARNING COMPLETE!")
print("="*50)
print(f"Original classes: Algal_Leaf_Spot_of_Jackfruit, Black_Spot_of_Jackfruit, Healthy_leaf_of_Jackfruit")
print(f"New classes: {new_class_names}")
print(f"Total classes in updated model: {new_num_classes}")
print(f"Final model saved as: {MODEL_SAVE_PATH}")
print(f"Best model saved as: best_model_updated.h5")