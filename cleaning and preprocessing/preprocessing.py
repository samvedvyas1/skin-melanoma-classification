import tensorflow as tf

# 1. Resize
def resize_image(image, label, target_size=(300, 300)):
    print("--- Resizing image...")
    image = tf.image.resize(image, target_size)
    return image, label

# 2. Normalize to [0, 1]
def normalize_image(image, label):
    print("--- Normalizing image to [0, 1]...")
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

# 3. Data Augmentation
def augment_image(image, label):
    print("--- Applying data augmentation...")
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, 0.9, 1.1)
    return image, label

# 4. Standardization
def standardize_image(image, label):
    print("--- Standardizing image (zero mean, unit variance)...")
    image = tf.image.per_image_standardization(image)
    return image, label

# Example dataset pipeline
def build_tf_dataset(dataset_directory, batch_size=32):
    training_dataset = tf.keras.utils.image_dataset_from_directory(
        directory=dataset_directory,
        labels='inferred',               # Infers labels from subfolder names
        label_mode='int',                # Labels are marked as integer indices
        batch_size=batch_size,           # Return individual (img, label) pairs for custom pipeline
        image_size=(300, 300),           # Optional: initially resizes (we’ll override this later)
        shuffle=True
    )

    training_dataset = training_dataset.map(resize_image)
    training_dataset = training_dataset.map(normalize_image)
    training_dataset = training_dataset.map(augment_image)
    training_dataset = training_dataset.map(standardize_image)
    return training_dataset

def main():
    path='dataset/train/'

    print("\n\n\n! PREPROCESSING STARTED")
    preprocessed_dataset = build_tf_dataset(path)
    # preprocessed_dataset will be an object of class tensorflow.python.data.ops.map_op._MapDataset
    
    print("! PREPROCESSING DONE")

    print("\n--- TYPE: ",type(preprocessed_dataset))


if __name__=="__main__":
    main()
