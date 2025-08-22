# Classification of Skin Melenoma using Deep Learning Techniques 
---
## What we are doing exactly:

System Architecture:

<img width="1920" height="1080" alt="System Architecture" src="https://github.com/user-attachments/assets/995da272-1d11-4d7a-a1c1-a1af97027b49" />

Block Diagram:

<img width="1920" height="1080" alt="Block Diagram" src="https://github.com/user-attachments/assets/9ff6fc1d-b051-44f9-86f7-7a9c5ec92ac2" />


1. We take an image dataset of benign and malignant skin melanoma
2. Perform below processes on the dataset:
    + Data Cleaning
    + Preprocessing
    + Advanced Processing
3. We make an ensemble of five(to be decided) models.
4. Use feature extraction(to be decided) to extract features from the dataset
5. Train the ensemble of models on the processed dataset of images.
6. Use a voting mechanism(to be decided) to decide if an image shows a benign or malignant skin melanoma.
7. ###### Publish the observations in our research paper.

---
### 1. Dataset Description:
Name: Melanoma Skin Cancer Dataset of 10000 Images
*Desc: This dataset contains two classes of melanoma cancer, malignant and benign.*

Total size: 103.28mb

Total files: 10.6k

*Dataset files:*
|*type*| train | test |
|----------|----------|----------|
|*benign*| 5000  | 500  |
|*malignant*| 4605  | 500  |

URL: https://doi.org/10.34740/kaggle/dsv/3376422

---

### 2. Dataset Cleaning and Processing
#### 2.1 Data Cleaning:
*Data cleaning aims to remove erroneous data from the dataset. We remove images that don't follow a strict set of rules (e.g. have entropy <8.0 and can be decoded correctly).*
Cleaning involves below checks to be performed on images:
```
1. file jpeg signature check using magic number
2. file size thresold
3. image dimesion (must be 300x300)
        3.1 aspect ratio (must be 1)
4. Decoding validity (can the image be loaded)
5. Entropy check:
       For 24-bit RGB images, where each channel is 8-bit, 
       entropy can theoretically go up to: Max Entropy = 24.0 (8 bits × 3 channels),
       where R, G & B are the 3 channels,
       generally, images with entropy 12-15 should be inspected and 
       those having entropy >15 should be discarded

Checks discarded:
1. EXIF data check - no image had exif data & it wasn't required for this use case
2. Uniform color check - gave faulty results for almost all images
3. Progressive Loading Test (JPEG) - not needed for this case
4. Extension mismatch - the images had '.jpg' extension & all the files already had     'jpeg' signatures
```
##### 2.1.1 File jpeg signature check using magic number
File Signature (Magic Number):
Check if the header bytes match the expected format (e.g., JPEG: FF D8, PNG: 89 50 4E 47).
```python
sig_map = { 'jpeg': [b'\xFF\xD8\xFF'] }
if any(header.startswith(s) for s in sig_map.get(fmt, [])):
        print(f"Magic: OK ({fmt})")
        return 1
```
##### 2.1.2 File size thresold
Unusually small or excessively large files might indicate corruption.
```python
min_size=1000
max_size=100000000
size = os.path.getsize(path)
if size < min_size:
    print(f"Size: Too small ({size} B)")
    return 0
elif size > max_size:
    print(f"Size: Too large ({size} B)")
    return 0
else:
    print(f"Size: OK ({size} B)")
    return 1
```
##### 2.1.3 Image dimension
For our case we identify the images that aren't 300x300.
Images not having size 300x300 are then checked if their `aspect ratio` is 1 or not, they can be then treated accordingly.
```python
w, h = img.size
if w == 300 and h == 300:
    return 1
else:
    if w/h == 1:
        return 0
    else:
        return -1
```
##### 2.1.4 Decoding validity
Attempt decoding using a reliable image library (e.g., OpenCV, Pillow). Any exceptions indicate corruption.
```python
try:
    img = Image.open(path)
    img.verify()         # structural check
    img = Image.open(path)
    img.load()           # pixel data check
    print("Decode: OK")
    return [img, 1]
except Exception as e:
    print(f"Decode: ERR ({e})")
    return [None, 0]
```
##### 2.1.5 Entropy check
Checks for very high detail or possibly random noise. Extremely low or high entropy may suggest compression issues or data corruption.

For 24-bit RGB Images (max 8 bits of entropy in each of R G B channels) —
Entropy | Interpretation | Action
|----------|----------|----------|
|< 7.0 | Very low detail, possibly blank or flat | 🔴 Discard or review
|7.0 – 12.0 | Normal range for real-world RGB images | 🟢 Accept
|12.0 – 15.0 | High detail or noise | 🟡 Inspect if unusual
|> 15.0 | Highly random or corrupted (e.g., noise) | 🔴 Suspicious, review
|> 24.0 | Impossible for 8-bit RGB | ❌ Discard (bug or corruption)

⚠️`After visual inspection of training images of the dataset chosen—` 
> Entropy of range 5.5 - 12.0 turned out fine.

```python
hist = img.histogram()
total = sum(hist)
probs = [h/total for h in hist if h > 0]
entropy = -sum(p * math.log2(p) for p in probs)
# 7.0 – 12.0 is the acceptable entropy for colored 24-bit RGB
# but on visual inspection of training images, entropy range of
# 5.5 - 12.0 is also fine
if entropy < 5.5 or entropy > 12.0:
    return 0
return 1
```

#### 2.2 Data Cleaning Verdict:
###### *All benign & malignant images in train and test were NOT Erroneous*
+ All images returned the [1, 1, 1, 1, 1] array. Where:
    + the array returned represent `[valid_magic_number, valid_file_size, valid_image_structure_and_pixel_data, valid_dimensions, valid_entropy]`

Verdict of benign training images —
```python
PATH: dataset/train/benign/
Print verdict of each image?[y/n]: n
Erroneous images:  0
Done.
```
Verdict of malignant training images —
```python
PATH: dataset/train/malignant/
Print verdict of each image?[y/n]: n
Erroneous images:  0
Done.
```

#### 2.3 Data Preprocessing
*To transform the data into a clean, consistent, and standardized format suitable for model training. 
Enhances data quality, reduce noise, and ensure uniformity across the dataset, thereby improving model performance and training efficiency.*
Below are the 4 methods used for data preprocessing
```
1. image resizing (spatial normalization) (optional: all images are 300x300)
2. min-max normalization (pixel scaling) to [0, 1]
3. data augmentation
4. zero-centered normalization (standardization)
```
> Library used - Tensorflow

##### 2.3.1 Image resizing (spatial normalization)
This resizes the image to a fixed width & height.
This was needed because:
+ ML models need a fixed-sized input tensors
+ Prevents dimension mismatch errors during training

```python
image = tf.image.resize(image, target_size)
```
Athough, for the dataset used, this was optional since all images were already 300x300.
##### 2.3.2 Normalize to [0, 1] (or, Pixel scaling)
Converts pixel values from the typical 0–255 range to a [0, 1] float range.
[0,1] normalization was needed because:
+ Makes convergence faster

```python
image = tf.cast(image, tf.float32) / 255.0
```
##### 2.3.3 Data augmentation
Process of artificially increasing the size and diversity of your training dataset by applying random, label-preserving transformations to input images.
Applies random transformations such as flipping, brightness, and contrast adjustments.
Data augmentation was needed because:
+ It increases the diversity of training data without needing more labeled samples.
+ Helps models learn invariant features (e.g., object orientation doesn’t matter).
+ Reduces overfitting.
```python
image = tf.image.random_flip_left_right(image)
image = tf.image.random_brightness(image, max_delta=0.1)
image = tf.image.random_contrast(image, 0.9, 1.1)
```

##### 2.3.4 Zero-centered Normalization (or, Standardization)
Centers pixel values around zero with unit variance `((x - mean) / std)`.
Transforms the pixel distribution to have mean = 0 and std = 1 for each channel.

Zero-centred normalization was needed because:
+ Ensures input distribution is balanced, which improves training stability.
+ Faster convergence
+ More stable gradient descent
```python
image = tf.image.per_image_standardization(image)
```

⚠️*We have performed normalization 2 times.*

Zero-centered normalization (or, Standardization) becomes neccessary after performing [0,1] Normalization:
+ The values are first scaled from [0-255] to [0-1] (pixel scaling) then we center them around a reference point (standardization) so the machine handles all inputs *symmetrically*.

> If we skip `scaling`, standardization might be skewed due to large initial values.
> If we skip `standardization`, optimization may be suboptimal — some channels may dominate learning.

#### 2.4 Building *image classification dataset pipeline* after preprocessing
```python
def build_tf_dataset_pipeline(dataset_directory, batch_size=32):
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory=dataset_directory,
        labels='inferred',    # Infers labels from subfolder names
        label_mode='int',     # Labels are marked as integer indices
        batch_size=batch_size,# Return individual (img, label) pairs for custom pipeline
        image_size=(300,300), # Optional: initially resizes (we’ll override this later)
        shuffle=True
    )

    dataset = dataset.map(resize_image)
    dataset = dataset.map(normalize_image)
    dataset = dataset.map(augment_image)
    dataset = dataset.map(standardize_image)
    return dataset
```
```
Returns: A dataset of image tensors.
The returned value is an object of the class tensorflow.python.data.ops.map_op._MapDataset
```
###### tf.keras.utils.image_dataset_from_directory() arguments:
`directory` - specify the directory of the dataset to be preprocessed, in our case, its, */dataset/train/*

`batch_size` - defines how many images to process at once

`label_mode` - set to `int`, defines that the labels will be marked like 0,1,2,3,....
e.g.
```
    (directory) dataset/train/
                             ├── benign      # label 0
                             ├── malignant   # label 1
```

`labels='inferred'` - infer the labels from the name of the subfolders

`image_size` - sets the image size

`shuffle` - randomizes the dataset order (buffer size = 1000), prevents learning sequence bias

The image is transformed in memory inside the pipeline — it's just passed along to the next `map()` stage.
Each stage of the pipeline (like `.map(normalize_image), .map(augment_image)`) creates a new transient image tensor for that step.
