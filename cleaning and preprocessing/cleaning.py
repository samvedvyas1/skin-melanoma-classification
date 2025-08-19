# This program goes through images in a folder, and checks for 5 things in an image
# header bytes (magic number) for checking image format
# file size of the image (checks if its in an appropriate range)
# if the image is decodable or not
# dimensions of the image are a fixed value or not
# entropy of the imge is in an appropriate range or not
# 
# For passing each check successfully the image_checker() function appends 1
# to a list, otherwise it appends -1 0 or -2 depending upon the situation, 
# therefore for a non-erroneous image the function returns [1, 1, 1, 1, 1],
# 
# This check is performed for every image in the specified folder and the images 
# not returning the array mentioned before are deemed erroneous and their name and 
# the array returned by those images are stored in a json file.

import os
import sys
import imghdr
from PIL import Image, ImageFile
import numpy as np
import math
import json

# Increase parser tolerance for truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

def check_magic_number(path) -> int:
    # Returns 1 if image signature is 'jpeg' (the format of the images in the database) otherwise 0

    sig_map = {
        'jpeg': [b'\xFF\xD8\xFF']
    }
    try:
        with open(path, 'rb') as f:
            header = f.read(8)
    except Exception as e:
        # print(f"Magic: ERR ({e})")
        return 0
    fmt = imghdr.what(path)
    if not fmt:
        # print("Magic: Unknown format")
        return 0
    elif any(header.startswith(s) for s in sig_map.get(fmt, [])):
        # print(f"Magic: OK ({fmt})")
        return 1
    else:
        # print(f"Magic: Mismatch ({fmt})")
        return 0


def check_file_size(path, min_size=1000, max_size=100_000_000) -> int:
    # Returns 1 if image has a valid size otherwise 0

    size = os.path.getsize(path)
    if size < min_size:
        # print(f"Size: Too small ({size} B)")
        return 0
    elif size > max_size:
        # print(f"Size: Too large ({size} B)")
        return 0
    else:
        # print(f"Size: OK ({size} B)")
        return 1


def load_image(path) -> list:
    # Returns a list of length 2, if the image can be verified and loaded correctly 
    # the returned list is [PIL.Image.Image object, 1]
    # in all other cases we return [None,0].
    # The returned PIL.Image.Image object is used by check_dim() & check_entropy().
    try:
        img = Image.open(path)
        img.verify()         # structural check
        img = Image.open(path)
        img.load()           # pixel data check
        # print("Decode: OK")
        return [img, 1]
    except Exception as e:
        # print(f"Decode: ERR ({e})")
        return [None, 0]


def check_dim(img) -> int:
    # Returns 1 if dimension is 300x300, since the image of the dataset follow the same format
    # else:
    #   if width and height arent equal return 0
    #   else return -1
    w, h = img.size
    if w == 300 and h == 300:
        return 1
    else:
        if w/h == 1:
            return 0
        else:
            return -1
    


def check_entropy(img) -> int:
    # Regarding entropy of image 7.0 – 12.0 is the acceptable entropy 
    # for colored 24-bit RGB otherwise its likely noisy or corrupted.
    # Thus, for valid images it returns 1 otherwise 0.
    
    # Entropy
    hist = img.histogram()
    total = sum(hist)
    probs = [h/total for h in hist if h > 0]
    entropy = -sum(p * math.log2(p) for p in probs)

    # 7.0 – 12.0 is the acceptable entropy for colored 24-bit RGB (the dataset images are 24bit RGB)
    if entropy < 5.5 or entropy > 12.0:
        return 0
    return 1

def image_checker(path) -> list:
    # Returns a list which has the following structure:
    #           [valid_magic_number, valid_file_size, valid_image_structure_and_pixel_data, valid_dimensions, valid_entropy]
    #
    # An image is VALID if this function returns the below list:
    #           [1, 1, 1, 1, 1]
    # Otherwise the image is NOT VALID, we separate those image and deal with them in main()

    valid_list=list()
    valid_list.append(check_magic_number(path))

    valid_list.append(check_file_size(path))

    img_load_result = load_image(path)

    valid_list.append(img_load_result[1])
    if img_load_result[0]:
        valid_list.append(check_dim(img_load_result[0]))
        valid_list.append(check_entropy(img_load_result[0]))
    else:
        # If the image can't even be loaded and verified by PIL then
        # checking for dimensions and entropy is useless
        # by placing -2 for (valid_dimensions, valid_entropy)
        # it is implied that the image can't be verified and loaded by PIL.
        #
        # So if an image has a valid magic number and file size but can't be loaded and verified by PIL
        # will return the below list:
        #       [1, 1, 0, -2, -2]
        valid_list.append(-2)
        valid_list.append(-2)
    return valid_list

def save_faulty_image_result(faulty_image_file, image_result):
    # faulty_image_file="test_benign.json"

    # Load existing JSON list or create a new one
    data=None
    if not os.path.exists(faulty_image_file):
        # Create the file with an empty list
        with open(faulty_image_file, 'w') as f:
            json.dump([], f)

    with open(faulty_image_file, 'r+') as f:
        data = json.load(f)
        data.append(image_result)
        f.seek(0)
        json.dump(data, f, indent=4)
        f.truncate()

def main():
    
    path = r"dataset/train/benign/"

    faulty_img_file="train_benign.json"
    print(f"PATH: {path}")
    ask=""
    print("Print verdict of each image?[y/n]: ",end='')
    ask=input()

    erroneous_img=0

    for i in range(0, 5000):
        img_name = "melanoma_"+str(i)+".jpg"
        img_path = path+img_name

        image_verdict = image_checker(img_path)

        if image_verdict != [1, 1, 1, 1, 1]:
            erroneous_img+=1
            data_to_append = {
                "name": img_name,
                "image_result": image_verdict
            }
            save_faulty_image_result(faulty_img_file, data_to_append)
        if ask=="y" or ask=="Y":
            print("Image : ",img_name,"| Vertict: ",image_verdict)
    print("Erroneous images: ", erroneous_img)
    print("Done.")


if __name__ == "__main__":
    main()
