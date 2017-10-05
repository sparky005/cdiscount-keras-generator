import os, sys, math, io
import argparse
import numpy as np
import pandas as pd
import multiprocessing as mp
import bson
import struct

from tensorflow.python.lib.io import file_io
import keras
from keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf

from collections import defaultdict
from tqdm import *

from keras.preprocessing.image import Iterator
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D

# FUNCTIONS: 
def make_category_tables():
    cat2idx = {}
    idx2cat = {}
    for ir in categories_df.itertuples():
        category_id = ir[0]
        category_idx = ir[4]
        cat2idx[category_id] = category_idx
        idx2cat[category_idx] = category_id
    return cat2idx, idx2cat

# We store the offsets and lengths of all items, allowing us random access to the items later.
# 
# 
# Note: this takes a few minutes to execute, but we only have to do it once (we'll save the table to a CSV file afterwards).
def read_bson(bson_path, num_records, with_categories):
    rows = {}
    with open(bson_path, "rb") as f, tqdm(total=num_records) as pbar:
        offset = 0
        while True:
            item_length_bytes = f.read(4)
            if len(item_length_bytes) == 0:
                break

            length = struct.unpack("<i", item_length_bytes)[0]

            f.seek(offset)
            item_data = f.read(length)
            assert len(item_data) == length

            item = bson.BSON.decode(item_data)
            product_id = item["_id"]
            num_imgs = len(item["imgs"])

            row = [num_imgs, offset, length]
            if with_categories:
                row += [item["category_id"]]
            rows[product_id] = row

            offset += length
            f.seek(offset)
            pbar.update()

    columns = ["num_imgs", "offset", "length"]
    if with_categories:
        columns += ["category_id"]

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "product_id"
    df.columns = columns
    df.sort_index(inplace=True)
    return df

# ## Create a random train/validation split
# 
# We split on products, not on individual images. Since some of the categories only have a few products, we do the split separately for each category.
# 
# This creates two new tables, one for the training images and one for the validation images. There is a row for every single image, so if a product has more than one image it occurs more than once in the table.
def make_val_set(df, split_percentage=0.2, drop_percentage=0.):
    # Find the product_ids for each category.
    category_dict = defaultdict(list)
    for ir in tqdm(df.itertuples()):
        category_dict[ir[4]].append(ir[0])

    train_list = []
    val_list = []
    with tqdm(total=len(df)) as pbar:
        for category_id, product_ids in category_dict.items():
            category_idx = cat2idx[category_id]

            # Randomly remove products to make the dataset smaller.
            keep_size = int(len(product_ids) * (1. - drop_percentage))
            if keep_size < len(product_ids):
                product_ids = np.random.choice(product_ids, keep_size, replace=False)

            # Randomly choose the products that become part of the validation set.
            val_size = int(len(product_ids) * split_percentage)
            if val_size > 0:
                val_ids = np.random.choice(product_ids, val_size, replace=False)
            else:
                val_ids = []

            # Create a new row for each image.
            for product_id in product_ids:
                row = [product_id, category_idx]
                for img_idx in range(df.loc[product_id, "num_imgs"]):
                    if product_id in val_ids:
                        val_list.append(row + [img_idx])
                    else:
                        train_list.append(row + [img_idx])
                pbar.update()
                
    columns = ["product_id", "category_idx", "img_idx"]
    train_df = pd.DataFrame(train_list, columns=columns)
    val_df = pd.DataFrame(val_list, columns=columns)   
    return train_df, val_df

# ## Lookup table for test set images
# 
# Create a list containing a row for each image. If a product has more than one image, it appears more than once in this list.
def make_test_set(df):
    test_list = []
    for ir in tqdm(df.itertuples()):
        product_id = ir[0]
        num_imgs = ir[1]
        for img_idx in range(num_imgs):
            test_list.append([product_id, img_idx])

    columns = ["product_id", "img_idx"]
    test_df = pd.DataFrame(test_list, columns=columns)
    return test_df

# The Keras generator is implemented by the `BSONIterator` class. It creates batches of images (and their one-hot encoded labels) directly from the BSON file. It can be used with multiple workers.
# 
# **Note:** For fastest results, put the train.bson and test.bson files on a fast drive (SSD).
# 
# See also the code in: https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py
class BSONIterator(Iterator):
    def __init__(self, bson_file, images_df, offsets_df, num_class,
                 image_data_generator, target_size=(180, 180), with_labels=True,
                 batch_size=32, shuffle=False, seed=None):

        self.file = bson_file
        self.images_df = images_df
        self.offsets_df = offsets_df
        self.with_labels = with_labels
        self.samples = len(images_df)
        self.num_class = num_class
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        self.image_shape = self.target_size + (3,)

        print("Found %d images belonging to %d classes." % (self.samples, self.num_class))

        super(BSONIterator, self).__init__(self.samples, batch_size, shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=K.floatx())
        if self.with_labels:
            batch_y = np.zeros((len(batch_x), self.num_class), dtype=K.floatx())

        for i, j in enumerate(index_array):
            # Protect file and dataframe access with a lock.
            with self.lock:
                image_row = self.images_df.iloc[j]
                product_id = image_row["product_id"]
                offset_row = self.offsets_df.loc[product_id]

                # Read this product's data from the BSON file.
                self.file.seek(offset_row["offset"])
                item_data = self.file.read(offset_row["length"])

            # Grab the image from the product.
            item = bson.BSON.decode(item_data)
            img_idx = image_row["img_idx"]
            bson_img = item["imgs"][img_idx]["picture"]

            # Preprocess the image.
            img = load_img(io.BytesIO(bson_img), target_size=self.target_size)
            x = img_to_array(img)
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)

            # Add the image and the label to the batch (one-hot encoded).
            batch_x[i] = x
            if self.with_labels:
                batch_y[i, image_row["category_idx"]] = 1

        if self.with_labels:
            return batch_x, batch_y
        else:
            return batch_x

    def next(self):
        with self.lock:
            index_array = next(self.index_generator)
        return self._get_batches_of_transformed_samples(index_array)

def build_model():
    model = Sequential()
    model.add(Conv2D(32, 3, padding="same", activation="relu", input_shape=(90, 90, 3)))
    model.add(MaxPooling2D())
    model.add(Conv2D(64, 3, padding="same", activation="relu"))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, 3, padding="same", activation="relu"))
    model.add(MaxPooling2D())
    model.add(GlobalAveragePooling2D())
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    model.summary()
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # input arguments
    parser.add_argument(
        '--train-file-dir',
        help='path to train.bson',
        nargs='+',
        required=True
    )
    args = parser.parse_args()
    data_dir = args.train_file_dir

    train_bson_path = os.path.join(data_dir, "train.bson")
    num_train_products = 7069896


    test_bson_path = os.path.join(data_dir, "test.bson")
    num_test_products = 1768172


    categories_path = os.path.join(data_dir, "category_names.csv")
    categories_df = pd.read_csv(categories_path, index_col="category_id")

    # Maps the category_id to an integer index. This is what we'll use to
    # one-hot encode the labels.
    categories_df["category_idx"] = pd.Series(range(len(categories_df)), index=categories_df.index)

    categories_df.to_csv("categories.csv")
    cat2idx, idx2cat = make_category_tables()
    train_offsets_df = read_bson(train_bson_path, num_records=num_train_products, with_categories=True)
    train_offsets_df.to_csv("train_offsets.csv")
    test_offsets_df = read_bson(test_bson_path, num_records=num_test_products, with_categories=False)
    test_offsets_df.to_csv("test_offsets.csv")

    train_images_df, val_images_df = make_val_set(train_offsets_df, split_percentage=0.2, 
                                                  drop_percentage=0.9)

    print("Number of training images:", len(train_images_df))
    print("Number of validation images:", len(val_images_df))
    print("Total images:", len(train_images_df) + len(val_images_df))

    train_images_df.to_csv("train_images.csv")
    val_images_df.to_csv("val_images.csv")


    test_images_df = make_test_set(test_offsets_df)


    print("Number of test images:", len(test_images_df))
    test_images_df.to_csv("test_images.csv")
    train_bson_file = open(train_bson_path, "rb")

    # Create a generator for training and a generator for validation.
    num_classes = 5270
    num_train_images = len(train_images_df)
    num_val_images = len(val_images_df)
    batch_size = 300
    target_size = (90, 90)

    # Tip: use ImageDataGenerator for data augmentation and preprocessing.
    train_datagen = ImageDataGenerator()
    train_gen = BSONIterator(train_bson_file, train_images_df, train_offsets_df,
                             num_classes, train_datagen, target_size=target_size, batch_size=batch_size, shuffle=True)

    val_datagen = ImageDataGenerator()
    val_gen = BSONIterator(train_bson_file, val_images_df, train_offsets_df,
                           num_classes, val_datagen, batch_size=batch_size)


    # How fast is the generator? Create a single batch:
    next(train_gen)  # warm-up

    bx, by = next(train_gen)

    cat_idx = np.argmax(by[-1])
    cat_id = idx2cat[cat_idx]
    categories_df.loc[cat_id]

    bx, by = next(val_gen)

    cat_idx = np.argmax(by[-1])
    cat_id = idx2cat[cat_idx]
    categories_df.loc[cat_id]

    model = build_model()

    # To train the model:
    model.fit_generator(train_gen,
                        steps_per_epoch = num_train_images // batch_size,
                        epochs = 1,
                        validation_data = val_gen,
                        validation_steps = num_val_images // batch_size,
                        workers = 8)

    # To evaluate on the validation set:
    model.evaluate_generator(val_gen, steps=num_val_images // batch_size, workers=8)


    # # Part 4: Test set predictions
    # 
    # Use `BSONIterator` to load the test set images in batches.



    test_bson_file = open(test_bson_path, "rb")

    test_datagen = ImageDataGenerator()
    test_gen = BSONIterator(test_bson_file, test_images_df, test_offsets_df,
                            num_classes, test_datagen, batch_size=batch_size, 
                            with_labels=False, shuffle=False)


    # Running `model.predict_generator()` gives a list of 3095080 predictions, one for each image. 
    # 
    # The indices of the predictions correspond to the indices in `test_images_df`. After making the predictions, you probably want to average the predictions for products that have multiple images.
    # 
    # Use `idx2cat[]` to convert the predicted category index back to the original class label.



    num_test_samples = len(test_images_df)
    predictions = model.predict_generator(test_gen, steps=num_test_samples // batch_size, workers=8)
