
# coding: utf-8

# This notebook contains a generator class for Keras called `BSONIterator` that can read directly from the BSON data. You can use it in combination with `ImageDataGenerator` for doing data augmentation.

# In[1]:


import os, sys, math, io
import numpy as np
import pandas as pd
import multiprocessing as mp
import bson
import struct


import keras
from keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf

from collections import defaultdict
from tqdm import *

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "./input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[2]:


keras.__version__, tf.__version__


# In[3]:


data_dir = "./input/"

train_bson_path = os.path.join(data_dir, "train.bson")
num_train_products = 7069896

# train_bson_path = os.path.join(data_dir, "train_example.bson")
# num_train_products = 82

test_bson_path = os.path.join(data_dir, "test.bson")
num_test_products = 1768172


# # Part 1: Create lookup tables
# 
# The generator uses several lookup tables that describe the layout of the BSON file, which products and images are part of the training/validation sets, and so on.
# 
# You only need to generate these tables once, as they get saved to CSV files. If you already have these CSV files, skip to part 2.

# ## Lookup table for categories

# In[4]:


categories_path = os.path.join(data_dir, "category_names.csv")
categories_df = pd.read_csv(categories_path, index_col="category_id")

# Maps the category_id to an integer index. This is what we'll use to
# one-hot encode the labels.
categories_df["category_idx"] = pd.Series(range(len(categories_df)), index=categories_df.index)

categories_df.to_csv("categories.csv")
categories_df.head()


# Create dictionaries for quick lookup of `category_id` to `category_idx` mapping.

# In[5]:


def make_category_tables():
    cat2idx = {}
    idx2cat = {}
    for ir in categories_df.itertuples():
        category_id = ir[0]
        category_idx = ir[4]
        cat2idx[category_id] = category_idx
        idx2cat[category_idx] = category_id
    return cat2idx, idx2cat


# In[6]:


cat2idx, idx2cat = make_category_tables()


# In[7]:


# Test if it works:
cat2idx[1000012755], idx2cat[4]


# ## Read the BSON files
# 
# We store the offsets and lengths of all items, allowing us random access to the items later.
# 
# Inspired by code from: https://www.kaggle.com/vfdev5/random-item-access
# 
# Note: this takes a few minutes to execute, but we only have to do it once (we'll save the table to a CSV file afterwards).

# In[8]:


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


# In[9]:


train_offsets_df = read_bson(train_bson_path, num_records=num_train_products, with_categories=True)


# In[10]:


train_offsets_df.head()


# In[11]:


train_offsets_df.to_csv("train_offsets.csv")


# In[12]:


# How many products?
len(train_offsets_df)


# In[13]:


# How many categories?
len(train_offsets_df["category_id"].unique())


# In[14]:


# How many images in total?
train_offsets_df["num_imgs"].sum()


# Also create a table for the offsets from the test set.

# In[15]:


test_offsets_df = read_bson(test_bson_path, num_records=num_test_products, with_categories=False)


# In[16]:


test_offsets_df.head()


# In[17]:


test_offsets_df.to_csv("test_offsets.csv")


# ## Create a random train/validation split
# 
# We split on products, not on individual images. Since some of the categories only have a few products, we do the split separately for each category.
# 
# This creates two new tables, one for the training images and one for the validation images. There is a row for every single image, so if a product has more than one image it occurs more than once in the table.

# In[18]:


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


# Create a 80/20 split. Also drop 90% of all products to make the dataset more manageable. (Note: if `drop_percentage` > 0, the progress bar doesn't go all the way.)

# In[19]:


train_images_df, val_images_df = make_val_set(train_offsets_df, split_percentage=0.2, 
                                              drop_percentage=0.99)


# In[20]:


train_images_df.head()


# In[21]:


val_images_df.head()


# In[22]:


print("Number of training images:", len(train_images_df))
print("Number of validation images:", len(val_images_df))
print("Total images:", len(train_images_df) + len(val_images_df))


# Are all categories represented in the train/val split? (Note: if the drop percentage is high, then very small categories won't have enough products left to make it into the validation set.)

# In[23]:


len(train_images_df["category_idx"].unique()), len(val_images_df["category_idx"].unique())


# Quickly verify that the split really is approximately 80-20:

# In[24]:


category_idx = 619
num_train = np.sum(train_images_df["category_idx"] == category_idx)
num_val = np.sum(val_images_df["category_idx"] == category_idx)
num_val / num_train


# Close enough. ;-) Remember that we split on products but not all products have the same number of images, which is where the slightly discrepancy comes from. (Also, there tend to be fewer validation images if `drop_percentage` > 0.)

# Save the lookup tables as CSV so that we don't need to repeat the above procedure again.

# In[25]:


train_images_df.to_csv("train_images.csv")
val_images_df.to_csv("val_images.csv")


# ## Lookup table for test set images
# 
# Create a list containing a row for each image. If a product has more than one image, it appears more than once in this list.

# In[26]:


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


# In[27]:


test_images_df = make_test_set(test_offsets_df)


# In[28]:


test_images_df.head()


# In[29]:


print("Number of test images:", len(test_images_df))


# In[30]:


test_images_df.to_csv("test_images.csv")


# # Part 2: The generator

# First load the lookup tables from the CSV files (you don't need to do this if you just did all the steps from part 1).

# In[31]:


categories_df = pd.read_csv("categories.csv", index_col=0)
cat2idx, idx2cat = make_category_tables()

train_offsets_df = pd.read_csv("train_offsets.csv", index_col=0)
train_images_df = pd.read_csv("train_images.csv", index_col=0)
val_images_df = pd.read_csv("val_images.csv", index_col=0)

test_offsets_df = pd.read_csv("test_offsets.csv", index_col=0)
test_images_df = pd.read_csv("test_images.csv", index_col=0)


# The Keras generator is implemented by the `BSONIterator` class. It creates batches of images (and their one-hot encoded labels) directly from the BSON file. It can be used with multiple workers.
# 
# **Note:** For fastest results, put the train.bson and test.bson files on a fast drive (SSD).
# 
# See also the code in: https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py

# In[32]:


from keras.preprocessing.image import Iterator
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

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


# In[33]:


train_bson_file = open(train_bson_path, "rb")


# Create a generator for training and a generator for validation.

# In[56]:


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

# In[57]:


next(train_gen)  # warm-up

bx, by = next(train_gen)


# Does it really output images and one-hot encoded class labels? Note that the images are pre-processed (and augmented) and therefore may look weird.

# In[58]:




# In[59]:


cat_idx = np.argmax(by[-1])
cat_id = idx2cat[cat_idx]
categories_df.loc[cat_id]


# In[60]:


bx, by = next(val_gen)


# In[61]:




# In[62]:


cat_idx = np.argmax(by[-1])
cat_id = idx2cat[cat_idx]
categories_df.loc[cat_id]


# # Part 3: Training
# 
# Create a very simple Keras model and train it, to test that the generators work.

# In[63]:


from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D

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


# In[ ]:


# To train the model:
model.fit_generator(train_gen,
                    steps_per_epoch = num_train_images // batch_size,
                    epochs = 1,
                    validation_data = val_gen,
                    validation_steps = num_val_images // batch_size,
                    workers = 8)


# In[ ]:


# To evaluate on the validation set:
model.evaluate_generator(val_gen, steps=num_val_images // batch_size, workers=8)


# # Part 4: Test set predictions
# 
# Use `BSONIterator` to load the test set images in batches.

# In[ ]:


test_bson_file = open(test_bson_path, "rb")


# In[ ]:


test_datagen = ImageDataGenerator()
test_gen = BSONIterator(test_bson_file, test_images_df, test_offsets_df,
                        num_classes, test_datagen, batch_size=batch_size, 
                        with_labels=False, shuffle=False)


# Running `model.predict_generator()` gives a list of 3095080 predictions, one for each image. 
# 
# The indices of the predictions correspond to the indices in `test_images_df`. After making the predictions, you probably want to average the predictions for products that have multiple images.
# 
# Use `idx2cat[]` to convert the predicted category index back to the original class label.

# In[ ]:


num_test_samples = len(test_images_df)
predictions = model.predict_generator(test_gen, steps=num_test_samples // batch_size, workers=8)
