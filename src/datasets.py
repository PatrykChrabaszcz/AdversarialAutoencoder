import tensorflow as tf
import numpy as np
import os
from PIL import Image

# http://stackoverflow.com/questions/34340489/tensorflow-read-images-with-labels
# CelebA Dataset
class Celeb:
    def __init__(self, dataset_path, attributes_path, batch_size=64, img_size=64):
        with tf.variable_scope("Input"):
            # % of images used for validation

            self.img_size = img_size
            self.batch_size = batch_size
            self.dataset_path = dataset_path
            self.attributes_path = attributes_path
            self.crop_size = 152

            self._f, self._l = self._read_paths()
            self._attr_names = self._read_attr_names()

            # Use 5 * batch_size for validation
            val_size = 5 * batch_size
            self.val_size = val_size
            self.train_size = len(self._f) - val_size

            val_images = tf.convert_to_tensor(self._f[-val_size:], dtype=tf.string)
            val_labels = tf.convert_to_tensor(self._l[-val_size:], dtype=tf.int32)

            train_images = tf.convert_to_tensor(self._f[:-val_size], dtype=tf.string)
            train_labels = tf.convert_to_tensor(self._l[:-val_size], dtype=tf.int32)

            val_queue = tf.train.slice_input_producer([val_images, val_labels],
                                                      shuffle=False, capacity=self.batch_size)
            train_queue = tf.train.slice_input_producer([train_images, train_labels],
                                                        shuffle=False, capacity=self.batch_size)

            self.val_batch = self._get_batch(val_queue, train=False)
            self.train_batch = self._get_batch(train_queue, train=True)

    def _get_batch(self, queue, train=True):
        image = tf.read_file(queue[0])
        image = tf.image.decode_jpeg(image, channels=3)
        label = queue[1]
        image = tf.image.resize_image_with_crop_or_pad(image, self.crop_size, self.crop_size)
        image = tf.image.resize_images(image, size=[self.img_size, self.img_size])

        # Apply 50% random horizontal flip
        if train:
            image = tf.image.random_flip_left_right(image)
        # Image should be in range -1, 1
        image = tf.cast(image, tf.float32)
        image = image / 127.5 - 1.0
        image = tf.image.resize_images(image, [self.img_size, self.img_size])
        # TODO: Add mean subtraction

        # NCHW is deafault for GPU (Should run faster)
        image = tf.transpose(image, [2, 0, 1])

        if train:
            return tf.train.shuffle_batch([image, label], self.batch_size, capacity=50000, min_after_dequeue=10000,
                                          num_threads=4)
        else:
            return tf.train.batch([image, label], self.batch_size, capacity=self.val_size, num_threads=1)

    # Returns paths to images and labels for images
    def _read_paths(self):
        files = []
        labels = []
        with open(self.attributes_path, 'r') as f:
            for i, line in enumerate(f):
                # First two lines contain meta information
                if i < 2:
                    continue
                line = line.split()
                files.append(os.path.join(self.dataset_path, line[0]))
                labels.append([int(x) for x in line[1:]])

        return np.array(files), np.array(labels)

    def _read_attr_names(self):
        with open(self.attributes_path, 'r') as f:
            f.readline()
            return f.readline()

    def create_embedding_metadata(self):
        if not os.path.exists('metadata'):
            os.mkdir('metadata')

        # Create Sprite file (Big Image file)
        val_images = self._f[-self.batch_size:]

        images = []
        for img_name in val_images:
            img = Image.open(img_name)
            x = (img.width-self.crop_size)//2
            y = (img.height-self.crop_size)//2
            img = img.crop((x, y, x + self.crop_size, y + self.crop_size))
            img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
            images.append(img)

        x_len = int(np.sqrt(self.batch_size))
        y_len = int(self.batch_size//x_len)
        x_len = x_len + 1 if (x_len * y_len < self.batch_size) else x_len

        sprite_img = Image.new('RGB', (x_len*self.img_size, y_len*self.img_size))

        for i, img in enumerate(images):
            sprite_img.paste(img, ((i % x_len)*self.img_size, (i//x_len)*self.img_size))

        sprite_img.save('metadata/sprite.png')

        # Create TSV file with labels

        val_labels = self._l[-self.batch_size:]

        with open('metadata/metadata.tsv', 'w') as f:
            f.write(self._attr_names)
            for label in val_labels:
                f.write('\t'.join([str(i) for i in label]))
                f.write('\n')
