from __future__ import division
import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize, imsave


def conv2d(input_image, num_output_channels, size_kernel=5, stride=2, name='conv2d'):
    with tf.variable_scope(name):
        # stddev = np.sqrt(2.0 / (np.sqrt(input_image.get_shape()[-1].value * num_output_channels) * size_kernel ** 2))
        stddev = .02
        kernel = tf.get_variable(
            name='w',
            shape=[size_kernel, size_kernel, input_image.get_shape()[-1], num_output_channels],
            dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=stddev)
        )
        biases = tf.get_variable(
            name='b',
            shape=[num_output_channels],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0.0)
        )
        conv = tf.nn.conv2d(input_image, kernel, strides=[1, stride, stride, 1], padding='SAME')
        return tf.nn.bias_add(conv, biases)
def linear_transform(input_image, tx = 0.1, ty = 0.3, tz = 0.3):
    for x in input_image:
        for y in x:
            for z in y:
                z = tx * ty * tz * z
    return input_image

def linear_transform_back(input_image, tx = 0.1, ty = 0.3, tz = 0.3):
    for x in input_image:
        for y in x:
            for z in y:
                z = z/(tx * ty * tz)
    return input_image


def fc(input_vector, num_output_length, name='fc'):
    with tf.variable_scope(name):
        # stddev = np.sqrt(1.0 / (np.sqrt(input_vector.get_shape()[-1].value * num_output_length)))
        stddev = .02
        w = tf.get_variable(
            name='w',
            shape=[input_vector.get_shape()[1], num_output_length],
            dtype=tf.float32,
            initializer=tf.random_normal_initializer(stddev=stddev)
        )
        b = tf.get_variable(
            name='b',
            shape=[num_output_length],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0.0)
        )
        return tf.matmul(input_vector, w) + b


def deconv2d(input_image, output_shape, size_kernel=5, stride=2, stddev=0.02, name='deconv2d'):
    with tf.variable_scope(name):
        # stddev = np.sqrt(1.0 / (np.sqrt(input_image.get_shape()[-1].value * output_shape[-1]) * size_kernel ** 2))
        stddev = .02
        # filter : [height, width, output_channels, in_channels]
        kernel = tf.get_variable(
            name='w',
            shape=[size_kernel, size_kernel, output_shape[-1], input_image.get_shape()[-1]],
            dtype=tf.float32,
            initializer=tf.random_normal_initializer(stddev=stddev)
        )
        biases = tf.get_variable(
            name='b',
            shape=[output_shape[-1]],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0.0)
        )
        deconv = tf.nn.conv2d_transpose(input_image, kernel, strides=[1, stride, stride, 1], output_shape=output_shape)
        return tf.nn.bias_add(deconv, biases)
       

def lrelu(logits, leak=0.2):
    return tf.maximum(logits, leak*logits)


def concat_label(x, label, duplicate=1):
    x_shape = x.get_shape().as_list()
    if duplicate < 1:
        return x
    # duplicate the label to enhance its effect, does it really affect the result?
    label = tf.tile(label, [1, duplicate])
    label_shape = label.get_shape().as_list()
    if len(x_shape) == 2:
        return tf.concat(axis=1, values=[x, label])
    elif len(x_shape) == 4:
        label = tf.reshape(label, [x_shape[0], 1, 1, label_shape[-1]])
        return tf.concat(axis=3, values=[x, label*tf.ones([x_shape[0], x_shape[1], x_shape[2], label_shape[-1]])])


def load_image(
        image_path,  # path of a image
        image_size=64,  # expected size of the image
        image_value_range=(-1, 1),  # expected pixel value range of the image
        is_gray=False,  # gray scale or color image
):
    if is_gray:
        image = imread(image_path, mode='L').astype(np.float32)
    else:
        image = imread(image_path, mode='RGB').astype(np.float32)
    image = imresize(image, [image_size, image_size])
    image = image.astype(np.float32) * (image_value_range[-1] - image_value_range[0]) / 255.0 + image_value_range[0]
    return image


def save_batch_images(
        batch_images,   # a batch of images
        save_path,  # path to save the images
        image_value_range=(-1,1),   # value range of the input batch images
        size_frame=None     # size of the image matrix, number of images in each row and column
):
    # transform the pixcel value to 0~1
    images = (batch_images - image_value_range[0]) / (image_value_range[-1] - image_value_range[0])
    if size_frame is None:
        auto_size = int(np.ceil(np.sqrt(images.shape[0])))
        size_frame = [auto_size, auto_size]
    img_h, img_w = batch_images.shape[1], batch_images.shape[2]
    frame = np.zeros([img_h * size_frame[0], img_w * size_frame[1], 3])
    for ind, image in enumerate(images):
        print ind
        print image
        ind_col = ind % size_frame[1]
        ind_row = ind // size_frame[1]
        frame[(ind_row * img_h):(ind_row * img_h + img_h), (ind_col * img_w):(ind_col * img_w + img_w), :] = image
    imsave(save_path, frame)


def save_images_single(
        batch_images,   # a batch of images
        save_path,  # path to save the images
        image_value_range=(-1,1),   # value range of the input batch images
        size_frame=None,     # size of the image matrix, number of images in each row and column
        index = 0
):
    # transform the pixcel value to 0~1
    images = (batch_images - image_value_range[0]) / (image_value_range[-1] - image_value_range[0])
    if size_frame is None:
        auto_size = int(np.ceil(np.sqrt(images.shape[0])))
        size_frame = [auto_size, auto_size]
    img_h, img_w = batch_images.shape[1], batch_images.shape[2]
    frame = np.zeros([img_h, img_w, 3])
    
    index = index * 10
    for ind, image in enumerate(images):
        if ind == index:
            print ind
            frame[:,:,:] = image
            break
    imsave(save_path, frame)





