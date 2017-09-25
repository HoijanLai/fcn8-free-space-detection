import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
from tqdm import tqdm
from argparse import ArgumentParser as argparser

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    # load the graph
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()

    # recover the tensor in the graph and return them
    t1 = graph.get_tensor_by_name(vgg_input_tensor_name)
    t2 = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    t3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    t4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    t5 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    return t1, t2, t3, t4, t5

# tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # layer7
    part7 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, 1, padding = 'SAME', 
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-2))

    part7_2x = tf.layers.conv2d_transpose(part7, num_classes, 4, 2, padding = 'SAME', 
                                          kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-2))
    # fuse layer 4 and layer 7
    part4 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, 1, padding = 'SAME',
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-2))

    part_4_7 = tf.add(part7_2x, part4)

    part_4_7_2x = tf.layers.conv2d_transpose(part_4_7, num_classes, 8, 2, padding = 'SAME', 
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-2))   

    # fuse layer 3 with layer 4 + layer 7
    part3 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, 1, padding = 'SAME',
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-2))   

    part_3_4_7 = tf.add(part3, part_4_7_2x)

    part_3_4_7_2x = tf.layers.conv2d_transpose(part_3_4_7, num_classes, 32, 8, padding = 'SAME',
                                               kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-2))   

    return part_3_4_7_2x 

# tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=correct_label, logits=logits))
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)
    return logits, train_op, cross_entropy_loss
# tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    init = tf.global_variables_initializer()
    sess.run(init)
    for ep in range(epochs):
        print("training epoch #%d"%ep)
        for batch_images, batch_labels in tqdm(get_batches_fn(batch_size)):
            feed_dict = {input_image   : batch_images, 
                         correct_label : batch_labels,
                         keep_prob     : 0.5}
            sess.run([train_op, cross_entropy_loss], feed_dict = feed_dict)

# tests.test_train_nn(train_nn)


def run(batch_size = 16, epochs = 6, lr = 0.001):
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network
        input_image, keep_prob, t3, t4, t7 = load_vgg(sess, vgg_path) 
        nn_last_layer = layers(t3, t4, t7, num_classes)
        correct_label = tf.placeholder(tf.float32, nn_last_layer.get_shape())
        learning_rate = tf.constant(lr)
        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)

        batch_size = batch_size
        epochs = epochs 
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
                 correct_label, keep_prob, learning_rate)

        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    parser = argparser()
    parser.add_argument('batch_size', type=int, default=16, nargs='?')
    parser.add_argument('epochs', type=int, default=6, nargs='?')
    parser.add_argument('learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    run(args.batch_size, args.epochs)
