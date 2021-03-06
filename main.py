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

tests.test_load_vgg(load_vgg, tf)




def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    init_style = tf.truncated_normal_initializer(stddev=0.01)

    # layer7
    part7 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, 1,
                             padding='SAME',
                             kernel_initializer=init_style, 
                             name='part7') 

    part7_2x = tf.layers.conv2d_transpose(part7, num_classes, 4, 2,
                                          padding='SAME',
                                          kernel_initializer=init_style,
                                          name='part7_2x') 
 

    # fuse layer 4 and layer 7
    part4 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, 1,
                             padding='SAME',
                             kernel_initializer=init_style,
                             name='part4') 


    part_4_7 = tf.add(part7_2x, part4)

    part_4_7_2x = tf.layers.conv2d_transpose(part_4_7, num_classes, 8, 2,
                                             padding='SAME',
                                             kernel_initializer=init_style,
                                             name='part_4_7_2x') 
 

    # fuse layer 3 with layer 4 & layer 7
    part3 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, 1,
                             padding='SAME',
                             kernel_initializer=init_style,
                             name='part3') 


    part_3_4_7 = tf.add(part3, part_4_7_2x)

    part_3_4_7_8x = tf.layers.conv2d_transpose(part_3_4_7, num_classes, 32, 8,
                                               padding='SAME',
                                               kernel_initializer=init_style,
                                               name='part_3_4_7_8x') 

    return part_3_4_7_8x 

tests.test_layers(layers)





def optimize(nn_last_layer, correct_label, learning_rate, num_classes, reg=1e-2, freeze=False):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate (now a constant tensor)
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    ## define loss: softmax + l2
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_const = reg 

    cross_entropy_loss = tf.reduce_mean(
                         tf.nn.softmax_cross_entropy_with_logits(labels=correct_label, logits=logits)
                         ) + reg_const * tf.reduce_sum(reg_losses)

    ## collect var_list to train 
    ## (if `freeze` is set to true only the newly defined part will be trained, see `layer`)
    var_prefices = ['part7', 'part7_2x',
                    'part4', 'part_4_7_2x',
                    'part3', 'part_3_4_7_8x']
    trainables = tf.trainable_variables()

    var_list = []
    for var_prefix in var_prefices:
        var_list += [var for var in trainables if var_prefix in var.name]

    if len(var_list) == 0 or not freeze: var_list = None

    ## finally define the optimizer 
    global_step = tf.Variable(0, trainable = False)
    lr_decay = tf.train.exponential_decay(learning_rate, global_step, 10000, 0.96)
    train_op = tf.train.AdamOptimizer(lr_decay).minimize(cross_entropy_loss,
                                                         global_step=global_step,
                                                         var_list=var_list)

    return logits, train_op, cross_entropy_loss

tests.test_optimize(optimize)




def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, keep_prob_init=0.5):
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
                          (now unused, the learning rate is handled by `lr_decay` in `optimize`)
    """
    init = tf.global_variables_initializer()
    sess.run(init)

    ## for tqdm description
    num_batches = 0
    loss_init = None 
    
    ## the training loop
    for ep in range(epochs):
        in_counting_msg = ' and counting items' if ep == 0 else ''
        print("training epoch #%d%s"%(ep+1, in_counting_msg))

        ## hand the generator to tqdm 
        gen_client = tqdm(get_batches_fn(batch_size), total=num_batches) 

        ## batch training        
        for batch_images, batch_labels in gen_client:
            feed_dict = {input_image   : batch_images, 
                         correct_label : batch_labels,
                         keep_prob     : keep_prob_init}
            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict=feed_dict)

            ## count items in the first epoch
            if ep == 0: num_batches += 1
            
            ## dynamically display the loss
            if not loss_init: loss_init = loss
            gen_client.set_description("LOSS_INIT: %.4f loss: %.4f"%(loss_init, loss))

###### END of train_nn
    
tests.test_train_nn(train_nn)







def run(batch_size=4, epochs=6, lr=0.001, kp=0.5, reg=1e-2):
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    ## Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        ## Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')

        ## Create function to get batches
        ## set the aug_size to make geometric augmnentations
        ## set channel_shift to allow channel shift augmentations
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'),
                                                   image_shape,
                                                   aug_size=0.6,
                                                   channel_shift=0.8)

        ## build the architecture
        input_image, keep_prob, t3, t4, t7 = load_vgg(sess, vgg_path) 
        nn_last_layer = layers(t3, t4, t7, num_classes)

        ## build the optimization part of the graph 
        correct_label = tf.placeholder(tf.float32, nn_last_layer.get_shape())
        learning_rate = tf.constant(lr)
        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes, reg)

        ## train the model 
        batch_size = batch_size
        epochs = epochs 
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
                 correct_label, keep_prob, kp)

        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    parser = argparser()
    parser.add_argument('batch_size', type=int, default=2, nargs='?')
    parser.add_argument('epochs', type=int, default=6, nargs='?')
    parser.add_argument('learning_rate', type=float, default=0.001, nargs='?')
    parser.add_argument('keep_prob', type=float, default=0.5, nargs='?')
    parser.add_argument('regularization', type=float, default=1e-2, nargs='?')
    args = parser.parse_args()
    print(args)
    run(args.batch_size, args.epochs, args.learning_rate, args.keep_prob, args.regularization)












