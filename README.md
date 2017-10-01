# Semantic Segmentation

## info
trained with tensorflow 1.3
python version: 3.5
platform: ubuntu 16.04

## usage
to start a train
```
python main.py <batch_size> <epochs> <learning_rate> <keep_prob> <regularization_factor>
```
(check main.py to see default parameters)


## some params worth trying 
batch_size=2, epochs=30, keep_prob=0.5, learning_rate=0.0001, regularization=0.01


## architecture
FCN 8, please check this [paper](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)
kernel size was 4x4 for the first-stage upsampling, 8x8 for the second and 32x32 for the final(8 times).

## details
* augmentations : rotation, flip, zoom, shift and channel shift. 
  [tf.contrib.keras.preprocessing.image](https://www.tensorflow.org/versions/master/api_docs/python/tf/keras/preprocessing/image)
  check `get_batch_function` in `helper.py`
  _NOTE: Instead of using keras's implementation, channel_shift is implemented by shearing the channel axis, which actually just shuffles along the channel axis._

* super high channel-shift probability in the most recent run:
  ```python
  get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'),
                                             image_shape,
                                             aug_size=0.6,
                                             channel_shift=0.8)
  ``` 
  since I found the model pretty confused on shaded road. It seems to work compared with the ones without channel shift, but need more epochs to train better (the not-so-shaded examples is well segmented with 20 epochs, but the shaded ones need extra 40 epochs to get a acceptable result).

* in `main.py`, set `freeze` argument to True to only train the skip layers:
```python
def optimize(nn_last_layer, correct_label, learning_rate, num_classes, reg=1e-2, freeze=False):
```

## the most recent trial

```
batch_size=2, epochs=60, keep_prob=0.5, learning_rate=0.0001, regularization=0.01
```

