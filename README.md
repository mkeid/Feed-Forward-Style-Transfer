# *Feed-Forward-Style-Transfer* implemented in TensorFlow

## Results

<table style="width:100%">
  <tr>
    <td><img src="" width="100%"></td>
    <td><img src="" width=100%"></td> 
    <td><img src="" width="100%"></td> 
    <td><img src="" width="100%"></td> 
  </tr>
  <tr>
    <td><img src="" width="100%"></td>
    <td><img src="" width="100%"></td> 
    <td><img src="" width="100%"></td> 
    <td><img src="" width="100%"></td> 
  </tr>
  <tr>
    <td><img src="" width="100%"></td>
    <td><img src="" width="100%"></td> 
    <td><img src="" width="100%"></td> 
    <td><img src="" width="100%"></td> 
  </tr>
  <tr>
    <td><img src="" width="100%"></td>
    <td><img src="" width=100%"></td> 
    <td><img src="" width="100%"></td> 
    <td><img src="" width="100%"></td> 
  </tr>
</table>

## Prerequisites

* [Python 3.5](https://www.python.org/downloads/release/python-350/)
* [TensorFlow](https://www.tensorflow.org/) (>= r0.12)
* [scikit-image](http://scikit-image.org/docs/dev/api/skimage.html)
* [NumPy](http://www.numpy.org/)

## Usage

```
python train.py /path/to/style/image

```

```
python test.py --input /path/to/input/image --style "style name"
```

```
python test.py --styles
```


## Files

[style.py](../bin/style.py)

[train.py](../bin/train.py)

[gen_net.py](../bin/gen_net.py)

[helpers.py](../bin/helpers.py)

[custom_vgg16.py](../bin/custom_vgg16.py)