# Keras BatchSizeAnnealing
This repository contains a wrapper class for adjusting the batch_size after aech epoch as shown on the paper [Don't Decay the Learning Rate, Increase the Batch Size](https://arxiv.org/abs/1711.00489) by by [Samuel L. Smith](https://arxiv.org/search?searchtype=author&query=Smith%2C+S+L), [Pieter-Jan Kindermans](https://arxiv.org/search?searchtype=author&query=Kindermans%2C+P), [Chris Ying](https://arxiv.org/search?searchtype=author&query=Ying%2C+C), [Quoc V. Le](https://arxiv.org/search?searchtype=author&query=Quoc%2C+V.%2C+Le).

## Train Example
A minimum example of working code would be:
```python
from BatchSizeAnnealing import BatchSizeAnnealing
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

def callback(epoch):
    return 32 * epoch / 10 + 32

model = createModel()
trainer = BatchSizeAnnealing(model, callback)
history = trainer.train( x_train, y_train, validation_data=(x_test, y_test), epochs=EPOCH, verbose=1)
``` 
The constructor takes as arguments the model and the callback to the annealing per epoch.
## Wrapper Example
This class can also be used as a wrapper for *keras.model* as it will redirect all methods to the model passed as parameter. ie:
```python
from BatchSizeAnnealing import BatchSizeAnnealing
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

def callback(epoch):
    return 32 * epoch / 10 + 32

model = createModel()
trainer = BatchSizeAnnealing(model, callback)
trainer.summary()
``` 
This last funtion is equivalent to to *model.summary()*
    
## Constuctor parameters

```python
class BatchSizeAnnealing(object):
    def __init__(self, model, callback, show_hist=False, keep_verbosity=False):
        ...
``` 
- **model**: The model to be used as training (Can be from the functional API).
- **callback**: Callback to get batch sizes in a specific epoch:
    ```python
    def callback(epoch):
        return ...
    ```
- **show_hist**: Show progress as bar while training,
- **keep_verbosity**: keep the "verbosity" parameter passed to *train*.
## TODO
- &#9745; Add verbose for training every batch.
- &#9744; Fix verbosity per epoch from *keras.model*.
- &#9744; Add parameter and return types.