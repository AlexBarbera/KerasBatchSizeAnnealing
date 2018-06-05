# Keras BatchSizeAnnealing
This repository contains a wrapper class for adjusting the batch_size after aech epoch as shown on the paper [Don't Decay the Learning Rate, Increase the Batch Size](https://arxiv.org/abs/1711.00489) by by [Samuel L. Smith](https://arxiv.org/search?searchtype=author&query=Smith%2C+S+L), [Pieter-Jan Kindermans](https://arxiv.org/search?searchtype=author&query=Kindermans%2C+P), [Chris Ying](https://arxiv.org/search?searchtype=author&query=Ying%2C+C), [Quoc V. Le](https://arxiv.org/search?searchtype=author&query=Quoc%2C+V.%2C+Le).

A minimum example of working code would be:

    from BatchSizeAnnealing import BatchSizeAnnealing
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    def callback(epoch):
	    return 32 * epoch / 10 + 32
    
    model = createModel()
    trainer = BatchSizeAnnealing(model, x_train, y_train, validation_data=(x_test, y_test), epochs=EPOCH, verbose=1)
    history = trainer.fit()
    
The constructor takes as arguments the model, training data and any arguments you may pass to the fit() funtion of the model.
    
    
## TODO
- Add verbose for training every batch

