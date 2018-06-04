# Keras BatchSizeAnnealing
This repository contains a wrapper class for adjusting the batch_size after aech epoch as shown on the paper [Don't Decay the Learning Rate, Increase the Batch Size](https://arxiv.org/abs/1711.00489) by by [Samuel L. Smith](https://arxiv.org/search?searchtype=author&query=Smith%2C+S+L), [Pieter-Jan Kindermans](https://arxiv.org/search?searchtype=author&query=Kindermans%2C+P), [Chris Ying](https://arxiv.org/search?searchtype=author&query=Ying%2C+C), [Quoc V. Le](https://arxiv.org/search?searchtype=author&query=Quoc%2C+V.%2C+Le).

A minimum example of working code would be:

    import KerasBatchSizeAnnealing
    def callback(epoch):
	    return 32 * epoch / 10 + 32
	model.fit(x,y, epochs=N, callbacks=[BatchSizeAnnealing(callback)])


