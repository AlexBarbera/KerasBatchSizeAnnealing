#Original paper by Samuel L. Smith, Pieter-Jan Kindermans, Chris Ying, Quoc V. Le "Don't Decay the Learning Rate, Increase the Batch Size"

import keras
import numpy

class BatchSizeAnnealing():
    def __init__(self, model, callback, x, y, **kwargs):
        self.model = model
        self.args = {}
	for (k,v) in kwargs.iteritems():
		if k == "batch_size":
			continue
		elif k == "epochs":
			self.args[k] = 1
		else:
			self.args[k] = v

        self.epochs = kwargs["epochs"]
	print self.args
        self.f = callback
        self.x = x
        self.y = y
        
    def train(self):
        output = []
        for i in xrange(self.epochs):
		    print "Epoch %d/%d" % (i+1,self.epochs)		
		    output.append( self.model.fit(self.x, self.y, batch_size=self.f(i), **self.args) )
		    output[-1].epoch = [i]
        
        return self.__mergeHistory(output)
        
    def __mergeHistory(self, history):
        output = keras.callbacks.History()
	print vars(output), history
        for h in history:
	    print vars(h)
            #soutput.epoch = output.epoch + h.epoch
            for key in vars(h):
                if type(getattr(h,key)) == list:
                    try:
                        aux = getattr(output, key) + getattr(h, key)
                    except:
                        aux = getattr(h, key)
                    setattr(output, key, aux)
                elif type(getattr(h,key)) == dict:
                    try:
                        aux = getattr(output, key)
                    except:
                        aux = {}
                        
                    temp = getattr(h, key)
                    for k1 in temp:
                        if type(temp[k1]) == list:
                            if k1 in aux:
                                aux[k1] = aux[k1] + temp[k1]
                            else:
                                aux[k1] = temp[k1]
                        else:
                            aux[k1] = temp[k1]
                            
                    setattr(output, key, aux)
                elif type(getattr(h, key)) == numpy.ndarray:
                    try:
                        aux = getattr(output, key)
                    except:
                        aux = numpy.asarray([])
                    aux = numpy.vstack(aux, getattr(h, key) )
                    setattr(output, key, aux)
                else:
                    setattr(output, key, getattr(h, key) )
                
        return output
