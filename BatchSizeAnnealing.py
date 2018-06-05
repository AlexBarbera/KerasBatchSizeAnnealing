#Original paper by Samuel L. Smith, Pieter-Jan Kindermans, Chris Ying, Quoc V. Le "Don't Decay the Learning Rate, Increase the Batch Size"

import keras
import numpy
import sys

class BatchSizeAnnealing():
    def __init__(self, model, callback, x, y, **kwargs):
        self.model = model
        self.args = {}
        self.verbose = False
        self.epochs = kwargs["epochs"]
        
        for (k,v) in kwargs.iteritems():
            if k == "batch_size":
                continue
            elif k == "epochs":
                self.args[k] = 1
            elif k == "verbose":
                self.verbose = True
                self.args[k] = 0
            else:
                self.args[k] = v

	print self.args
        self.f = callback
        self.x = x
        self.y = y
        
    def train(self):
        output = []
        for i in xrange(self.epochs):
            if self.verbose:
		        sys.stdout.write("Epoch %d/%d:\n" % (i+1, self.epochs))
		        sys.stdout.flush()
		        sys.stdout.write( "[%-60s] %d%%\n" % ("="*(60*(i+1)/self.epochs), (100*(i+1)/self.epochs) ))
		        sys.stdout.flush()
		    
            output.append( self.model.fit(self.x, self.y, batch_size=self.f(i), **self.args) )
            output[-1].epoch = [i]
            
            if self.verbose:
                print output[-1].history
                
        
        return self.__mergeHistory(output)
        
    def __mergeHistory(self, history):
        output = keras.callbacks.History()
	print vars(output), history
        for h in history:
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
