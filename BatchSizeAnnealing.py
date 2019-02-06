#Original paper by Samuel L. Smith, Pieter-Jan Kindermans, Chris Ying, Quoc V. Le "Don't Decay the Learning Rate, Increase the Batch Size"

import keras
import numpy
import sys

class BatchSizeAnnealing(object):
	def __init__(self, model, callback, show_hist=False, keep_verbosity=False):
		self.model = model
		"""
		for (k,v) in kwargs.items():
			if k == "batch_size":
				continue
			elif k == "epochs":
				self.args[k] = 1
			elif k == "verbose":
				self.verbose = True
				self.args[k] = 0
			else:
				self.args[k] = v

		print(self.args)
		self.x = x
		self.y = y
		"""
		self.f = callback
		self.verbose=show_hist
		self.keep_verb = keep_verbosity

	def __getattr__(self, name):
		if hasattr(self.model, name):
			def wrapper(*args, **kwargs):
				return getattr(self.model, name)(*args, **kwargs)
			return wrapper       
		else:
			return object.__getattribute__(self, name)  


	def train(self, x, y, **kwargs):
		output = []
		epochs = kwargs["epochs"]
		del kwargs["epochs"]

		if "verbose" in kwargs and not self.keep_verb:
			kwargs["verbose"] = 0

		for i in range(epochs):
			output.append( self.model.fit(x, y, batch_size=self.f(i), initial_epoch=i-1, epochs=i, **kwargs) )
			output[-1].epoch = [i]

			if self.verbose:
				sys.stdout.write("Epoch %d/%d:\n" % (i+1, epochs))
				sys.stdout.flush()
				sys.stdout.write( "[%-60s] %d%%\n" % (("="*int(60*(i+1)/epochs) + ">"), (100*(i+1)/epochs) ))
				sys.stdout.flush()
				sys.stdout.write(",".join([key + ": " + str(value[0]) for key, value in output[-1].history.items()]) )
		
		return self.__mergeHistory(output)
        
	def __mergeHistory(self, history):
		output = keras.callbacks.History()
		# print(vars(output), history)
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
