	#!/bin/python

from numpy import *

class NN():
	def __init__( self, size ):
		random.seed( 1 )
		self.synaptic_weights = 2 * random.random(( size, 1 )) - 1
	
	def __sigmoid( self, x ):
		return 1 / ( 1 + exp( -x ))
	
	def __sigmoid_derivative( self, x ):
		return x * ( 1 - x )
	
	def predict( self, inputs ):
		return self.__sigmoid( dot( inputs, self.synaptic_weights ))
	
	def train( self, train_in, train_out, n ):
		for iteration in range( n ):
			output 	= self.predict( train_in )
			error 	= train_out - output
			adjust	= dot( train_in.T, error * self.__sigmoid_derivative( output ))
			self.synaptic_weights += adjust 


def compare( data ):
	sdr = []
	for i in  range(len(data)):
		if i == 0:
			continue 
		sdr.append(1 if(data[i]>data[i-1]) else 0)
	return sdr

buy  = lambda money, price: money / price # returns stock
sell = lambda stock, price: stock * price # returns money

def prepare_data( m,t, f, nn ):
	a  = genfromtxt(f)[:m+1]
	b  = compare(a)
	t_in  = []
	t_out = []

	for i in range(len(b)):
		if i > t:
			t_in.append(b[i-t-1:i-1])
	

	for i in range(len(t_in)):
		if i == 0:
			continue

		t_out.append(t_in[i][0])


	x=array(t_in[:m][:-1])
	y=array([t_out[:m]]).T
	return [x,y]

if __name__ == '__main__':
	m  = 1000
	error = .0005
	e  = 2
	t  = 5
	got = 100
	stock = 0
	mode = 1
	f  = 'PETR32008.txt'

	#train
	nn = NN(t)
	x,y = prepare_data( m,t,f, nn )
	nn.train( x, y, e )
	print(nn.synaptic_weights)

	#use
	f = 'PETR32017.txt'
	a = genfromtxt(f)
	t_in = []
	t_out = []
	for i in range(len(a)):
		if i > t:
			x = a[i-t-1:i-1]
			if i == len(a)-1:
				break
			n= a[i-t]
			y=compare(a[i-t-1:i])
			p=nn.predict(array(y)) 
			if p > error:
				if  mode == 1:
					stock = buy(got,n)
					mode   = 0
				else:
					got = sell(stock,n)
					mode = 1

print(got)




