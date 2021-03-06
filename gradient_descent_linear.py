import numpy as np
import matplotlib.pyplot as plt

# y = ax + b 

def calc_error(a,b,data):
	error = 0.0
	N = len(data)
	for i in range(N):
		x = data[i,0]
		y = data[i,1]
		error += (1 / N) * ((a * x + b) - y) ** 2
	return error

def step_gradient_descent(a,b,data,learning_rate):
	step_a = 0
	step_b = 0
	N = len(data)
	for i in range(N):
		x = data[i,0]
		y = data[i,1]
		step_a += (2/N) * (a * x + b - y) * x
		step_b += (2/N) * (a * x + b - y)
	new_a = a - learning_rate * step_a
	new_b = b - learning_rate * step_b
	return (new_a,new_b) 


def gradient_descent_runner(init_a,init_b,data,learning_rate,num_iterations):
	a = init_a
	b = init_b
	for i in range(num_iterations):
		(a,b) = step_gradient_descent(a,b,data,learning_rate)
		print('step {}: a:{} b:{} error:{}'.format(i,a,b,calc_error(a,b,data)))
	return (a,b)

def main():
	data = np.genfromtxt("data.csv", delimiter=",")
	learning_rate = 0.00001
	init_a = 0
	init_b = 0
	num_iterations = 1000
	print('init a:{} b:{} error:{}'.format(init_a,init_b,calc_error(init_a,init_b,data)))
	(new_a,new_b) = gradient_descent_runner(init_a,init_b,data,learning_rate,num_iterations)
	plt.scatter(data[:,0],data[:,1])
	pt_x = np.arange(20,80,0.1)
	pt_y = new_a * pt_x + new_b
	plt.plot(pt_x,pt_y,color='red')
	plt.show()
	print('answer a:{} b:{} error:{}'.format(new_a,new_b,calc_error(new_a,new_b,data)))

if __name__ == '__main__':
	main()