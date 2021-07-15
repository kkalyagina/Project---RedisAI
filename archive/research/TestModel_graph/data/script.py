from redisai import Client

#connecting to Redis server
con = Client(host='viper', port=6379)

#reading model and setting it to redisai
model_graph = open('/data/graph.pb', 'rb').read()
con.modelset("graph1", 'TF', 'CPU', model_graph, inputs = ['a', 'b'], outputs = ['c'])

#in case of stdin input
#tensor_a = [int(i) for i in input('Enter tensor a (default \'2 3\'):  ').split(' ')]
#tensor_b = [int(i) for i in input('Enter tensor b (default \'3 5\'):  ').split(' ')]

#reading input tensors
tensor_a_file = open('/data/tensor_a.txt').read()
tensor_b_file = open('/data/tensor_b.txt').read()
tensor_a = [int(i) for i in tensor_a_file.split(' ')]
tensor_b = [int(i) for i in tensor_b_file.split(' ')]

#setting tensors, running model, printing output
con.tensorset('a', tensor_a, dtype='float')
con.tensorset('b', tensor_b, dtype='float')
con.modelrun('graph1', ['a', 'b'], ['c'])
print('model run output tensor:   ', con.tensorget('c'))