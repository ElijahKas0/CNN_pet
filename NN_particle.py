import numpy as np

np.random.seed(1)

streetlights = np.array( [[ 1, 0, 1 ],
    [ 0, 1, 1],
    [ 0, 0, 1],
    [1,1,1]])
walk_vs_stop = np.array([[ 1, 1, 0, 0]]).T

def relu(x):
    return (x>0)*x

def relu2deriv(x):
    return x>0

layer_1_values = 4
hidden_size = 4
weights_0_1 = 2*np.random.random((3,hidden_size)) - 1
weights_1_2 = 2*np.random.random((hidden_size,1)) - 1
print(weights_0_1)
print(weights_1_2)

alpha = 0.2

for i in range(60):
    error_sum = 0
    for i in range(len(streetlights)):
        goal = walk_vs_stop[i:i+1]
        # print(goal,type(goal))
        layer_0 = streetlights[i:i+1]
        # print(layer_0,type(layer_0))
        layer_1 = relu(layer_0.dot(weights_0_1.T))
        # print(layer_1,type(layer_1))
        layer_2 = layer_1.dot(weights_1_2.T)
        # print(layer_2,type(layer_1))
        error = (layer_2-goal)**2
        error_sum += np.sum(error)
        delta_2 = np.array(layer_2 - goal) 
        delta_1 = np.array(delta_2)*weights_1_2
        # print(weight_layer_1_2)
        weights_1_2 -= alpha*delta_2.dot(layer_1)
        # print(alpha*delta_1.T.dot(layer_0))
        weights_0_1 -= alpha*delta_1.T.dot(layer_0)*relu2deriv(layer_1.T)
    
    print(error_sum)



