import numpy as np
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

weight_layer_0_1 = []
for i in range(layer_1_values):
    weight_layer_0_1.append(np.random.rand(3))
weight_layer_0_1 = np.array(weight_layer_0_1)
weight_layer_1_2 = np.random.rand(layer_1_values)

# print(weight_layer_0_1)
# print(weight_layer_1_2)
alpha = 0.01

for i in range(60):
    error_sum = 0
    for i in range(len(streetlights)):
        goal = walk_vs_stop[i:i+1]
        layer_0 = streetlights[i:i+1]
        print(layer_0,goal,weight_layer_0_1)
        layer_1 = np.array(relu(layer_0.dot(weight_layer_0_1.T)))
        print(layer_1)
        layer_2 = np.array(layer_1.dot(weight_layer_1_2.T))
        print(layer_2)
        error = (layer_2-goal)**2
        error_sum += error
        delta_2 = np.array(layer_2 - goal) 
        delta_1 = np.array(delta_2)*weight_layer_1_2
        print(delta_2,layer_1)
        weight_layer_1_2 -= alpha*(delta_2[0]*layer_1[0])
        print(delta_1[0],layer_0[0])
        weight_layer_0_1 -= alpha*(np.array(delta_1[0])*np.array(layer_0[0])*relu2deriv(layer_1))
    if i % 10 == 9:
        print(error,error_sum)



