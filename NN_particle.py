# import numpy as np

# # weights_0_1 = np.array([[-0.3,-0.22,-0.29,-0.07]])
# # print(weights_0_1)
# # input = np.array([[1,0,1]])
# # print(weights_0_1.T.dot(input))
# # print(weights_0_1)

# np.random.seed(42)


# streetlights = np.array( [[ 1, 0, 1],
#     [ 0, 1, 1 ],
#     [ 0, 0, 1],
#     [1,1,1]])
# walk_vs_stop = np.array([[ 1, 1, 0, 0]]).T

# def relu(x):
#     return (x>0)*x

# def relu2deriv(x):
#     return x>0

# weights_0_1 = np.array([[-0.3, 0.5, -0.8],[-0.4,0.2,0.6],[0.1,-0.5,0.7],[0.6,-0.3,0.9]])
# weights_1_2 = np.array([[0.5],[-0.3],[0.4],[0.1]])
# print(weights_0_1,weights_1_2)
# weights_0_1 = 2*np.random.rand(4,3)-1
# weights_1_2 = 2*np.random.rand(4,1)-1
# print(weights_0_1,weights_1_2)
# alpha = 0.05

# for iteration in range(1000):
#     layer_2_error = 0
#     for iter in range(len(streetlights)):
#         layer_0 = streetlights[iter:iter+1]
        
#         goal = walk_vs_stop[iter:iter+1]
        
#         layer_1 = relu(layer_0.dot(weights_0_1.T))
        
#         layer_2 = layer_1.dot(weights_1_2)
        
#         error = (layer_2-goal)**2
        
#         layer_2_error +=np.sum(error)
#         delta_2 = layer_2-goal
        
        
        
#         delta_1 = delta_2.dot(weights_1_2.T)*relu2deriv(layer_1)
#         # print(delta_1)
#         weighted_delta_2 = alpha*(layer_1.T.dot(delta_2))
        
#         weighted_delta_1 = alpha*(delta_1.T.dot(layer_0))
#         # print(weighted_delta_1)
        
        
        
#         weights_1_2 -= weighted_delta_2

#         weights_0_1 -= weighted_delta_1
#         # print(error,layer_1,layer_2,layer_0)
        
#     if iteration % 10 == 9:
#         print(f"Error{layer_2_error}")


import numpy as np

# weights_0_1 = np.random.random((4,3))
# weights_1_2 = np.random.random((4,1))
# print(weights_0_1)
# print(weights_1_2)

np.random.seed(42)

streetlights = np.array( [[ 1, 0, 1 ],
[ 0, 1, 1],
[ 0, 0, 1],
[1,1,1]])
walk_vs_stop = np.array([[ 1, 1, 0, 0]]).T

def relu(x):
    return (x>0)*x

def relu2deriv(x):
    return x>0

weights_0_1 = 2*np.random.rand(4,3)-1
weights_1_2 = 2*np.random.rand(4,1)-1

alpha = 0.2

for iteration in range(60):
    layer_2_error = 0
    for iter in range(len(streetlights)):
        layer_0 = streetlights[iter:iter+1]
        
        goal = walk_vs_stop[iter:iter+1]
        
        layer_1 = relu(layer_0.dot(weights_0_1.T))
        
        layer_2 = layer_1.dot(weights_1_2)
        
        error = (layer_2-goal)**2
        # print(error)
        layer_2_error +=np.sum(error)
        delta_2 = layer_2-goal
        
        
        # print(delta_2)
        delta_1 = delta_2.dot(weights_1_2.T)*relu2deriv(layer_1)
        
        weighted_delta_2 = alpha*(layer_1.T.dot(delta_2))
        
        weighted_delta_1 = alpha*(delta_1.T.dot(layer_0))
        
        # print(weighted_delta_1)
        weights_1_2 -= weighted_delta_2
        weights_0_1 -= weighted_delta_1
    if iteration % 10 == 9:
        print(f"Error{layer_2_error}")


