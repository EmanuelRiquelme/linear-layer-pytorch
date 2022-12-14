import numpy as np
np.set_printoptions(suppress=True)

input_data = np.random.rand(28,28)
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def relu(x):
    return x * (x > 0)

def linear_network(input_data = input_data,num_classes = 3):
    hidden_layer = np.random.rand(28**2,num_classes)
    input_data = input_data.reshape(-1)
    product = np.matmul(input_data,hidden_layer)
    product = relu(product)
    percentage = product/np.sum(product)
    return percentage,softmax(product)


if __name__ == '__main__':
    print(linear_network())

