import numpy as np

def test_load_data():
    x_train = [np.random.rand(10,12,10,1)] * 10 + [np.random.rand(26,20,12,1)] * 10
    
    y_train = [np.array([1, 0])] * 10 + [np.array([0, 1])] * 10
    x_val = [np.random.rand(10,12,10,1)] * 5 + [np.random.rand(10,19,25,1)] * 5
    y_val = [np.array([1, 0])] * 5 + [np.array([0, 1])] * 5
    return x_train, y_train, x_val, y_val

def test_load_test():
    x_test = [np.random.rand(10,23,15,1)] * 5 + [np.random.rand(21,19,25,1)] * 5
    y_test = [np.array([1, 0])] * 5 + [np.array([0, 1])] * 5
    return x_test, y_test
    
        
if __name__ == '__main__':
    x_train, y_train, x_val, y_val = test_load_data()
    print(x_train[0:2], y_train[0].shape, x_val[0].shape, y_val[0].shape)
    print(y_val)
