import numpy as np
def make_predict(model, test, target):
    test_x = np.array(test.input.tolist())
    test[target] = model.predict(test_x, batch_size=128)
    return test