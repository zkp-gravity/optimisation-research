import ctypes as c
import numpy as np
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from scipy.stats import norm




def random_shift_x(x, max_shift=5):
    '''
    Shift image randomly
    
    Parameters
    -x: Observation
    -max_shift: Maximum possible shift to one direction
    '''
    
    n = int(np.sqrt(len(x)))
    img = x.reshape(n, n)
    
    h_shift = 0
    v_shift = 0
    
    while h_shift == 0 and v_shift == 0:
        h_shift = np.random.randint(-max_shift, max_shift+1)
        v_shift = np.random.randint(-max_shift, max_shift+1)
    
    new_img = np.zeros_like(img)
    
    if v_shift == 0 and h_shift > 0:
        new_img[:, h_shift:] = img[:, :-h_shift]
    
    if v_shift == 0 and h_shift < 0:
        new_img[:, :h_shift] = img[:, -h_shift:]
        
    if v_shift < 0 and h_shift == 0:
        new_img[:v_shift, :] = img[-v_shift:, :]
    
    if v_shift > 0 and h_shift == 0:
        new_img[v_shift:, :] = img[:-v_shift, :]
    
    if v_shift > 0 and h_shift > 0:
        new_img[v_shift:, h_shift:] = img[:-v_shift, :-h_shift]
    
    if v_shift > 0 and h_shift < 0:
        new_img[v_shift:, :h_shift] = img[:-v_shift, -h_shift:]
        
    if v_shift < 0 and h_shift > 0:
        new_img[:v_shift, h_shift:] = img[-v_shift:, :-h_shift]
    
    if v_shift < 0 and h_shift < 0:
        new_img[:v_shift, :h_shift] = img[-v_shift:, -h_shift:]
    
    return np.ravel(new_img)
   

def random_corrupt_x(x, p_crit):
    '''
    Inverse bits in the image.
    Every black pixel which has at least one white pixel neighbour can become white with probability p_crit
    Every white pixel which has at least one black pixel neighbour can become black with probability p_crit
    
    Parameters
    -x: Observation
    -p_crit: probability to change the pixel
    '''
    
    x_cor = x.copy()
    ps = np.random.uniform(0,1,len(x))
    ps = ps < p_crit
    for i, p in enumerate(ps[:-1]):
        if p:
            if x_cor[i] == 1 and (x_cor[i-1] == 0 or x_cor[i+1] == 0):
                x_cor[i] = 0
            elif x_cor[i] == 0 and (x_cor[i-1] == 1 or x_cor[i+1] == 1):
                x_cor[i] = 1
    return x_cor


def get_mnist_dataset():
    '''
    Loading MNIST dataset from torchvision library
    '''
    
    train_dataset = dsets.MNIST(
        root='./data',
        train=True,
        transform=transforms.ToTensor(),
        download=True)
    new_train_dataset = []
    for d in train_dataset:
        new_train_dataset.append((d[0].numpy().flatten(), d[1]))
    train_dataset = new_train_dataset
    test_dataset = dsets.MNIST(
        root='./data',
        train=False,
        transform=transforms.ToTensor())
    new_test_dataset = []
    for d in test_dataset:
        new_test_dataset.append((d[0].numpy().flatten(), d[1]))
    test_dataset = new_test_dataset

    return train_dataset, test_dataset


# Convert input dataset to binary representation
# Use a thermometer encoding with a configurable number of bits per input
# A thermometer encoding is a binary encoding in which subsequent bits are set as the value increases
#  e.g. 0000 => 0001 => 0011 => 0111 => 1111
def binarize_datasets(train_dataset, test_dataset, bits_per_input, separate_validation_dset=None, train_val_split_ratio=0.9):
    # Given a Gaussian with mean=0 and std=1, choose values which divide the distribution into regions of equal probability
    # This will be used to determine thresholds for the thermometer encoding
    std_skews = [norm.ppf((i+1)/(bits_per_input+1))
                 for i in range(bits_per_input)]

    print("Binarizing train/validation dataset")
    train_inputs = []
    train_labels = []
    for d in train_dataset:
        # Expects inputs to be already flattened numpy arrays
        train_inputs.append(d[0])
        train_labels.append(d[1])
    train_inputs = np.array(train_inputs)
    train_labels = np.array(train_labels)
    use_gaussian_encoding = True
    if use_gaussian_encoding:
        mean_inputs = train_inputs.mean(axis=0)
        std_inputs = train_inputs.std(axis=0)
        train_binarizations = []
        for i in std_skews:
            train_binarizations.append(
                (train_inputs >= mean_inputs+(i*std_inputs)).astype(c.c_ubyte))
    else:
        min_inputs = train_inputs.min(axis=0)
        max_inputs = train_inputs.max(axis=0)
        train_binarizations = []
        for i in range(bits_per_input):
            train_binarizations.append(
                (train_inputs > min_inputs+(((i+1)/(bits_per_input+1))*(max_inputs-min_inputs))).astype(c.c_ubyte))

    # Creates thermometer encoding
    train_inputs = np.concatenate(train_binarizations, axis=1)

    # Ideally, we would perform bleaching using a separate dataset from the training set
    #  (we call this the "validation set", though this is arguably a misnomer),
    #  since much of the point of bleaching is to improve generalization to new data.
    # However, some of the datasets we use are simply too small for this to be effective;
    #  a very small bleaching/validation set essentially fits to random noise,
    #  and making the set larger decreases the size of the training set too much.
    # In these cases, we use the same dataset for training and validation
    if separate_validation_dset is None:
        separate_validation_dset = (len(train_inputs) > 10000)
    if separate_validation_dset:
        split = int(train_val_split_ratio*len(train_inputs))
        val_inputs = train_inputs[split:]
        val_labels = train_labels[split:]
        train_inputs = train_inputs[:split]
        train_labels = train_labels[:split]
    else:
        val_inputs = train_inputs
        val_labels = train_labels

    print("Binarizing test dataset")
    test_inputs = []
    test_labels = []
    for d in test_dataset:
        # Expects inputs to be already flattened numpy arrays
        test_inputs.append(d[0])
        test_labels.append(d[1])
    test_inputs = np.array(test_inputs)
    test_labels = np.array(test_labels)
    test_binarizations = []
    if use_gaussian_encoding:
        for i in std_skews:
            test_binarizations.append(
                (test_inputs >= mean_inputs+(i*std_inputs)).astype(c.c_ubyte))
    else:
        for i in range(bits_per_input):
            test_binarizations.append(
                (test_inputs > min_inputs+(((i+1)/(bits_per_input+1))*(max_inputs-min_inputs))).astype(c.c_ubyte))
    test_inputs = np.concatenate(test_binarizations, axis=1)

    return train_inputs, train_labels, val_inputs, val_labels, test_inputs, test_labels



def pooling(x, window_size, type):
    '''
    Standard pooling operation
    
    Parameters:
    -x: Observation
    -window_size: size of a poolinf window
    -type: type of pooling: max, min, avg
    '''
    
    if type == 'max':
        f = np.max
    if type == 'min':
        f = np.min
    if type == 'avg':
        f = np.mean

    step = window_size
    step = window_size

    h_pad = x.shape[1] % step
    v_pad = x.shape[0] % step

    x = np.pad(x, ((0,h_pad), (0, v_pad)))

    v_steps = x.shape[0] // step
    h_steps = x.shape[1] // step


    x_pooled = np.empty((v_steps, h_steps))

    for i in range(v_steps):
        for j in range(h_steps):
            x_pooled[i,j] = f(x[i*step:(i+1)*step, j*step:(j+1)*step])

    return x_pooled


# def train_model(model, train_inputs, train_labels):
#     '''
#     Training the model
    
#     Parameters:
#     -model: Wisard model
#     -train_inputs: training observations
#     -train_labels: training labels
#     '''
    
#     for d in range(len(train_inputs)):
#         model.train(train_inputs[d], train_labels[d])
#     return model


# def run_inference(model, inputs, labels, bleach=1):
#     '''
#     Runing inference for the model
    
#     Parameters:
#     -model: Wisard model
#     -inputs: test observations
#     -labels: test labels
#     -bleach: bleach value to set for the model
#     '''
    
#     num_samples = len(inputs)
#     correct = 0
#     ties = 0
#     model.set_bleaching(bleach)
#     for d in range(num_samples):
#         prediction = model.predict(inputs[d])
#         label = labels[d]
#         if len(prediction) > 1:
#             ties += 1
#         if prediction[0] == label:
#             correct += 1
#     correct_percent = round((100 * correct) / num_samples, 4)
#     tie_percent = round((100 * ties) / num_samples, 4)
#     print(f"With bleaching={bleach}, accuracy={correct}/{num_samples} ({correct_percent}%); ties={ties}/{num_samples} ({tie_percent}%)")
#     return correct_percent
    

# def find_best_bleach(model, val_inputs, val_labels):
#     '''
#     Finding the best bleaching values
    
#     Parameters:
#     -model: Wisard model
#     -inputs: test observations
#     -labels: test labels
#     -bleach: bleach value to set for the model
#     '''
    
#     max_val = 0
#     for d in model.discriminators:
#         for f in d.filters:
#             max_val = max(max_val, f.data.max())
#     print(f"Maximum possible bleach value is {max_val}")
#     # Use a binary search-based strategy to find the value of b that maximizes accuracy on the validation set
#     best_bleach = max_val // 2
#     step = max(max_val // 4, 1)
#     bleach_accuracies = {}
#     while True:
#         values = [best_bleach-step, best_bleach, best_bleach+step]
#         accuracies = []
#         for b in values:
#             if b in bleach_accuracies:
#                 accuracies.append(bleach_accuracies[b])
#             elif b < 1:
#                 accuracies.append(0)
#             else:
#                 accuracy = run_inference(model, val_inputs, val_labels, b)
#                 bleach_accuracies[b] = accuracy
#                 accuracies.append(accuracy)
#         new_best_bleach = values[accuracies.index(max(accuracies))]
#         if (new_best_bleach == best_bleach) and (step == 1):
#             break
#         best_bleach = new_best_bleach
#         if step > 1:
#             step //= 2
#     print(f"Best bleach: {best_bleach}")
    
#     return best_bleach

# def find_responses(model, inputs):
#     responses = []
#     for x in inputs:
#         r = model.predict_proba(x)
#         responses.append(r)
#     return np.array(responses)
    