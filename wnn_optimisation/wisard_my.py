#!/usr/bin/false

import numpy as np
from numba import jit

from bloom_filter import BloomFilter

# Generates a matrix of random values for use as m-arrays for H3 hash functions
def generate_h3_values(num_inputs, num_entries, num_hashes):
    assert(np.log2(num_entries).is_integer())
    shape = (num_hashes, num_inputs)
    values = np.random.randint(0, num_entries, shape)
    return values

    
class WiSARD1:
    def __init__(self, num_classes, unit_inputs, unit_entries, unit_hashes, random_values, input_order=None):
        
        self.input_order = input_order
        self.random_values = random_values
        self.filters = [BloomFilter(unit_inputs, unit_entries, unit_hashes, random_values) for i in range(num_classes)]

    def train(self, xv, label):
        self.filters[label].add_member(xv[self.input_order])

    def predict(self, xv):
        xv = xv[self.input_order]
        responses = np.array([f.check_membership(xv) for f in self.filters], dtype=int)
        return responses

    def set_bleaching(self, bleach):
        for f in self.filters:
            f.set_bleaching(bleach)


# Implementes a single discriminator in the WiSARD model
# A discriminator is a collection of boolean LUTs with associated input sets
# During inference, the outputs of all LUTs are summed to produce a response
# Slightly modified version which allows to construct model with arbitrary number of features
class Discriminator:
    def __init__(self, n_filters, unit_inputs, unit_entries, unit_hashes, random_values):
        # Constructor
        # Inputs:
        #  n_filters:     The total number of filter in the discriminator
        #  unit_inputs:   The number of boolean inputs to each LUT/filter in the discriminator
        #  unit_entries:  The size of the underlying storage arrays for the filters. Must be a power of two.
        #  unit_hashes:   The number of hash functions for each filter.
        #  random_values: If provided, is used to set the random hash seeds for all filters.
        self.num_filters = n_filters
        self.unit_inputs = unit_inputs
        self.unit_entries = unit_entries
        self.unit_hashes = unit_hashes
        self.random_values = random_values

        self.filters = [BloomFilter(unit_inputs, unit_entries, unit_hashes, random_values) for i in range(self.num_filters)]

    def train(self, xv):
        filter_inputs = xv.reshape(self.num_filters, -1)
        for idx, inp in enumerate(filter_inputs):
            self.filters[idx].add_member(inp)
            
    def predict(self, xv):
        filter_inputs = xv.reshape(self.num_filters, -1)
        response = 0
        for idx, inp in enumerate(filter_inputs):
            response += int(self.filters[idx].check_membership(inp))
        return response

    def set_bleaching(self, bleach):
        if type(bleach) == int:
            for f in self.filters:
                f.set_bleaching(bleach)
        else:
            for f, b in zip(self.filters, bleach):
                f.set_bleaching(b)

    def binarize(self):
        for f in self.filters:
            f.binarize()

# Top-level class for the WiSARD weightless neural network model with our modifications
# This model allows to explicitly define features and individual bleaching value for each feature.
class WiSARD2:
    def __init__(self, num_classes, unit_inputs, unit_entries, unit_hashes, random_values, features):
        # Constructor
        # Inputs:
        #  num_classes:      The number of distinct possible outputs of the model; the number of classes in the dataset
        #  unit_inputs:      The number of boolean inputs to each LUT/filter in the model
        #  unit_entries:     The size of the underlying storage arrays for the filters. Must be a power of two.
        #  unit_hashes:      The number of hash functions for each filter.
        #  random_values:    If provided, is used to set the random hash seeds for all filters.
        #  features:         Indices of bits which are considered as single feature.
        self.features = features
        self.input_order = np.ravel(self.features)
        self.discriminators = [Discriminator(len(features), unit_inputs, unit_entries, unit_hashes, random_values) for i in range(num_classes)]

    def train(self, xv, label):
        xv = xv[self.input_order] # Reorder input
        self.discriminators[label].train(xv)

    def predict(self, xv):
        xv = xv[self.input_order] # Reorder input
        responses = np.array([d.predict(xv) for d in self.discriminators], dtype=int)
        max_response = responses.max()
        return np.where(responses == max_response)[0]

    def set_bleaching(self, bleach):
        for d in self.discriminators:
            d.set_bleaching(bleach)

    def binarize(self):
        for d in self.discriminators:
            d.binarize()