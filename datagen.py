# Data generator(s)

import math

import numpy as np

import tensorflow as tf

import improc

class SimpleSequence(tf.keras.utils.Sequence):
    
    def __init__(self, p, IDs, shuffle = True, data = {}):
        """
        Create a generator object.

        Parameters
        ----------
        p : parameters (an OmegaConf object)
        IDs : an iterable object
            Sample IDs; useful when data needs to be loaded online.
        shuffle : bool, optional
            Whether to shuffle samples every epoch.
        data : dictionary
            Pre-loaded data.

        Returns
        -------
        None.

        """
        
        self.p = p
        self.IDs = IDs
        self.shuffle = shuffle
        self.data = data
        
        self.reset()
        
    def __len__(self):
        'Returns number of batches per epoch.'
        return math.ceil(len(self.IDs) / self.p.batch_size)
    
    def __getitem__(self, idx):
        'Is called by keras in model.fit(); idx is batch index.'
        
        batch_idx = self.indexes[idx * self.p.batch_size:(idx + 1) *
            self.p.batch_size]
        
        # the following is to ensure all batches have the same size:
        if batch_idx.shape[0] < self.p.batch_size:
            num_to_add = self.p.batch_size - batch_idx.shape[0]
            batch_idx = np.concatenate((batch_idx, self.indexes[:num_to_add]), axis = 0)

        batch_x = self.data['x'][batch_idx, ...]
        batch_y = self.data['y'][batch_idx, ...]
        batch_labeled = self.data['labeled'][batch_idx]

        # normalize image values between 0 and 1
        batch_x = batch_x / 255
        
        # putting labeled images in the beginning of the batch
        # (makes things easier later)
        sortd = np.argsort(np.logical_not(batch_labeled))
        
        batch_x = batch_x[sortd, ...]
        batch_y = batch_y[sortd, ...]
        batch_labeled = batch_labeled[sortd, ...]
        
        # generate transform parameters (e.g., noise or displacement map)
        # using a function given by p.transform.gen_func parameter
        transform_parameters = getattr(improc, 'get_' + self.p.transform.gen_func) \
            (batch_x.shape, **self.p.transform.params_gen)
        
        # return training samples + transform parameters, labels, indicators of whether images are labeled
        return [batch_x, transform_parameters], batch_y, batch_labeled
    
    def reset(self):
        self.indexes = np.arange(len(self.IDs), dtype = int)
        if self.shuffle:
            np.random.shuffle(self.indexes)
        
    def on_epoch_end(self):
        'Executes in the end of every epoch.'
        self.reset()
