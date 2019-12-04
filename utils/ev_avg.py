import os
import yaml
import numpy as np
import torch
import shutil
from collections import namedtuple

class EVLocalAvg(object):
    def __init__(self, window=5, ev_freq=2, total_epochs=50):
        """ Keep track of the eigenvalues local average.
        Args:
            window (int): number of elements used to compute local average.
                Default: 5
            ev_freq (int): frequency used to compute eigenvalues. Default:
                every 2 epochs
            total_epochs (int): total number of epochs that DARTS runs.
                Default: 50
        """
        self.window = window
        self.ev_freq = ev_freq
        self.epochs = total_epochs

        self.stop_search = False
        self.stop_epoch = total_epochs - 1
        self.stop_genotype = None

        self.ev = []
        self.ev_local_avg = []
        self.genotypes = {}
        self.la_epochs = {}

        # start and end index of the local average window
        self.la_start_idx = 0
        self.la_end_idx = self.window

    def reset(self):
        self.ev = []
        self.ev_local_avg = []
        self.genotypes = {}
        self.la_epochs = {}

    def update(self, epoch, ev, genotype):
        """ Method to update the local average list.
        Args:
            epoch (int): current epoch
            ev (float): current dominant eigenvalue
            genotype (namedtuple): current genotype
        """
        self.ev.append(ev)
        self.genotypes.update({epoch: genotype})
        # set the stop_genotype to the current genotype in case the early stop
        # procedure decides not to early stop
        self.stop_genotype = genotype

        # since the local average computation starts after the dominant
        # eigenvalue in the first epoch is already computed we have to wait
        # at least until we have 3 eigenvalues in the list.
        if (len(self.ev) >= int(np.ceil(self.window/2))) and (epoch <
                                                              self.epochs - 1):
            # start sliding the window as soon as the number of eigenvalues in
            # the list becomes equal to the window size
            if len(self.ev) < self.window:
                self.ev_local_avg.append(np.mean(self.ev))
            else:
                assert len(self.ev[self.la_start_idx: self.la_end_idx]) == self.window
                self.ev_local_avg.append(np.mean(self.ev[self.la_start_idx:
                                                         self.la_end_idx]))
                self.la_start_idx += 1
                self.la_end_idx += 1

            # keep track of the offset between the current epoch and the epoch
            # corresponding to the local average. NOTE: in the end the size of
            # self.ev and self.ev_local_avg should be equal
            self.la_epochs.update({epoch: int(epoch -
                                              int(self.ev_freq*np.floor(self.window/2)))})

        elif len(self.ev) < int(np.ceil(self.window/2)):
          self.la_epochs.update({epoch: -1})

        # since there is an offset between the current epoch and the local
        # average epoch, loop in the last epoch to compute the local average of
        # these number of elements: window, window - 1, window - 2, ..., ceil(window/2)
        elif epoch == self.epochs - 1:
            for i in range(int(np.ceil(self.window/2))):
                assert len(self.ev[self.la_start_idx: self.la_end_idx]) == self.window - i
                self.ev_local_avg.append(np.mean(self.ev[self.la_start_idx:
                                                         self.la_end_idx + 1]))
                self.la_start_idx += 1

    def early_stop(self, epoch, factor=1.3, es_start_epoch=10, delta=4):
        """ Early stopping criterion
        Args:
            epoch (int): current epoch
            factor (float): threshold factor for the ration between the current
                and prefious eigenvalue. Default: 1.3
            es_start_epoch (int): until this epoch do not consider early
                stopping. Default: 20
            delta (int): factor influencing which previous local average we
                consider for early stopping. Default: 2
        """
        if int(self.la_epochs[epoch] - self.ev_freq*delta) >= es_start_epoch:
            # the current local average corresponds to 
            # epoch - int(self.ev_freq*np.floor(self.window/2))
            current_la = self.ev_local_avg[-1]
            # by default take the local average corresponding to epoch
            # delta*self.ev_freq
            previous_la = self.ev_local_avg[-1 - delta]

            self.stop_search = current_la / previous_la > factor
            if self.stop_search:
                self.stop_epoch = int(self.la_epochs[epoch] - self.ev_freq*delta)
                self.stop_genotype = self.genotypes[self.stop_epoch]
