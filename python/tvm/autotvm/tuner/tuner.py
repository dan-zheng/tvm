# pylint: disable=unused-argument, no-self-use, invalid-name
"""Base class of tuner"""
import logging
import time

import numpy as np

from ..measure import MeasureInput, create_measure_batch

from ..env import GLOBAL_SCOPE

logger = logging.getLogger('autotvm')

class Tuner(object):
    """Base class for tuners

    Parameters
    ----------
    task: autotvm.task.Task
        Tuning Task
    """

    def __init__(self, task, **kwargs):
        self.param = kwargs
        self.recorder = None

        self.task = task

        # keep the current best
        self.best_config = None
        self.best_flops = 0
        self.best_measure_pair = None
        self.best_iter = 0
        # NOTE(dan-zheng): keep track of best raw cost, in addition to flops.
        self.best_cost = 0

        # time to leave
        self.ttl = None
        self.n_trial = None
        self.early_stopping = None

    def has_next(self):
        """Whether has next untried config in the space

        Returns
        -------
        has_next: bool
        """
        raise NotImplementedError()

    def next_batch(self, batch_size):
        """get the next batch of configs to be measure on real hardware

        Parameters
        ----------
        batch_size: int
            The size of the batch

        Returns
        -------
        a batch of configs
        """
        raise NotImplementedError()

    def update(self, inputs, results):
        """Update parameters of the tuner according to measurement results

        Parameters
        ----------
        inputs: Array of autotvm.measure.MeasureInput
            The input for measurement
        results: Array of autotvm.measure.MeasureResult
            result for measurement
        """
        pass

    def tune(self, n_trial, measure_option, early_stopping=None, callbacks=()):
        """Begin tuning

        Parameters
        ----------
        n_trial: int
            Maximum number of configs to try (measure on real hardware)
        measure_option: dict
            The options for how to measure generated code.
            You should use the return value ot autotvm.measure_option for this argument.
        early_stopping: int, optional
            Early stop the tuning when not finding better configs in this number of trials
        callbacks: List of callable
            A list of callback functions. The signature of callback function is
            (Tuner, List of MeasureInput, List of MeasureResult)
            with no return value. These callback functions will be called on
            every measurement pair. See autotvm/tuner/callback.py for some examples.
        """
        measure_batch = create_measure_batch(self.task, measure_option)
        n_parallel = getattr(measure_batch, 'n_parallel', 1)
        early_stopping = early_stopping or 1e9
        self.n_trial = n_trial
        self.early_stopping = early_stopping

        old_level = logger.level

        GLOBAL_SCOPE.in_tuning = True
        i = error_ct = 0

        logger.debug('TOTAL TRIAL COUNT = {}'.format(n_trial))
        start_time = time.time()
        while i < n_trial:
            logger.debug('TIME: {}, number of trials = {}'.format(time.time() - start_time, i))

            if not self.has_next():
                break

            configs = self.next_batch(min(n_parallel, n_trial - i))
            logger.debug('TIME: {}, get next batch count = {}'.format(time.time() - start_time, len(configs)))

            inputs = [MeasureInput(self.task.target, self.task, config) for config in configs]
            logger.debug('TIME: {}, constructed inputs'.format(time.time() - start_time))
            results = measure_batch(inputs)
            logger.debug('TIME: {}, measured batch'.format(time.time() - start_time))

            # NOTE(dan-zheng): Configurations are logged at the end.

            # keep best config
            for k, (inp, res) in enumerate(zip(inputs, results)):
                config = inp.config
                if res.error_no == 0:
                    # NOTE(dan-zheng): keep track of best cost.
                    cost = np.mean(res.costs)
                    flops = inp.task.flop / np.mean(res.costs)
                    error_ct = 0
                else:
                    cost = 0
                    flops = 0
                    error_ct += 1

                if flops > self.best_flops:
                    self.best_cost = cost
                    self.best_flops = flops
                    self.best_config = config
                    self.best_measure_pair = (inp, res)
                    self.best_iter = i + k

                # Old logging statement.
                # logger.debug("No: %d\tGFLOPS: %.2f/%.2f\tresult: %s\t%s",
                #              i + k + 1, flops / 1e9, self.best_flops / 1e9,
                #              res, config)

                # Log raw cost, in addition to flops.
                logger.debug("No: %d\tGFLOPS: %.4f/%.4f\tcost: %.4f\tresult: %s\t%s",
                             i + k + 1, flops / 1e9, self.best_flops / 1e9, self.best_cost,
                             res, config)

                # logger.debug('BEST_FLOPS %.10f', self.best_flops / 1e9)

                # if cost > self.best_cost:
                #     self.best_cost = cost
                #     self.best_flops = flops
                #     self.best_config = config
                #     self.best_measure_pair = (inp, res)
                #     self.best_iter = i + k

                # logger.debug("No: %d\tCOST: %.2f/%.2f\tresult: %s\t%s",
                #              i + k + 1, cost, self.best_cost, res, config)

            i += len(results)
            self.ttl = min(early_stopping + self.best_iter, n_trial) - i

            logger.debug('TIME: {}, about to update'.format(time.time() - start_time))
            self.update(inputs, results)
            logger.debug('TIME: {}, done update'.format(time.time() - start_time))
            for callback in callbacks:
                callback(self, inputs, results)
            logger.debug('TIME: {}, done callbacks'.format(time.time() - start_time))

            if i >= self.best_iter + early_stopping:
                logger.debug("Early stopped. Best iter: %d.", self.best_iter)
                break

            if error_ct > 150:
                logging.basicConfig()
                logger.warning("Too many errors happen in the tuning. Now is in debug mode")
                logger.setLevel(logging.DEBUG)
            else:
                logger.setLevel(old_level)

        GLOBAL_SCOPE.in_tuning = False
        del measure_batch

    def reset(self):
        """reset the status of tuner"""
        self.best_config = None
        self.best_flops = 0
        self.best_measure_pair = None

    def load_history(self, data_set):
        """load history data for transfer learning

        Parameters
        ----------
        data_set: Array of (MeasureInput, MeasureResult) pair
            Previous tuning records
        """
        raise NotImplementedError()
