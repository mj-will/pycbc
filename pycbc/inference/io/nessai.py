"""Provides IO for the nessai sampler"""
import numpy

from .base_nested_sampler import BaseNestedSamplerFile

from ...io.hdf import dump_state, load_state
from .posterior import read_raw_samples_from_file, write_samples_to_file
from .dynesty import CommonNestedMetadataIO


class NessaiFile(CommonNestedMetadataIO, BaseNestedSamplerFile):
    """Class to handle file IO for the ``nessai`` sampler."""

    name = "nessai_file"

    def read_raw_samples(self, fields, raw_samples=False, seed=0):
        """Reads samples from a nessai file and constructs a posterior.

        Using rejection sampling to resample the nested samples

        Parameters
        ----------
        fields : list of str
            The names of the parameters to load. Names must correspond to
            dataset names in the file's ``samples`` group.
        raw_samples : bool, optional
            Return the raw (unweighted) samples instead of the estimated
            posterior samples. Default is False.

        Returns
        -------
        dict :
            Dictionary of parameter fields -> samples.
        """
        samples = read_raw_samples_from_file(self, fields)
        logwt = read_raw_samples_from_file(self, ['logwt'])['logwt']
        loglikelihood = read_raw_samples_from_file(
            self, ['loglikelihood'])['loglikelihood']
        if not raw_samples:
            n = len(logwt)
            # Rejection sample
            rng = numpy.random.default_rng(seed)
            logwt -= logwt.max()
            logu = numpy.log(rng.random(n))
            keep = logwt > logu
            post = {'loglikelihood': loglikelihood[keep]}
            for param in fields:
                post[param] = samples[param][keep]
            return post
        else:
            return samples

    def write_pickled_data_into_checkpoint_file(self, data):
        """Write the pickled data into a checkpoint file"""
        if "sampler_info/saved_state" not in self:
            self.create_group("sampler_info/saved_state")
        dump_state(data, self, path="sampler_info/saved_state")

    def read_pickled_data_from_checkpoint_file(self):
        """Read the pickled data from a checkpoint file"""
        return load_state(self, path="sampler_info/saved_state")

    def write_raw_samples(self, data, parameters=None):
        """Write the nested samples to the file"""
        if "samples" not in self:
            self.create_group("samples")
        write_samples_to_file(
            self, data, parameters=parameters, group="samples"
        )
