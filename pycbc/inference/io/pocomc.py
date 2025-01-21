import numpy as np

from .dynesty import CommonNestedMetadataIO
from .base_sampler import BaseSamplerFile
from .posterior import read_raw_samples_from_file, write_samples_to_file


class PocoMCFile(CommonNestedMetadataIO, BaseSamplerFile):
    """Class to handle IO for the ``poco`` sampler."""

    name = "pocomc_file"

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
            n_samples = len(logwt)
            # Rejection sample
            rng = np.random.default_rng(seed)
            logwt -= logwt.max()
            logu = np.log(rng.random(n_samples))
            keep = logwt > logu
            post = {'loglikelihood': loglikelihood[keep]}
            for param in fields:
                post[param] = samples[param][keep]
            return post
        return samples

    def write_sampler_metadata(self, sampler):
        self.attrs['sampler'] = sampler.name
        if self.sampler_group not in self.keys():
            self.create_group(self.sampler_group)
        if sampler._sampler is not None:
            self[self.sampler_group].attrs['n_active'] = sampler._sampler.n_active
        sampler.model.write_metadata(self)

    def write_samples(self, samples, parameters=None):
        write_samples_to_file(self, samples, parameters=parameters)

    def write_resume_point(self):
        pass
