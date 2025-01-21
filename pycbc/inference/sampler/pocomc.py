import logging
import numpy as np
import os
import pocomc
from pocomc.prior import Prior

from .base import BaseSampler, setup_output
from .base_cube import setup_calls
from .base_mcmc import get_optional_arg_from_config
from ...pool import choose_pool
from ..io.pocomc import PocoMCFile


class PocoMCSampler(BaseSampler):
    """Wrapper for the PocoMC sampler from the pocomc package."""

    name = "pocomc"
    _io = PocoMCFile

    def __init__(
        self,
        model,
        loglikelihood_function,
        nprocesses=1,
        use_mpi=False,
        run_kwds=None,
        extra_kwds=None
    ):
        super().__init__(model)

        self.log_likelihood_call, _ = setup_calls(
            self.model, loglikelihood_function=loglikelihood_function,
        )
        self.pool = choose_pool(mpi=use_mpi, processes=nprocesses)
        self.nprocesses = nprocesses

        self.prior = PocoMCPriorWrapper(model)
        self._sampler = None

        self.extra_kwds = extra_kwds or {}
        self.run_kwds = run_kwds or {}

        self.checkpoint_file = None

    def run(self):

        output = os.path.join(
            os.path.dirname(os.path.abspath(self.checkpoint_file)),
        )

        logging.info(f"Initializing PocoMC sampler with {self.extra_kwds}")

        self._sampler = pocomc.Sampler(
            prior=self.prior,
            likelihood=self.log_likelihood_call,
            n_dim=self.prior.dims,
            output_label="pocomc",
            output_dir=output,
            pool=self.pool,
            vectorize=False,
            **self.extra_kwds
        )

        logging.info(f"Running PocoMC sampler with {self.run_kwds}")

        self._sampler.run(**self.run_kwds)

        samples, weights, logl, logp = self._sampler.posterior()
        logz, logzerr = self._sampler.evidence()

        self.result = {
            "samples": samples,
            "weights": weights,
            "logl": logl,
            "logp": logp,
            "logz": logz,
            "logzerr": logzerr,
        }

    @classmethod
    def from_config(
        cls,
        cp,
        model,
        output_file=None,
        nprocesses=1,
        use_mpi=False,
        **kwargs
    ):
        if kwargs:
            logging.warning(f"Ignoring extra arguments: {kwargs}")
        opts = {
            "n_active": int,
            "n_effective": int,
            "train_config": dict,
            "flow": str,
            "n_steps": int,
            "n_max_steps": int,
        }
        run_opts = {
            "n_total": int,
            "n_evidence": int,
            "progress": bool,
        }
        extra_kwds = {}
        run_kwds = {}
        for opt_name in opts:
            if cp.has_option('sampler', opt_name):
                value = cp.get('sampler', opt_name)
                extra_kwds[opt_name] = opts[opt_name](value)
        for opt_name in run_opts:
            if cp.has_option('sampler', opt_name):
                value = cp.get('sampler', opt_name)
                run_kwds[opt_name] = run_opts[opt_name](value)

        loglikelihood_function = get_optional_arg_from_config(
            cp, "sampler", "loglikelihood-function"
        )

        sampler = cls(
            model,
            loglikelihood_function,
            nprocesses=nprocesses,
            use_mpi=use_mpi,
            extra_kwds=extra_kwds,
            run_kwds=run_kwds,
        )

        setup_output(sampler, output_file, check_nsamples=False)
        return sampler

    @property
    def io(self):
        return self._io

    def checkpoint(self):
        """ There is currently no checkpointing implemented"""
        pass

    def resume_from_checkpoint(self):
        """ There is currently no checkpointing implemented"""
        pass

    @property
    def model_stats(self):
        {}

    def finalize(self):
        for fn in [self.checkpoint_file, self.backup_file]:
            self.write_results(fn)

    @property
    def samples(self):
        samples = self.result["samples"]
        samples_dict = {
            p: samples[:, i] for i, p in enumerate(self.model.variable_params)
        }
        samples_dict["loglikelihood"] = self.result["logl"]
        samples_dict["logprior"] = self.result["logp"]
        samples_dict["logwt"] = self.result["weights"]
        return samples_dict

    def write_results(self, filename):
        with self.io(filename, "a") as f:
            f.write_samples(self.samples)
            f.write_logevidence(self.result["logz"], self.result["logzerr"])


class PocoMCPriorWrapper(Prior):
    """Wrapper for the prior distribution of a PyCBC model.

    PocoMC requires a custom prior distribution class that can be used to
    evaluate and sample from the prior.
    """
    def __init__(self, model):
        self.model = model
        self._dims = len(model.sampling_params)
        bounds = []
        for dist in model.prior_distribution.distributions:
            bounds += [
                    [v.min, v.max]
                    for k, v in dist.bounds.items()
                    if k in self.model.sampling_params
            ]
        self._bounds = np.array(bounds)

    @property
    def bounds(self):
        return self._bounds

    @property
    def dims(self):
        return self._dims

    def to_dict(self, x):
        return dict(zip(self.model.sampling_params, x.T))

    def from_dict(self, x):
        return np.array([x[k] for k in self.model.sampling_params]).T

    def logpdf(self, x):
        logp = np.zeros(len(x))
        # PyCBC prior is not vectorized
        for i, xx in enumerate(x):
            self.model.update(**self.to_dict(xx))
            logp[i] = self.model.logprior
        return logp

    def rvs(self, size=1):
        return self.from_dict(self.model.prior_rvs(size=size))
