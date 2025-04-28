import logging

# LOGGER = logging.getLogger(__name__)


class BaseSynthesizer:
    """Base class for all default synthesizers of ``SDGym``."""

    def fit(self, data, categorical_columns=tuple(), ordinal_columns=tuple(), bimodal_columns={}):
        print("base fit")
        self.fit(data, categorical_columns, ordinal_columns, bimodal_columns)

    def sample(self, samples):
        print("base sample")
        return self.sample(self, samples)

    def sample_indicating_column(self, samples, column):
        print("base sample")
        return self.sample_indicating_column(self, samples, column)

    def sample_indicating_pair(self, samples, pair):
        print("base sample")
        return self.sample_new(self, samples, column)

    def fit_sample(self, data, categorical_columns=tuple(), ordinal_columns=tuple(), bimodal_columns=tuple()):
        # LOGGER.info("Fitting %s", self.__class__.__name__)
        self.fit(data, categorical_columns, ordinal_columns, bimodal_columns)

        # LOGGER.info("Sampling %s", self.__class__.__name__)
        return self.sample(data.shape[0])
