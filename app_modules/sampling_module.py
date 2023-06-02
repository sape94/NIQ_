import pandas as pd


class SamplingMachine:

    def __init__(self,
                 sample_portion: int = 50,
                 confidence_level: int = 99,
                 standard_error: int = 1):
        self.p = float(sample_portion)/100
        self.cl = int(confidence_level)
        self.e = float(standard_error)/100

        z_score_dict = {99: 2.576,
                        98: 2.326,
                        95: 1.96,
                        90: 1.645,
                        85: 1.44,
                        80: 1.282}
        self.q = 1 - self.p
        self.Z = z_score_dict[self.cl]

    def calc_samp(self, population_size: int):
        self.nn = population_size
        aa = self.nn*(self.Z**2)*self.p*self.q
        bb = (self.e**2)*(self.nn-1)+(self.Z**2)*self.p*self.q
        n = int(aa/bb)
        return n

    def rand_samp(self, df):
        self.o_df = df
        self.nm = self.o_df.shape[0]
        aa = self.nm*(self.Z**2)*self.p*self.q
        bb = (self.e**2)*(self.nm-1)+(self.Z**2)*self.p*self.q
        n = int(aa/bb)
        self.s_df = self.o_df.sample(n=n)
        self.o_df_2 = self.o_df.copy().sample(frac=1)
        self.s_df_2 = self.o_df_2.sample(n=n)

        return self.s_df
    # , self.s_df_2
