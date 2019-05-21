import math
import csv
import random
import sys
import numpy as np

# ----------- DS --------------
# Dawid-Skene Algo adapted from https://github.com/SDey96/Fast-Dawid-Skene/blob/master/fast_dawid_skene/algorithms.py
class DS:
    def __init__(self, counts):
        self.counts = counts
        self.class_marginals = 0
        self.error_rates = 0

    def run(self, tol = 1e-5):
        # Using majority rating to initialize labels
        influencers_label = self.majority_voting(self.counts)
        # initialize
        nIter = 0
        converged = False
        old_class_marginals = None
        old_error_rates = None
        max_iter = 200
        print("iteration", "likelihood", "class_marginals_diff", "error_rates_diff")
        while not converged:
            nIter += 1
            (class_marginals, error_rates) = self.m_step(self.counts, influencers_label)

            influencers_label = self.e_step(self.counts, class_marginals, error_rates)

            log_L = self.calc_likelihood(self.counts, class_marginals, error_rates)

            if old_class_marginals is not None:
                class_marginals_diff = np.sum(np.abs(class_marginals - old_class_marginals))
                error_rates_diff = np.sum(np.abs(error_rates - old_error_rates))
                print(nIter, '\t', '\t%.3f\t%.6f\t%.6f' % (log_L, class_marginals_diff, error_rates_diff))
                if (class_marginals_diff < tol) or nIter >= max_iter:
                    converged = True
            else:
                print(nIter, '\t', log_L)

            old_class_marginals = class_marginals
            old_error_rates = error_rates
            self.class_marginals = old_class_marginals
            self.error_rates = old_error_rates
        np.set_printoptions(precision=2, suppress=True)

        result = np.argmax(influencers_label, axis=1)
        
        return result
    def majority_voting(self, counts):
        [nInfluencers, nWorkers, nChoices] = np.shape(counts)
        responses_sums = np.sum(counts, 1)
        result = np.zeros([nInfluencers, nChoices])
        for p in range(nInfluencers):
            result[p, :] = responses_sums[p, :] / np.sum(responses_sums[p, :], dtype=float)
        return result


    def m_step(self, counts, influencers_label):
        [nInfluencers, nWorkers, nChoices] = np.shape(counts)

        # compute class marginals --- current estimation of class prior
        class_marginals = np.sum(influencers_label, 0) / float(nInfluencers)

        # compute error rates
        error_rates = np.zeros([nWorkers, nChoices, nChoices])
        for k in range(nWorkers):
            for j in range(nChoices):
                for l in range(nChoices):
                    error_rates[k, j, l] = np.dot(influencers_label[:, j], counts[:, k, l])
                sum_over_responses = np.sum(error_rates[k, j, :])
                if sum_over_responses > 0:
                    error_rates[k, j, :] = error_rates[k, j, :] / float(sum_over_responses)

        return class_marginals, error_rates


    def e_step(self, counts, class_marginals, error_rates):
        [n_influencers, m_workers, m_choices] = np.shape(counts)
        influencers_label = np.zeros([n_influencers, m_choices])

        for i in range(n_influencers):
            for j in range(m_choices):
                estimate = class_marginals[j]
                estimate *= np.prod(np.power(error_rates[:, j, :], counts[i, :, :]))
                influencers_label[i, j] = estimate
            question_sum = np.sum(influencers_label[i, :])
            if question_sum > 0:
                influencers_label[i, :] = influencers_label[i, :] / float(question_sum)
        return influencers_label

    def calc_likelihood(self, counts, class_marginals, error_rates):
        [n_influencers, m_workers, m_choices] = np.shape(counts)
        log_L = 0.0

        for i in range(n_influencers):
            influencers_likelihood = 0.0
            for j in range(m_choices):
                class_prior = class_marginals[j]
                influencers_class_likelihood = np.prod(np.power(error_rates[:, j, :], counts[i, :, :]))
                influencers_class_posterior = class_prior * influencers_class_likelihood
                influencers_likelihood += influencers_class_posterior
            temp = log_L + np.log(influencers_likelihood)
            if np.isnan(temp) or np.isinf(temp):
                sys.exit()
            log_L = temp

        return log_L