import numpy as np
import sys

def main(args):
    fashion = np.genfromtxt('data/aij_fashion.csv', delimiter=",",dtype=int)
    fashion = format_data(fashion)
    (influencers, workers, choices, counts) = fashion_to_counts(fashion)
    influencers_label = majority_voting(counts)

    # initialize
    nIter = 0
    converged = False
    old_class_marginals = None
    old_error_rates = None
    tol=0.0000001
    max_iter=100

    while not converged:
        nIter += 1
        (class_marginals, error_rates) = m_step(counts,influencers_label)
        influencers_label = e_step(counts, class_marginals, error_rates)
        log_L = calc_likelihood(counts, class_marginals, error_rates)
        if old_class_marginals is not None:
            class_marginals_diff = np.sum(
                np.abs(class_marginals - old_class_marginals))
            error_rates_diff = np.sum(np.abs(error_rates - old_error_rates))
            print(nIter, '\t', log_L, '\t%.6f\t%.6f' % (class_marginals_diff, error_rates_diff))
            if (class_marginals_diff < tol) or nIter >= max_iter:
                converged = True
        else:
            print(nIter, '\t', log_L)

        old_class_marginals = class_marginals
        old_error_rates = error_rates
    np.set_printoptions(precision=2, suppress=True)
    result = np.argmax(influencers_label, axis=1)
    i = 0
    print(result)
    # for data in result:
    #     if data == 1:
    #         print(i)
    #     i = i + 1

def format_data(aij_fashion):
    data = {}
    for (worker,influencer,label) in aij_fashion:
        if influencer not in data:
            data[influencer] = {}
        result_label = [0]
        if label == 1:
             result_label = [1]
        data[influencer][worker] = result_label
    return data
  

def fashion_to_counts(fashions):
    """
    Convert a matrix of fashions to count data
    Args:
        fashions: dictionary of label {influencers:{workers:[labels]}}
    Returns:
        influencers: list of influencers
        workers: list of workers
        choices: list of choices (1 or 0 in our case)
        counts: 3d array of counts: [influencers x workers]
    """
    influencers = list(fashions)
    nInfluencers = len(influencers)

    workers = set()
    choices = set()
    for i in influencers:
        i_workers = list(fashions[i])
        for k in i_workers:
            if k not in workers:
                workers.add(k)
            ik_label = fashions[i][k]
            choices.update(ik_label)

    choices = list(choices)
    choices.sort()
    nChoices = len(choices)
    
    workers = list(workers)
    nWorkers = len(workers)

    # create a 3d array to hold counts
    counts = np.zeros([nInfluencers, nWorkers, nChoices])

    # convert responses to counts
    for influencer in influencers:
        i = influencers.index(influencer)
        for worker in fashions[influencer].keys():
            k = workers.index(worker)
            for label in fashions[influencer][worker]:
                j = choices.index(label)
                counts[i, k, j] += 1

    return (influencers, workers, choices, counts)

def majority_voting(counts):
    [nInfluencers, nWorkers, nChoices] = np.shape(counts)
    responses_sums = np.sum(counts, 1)
    result = np.zeros([nInfluencers, nChoices])
    for p in range(nInfluencers):
        result[p, :] = responses_sums[p, :] / np.sum(responses_sums[p, :], dtype=float)
    return result

def m_step(counts, influencers_label):
    [nInfluencers, nWorkers, nChoices] = np.shape(counts)

    # compute class marginals
    class_marginals = np.sum(influencers_label, 0) / float(nInfluencers)

    # compute error rates
    error_rates = np.zeros([nWorkers, nChoices, nChoices])
    for k in range(nWorkers):
        for j in range(nChoices):
            for l in range(nChoices):
                error_rates[k, j, l] = np.dot(
                    influencers_label[:, j], counts[:, k, l])
            sum_over_responses = np.sum(error_rates[k, j, :])
            if sum_over_responses > 0:
                error_rates[k, j, :] = error_rates[
                    k, j, :] / float(sum_over_responses)

    return (class_marginals, error_rates)

def e_step(counts, class_marginals, error_rates):
    [nInfluencers, nWorkers, nChoices] = np.shape(counts)

    influencers_label = np.zeros([nInfluencers, nChoices])

    for i in range(nInfluencers):
        for j in range(nChoices):
            estimate = class_marginals[j]
            estimate *= np.prod(np.power(error_rates[:, j, :], counts[i, :, :]))
            influencers_label[i, j] = estimate
        question_sum = np.sum(influencers_label[i, :])
        if question_sum > 0:
            influencers_label[i, :] = influencers_label[i, :] / float(question_sum)
    return influencers_label

def calc_likelihood(counts, class_marginals, error_rates):
    [nInfluencers, nWorkers, nChoices] = np.shape(counts)
    log_L = 0.0

    for i in range(nInfluencers):
        influencers_likelihood = 0.0
        for j in range(nChoices):

            class_prior = class_marginals[j]
            influencers_class_likelihood = np.prod(
                np.power(error_rates[:, j, :], counts[i, :, :]))
            influencers_class_posterior = class_prior * influencers_class_likelihood
            influencers_likelihood += influencers_class_posterior

        temp = log_L + np.log(influencers_likelihood)

        if np.isnan(temp) or np.isinf(temp):
            sys.exit()

        log_L = temp

    return log_L

main(None)
