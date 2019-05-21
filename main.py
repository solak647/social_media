import numpy as np
import sys

import math
import csv
import random
import sys
import numpy as np


class EM:
    def __init__(self,e2wl,w2el,label_set,beta_param={}):
        self.e2wl = e2wl
        self.w2el = w2el
        self.workers = self.w2el.keys()
        self.label_set = label_set
        self.initalquality = 0.7
        self.P_beta_param = beta_param
        self.use_P_beta = False
        self.sensitivity = 0.684
        self.specificity = 0.73
        if self.P_beta_param:
            self.use_P_beta = True

    # E-step
    def Update_e2lpd(self):
        self.e2lpd = {}

        for example, worker_label_set in e2wl.items():
            lpd = {}
            total_weight = 0

            for tlabel, prob in self.l2pd.items():
                weight = prob
                for (w, label) in worker_label_set:
                    weight *= self.w2cm[w][tlabel][label]

                lpd[tlabel] = weight
                total_weight += weight

            for tlabel in lpd:
                if total_weight == 0:
                    # uniform distribution
                    lpd[tlabel] = 1.0/len(self.label_set)
                else:
                    lpd[tlabel] = lpd[tlabel]*1.0/total_weight

            self.e2lpd[example] = lpd
    #M-step

    def Update_l2pd(self):
        for label in self.l2pd:
            self.l2pd[label] = 0

        for _, lpd in self.e2lpd.items():
            for label in lpd:
                self.l2pd[label] += lpd[label]

        for label in self.l2pd:
            if self.use_P_beta:
                self.l2pd[label] = 1.0 * (self.P_beta_param[label][0] - 1 + self.l2pd[label]) / (sum(self.P_beta_param[label]) - 2 + len(self.e2lpd))
            else:
                self.l2pd[label] *= 1.0/len(self.e2lpd)

    def Update_w2cm(self):

        for w in self.workers:
            for tlabel in self.label_set:
                for label in self.label_set:
                    self.w2cm[w][tlabel][label] = 0

        w2lweights = {}
        for w in self.w2el:
            w2lweights[w] = {}
            for label in self.label_set:
                w2lweights[w][label] = 0
            for example, _ in self.w2el[w]:
                for label in self.label_set:
                    w2lweights[w][label] += self.e2lpd[example][label]

            for tlabel in self.label_set:
                if w2lweights[w][tlabel] == 0:
                    if tlabel == "0":
                        self.w2cm[w]["0"]["0"] = self.specificity
                        self.w2cm[w]["0"]["1"] = 1 - self.specificity
                    elif tlabel == "1":
                        self.w2cm[w]["1"]["1"] = self.sensitivity
                        self.w2cm[w]["1"]["0"] = 1 - self.sensitivity
                    continue

                for example, label in self.w2el[w]:
                    self.w2cm[w][tlabel][label] += self.e2lpd[example][tlabel]*1.0/w2lweights[w][tlabel]

        return self.w2cm

    #initialization
    def Init_e2lpd(self):
        e2lpd = {}
        for example, worker_label_set in e2wl.items():
            lpd = {}
            total = 0
            for label in self.label_set:
                lpd[label] = 0

            for (w, label) in worker_label_set:
                lpd[label] += 1
                total+= 1

            if not total:
                for label in self.label_set:
                    lpd[label] = 1.0 / len(self.label_set)
            else:
                for label in self.label_set:
                    lpd[label] = lpd[label] * 1.0 / total
            e2lpd[example] = lpd
        return e2lpd

    def Init_l2pd(self):
        #uniform probability distribution
        l2pd = {}
        for label in self.label_set:
            l2pd[label] = 1.0/len(self.label_set)
        return l2pd

    def Init_w2cm(self):
        w2cm = {}
        for worker in self.workers:
            w2cm[worker] = {"0": {}, "1": {}}
            w2cm[worker]["0"]["0"] = self.specificity
            w2cm[worker]["0"]["1"] = 1 - self.specificity
            w2cm[worker]["1"]["1"] = self.sensitivity
            w2cm[worker]["1"]["0"] = 1 - self.sensitivity
            # for tlabel in self.label_set:
            #     w2cm[worker][tlabel] = {}
            #     for label in self.label_set:
            #         if tlabel == label:
            #             w2cm[worker][tlabel][label] = self.initalquality
            #         else:
            #             w2cm[worker][tlabel][label] = (1-self.initalquality)/(len(label_set)-1)


        return w2cm

    def Run(self, iterr = 20):
        self.e2lpd = self.Init_e2lpd()
        self.l2pd = self.Init_l2pd()
        self.w2cm = self.Init_w2cm()

        while iterr > 0:

            # M-step
            self.Update_l2pd()
            self.Update_w2cm()

            # E-step
            self.Update_e2lpd()

            # compute the likelihood
            print (self.computelikelihood())

            iterr -= 1

        return self.e2lpd, self.w2cm

    def computelikelihood(self):
        lh = 0

        for _, worker_label_set in self.e2wl.items():
            temp = 0
            for tlabel, prior in self.l2pd.items():
                inner = prior
                for worker, label in worker_label_set:
                    inner *= self.w2cm[worker][tlabel][label]
                temp += inner

            lh += math.log(temp)

        return lh

###################################
# The above is the EM method (a class)
# The following are several external functions
###################################

def getaccuracy(truthfile, e2lpd, label_set):
    e2truth = {}
    f = open(truthfile, 'r')
    reader = csv.reader(f)
    next(reader)

    for line in reader:
        example, truth = line
        e2truth[example] = truth

    tcount = 0
    count = 0

    for e in e2lpd:

        if e not in e2truth:
            continue

        temp = 0
        for label in e2lpd[e]:
            if temp < e2lpd[e][label]:
                temp = e2lpd[e][label]

        candidate = []

        for label in e2lpd[e]:
            if temp == e2lpd[e][label]:
                candidate.append(label)

        truth = random.choice(candidate)

        count += 1

        if truth == e2truth[e]:
            tcount += 1

    return tcount*1.0/count


def gete2wlandw2el(datafile):
    e2wl = {}
    w2el = {}
    label_set=[]

    f = open(datafile, 'r')
    reader = csv.reader(f)
    next(reader)

    for line in reader:
        worker, potential_influencer, label = line
        if potential_influencer not in e2wl:
            e2wl[potential_influencer] = []
        e2wl[potential_influencer].append([worker,label])

        if worker not in w2el:
            w2el[worker] = []
        w2el[worker].append([potential_influencer,label])

        if label not in label_set:
            label_set.append(label)

    return e2wl,w2el,label_set

# ----------- DS --------------


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

    return influencers, workers, choices, counts


def majority_voting(counts):
    [nInfluencers, nWorkers, nChoices] = np.shape(counts)
    responses_sums = np.sum(counts, 1)
    result = np.zeros([nInfluencers, nChoices])
    for p in range(nInfluencers):
        result[p, :] = responses_sums[p, :] / np.sum(responses_sums[p, :], dtype=float)
    return result


def m_step(counts, influencers_label):
    [nInfluencers, nWorkers, nChoices] = np.shape(counts)

    # compute class marginals --- current estimation of class prior
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

    return class_marginals, error_rates


def e_step(counts, class_marginals, error_rates):
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


def calc_likelihood(counts, class_marginals, error_rates):
    [n_influencers, m_workers, m_choices] = np.shape(counts)
    log_L = 0.0

    for i in range(n_influencers):
        influencers_likelihood = 0.0
        for j in range(m_choices):

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


def data_structure(counts) :
    [n_influencers, m_workers, m_choices] = np.shape(counts)
    print("potential influencers :", n_influencers)
    print("number of workers :", m_workers)
    print("number of choices :", m_choices)


def fashion_split_train_test_val():
    fashion = np.genfromtxt('data/aij_fashion.csv', delimiter=",", dtype=int)
    fashion = format_data(fashion)
    train = {}
    test = {}
    validation = {}
    for line in fashion:
        if line < 181:
            validation[line]=fashion[line]
        elif line < 363:
            test[line]=fashion[line]
        else :
            train[line]=fashion[line]
    return train, test, validation


if __name__ == "__main__":
    train, test, validation = fashion_split_train_test_val()
    (influencers_validation, workers, choices, counts_validation) = fashion_to_counts(validation)
    (influencers_test, workers, choices, counts_test) = fashion_to_counts(test)
    (influencers_train, workers, choices, counts_train) = fashion_to_counts(train)

    influencers_label = majority_voting(counts_train)

    # initialize
    nIter = 0
    converged = False
    old_class_marginals = None
    old_error_rates = None
    tol = 0.0000001
    max_iter = 100
    print("iteration", "likelihood", "class_marginals_diff", "error_rates_diff")
    while not converged:
        nIter += 1
        (class_marginals, error_rates) = m_step(counts_train, influencers_label)
        influencers_label = e_step(counts_train, class_marginals, error_rates)
        log_L = calc_likelihood(counts_train, class_marginals, error_rates)
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

    # WITH TEST SET
    influencers_label = e_step(counts_test,class_marginals, error_rates)

    np.set_printoptions(precision=2, suppress=True)
    result = np.argmax(influencers_label, axis=1)

    i = 0
    result_DS = {}
    for line in influencers_test:
        result_DS[line] = result[i]
        #    if data == 1:
        #        print(i)
        i = i + 1
    print(result_DS)

    truth = np.genfromtxt('data/labels_fashion.csv', delimiter=",", dtype=int)

    array_truth = []
    for item in truth:
        array_truth.append(item[2])
    array_truth=array_truth[182:]

    from sklearn.metrics import accuracy_score, f1_score

    array_DS = list(result_DS.values())

    print("accuracy DS", accuracy_score(array_truth, array_DS, normalize=True))
    print("F1 DS", f1_score(array_truth, array_DS))


if __name__ == "__min__":
    fashion = np.genfromtxt('data/aij_fashion.csv', delimiter=",", dtype=int)
    fashion = format_data(fashion)
    (influencers, workers, choices, counts) = fashion_to_counts(fashion)
    data_structure(counts)
    print("------------------ DS -----------------")
    # Using majority rating to initialize labels
    influencers_label = majority_voting(counts)

    # initialize
    nIter = 0
    converged = False
    old_class_marginals = None
    old_error_rates = None
    tol = 0.0000001
    max_iter = 100
    print("iteration", "likelihood", "class_marginals_diff", "error_rates_diff")
    while not converged:
        nIter += 1
        (class_marginals, error_rates) = m_step(counts, influencers_label)
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
    #print(influencers_label)
    result = np.argmax(influencers_label, axis=1)
    i = 0
    #print("------------")
    #print("Result Matrix")
    #print(result_LFC)
    result_DS = {}
    #print("Supposed influencers Nr. ")
    for data in result:
        result_DS[i]=data
    #    if data == 1:
    #        print(i)
        i = i + 1

    print(result_DS)

    print("------------------ LFC -----------------")
    datafile = "data/aij_fashion.csv"
    e2wl,w2el,label_set = gete2wlandw2el(datafile) # generate structures to pass into EM
    iterations = 20 # EM iteration number
    e2lpd, w2cm = EM(e2wl,w2el,label_set).Run(iterations)

    result_LFC = {}
    for i in e2lpd:
        # print("potential influencer",i,e2lpd[i])
        if e2lpd[i]['0']<e2lpd[i]['1']:
            result_LFC[i]= 1
        else:
            result_LFC[i]= 0

    print(result_LFC)

    #print("Supposed influencers Nr. ")

    #for data in result_LFC:
    #    if result_LFC[data] == 1:
    #       print(data)

    truth = np.genfromtxt('data/labels_fashion.csv', delimiter=",", dtype=int)

    array_truth = []
    for item in truth:
        array_truth.append(item[2])
    array_truth.remove(array_truth[0])

    from sklearn.metrics import accuracy_score, f1_score
    array_DS = list(result_DS.values())
    array_LFC = list(result_LFC.values())
    array_LFC = array_LFC[0:len(array_truth)]
    array_DS = array_DS[0:len(array_truth)]

    print("accuracy DS",accuracy_score(array_truth,array_DS,normalize=True))
    print("accuracy LFC",accuracy_score(array_truth,array_LFC,normalize=True))
    print("F1 DS",f1_score(array_truth,array_DS))
    print("F1 LFC",f1_score(array_truth,array_LFC))





