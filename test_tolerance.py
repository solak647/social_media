import math
import csv
import random
import sys
import numpy as np
from LFS import EM
from DS import DS
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

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
        counts: 3d array of counts: [influencers x workers x choices]
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

# MAIN
if __name__ == "__main__":
    fashion = np.genfromtxt('data/aij_fashion.csv', delimiter=",", dtype=int)
    fashion = format_data(fashion)
    (influencers, workers, choices, counts) = fashion_to_counts(fashion)
    data_structure(counts)

    print("--------- PART WITH 60% training, 20% test, 20% validation ----------")
    train, test, validation = fashion_split_train_test_val()
    (influencers_validation, workers, choices, counts_validation) = fashion_to_counts(validation)
    (influencers_test, workers, choices, counts_test) = fashion_to_counts(test)
    (influencers_train, workers, choices, counts_train) = fashion_to_counts(train)
    truth = np.genfromtxt('data/labels_fashion.csv', delimiter=",", dtype=int)

    array_truth = []
    for item in truth:
        array_truth.append(item[2])
    array_truth = array_truth[182:]

    ds2 = DS(counts_train)
    tols = [0.1,0.01,0.001,0.0001,0.00001]
    accuracy_scores = []
    f1_scores = []
    for tol in tols:
      _ = ds2.run(tol)

      # WITH TEST SET
      influencers_label = ds2.e_step(counts_test, ds2.class_marginals, ds2.error_rates)

      result = np.argmax(influencers_label, axis=1)

      i = 0
      result_DS = {}
      for line in influencers_test:
          result_DS[line] = result[i]
          i = i + 1

      array_DS = list(result_DS.values())
      accuracy_scores.append(accuracy_score(array_truth, array_DS, normalize=True))
      f1_scores.append(f1_score(array_truth, array_DS))
    fig = plt.figure(figsize=(11,8))
    ax1 = fig.add_subplot(111)
    ax1.plot(accuracy_scores)
    ax1.plot(f1_scores)
    plt.xticks([0,1,2,3,4], tols)
    plt.xlabel('Tolerance')
    plt.ylabel('Value')
    plt.show()