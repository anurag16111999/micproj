# Random Forest Algorithm on Sonar Dataset
from random import seed
from random import randrange
from csv import reader
from math import sqrt
from sklearn.preprocessing import normalize
import random as rd
import statistics

import numpy
import time

# Load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

def str_column_to_int(dataset, column):
    for row in dataset:
        if row[column].strip() == '?':
            row[column] = 0
        else:
            row[column] = int(row[column].strip())

# Convert string column to integer
def str_column_to_cls(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup


# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0

    for i in range(len(actual)):
        if actual[i] == predicted[i][0]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


# Calculate the Gini index for a split dataset
def gini_index(groups, classes):
    # count all samples at split point
    n_instances = float(sum([len(group) for group in groups]))
    # sum weighted Gini index for each group
    gini = 0.0
    for group in groups:
        size = float(len(group))
        # avoid divide by zero
        if size == 0:
            continue
        score = 0.0
        # score the group based on the score for each class
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        # weight the group score by its relative size
        gini += (1.0 - score) * (size / n_instances)
    return gini


# Select the best split point for a dataset
def get_split(dataset, n_features):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    features = list()
    while len(features) < n_features:
        index = randrange(len(dataset[0]) - 1)
        if index not in features:
            features.append(index)
    # print(features)s

    # print(hist1)
# n_features are the number of features choosen randomly
# we do split on a feature and gini_index calculation on entire dataset
# features is a set of indices
# each row of the dataset is an entry
# I have selectedd features, then i will use only these features to calculate gini index on 
# here i have the features , now i have to generate outliers and append to features

    for index in features:
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            # print(type(groups))
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index': b_index, 'value': b_value, 'groups': b_groups}


# Create a terminal node value
def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)


# Create child splits for a node or make terminal
def split(node, max_depth, min_size, n_features, depth):
    left, right = node['groups']
    del (node['groups'])
    # check for a no split
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    # check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    # process left child
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left, n_features)
        split(node['left'], max_depth, min_size, n_features, depth + 1)
    # process right child
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right, n_features)
        split(node['right'], max_depth, min_size, n_features, depth + 1)


# Build a decision tree
def build_tree(train, max_depth, min_size, n_features):
    root = get_split(train, n_features)
    split(root, max_depth, min_size, n_features, 1)
    return root


# Make a prediction with a decision tree
def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']


# Create a random subsample from the dataset with replacement
def subsample(dataset, ratio):
    sample = list()
    n_sample = round(len(dataset) * ratio)
    while len(sample) < n_sample:
        index = randrange(len(dataset))
        sample.append(dataset[index])
    return sample


# Make a prediction with a list of bagged trees
def bagging_predict(trees, row):
    # actualda = 
    predictions = []
    for tree in trees:
        tr = tree[0]
        fts = tree[1]
        nRow = []
        for ft in fts:
            nRow.append(row[ft])
        nRow.append(row[-1])
        predictions.append(predict(tr, nRow))
    #predictions = [predict(tree, row) for tree in trees]
    
    return [max(set(predictions), key=predictions.count)]



def valfromhist(index):
    value = 0
    feature = hist1[index]
    prob = (feature[0])[0]
    bins = feature[1]

    # print(prob)
    # print(bins)
    
    chosenbin=numpy.random.choice(range(len(prob)),1,p=prob)
    
    # print("chosenbin")

    # print(chosenbin)

    binindexlower = chosenbin[0]
    binindexupper = binindexlower + 1
    value = numpy.random.uniform(bins[binindexlower], bins[binindexupper])
    return value




def giveoutliers(ratio,sample,rsmlist):
# change last outlier element to be -1
    val = []
    for x in range(ratio):
        val1 = list(sample)
        # print("val1")
        
        # print(val1)
        val1[-1] = 1
        for y in rsmlist:
            val1[y] = valfromhist(y)
        val.append(val1)
    # print(".")
    return val



# Random Forest Algorithm
def random_forest(train, test, max_depth, min_size, sample_size, n_trees, n_features):
    trees = list()
    # rsmnum = 10
    # print("in rf")
    # print(len(train[0]))

    for i in range(n_trees):
        rsmlist = list()
        while len(rsmlist) < rsmnum:
            index = randrange(len(train[0]) - 1)
            if index not in rsmlist:
                rsmlist.append(index)
        sample = subsample(train, sample_size)
        # outlierset = []
        # print("sample")
        # print(len(sample))
        sample11 = []
        # print(sample)
        i1 = 1;
        for x in sample:
            # print(x)
            i1 = i1 + 1;
            x1 = giveoutliers(outlierratio,x,rsmlist)
            # print("x1 => " + str(len(sample)))
            # print(x1)
            sample11.extend(x1)

        sample.extend(sample11)
        # print("x1 => " + str(len(sample)))
        # print("tree " + str(i))
# n generate the data here
# THROUGH RSM
        tempFea = list(rsmlist)
        newS = []
        for sm in sample:
            templ = []
            for ft in rsmlist:
                templ.append(sm[ft])
            templ.append(sm[-1])
            newS.append(templ)
            
        tree = build_tree(newS, max_depth, min_size, n_features)
        #tree = build_tree(sample, max_depth, min_size, n_features)
        trees.append([tree,tempFea])
        #trees.append(tree)
    # print("out rf")

    predictions = [bagging_predict(trees, row) for row in test]

    MCC = 0
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    # print(predictions)
    # print(actuald)
    # print(len(predictions))
    # print(len(actuald))
    

    for i in range(len(predictions)):
        if(predictions[i][0] == 0):
            if(actuald[i] == 0):
                TN = TN + 1;
            else:
                FP = FP + 1;
        else:
            if(actuald[i] == 0):
                FN = FN + 1;
            else:
                TP = TP + 1;

    print(TP)
    print(FP)
    print(TN)
    print(FN)

    MCC = float((TP*TN-FP*FN))/sqrt(float(((TP + FP)*(TP + FN)*(TN + FP)*(TN + FN))))


    return ([predictions,MCC,(FN/float(FN + TN))])


# Test the random forest algorithm
# seed(2)

# load and prepare data

def new_evaluate_algorithm(train_set,test_set, algorithm, *args):
    # folds = cross_validation_split(dataset, n_folds)
    scores = list()
    # for fold in folds:
        # train_set = list(folds)
        # train_set.remove(fold)
        # train_set = sum(train_set, [])
        # test_set = list()
        # for row in fold:
            # row_copy = list(row)
            # test_set.append(row_copy)
            # row_copy[-1] = None
    predicted = algorithm(train_set, test_set, *args)
    # print("predicted: ")
    # print(predicted)

    # print("actual: ")
    # print(actuald)
    # actual = actuald
    accuracy = accuracy_metric(actuald, predicted[0])
    scores.append([accuracy,predicted[1:]])
    return scores


validave = 0
testave = []
testmcc = []
for x2234 in range(30):
    filename = 'breast-cancer-wisconsin.data'
    dataset = load_csv(filename)
    for i,rw in enumerate(dataset):
        rw = rw[1:]
        dataset[i] = rw
    # convert string attributes to integers
    for i in range(0, len(dataset[0]) - 1):
        str_column_to_int(dataset, i)

    # convert class column to integers
    str_column_to_cls(dataset, len(dataset[0]) - 1)

    # evaluate algorithm

    # for x in dataset:
    #     print(x[len(x) -1 ]);

    y22=numpy.array([numpy.array(xi) for xi in dataset])
    print(y22)

    yt = []
    yf = []


    # print(y)
    for z11 in y22:
        if(z11[-1] == 0):
            yt.append(list(z11))
        else:
            yf.append(list(z11))

    time.sleep(1)


    rd.shuffle(yt)
    # train_set = yt[0:200]
    # y = numpy.array(train_set)
    # test_set = yt[200:]
    # test_set.extend(yf)
    # validationset
    train_set = yt[0:40]
    y = numpy.array(train_set)
    validationset = yt[40:240]
    validationset.extend(yf[:100])

    test_set = yt[240:]
    test_set.extend(yf[100:])



    print([row[-1] for row in test_set])
    print([row[-1] for row in train_set])


    # print("train_set")
    # print(train_set)

    # print("test_set")
    # print(test_set)

    bins = 20;

    column1 = len(dataset[0])
    row1 = len(y)
    hist1 = []
    outlierratio = 10;

    # train and testset have to be list of lists

    # train_set = y 



    for i in range(column1-1):
        y1 = y[:,[i]]
        # print(y1)
        # print( str(i) + " => " + str(max(y1)))
        # print(min(y1))
        #k1 = numpy.histogram(y1,range = (y1.min() - ((y1.max()-y1.min())/10),y1.max() + ((y1.max()-y1.min())/10)),bins = bins)
        k1 = numpy.histogram(y1,range = (y1.min() ,y1.max()),bins = bins)
        r1 = k1[0];
        # print(k1)
        r1 = normalize(r1.reshape(1,-1), norm="l1")
        r1 = r1[0]

        for j in range(len(r1)):
            r1[j] = 1 - r1[j]
        r1 = normalize(r1.reshape(1,-1), norm="l1")

        # print(r1)
        hist1.append([r1,k1[1]])
        # k1[0] = r1


    # print(hist1)


    # for x in 


    # n_folds = 5
    max_depth = 10# previously 100 
    min_size = 1
    sample_size = 0.1
    #n_features = in(sqrt(len(dataset[0]) - 1))
    # n_features = 3
    rsmnum = 3;
    # n_features = int(sqrt(rsmnum))
    n_features = 2;

    dataset = validationset
    actuald = [row[-1] for row in dataset]


    # for n_trees in [20 , 30 , 40 , 50 , 60 , 70 , 80 , 90 , 100 ,120 ,130 ,140 , 150 ]:
    #     scores = new_evaluate_algorithm(train_set,dataset, random_forest, max_depth, min_size, sample_size, n_trees, n_features)
    #     print('Trees: %d' % n_trees)
    #     print('Scores: %s' % scores)
    #     time.sleep(0.1)
        # print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))

    n_trees = 100
    # for max_depth in [3 , 6 , 12,20,40,60,80,100,140,200,250 ]:
    outlierratio = 5
    # for max_depth in [200]:
    max_depth = 10
    # for max_depth in [1,2,3,4,5,6,8,10,15,20,30,40,60,100,150,200,300]:
    # ave = ave +
    scores = new_evaluate_algorithm(train_set,dataset, random_forest, max_depth, min_size, sample_size, n_trees, n_features)
    validave = validave + scores[0][0]
    # print('outlierratio %d' % max_depth)
    print('Scores: validationset%s' % scores)

    dataset = test_set
    actuald = [row[-1] for row in dataset]

    # time.sleep(3)
    scores = new_evaluate_algorithm(train_set,dataset, random_forest, max_depth, min_size, sample_size, n_trees, n_features)
    testave.append(scores[0][0])
    testmcc.append(scores[0][1][0])
    
    # print('outlierratio %d' % max_depth)
    print('Scores: test_set%s' % scores)



validave = validave/20
# testave = testave/20

print("validave")
print(validave)
print("testave")
print(statistics.mean(testave))
print(statistics.stdev(testave))
print("testmcc")
print(statistics.mean(testmcc))
print(statistics.stdev(testmcc))

# scores = new_evaluate_algorithm(train_set,dataset, random_forest, max_depth, min_size, sample_size, n_trees, n_features)
    # ave = ave + scores
    # print('outlierratio %d' % max_depth)
    # print('Scores: %s' % scores)


# evaluate_algorithm(dataset, random_forest, n_folds, max_depth, min_size, sample_size, n_trees, n_features);
#predicted = algorithm(train_set, test_set, *args);


# def bagging_predict(trees, row);
