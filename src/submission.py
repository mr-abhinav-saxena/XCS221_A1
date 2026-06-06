#!/usr/bin/python

import random
from typing import Callable, Dict, List, Tuple, TypeVar, DefaultDict
from util import *

FeatureVector = Dict[str, int]
WeightVector = Dict[str, float]
Example = Tuple[FeatureVector, int]

############################################################
# Problem 1: binary classification
############################################################

############################################################
# Problem 1a: feature extraction


def extractWordFeatures(x: str) -> FeatureVector:
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x:
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    pass
    # ### START CODE HERE ###
    word_counts = DefaultDict(int)
    for word in x.split(' '):
        word_counts[word] += 1
    return word_counts
    # ### END CODE HERE ###


############################################################
# Problem 1b: stochastic gradient descent

T = TypeVar("T")


def learnPredictor(
    trainExamples: List[Tuple[T, int]],
    validationExamples: List[Tuple[T, int]],
    featureExtractor: Callable[[T], FeatureVector],
    numEpochs: int,
    eta: float,
) -> WeightVector:
    """
    Given |trainExamples| and |validationExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of epochs to
    train |numEpochs|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Notes:
    - Only use the trainExamples for training!
    - You should call evaluatePredictor() on both trainExamples and validationExamples
    to see how you're doing as you learn after each epoch.
    - The predictor should output +1 if the score is precisely 0.
    """
    weights = {}  # feature => weight
    # ### START CODE HERE ###
    for epoch in range(numEpochs):
        for x, y in trainExamples:
            phi_x = featureExtractor(x)
            score = dotProduct(weights, phi_x)
            margin = y * score
            predicted_y = 1 if score >= 0 else -1
            if margin < 1: # hinge loss is non-zero
                increment(weights, eta * y, phi_x) # weight <- weight - eta * (-1 * y * phi(x))
        train_error = evaluatePredictor(trainExamples, lambda x: 1 if dotProduct(weights, featureExtractor(x)) >= 0 else -1)
        validation_error = evaluatePredictor(validationExamples, lambda x: 1 if dotProduct(weights, featureExtractor(x)) >= 0 else -1)
        print(f"Epoch {epoch + 1}: Train Error = {train_error}, Validation Error = {validation_error}")
    # ### END CODE HERE ###
    return weights


############################################################
# Problem 1c: generate test case


def generateDataset(numExamples: int, weights: WeightVector) -> List[Example]:
    """
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    """
    random.seed(42)

    # Return a single example (phi(x), y).
    # phi(x) should be a dict whose keys are a subset of the keys in weights
    # and values can be anything (randomize!) with a score for the given weight vector.
    # y should be 1 or -1 as classified by the weight vector.
    # y should be 1 if the score is precisely 0.

    # Note that the weight vector can be arbitrary during testing.
    def generateExample() -> Tuple[Dict[str, int], int]:
        phi = None
        y = None
        # ### START CODE HERE ###
        phi = {}
        for feature in weights.keys():
            if random.random() < 0.5: # 5% chance to include each feature
                phi[feature] = random.randint(1, 5)  # Random integer weight between 1 and 5
        score = dotProduct(weights, phi)
        y = 1 if score >= 0 else -1
        # ### END CODE HERE ###
        return (phi, y)

    return [generateExample() for _ in range(numExamples)]


############################################################
# Problem 1d: character features


def extractCharacterFeatures(n: int) -> Callable[[str], FeatureVector]:
    """
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces mapped to their n-gram counts.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    """

    def extract(x):
        pass
        # ### START CODE HERE ###
        ngram_counts = DefaultDict(int)
        x_no_spaces = x.replace(" ", "")
        for i in range(len(x_no_spaces) - n + 1):
            ngram = x_no_spaces[i:i+n]
            ngram_counts[ngram] += 1
        return ngram_counts
        # ### END CODE HERE ###

    return extract


############################################################
# Problem 1e:
#
# Helper function to test 1e.
#
# To run this function, run the command from termial with `n` replaced
#
# $ python -c "from submission import *; testValuesOfN(n)"
#


def testValuesOfN(n: int):
    """
    Use this code to test different values of n for extractCharacterFeatures
    This code is exclusively for testing.
    Your full written solution for this problem must be submitted.
    """
    trainExamples = readExamples("polarity.train")
    validationExamples = readExamples("polarity.dev")
    featureExtractor = extractCharacterFeatures(n)
    weights = learnPredictor(
        trainExamples, validationExamples, featureExtractor, numEpochs=20, eta=0.01
    )
    outputWeights(weights, "weights")
    outputErrorAnalysis(
        validationExamples, featureExtractor, weights, "error-analysis"
    )  # Use this to debug
    trainError = evaluatePredictor(
        trainExamples,
        lambda x: (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1),
    )
    validationError = evaluatePredictor(
        validationExamples,
        lambda x: (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1),
    )
    print(
        (
            "Official: train error = %s, validation error = %s"
            % (trainError, validationError)
        )
    )


############################################################
# Problem 2b: K-means
############################################################


def kmeans(
    examples: List[Dict[str, float]], K: int, maxEpochs: int
) -> Tuple[List, List, float]:
    """
    examples: list of examples, each example is a string-to-float dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxEpochs: maximum number of epochs to run (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments (i.e. if examples[i] belongs to centers[j], then assignments[i] = j),
            final reconstruction loss)
    """
    # ### START CODE HERE ###
    centroids = random.sample(examples, K)
    examples_squared = [dotProduct(x, x) for x in examples]
    prev_assignments = [0]*len(examples)
    for epoch in range(maxEpochs):

        # Assignment step
        assignments_list = []
        assignment_dict = DefaultDict(list)
        centroids_squared = [dotProduct(c, c) for c in centroids]
        for i, example in enumerate(examples):
            min_distance = float('inf')
            for k in range(K):
                distance = examples_squared[i] + centroids_squared[k] - 2 * dotProduct(example, centroids[k])
                if distance < min_distance:
                    min_distance = distance
                    closest_cluster_index = k
            assignments_list.append(closest_cluster_index)
            assignment_dict[closest_cluster_index].append(example)
        
        # Update centroids step
        for cluster_index, cluster_points in assignment_dict.items():
            if cluster_points:
                new_centroid = DefaultDict(float)
                for point in cluster_points:
                    increment(new_centroid, 1.0 / len(cluster_points), point)
                centroids[cluster_index] = new_centroid
        
        # Calculate reconstruction loss
        loss = 0.0
        for x_in, x in enumerate(examples):
            assigned_centroid = assignments_list[x_in]
            loss += examples_squared[x_in] + centroids_squared[assigned_centroid] - 2 * dotProduct(x, centroids[assigned_centroid])

        print(f"Epoch {epoch + 1}: Loss = {loss}")

        #Check for convergence (if assignments do not change)
        if assignments_list == prev_assignments:
            break
        prev_assignments = assignments_list.copy()
        
    return centroids, assignments_list, loss
    # ### END CODE HERE ###
