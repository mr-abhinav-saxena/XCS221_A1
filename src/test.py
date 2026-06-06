from submission import *

#####Testing different values of n for extractCharacterFeatures
n=15
for i in range(n):
    print("Testing for n =", i+1)
    testValuesOfN(i+1)

#####Checking average length of words in reviews
#def average_word_length(examples):
#    total_length = 0
#    total_words = 0
#    for example in examples:
#        words = example[0].split()
#        total_words += len(words)
#        total_length += sum(len(word) for word in words[1:]) # Exclude the label
#    return total_length / (total_words-len(examples)) if total_words > 0 else 0 # Exclude labels from word count

#validation_examples = readExamples("polarity.dev")
#print(average_word_length(validation_examples)) #4.49

#####Test pediction for a custom review example
#test_example_p = ['Acting is good , story is not bad', 1]  # Example review with positive sentiment
#test_example_n = ['Acting is bad , story is not good', -1]  # Example review with negative sentiment

#def test_prediction(example1, example2, trainExamples, validationExamples, featureExtractor):
#    weights = learnPredictor(
#        trainExamples, validationExamples, featureExtractor, numEpochs=20, eta=0.01
#    )
#    x1 = example1[0]
#    y1 = example1[1]
#    phi_x1 = featureExtractor(x1)
#    score1 = dotProduct(weights, phi_x1)
#    margin1 = y1 * score1
#    if margin1 >= 0:
#        pred1 = 1*y1
#        print(f"Correctly classified review example {example1} as {pred1} with margin {margin1}")
#    else:
#        pred1 = -1*y1
#        print(f"Incorrectly classified review example {example1} as {pred1} with margin {margin1}")
#
#    x2 = example2[0]
#    y2 = example2[1]
#    phi_x2 = featureExtractor(x2)
#    score2 = dotProduct(weights, phi_x2)
#    margin2 = y2 * score2
#    if margin2 >= 0:
#        pred2 = 1*y2
#        print(f"Correctly classified review example {example2} as {pred2} with margin {margin2}")
#    else:
#        pred2 = -1*y2
#        print(f"Incorrectly classified review example {example2} as {pred2} with margin {margin2}")

#print("Testing with extractCharacterFeatures(n=7):")
#test_prediction(example1 = test_example_p, example2 = test_example_n, trainExamples = readExamples("polarity.train"), validationExamples = readExamples("polarity.dev"), featureExtractor = extractCharacterFeatures(7))
#print("\nTesting with extractWordFeatures():")
#test_prediction(example1 = test_example_p, example2 = test_example_n, trainExamples = readExamples("polarity.train"), validationExamples = readExamples("polarity.dev"), featureExtractor = extractWordFeatures)
