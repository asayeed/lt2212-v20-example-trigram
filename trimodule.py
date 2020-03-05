import sys
import os
import random

class TrigramModel:
    def __init__(self, inputstring):
        self.tridict = {}
        for position in range(len(inputstring) - 2):
            char0 = inputstring[position]
            char1 = inputstring[position + 1]
            char2 = inputstring[position + 2]

            if char0 in self.tridict:
                if char1 in self.tridict[char0]:
                    if char2 in self.tridict[char0][char1]:
                        self.tridict[char0][char1][char2] += 1
                    else:
                        self.tridict[char0][char1][char2] = 1
                else:
                    self.tridict[char0][char1] = {}
                    self.tridict[char0][char1][char2] = 1
            else:
                self.tridict[char0] = {}
                self.tridict[char0][char1] = {}
                self.tridict[char0][char1][char2] = 1

        self.probdict = {}
        for char0 in self.tridict.keys():
            if char0 not in self.probdict:
                self.probdict[char0] = {}

            for char1 in self.tridict[char0].keys():
                if char1 not in self.probdict[char0]:
                    self.probdict[char0][char1] = {}

                for char2 in self.tridict[char0][char1].keys():
                    fullcount = sum(self.tridict[char0][char1].values())
                    self.probdict[char0][char1][char2] = self.tridict[char0][char1][char2]/fullcount
        
    def __getitem__(self, item):
        if len(item) != 3:
            raise ValueError("Must be exactly 3 chars.")
        return self.probdict[item[0]][item[1]][item[2]]

class TrigramModelWithDistribution(TrigramModel):
    def predict(self, n, seed):
        '''
        Predicts n characters using random sampling from the 
        distribution starting with the seed.
        '''
        if len(seed) != 2:
            raise ValueError("Need exactly two characters for prediction.")

        inputchar0 = seed[0]
        inputchar1 = seed[1]

        outputstring = "{}{}".format(inputchar0,inputchar1)
        for output in range(n):
            choices = self.probdict[inputchar0][inputchar1]
            randomval = random.random()
            total = 0
            mychoice = ''
            for key in choices:
                total += choices[key]
                if randomval < total:
                    mychoice = key
                    break
        #options = sorted(choices.keys(), key=lambda x: choices[x], reverse=True)
        #print(options)
            outputstring += mychoice
            inputchar0 = inputchar1
            inputchar1 = mychoice

        return outputstring

class TrigramModelWithTopK(TrigramModel):
    def __init__(self, inputstring, k=5):
        super().__init__(inputstring)
        self.k = k

    def get_choices(self, inputchar0, inputchar1):
        choices = self.probdict[inputchar0][inputchar1]
        return sorted(choices.keys(), key=lambda x: choices[x], reverse=True)
    
    def predict(self, n, seed):
        '''
        Predicts n characters using random sampling from the 
        distribution starting with the seed.
        '''
        if len(seed) != 2:
            raise ValueError("Need exactly two characters for prediction.")

        inputchar0 = seed[0]
        inputchar1 = seed[1]

        outputstring = "{}{}".format(inputchar0,inputchar1)
        for output in range(n):
            options = self.get_choices(inputchar0, inputchar1)
            mychoice = random.choice(options[:5])
        #print(options)
            outputstring += mychoice
            inputchar0 = inputchar1
            inputchar1 = mychoice

        return outputstring

import numpy as np
import random
from sklearn.linear_model import LogisticRegression
    
class TrigramMaxEnt(TrigramModelWithTopK):
    def sample(self, n):
        samples = []
        for i in range(n):
            char1 = random.choice(list(self.tridict.keys()))
            char2 = random.choice(list(self.tridict[char1].keys()))
            char3 = random.choice(list(self.tridict[char1][char2].keys()))
            count = self.tridict[char1][char2][char3]
            
            samples.append((char1, char2, char3, count))
            
        return samples
    
    def vectorize(self, feature, features):
        empty = np.zeros(len(features))
        empty[features.index(feature)] = 1
        
        return empty
    
    def __init__(self, inputstring):
        super().__init__(inputstring)
        self.model = LogisticRegression()
        
        self.features = set()
        for x in self.tridict.keys():
            self.features.add(x)
        self.features = list(self.features)

    def process_samples(self, instances):
        X = [x[0] for x in instances]
        y = [x[1] for x in instances]
        sample_weights = [x[2] for x in instances]
        return X, y, sample_weights
        
    def train(self, num_samples):
        samples = self.sample(num_samples)
        #print(samples, len(samples))
        instances = []
        for s in samples:
            char1inst = self.vectorize(s[0], self.features)
            char2inst = self.vectorize(s[1], self.features)
            
            instances.append((np.concatenate([char1inst, char2inst]), s[2], s[3]))
            
        X, y, sample_weights = self.process_samples(instances)
        
        self.model.fit(X, y, sample_weights)
        
    def get_choices(self, inputchar0, inputchar1):
        inputvec = np.concatenate([self.vectorize(inputchar0, self.features), self.vectorize(inputchar1, self.features)])
        
        predictions = self.model.predict_log_proba([inputvec])
        #print(-predictions, len(-predictions))
        sortedargs = np.argsort(-predictions[0])
        #print(sortedargs, len(sortedargs))
        return [self.model.classes_[x] for x in sortedargs]

    def perplexity(self, text):
        samples_X = []
        samples_y = []
        for i in range(len(text) - 2):
            char0 = text[i]
            char1 = text[i+1]
            char2 = text[i+2]

            samples_X.append(np.concatenate([self.vectorize(char0, self.features), self.vectorize(char1, self.features)]))
            samples_y.append(self.model.classes_.index(char2))

        predictions = self.model.predict_log_proba(samples_X)
        logprobs = [x[0][x[1]] for x in zip(predictions, samples_y)]
        

class TrigramMaxEntExpandSamples(TrigramMaxEnt):
    def process_samples(self, instances):
        X = [x[0] for x in instances]
        y = [x[1] for x in instances]
        sample_weights = [x[2] for x in instances]

        real_X = []
        real_y = []
        real_weights = []
        for i in range(len(instances)):
            multiplier = sample_weights[i]
            real_X += [X[i]] * multiplier
            real_y += [y[i]] * multiplier
            real_weights += [1] * multiplier

        return (real_X, real_y, real_weights)
