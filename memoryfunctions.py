import numpy as np 


def update_word_probabilities(word_probabilities, index, reward):
	word_probabilities[index]+=reward
	summ = sum(word_probabilities)
	corrected_word_probabilities = list(map(lambda i: float(i)/summ, word_probabilities))
	return corrected_word_probabilities



def normalize(vector):
    summ =  sum(vector) 
    return list(map(lambda i: float(i)/summ, vector))


#TEST
#print(update_word_probabilities([0.5,0.5],1,1))
