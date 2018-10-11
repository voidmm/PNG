import random
import numpy as np
import time
from memoryfunctions import update_word_probabilities, normalize


def language_game(number_of_agents, number_of_rounds, reward):
    word_counters = {'number of different words': [], 'total number of words': []}
    number_of_unique_words = 0
    total_number_of_words = 0
    history = []
    words = []
    agents = {}

    for agentint in range(number_of_agents):
        agents[str(agentint)] = {'word memory': [], 'word probabilities': []}

    def add_word(agent, word):
        agent['word probabilities'] = normalize(np.append(agent['word probabilities'], 1))
        agent['word memory'].append(word)

    def get_random_word(agent):
        if len(agent['word memory']) != 1:
            word = np.random.choice(a=agent['word memory'], p=agent['word probabilities'])
        else:
            word = agent['word memory'][0]

        return word

    def communicate(speaker, hearer, word, total_number_of_words, reward):
        success = 0
        if word not in hearer['word memory']:
            hearer = add_word(hearer, word)
            total_number_of_words += 1
        else:
            hearer_word = get_random_word(hearer)

            if hearer_word != word:
                pass
            else:
                success = 1
                hearer['word probabilities'] = update_word_probabilities(hearer['word probabilities'],
                                                                         hearer['word memory'].index(word),
                                                                         reward)
                speaker['word probabilities'] = update_word_probabilities(speaker['word probabilities'],
                                                                          speaker['word memory'].index(word),
                                                                          reward)
        return success, total_number_of_words

    def communicate_teach(speaker, hearer, word, total_number_of_words, reward):
        success = 0
        if word not in hearer['word memory']:
            hearer = add_word(hearer, word)
            total_number_of_words += 1
        else:
            hearer_word = get_random_word(hearer)
            if hearer_word != word:
                hearer['word probabilities'] = update_word_probabilities(hearer['word probabilities'],
                                                                         hearer['word memory'].index(word),
                                                                         reward)
            else:
                success = 1
                hearer['word probabilities'] = update_word_probabilities(hearer['word probabilities'],
                                                                         hearer['word memory'].index(word),
                                                                         reward)
                speaker['word probabilities'] = update_word_probabilities(speaker['word probabilities'],
                                                                          speaker['word memory'].index(word),
                                                                          reward)
        return success, total_number_of_words

    for i in range(number_of_rounds):

        random_agents = random.sample(range(number_of_agents), 2)
        speaker_id = random_agents[0]
        hearer_id = random_agents[1]
        speaker = agents[str(speaker_id)]
        hearer = agents[str(hearer_id)]

        if len(speaker['word memory']) != 0:
            word = get_random_word(speaker)
        else:
            word = number_of_unique_words
            number_of_unique_words += 1
            total_number_of_words += 1
            add_word(speaker, word)

        success, total_number_of_words = communicate(speaker, hearer, word, total_number_of_words, reward)
        history.append(success)

        word_counters['number of different words'].append(number_of_unique_words)
        word_counters['total number of words'].append(total_number_of_words)

    return history, word_counters

# TEST
# language_game(10,20,1)
