import time
import numpy as np
from plot import Plot
from probabilisticnaminggame import language_game
from multiprocessing import Pool, cpu_count



CPU_COUNT = cpu_count()

params = eval(open('parameters.txt').read())

reward = params['reward']
number_of_repeats = params['number_of_repeats']
number_of_agents = params['number_of_agents']
number_of_rounds = params['number_of_rounds']


if number_of_repeats % CPU_COUNT != 0:
	number_of_repeats -= number_of_repeats % CPU_COUNT


word_counter_container = {'different words of all games': [], 'total number of words of all games': []}


def games(number_of_repeats, word_counter_container):
	aggregated_history=[]

	for i in range(number_of_repeats):
		history, word_counters = language_game(number_of_agents, number_of_rounds, reward)
		aggregated_history.append(history)
		word_counter_container['different words of all games'].append(word_counters['number of different words'])
		word_counter_container['total number of words of all games'].append(word_counters['total number of words'])

	return aggregated_history, word_counter_container


if __name__ == "__main__":
    start = time.time()

    p = Pool()
    size_of_game_batch = number_of_repeats / CPU_COUNT
    all_parallel_games = [int(size_of_game_batch) for i in range(CPU_COUNT)]

    output = [p.apply_async(games, args=(x, word_counter_container)) for x in all_parallel_games]
    p.close()
    p.join()

    history = [p.get()[0] for p in output]
    history = np.reshape(np.ravel(history, order='A'), (number_of_repeats, number_of_rounds))
    mean_number_of_different_words = np.mean([p.get()[1]['different words of all games'] for p in output], axis=0, dtype=int)[0]
    mean_number_of_total_words = np.mean([p.get()[1]['total number of words of all games'] for p in output], axis=0, dtype=int)[0]

    end = time.time()
    print('Total time in seconds: '+ str( end - start))

    Plot = Plot()
    Plot.plot_success_and_word_numbers(history, mean_number_of_different_words, mean_number_of_total_words)




