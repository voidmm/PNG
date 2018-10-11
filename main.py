import time
import numpy as np
from plot import Plot
from probabilisticnaminggame import language_game
from multiprocessing import Pool, cpu_count


"""

Define parameters of the Probabilistic Naming Game

insert the path where the plots of the success history should be saved
decide whether you want to save the figure
the number of agents that partake in each game
the number of rounds defined the number of rounds of one probabilistic naming game
the reward that is applied to the probability of a word that was used by both agents to
name the object
the number of repeats defines the number of total, individual games that are played

"""

CPU_COUNT = cpu_count()

plot_path = ''

save_figure = False

number_of_agents = 100

number_of_rounds = 20000

reward = 10

number_of_repeats = 1000

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

    Plot = Plot(history, save_figure, plot_path, number_of_agents, number_of_repeats,
				 number_of_rounds, reward, mean_number_of_different_words, mean_number_of_total_words)
    Plot.plot_success_and_word_numbers()




