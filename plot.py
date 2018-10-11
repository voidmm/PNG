import string
import matplotlib
matplotlib.use('TKAgg')
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.optimize import curve_fit
import pylab
import numpy as np
import pandas as pd


class Plot():

    def __init__(self, history, save_figure, path, number_of_agents, number_of_repeats,
                 number_of_rounds, reward, mean_number_of_different_words, mean_number_of_total_words):
        self.history = history
        self.save_figure = save_figure
        self.path = path
        self.number_of_agents = number_of_agents
        self.number_of_repeats = number_of_repeats
        self.number_of_rounds = number_of_rounds
        self.reward = reward
        self.extrapolation_factor = int(np.ceil(self.number_of_rounds / 10000))
        self.mean_number_of_different_words = mean_number_of_different_words
        self.mean_number_of_total_words = mean_number_of_total_words


    def sigmoid(self, x, a, b, c):
        np.seterr(all='warn')
        return 1 / (a * np.exp(-b * x) + 1)

    def markov_sigmoid(self, x, a, b, c, d):
        np.seterr(all='warn')
        return (1 / (a * np.exp(-b * x) + 1)) + d

    def fit_sigmoid(self, x, y):
        popt, pcov = curve_fit(self.sigmoid, x, y)
        a, b, c = popt
        return a, b, c

    def nonconvergent_sigmoid(self, x, a, b, c):
        np.seterr(all='warn')
        return c / (a * np.exp(-b * x) + 1)

    def fit_nonconvergent_sigmoid(self, x, y):
        popt, pcov = curve_fit(self.nonconvergent_sigmoid, x, y, p0=[900.64093468e+00, 1.25013021e-06, 3.37716955e-01])
        np.seterr(all='warn')
        return popt


    def save_figure(self, **kwargs):
        file_string = 'Agents {}, repeats {}, rounds {}, reward {}, graph {}, connects {}'

        for k, v in kwargs.items():
            file_string += ', ' + k + ' {' + k + '}'
        if self.save_figure:
            plt.savefig(self.path + file_string.format(self.number_of_agents, self.number_of_repeats,
                                                        self.number_of_rounds, self.reward,
                                                        self.language_construct_filename, **kwargs) + '.png',
                                                        dpi=180)

    def plot_success_and_word_numbers(self):
        original_number_of_rounds = self.number_of_rounds
        mean_history = np.sum(self.history, axis=0)
        
        if self.extrapolation_factor >= 2:
            mean_history = mean_history[0::self.extrapolation_factor]
            self.mean_number_of_different_words = self.mean_number_of_different_words[0::self.extrapolation_factor]
            self.mean_number_of_total_words = self.mean_number_of_total_words[0::self.extrapolation_factor]
            self.number_of_rounds = len(self.mean_number_of_total_words)
        
        _, ax = plt.subplots(3, 1, figsize=(8, 4))
        ml = AutoMinorLocator()

        total_number_of_games = np.multiply(np.ones(self.number_of_rounds), self.number_of_repeats)

        y = np.divide(mean_history, total_number_of_games)
        x = np.linspace(0, original_number_of_rounds, self.number_of_rounds)

        ax[2].plot(x, y, 'silver', linewidth=1.5)

        popt = self.fit_nonconvergent_sigmoid(x, y)
        a, b, c = popt
        print(popt)

        self.save_steels_to_archive(x, y, self.mean_number_of_different_words, self.mean_number_of_total_words)

        ax[2].plot(x, self.nonconvergent_sigmoid(x, a, b, c), color='navy', linewidth=1)
        ax[2].legend(('averaged success', 'success'))

        ax[2].set_xlabel('$t$')
        ax[2].set_ylabel('$S(t)$', rotation=90)
        ax[2].xaxis.set_minor_locator(ml)
        ax[2].set_yticks(np.arange(0, 1.25, 0.25))

        ax[1].plot(x, self.mean_number_of_different_words, 'navy')
        ax[1].set_ylabel('$N_{d}(t)$', rotation=90)
        ax[1].xaxis.set_minor_locator(ml)
        ax[1].set_yticks(np.linspace(0, self.mean_number_of_different_words[-1], 5, dtype=int))

        ax[0].plot(x, self.mean_number_of_total_words, 'navy')
        ax[0].set_ylabel('$N_{w}(t)$', rotation=90)
        ax[0].set_yticks(np.linspace(0, self.mean_number_of_total_words[-1], 5, dtype=int))
        ax[0].xaxis.set_minor_locator(ml)


        plt.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, wspace=0.2, hspace=0.3)
        plt.show()

    
    def save_steels_to_archive(self, x, y, diffWordNums, totalWordNums):
        filename = 'sigmoid data/'+ 'PNG  ' + str(self.number_of_agents)+', '+str(self.number_of_repeats)+', '+str(self.number_of_rounds)+', '+str(self.reward)
        np.savez(filename+'.npz',name1=x,name2=y, name3=diffWordNums,name4=totalWordNums)