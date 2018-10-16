import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('TKAgg')
from matplotlib import rcParams
from matplotlib.ticker import AutoMinorLocator
from matplotlib import pyplot as plt
import os
rcParams['font.family'] = 'Arial'
from scipy import stats


class CurveAnalysis():

    def __init__(self, foldername, convergent):
        self.foldername = foldername
        self.convergent = convergent
        self.figfoldername = 'fitted sigmoid plots/'
        self.filename = ''

    def read_steels(self, filename):
        container = np.load(filename)
        data = [container[key] for key in container]
        self.filename = filename
        return data

    def sigmoid(self, x, a, b, c):
        np.seterr(all='warn')
        return 1/(a * np.exp(-b * x) + c)

    def nonconvergent_sigmoid(self, x, a, b, c):
        np.seterr(all='warn')
        return c/(a * np.exp(-b * x) + 1)

    def fit_sigmoid(self,x,y):
        popt, pcov = curve_fit(self.sigmoid, x, y, p0=[2000,0.0000936072, 0.999])
        np.seterr(all='warn')
        return popt, pcov

    def fit_nonconvergent_sigmoid(self,x,y):
        if len(x)>100000:
            popt, pcov = curve_fit(self.nonconvergent_sigmoid, x, y,p0=[  9.64093468e+00,   1.25013021e-03,  3.37716955e-01])
        else:
            popt, pcov = curve_fit(self.nonconvergent_sigmoid, x, y, p0=[900.64093468e+00, 1.25013021e-06, 3.37716955e-01])
        np.seterr(all='warn')
        return popt, pcov

    def custom_div_cmap(self,numcolors=80, name='custom_div_cmap',
                        mincol='white', midcol='#9dbfeb', maxcol='navy'):
        cmap = LinearSegmentedColormap.from_list(name=name,
                                                 colors=[mincol, midcol, maxcol],
                                                 N=numcolors)
        return cmap


    def save_figure(self, **kwargs):
        file_string = '{}'

        for k, v in kwargs.items():
            file_string += ', ' + k + ' {' + v + '}'
        
        plt.savefig(self.figfoldername + file_string.format(self.filename.split('.npz')[0].rsplit('/')[1], **kwargs) + '.png',
                        dpi=300)


    def plot_sigmoid(self,x,y,popt, diffwordNums, totalWordNums, save):
        _, ax = plt.subplots(3,1,figsize=(8,4))
        ml = AutoMinorLocator()
        ax[2].plot(x,y, 'silver', linewidth=1.5)

        if self.convergent:
            ax[2].plot(x, self.sigmoid(x, *popt), color='navy', linewidth=1)
        else:
            ax[2].plot(x, self.nonconvergent_sigmoid(x, *popt), color='navy', linewidth=1)

        ax[2].legend(('averaged success', 'success'))
        ax[2].set_xlabel('$t$')
        ax[2].set_ylabel('$S(t)$', rotation=90)
        ax[2].xaxis.set_minor_locator(ml)
        ax[2].set_yticks(np.arange(0, 1.25, 0.25))

        ax[1].plot(x, diffWordNums, 'navy')
        ax[1].set_ylabel('$N_{d}(t)$', rotation=90)
        ax[1].xaxis.set_minor_locator(ml)
        def rnd(x): return int(round(x,-1))
        ax[1].set_yticks(np.linspace(0, diffWordNums[-1], 5,dtype=int))

        ax[0].plot(x, totalWordNums, 'navy')
        ax[0].set_ylabel('$N_{w}(t)$', rotation=90)
        ax[0].set_yticks(list(map(rnd,np.linspace(0, totalWordNums[-1], 5, dtype=int))))
        ax[0].xaxis.set_minor_locator(ml)
        ax[0].tick_params(axis='y', which='major', labelsize=8)

        plt.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, wspace=0.2, hspace=0.3)

        if save:
            kwargs = {'type': 'sigmoid'}
            self.save_figure(**kwargs)

        plt.show()


    def extract_t_conv(self,x, y):
        try:
            first_conv = int(np.where(y == 1.0)[0][0])
            y_conv = int(np.mean(np.where(y == 1.0)[0]))
            print('variance: ', np.var(y[y_conv:], dtype=np.float32), 'mean: ', np.mean(y[y_conv:]))
            #print('\n')
            t_conv=x[y_conv]
            print('first_conv: ',x[first_conv])
            print('mean_conv: ', t_conv)
        except IndexError:
            print('not entirely converged, final value: ')
            print(y[-10:])
            y_conv = len(y)-20
            print('variance: ', np.var(y[y_conv:], dtype=np.float32), 'mean: ', np.mean(y[y_conv:]))
            t_conv=x[y_conv]
        return t_conv


    def lin_regress(self, x, y):
        x = np.log(x)
        y = np.log(y)
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        line = slope * x + intercept
        return line, x, y


    def plot_multiple_sigmoid(self,save):
        _, ax = plt.subplots(figsize=(12, 6))
        ml = AutoMinorLocator()
        colors = [ 'navy','#008080', '#800000', '#008040', '#800080',  '#808000']
        labels = ['1.0', '1.2', '1.4', '1.6', '1.8', '2.0']
        #labels = ['0.4', '2', '10']
        i=0
        for file in sorted(os.listdir(self.foldername)):
            filename = os.fsdecode(file);print(filename)
            if filename.endswith(".npz"):
                file = self.read_steels(self.foldername+filename)
                x = file[0]
                y = file[1];print(len(y))

                if self.convergent:
                    popt, pcov = self.fit_sigmoid(x, y)
                    ax.plot(x, y, 'silver', linewidth=1.5, alpha=0.5, zorder=0)
                    ax.plot(x, self.sigmoid(x, *popt), color=colors[i], linewidth=1.5, label='$\delta$='+labels[i], zorder=10)
                    #ax.plot(x, ca.sigmoid(x, *popt),  linewidth=1.5)
                else:
                    popt, pcov = self.fit_nonconvergent_sigmoid(x, y)
                    ax.plot(x, y, 'silver', linewidth=1.5, alpha=0.5, zorder=0);print(popt)
                    ax.plot(x, self.nonconvergent_sigmoid(x, *popt), color=colors[i],  linewidth=1.5,  label='$\delta$='+labels[i], zorder=10)
                i+=1
                
        ax.set_xlabel('$t$')
        ax.set_ylabel('$S(t)$', rotation=90)
        ax.xaxis.set_minor_locator(ml)
        ax.legend(prop={'size': 15})
        ax.set_yticks(np.arange(0, 1.25, 0.25))
        if save:
            plt.savefig(self.figfoldername + '/different_rewards_Arial.png', dpi=300)
        plt.show()


    def plot_t_max(self, save):
        rewards = []
        maxWords = []

        for file in os.listdir(self.foldername):
            filename = os.fsdecode(file)
            if filename.endswith(".npz"):
                file = self.read_steels(self.foldername+filename)
                totalWordNums=file[3]
                maxWords.append(totalWordNums[-1])
                rewards.append(float(filename.split('.npz')[0].split('10000, ')[1].split(', 0')[0]))
        print(maxWords)

        a = [[x, y] for (y, x) in sorted(zip(maxWords, rewards), key=lambda pair: pair[0])]
        a = np.transpose(a)
        rewards = a[0]
        maxWords = a[1]
        fig, ax = plt.subplots(figsize=(5, 5))

        ml = AutoMinorLocator()

        ax.scatter(maxWords, rewards, marker='.', color='navy', s=50)
        
        ax.grid(which='major', linestyle='-', linewidth='0.5', color='silver')
        ax.grid(which='minor', linestyle='--', linewidth='0.5', color='silver')
        ax.set_ylabel('$\delta$', rotation=90)
        ax.set_xlabel('$N_w(t)$')

        ax.set_axisbelow(True)
        ax.xaxis.set_minor_locator(ml)
        rp = [1.0, 1.3999999999999999, 2.0, 3.0, 4.0, 5.0, 6.0,8.0, 10.0]
        ax.set_yticks(rp)

        if save:
            plt.savefig(self.figfoldername + '/tmaxAndrewardArial2.png', dpi=300)
        plt.show()

    def plot_a_reward(self, save):
        ml = AutoMinorLocator()
        rewards = []
        aas = []
        for file in os.listdir(self.foldername):
            filename = os.fsdecode(file)
            if filename.endswith(".npz"):
                file = self.read_steels(self.foldername+filename)
                x = file[0]
                y = file[1]
                if self.convergent:
                    popt, pcov = self.fit_sigmoid(x, y);print(popt, filename)
                    #self.plot_sigmoid(x, y, popt, file[2], file[3])
                else:
                    popt, pcov = self.fit_nonconvergent_sigmoid(x, y)
                aas.append(popt[0])
                rewards.append(float(filename.split('.npz')[0].split('10000, ')[1].split(', 0')[0]))
        
        a = [[x, y] for (y, x) in sorted(zip(aas, rewards), key=lambda pair: pair[0])]
        a = np.transpose(a)
        rewards = a[0]
        a_values = a[1]
        fig, ax = plt.subplots(figsize=(5, 5))
        
        line, maxWords, rewards = self.lin_regress(a_values, rewards)
        ax.scatter(a_values, rewards, marker='.', color='navy', s=50)
        

        #ax.plot(a_values, line, linewidth=1, color='gray')
        ax.set_axisbelow(True)
        ax.grid(which='major', linestyle='-', linewidth='0.5', color='silver')
        ax.grid(which='minor', linestyle='--', linewidth='0.5', color='silver')
        ax.set_ylabel('$\delta$', rotation=90)
        ax.set_xlabel('$a$')
        ax.xaxis.set_minor_locator(ml)

        rp = [1.0, 1.4, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]

        #ax.set_yticks(np.log([1.0, 1.4, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]))
        ax.set_yticklabels(rp)
        #ax.set_xticks(np.log([500, 1000, 1500, 2000, 2500]))
        ax.set_xticklabels([500, 1000, 1500, 2000, 2500])

        #ax.ticklabel_format(style='sci',scilimits=(0,0),axis='x')
        if save:
            plt.savefig(self.figfoldername + '/PNGreward_220000rounds_Arial.png', dpi=300)
        plt.show()


    def plot_inflectionpoints_reward(self, save):
        rewards = []
        inflection_points = []
        for file in os.listdir(self.foldername):
            filename = os.fsdecode(file)
            if filename.endswith(".npz"):
                file = self.read_steels(self.foldername+filename)
                x = file[0]
                y = file[1]
                if self.convergent:
                    popt, pcov = self.fit_sigmoid(x, y);print(popt, filename)
                else:
                    popt, pcov = self.fit_nonconvergent_sigmoid(x, y)
                inflection_points.append(np.log(popt[0])/popt[1])
                rewards.append(float(filename.split('.npz')[0].split('10000, ')[1].split(', 0')[0]))
        
        ip = [[x, y] for (y, x) in sorted(zip(inflection_points, rewards), key=lambda pair: pair[0])]
        ip = np.transpose(ip)
        rewards = ip[0]
        maxWords = ip[1]
        fig, ax = plt.subplots(figsize=(5, 5))

        line, maxWords, rewards = self.lin_regress(maxWords, rewards)

        ml = AutoMinorLocator()
        ax.scatter(maxWords, rewards, marker='.', color='navy', s=50)

        ax.plot(maxWords, line, linewidth=1, color='gray')
        ax.set_axisbelow(True)
        ax.grid(which='major', linestyle='-', linewidth='0.5', color='silver')
        ax.grid(which='minor', linestyle='--', linewidth='0.5', color='silver')
        ax.set_ylabel('$\delta$', rotation=90)
        ax.set_xlabel('$ \\frac{ln(a)}{b}$')
        ax.xaxis.set_minor_locator(ml)

        rp = [1.0, 1.4, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]

        ax.set_yticks(np.log([1.0, 1.4, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]))
        ax.set_yticklabels(rp)

        ax.set_xticks(np.log([40000, 60000, 70000, 80000, 90000, 110000]))
        
        ax.set_xticklabels([40000, 60000, 70000, 80000, 90000, 110000])

        #ax.ticklabel_format(style='sci',scilimits=(0,0),axis='x')

        if save:
            plt.savefig(self.figfoldername +'/logPNGreward_inflectionpoints_Arial.png', dpi=300)
        plt.show()


    def plot_c_reward(self, save):
        rewards = []
        cs = []
        for file in os.listdir(self.foldername):
            filename = os.fsdecode(file)
            if filename.endswith(".npz"):
                file = self.read_steels(self.foldername+filename)
                x = file[0]
                y = file[1]
                if self.convergent:
                    popt, pcov = self.fit_sigmoid(x, y)
                else:
                    popt, pcov = self.fit_nonconvergent_sigmoid(x, y)
                cs.append(popt[2])
                rewards.append(float(filename.split(', 0.npz')[0].split(', ')[-1]))
        c = [[x, y] for (y, x) in sorted(zip(cs, rewards), key=lambda pair: pair[0])]
        c = np.transpose(c)
        rewards = c[0] 
        c = c[1]
        fig, ax = plt.subplots(figsize=(5, 5))
        
        line, c, rewards = self.lin_regress(c, rewards)
        ml = AutoMinorLocator()
        ax.scatter(c, rewards, marker='.', color='navy', s=50)

        ax.plot(c, line, linewidth=1, color='gray')
        ax.set_axisbelow(True)
        ax.grid(which='major', linestyle='-', linewidth='0.5', color='silver')
        ax.grid(which='minor', linestyle='--', linewidth='0.5', color='silver')
        ax.set_ylabel('$\delta$', rotation=90)
        ax.set_xlabel('$c$')

        rp = [0.25,0.275,0.3,0.325,0.35,0.375,0.4]

        ax.set_yticks(np.log([0.5,1,2,3,4,5,6,7,8,9,10]))
        ax.set_xticks(np.log(rp))
        ax.set_xticklabels(rp)
        ax.set_yticklabels([0.5,1,2,3,4,5,6,7,8,9,10])
        ax.xaxis.set_minor_locator(ml)

        if save:
            plt.savefig(self.figfoldername + '/logcrewardArial.png', dpi=300)
        plt.show()


    def plot_c_agents(self, save):
        agents = []
        cs = []
        for file in os.listdir(self.foldername):
            filename = os.fsdecode(file)
            if filename.endswith(".npz"):
                file = self.read_steels(self.foldername+filename)
                x = file[0]
                y = file[1]
                popt, pcov = self.fit_nonconvergent_sigmoid(x, y)
                cs.append(popt[2]);print(popt[2])
                agents.append(int(filename.split('NonConvsteels ')[1].split(', 1000,')[0]));print(agents[-1])
        
        c = [[x, y] for (y, x) in sorted(zip(cs, agents), key=lambda pair: pair[0])]
        c = np.transpose(c)
        agents = c[0]
        c = c[1]
        fig, ax = plt.subplots(figsize=(5, 5))

        line, c, agents = self.lin_regress(c, agents)

        ml = AutoMinorLocator()
        ax.scatter(c, agents, marker='.', color='navy', s=50)
        ax.plot(c, line, linewidth=1, color='gray')
        ax.set_axisbelow(True)
        ax.grid(which='major', linestyle='-', linewidth='0.5', color='silver')
        ax.grid(which='minor', linestyle='--', linewidth='0.5', color='silver')
        ax.set_xlabel('$c$')
        ax.set_ylabel('$N$')

        rp =[50,100,150, 200, 300, 400, 500, 600, 700, 800, 1000]
        ax.set_yticks(np.log(rp))
        ax.set_yticklabels(rp)
        ax.set_xticks(np.log([0.125,0.15,0.2,0.25,0.3,0.35,0.4]))
        ax.set_xticklabels([0.125,0.15,0.2,0.25,0.3,0.35,0.4])
        ax.xaxis.set_minor_locator(ml)

        if save:
            plt.savefig(self.figfoldername + '/logc_agentsArial.png', dpi=300)

        plt.show()


if __name__ == "__main__":
    foldername='rewards_leading_figure/'
    CA = CurveAnalysis(foldername, convergent=True)
    file = CA.read_steels(foldername+'PNG  1000, 504, 10000, 1.6.npz')
    x = file[0]
    y = file[1]
    popt, pcov = CA.fit_nonconvergent_sigmoid(x,y);print(popt)

    diffWordNums = file[2]
    totalWordNums = file[3]

    #CA.plot_sigmoid(x,y,popt,diffWordNums,totalWordNums, save=False)
    #CA.plot_c_reward(save=False)
    #CA.plot_c_agents(save=False)
    #CA.plot_a_reward(save=False)
    #CA.plot_t_max(save=False)
    #CA.plot_inflectionpoints_reward('transition_rounds/')
    CA.plot_multiple_sigmoid(save=False)
