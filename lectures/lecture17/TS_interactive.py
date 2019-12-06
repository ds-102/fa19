import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from matplotlib.widgets import Button
from matplotlib.widgets import CheckButtons
from matplotlib import gridspec
import functools
import scipy
matplotlib.use('Qt5Agg') 

sns.set()

class BanditEnv():

	def __init__(self,num_arms,means=None,standard_deviations=None):
		
		self.num_arms=num_arms
		
		if means is None:
			self.num_arms=num_arms
			self.means=list(10*np.random.random((num_arms-1)))
			self.means.insert(0,10)
			self.standard_deviations=list(1.5*np.ones((num_arms)))
		elif means=='rand':
			self.means=list(10*np.random.random((num_arms)))
			self.standard_deviations=list(1.5*np.ones((num_arms)))
		else:
			assert len(means)==len(standard_deviations)
			self.num_arms=len(means)
			self.means=means
			self.standard_deviations=standard_deviations

		return

	def pull_arm(self,arm):
		
		mu=self.means[arm]
		sigma=self.standard_deviations[arm]
		
		return np.random.normal(mu,sigma**2)

	def initialize(self):

		self.COLOURS=sns.color_palette("Set2",self.num_arms)
		self.COLOURS2=sns.color_palette("Paired")

		self.figure=plt.figure()
		gs = gridspec.GridSpec(2, 1, width_ratios=[1], height_ratios=[3, 1]) 
		gs2 = gridspec.GridSpec(1, 1, width_ratios=[1], height_ratios=[1]) 
		self.ax = self.figure.add_subplot(gs[0])
		self.ax2= self.figure.add_subplot(gs[1])
		self.axL= self.figure.add_subplot(gs2[0])
		self.axes=[self.ax,self.axL]

		self.ax.set_xticks([])
		self.ax.set_ylabel('Rewards',fontsize=15)
		self.axL.set_xticks([])
		self.axL.set_ylabel('Rewards',fontsize=15)
		self.figure.canvas.set_window_title('UCB Demo') 
		self.plot_dists()
		self.ax2.set_xlabel('Time (t)',fontsize=10)
		self.ax2.set_ylabel('Pseudo-regret',fontsize=10)

		return

	def plot_dists(self):
		separation=1.0/self.num_arms
		dist_maxes=[]
		for ax in self.axes:
			for arm in range(self.num_arms):
				mu=self.means[arm]
				sigma=self.standard_deviations[arm]
				rect = patches.Rectangle((0,-10),1,25,facecolor='w',zorder=0)
				ax.add_patch(rect)
				a=sns.kdeplot(np.random.normal(mu,sigma**2,(100000)), color=self.COLOURS[arm],shade=False,vertical=True,alpha=1.0,ax=ax,kernel='gau',lw=1.2,ls='-')
				ax.plot([-separation*(self.num_arms+0.5),0],[mu,mu],':',lw=3,c=self.COLOURS[arm])
				dist_maxes.append(np.max(a.get_lines()[0].get_data()[0]))

		self.dist_height=np.max(dist_maxes)
		return


	def toggle_dists(self,show_dists):
		separation=1.0/self.num_arms

		for ax in self.axes:
			if show_dists:
				ax.set_xlim([-separation*(self.num_arms+0.5),0.35])
			else:
				ax.set_xlim([-separation*(self.num_arms+0.5),-0.05])
		return
		
	def make_axes(self,show_dists,show_regret):
		
		self.toggle_dists(show_dists)

		if show_regret:
			self.ax.set_visible(True)
			self.ax2.set_visible(True)
			self.axL.set_visible(False)
			true_ax=self.ax
		else:
			self.ax.set_visible(False)
			self.ax2.set_visible(False)
			self.axL.set_visible(True)
			true_ax=self.axL
		self.ax.set_xticks([])
		self.axL.set_xticks([])
		return true_ax

	def get_axes(self):
		return self.axes

	def get_regret_ax(self,label):
		return self.ax2


class ThompsonSampling():
	
	def __init__(self,bandit_env,prior_means=None,prior_stds=None):
		
		self.bandit_env=bandit_env
		self.num_arms=self.bandit_env.num_arms
					
		self.times_pulled=None
		self.rewards=None
		self.buttons=[]

		
		if prior_means is None:
			self.prior_means=list(10*np.random.random((self.num_arms)))
			self.prior_std=list(np.random.random((self.num_arms))+2.0)
		else:
			assert(len(prior_means)==len(prior_stds) and len(prior_means)==self.num_arms)
			self.prior_means=prior_means
			self.prior_std=prior_stds

	def initialize(self,show=1):
		
		self.plot_content=None
		self.pseudo_regret=0
		self.regret=[0]
		self.times_pulled=[0 for arm in range(self.num_arms)]
		self.rewards=[[0] for arm in range(self.num_arms)]
		self.total_rewards=0
		self.means=[0 for arm in range(self.num_arms)]
		self.variances=[0 for arm in range(self.num_arms)]
		if show:
			self.bandit_env.initialize()


		for arm in range(self.num_arms): 
			self._update_posterior(arm)

		self._generate_samples(-1)
		return 

	def _generate_samples(self,arm):

		self.samples=[]
		for a in range(self.num_arms):
			self.samples.append(np.random.normal(self.means[a],self.variances[a],1))
		return

	def _update_posterior(self,arm):

		self.variances[arm]=1/(1/(self.prior_std[arm]**2)+self.times_pulled[arm]/(self.bandit_env.standard_deviations[arm]**2))
		
		self.means[arm]=self.variances[arm]*(self.prior_means[arm]/(self.prior_std[arm]**2)+np.sum(self.rewards[arm])/(self.bandit_env.standard_deviations[arm]**2))
		return

	def _log_info(self,arm,reward):
		self.times_pulled[arm]+=1
		self.rewards[arm].append(reward)
		self.total_rewards+=reward
		self.pseudo_regret+=np.max(self.bandit_env.means)-self.bandit_env.means[arm]
		self.regret.append(self.pseudo_regret)
		return


	def _TS_arm(self, event,make_buttons=1,plot=True):

		
		arm=np.argmax(self.samples)
		reward=self.bandit_env.pull_arm(arm)
		self._log_info(arm,reward)
		self._update_posterior(arm)
		self._generate_samples(arm)
		if plot:
			self._make_Plot(0,make_buttons)

		if make_buttons and plot:
			plt.draw()
		return 
		

	def _interactive_arm(self,event,pull=-1):

		arm=pull
		self.samples=[]
		self._generate_samples(arm)

		reward=self.bandit_env.pull_arm(arm)
		self._log_info(arm,reward)
		self._update_posterior(-1)
		self._make_Plot(0,make_buttons=1)
		return 

	def run_Interactive(self,dists=1,reg=1):
		
		self.initialize()
		self.show_regret=reg
		self.show_dists=dists
		self._make_Plot('start')
		plt.show()
		return self.regret

	def run_TS(self,T,dists=1,reg=1,init=1,show=0):
		
		if init:
			self.initialize(show)
		self.show_regret=reg
		self.show_dists=dists
		if show:
			self._make_Plot('start',0)
			plt.pause(1)

		for i in range(T):
			self._TS_arm(None,make_buttons=0,plot=show)
			if show:
				plt.pause(0.1)

		if show:
			plt.show()
		return self.regret



	def _make_buttons(self,true_ax):

		self.buttons=[]
		width,height, locations=self._get_Button_Locations(true_ax)

		for arm in range(self.num_arms):
			self.buttons.append(Button(plt.axes(locations[arm]),str(arm),color=self.bandit_env.COLOURS[arm], hovercolor=self.bandit_env.COLOURS2[1]))

			self.buttons[-1].on_clicked(functools.partial(self._interactive_arm, pull=arm))

		self.buttons.append(Button(plt.axes(locations[-1]),'TS',color=self.bandit_env.COLOURS2[0], hovercolor=self.bandit_env.COLOURS2[1]))

		self.buttons[-1].on_clicked(functools.partial(self._TS_arm))

		self.buttons.append(CheckButtons(plt.axes([0.2,0.88,0.12,0.12],facecolor='w'),('Distributions',),[self.show_dists]))
		self.buttons[-1].on_clicked(self._make_Plot)
		self.buttons.append(CheckButtons(plt.axes([0.5,0.88,0.12,0.12],facecolor='w'),('Regret',),[self.show_regret]))
		self.buttons[-1].on_clicked(self._make_Plot)
		return

	def remove_buttons(self):

		for button in self.buttons:
			self.bandit_env.figure.delaxes(button.ax)
		return

	def plot_Prior_Means(self):

		for ax in self.bandit_env.get_axes():
			for arm in range(self.num_arms):
				ax.plot([0,self.bandit_env.dist_height],[self.prior_means[arm],self.prior_means[arm]],color=self.bandit_env.COLOURS[arm],lw=2,ls='-')
		return

	def _get_Button_Locations(self,true_ax):

		locations=[]
		separation=1.0/self.num_arms
		axes_height=2.5
		axes_width=0.5*separation

		width,h=self.bandit_env.figure.transFigure.inverted().transform(true_ax.transData.transform([axes_width,0]))-self.bandit_env.figure.transFigure.inverted().transform(true_ax.transData.transform([0,0]))

		w,height=self.bandit_env.figure.transFigure.inverted().transform(true_ax.transData.transform([0,axes_height]))-self.bandit_env.figure.transFigure.inverted().transform(true_ax.transData.transform([0,0]))

		for arm in range(self.num_arms):
			location=-separation*(self.num_arms-arm)
			t1,t2=self.bandit_env.figure.transFigure.inverted().transform(true_ax.transData.transform([location-axes_width/2.0,-8.5]))
			locations.append([t1,t2,width,height])


		if self.show_dists:
			t1,t2=self.bandit_env.figure.transFigure.inverted().transform(true_ax.transData.transform([0.1,-8.5]))
			locations.append([t1, t2, 2*width, height])
		else:
			t1,t2=self.bandit_env.figure.transFigure.inverted().transform(true_ax.transData.transform([-0.5-2*axes_width,-12.5]))
			locations.append([t1, t2, 2*width, height])

		return width, height, locations

	def viewBounds(self,true_ax,make_buttons=1, fs=None):

		if fs is None:
			fs=min(60/(self.num_arms),10)

		if self.plot_content is not None:
			for line in self.plot_content:
				line.remove()

		self.plot_content=[]
		
		for arm in range(self.num_arms):
			separation=1.0/self.num_arms
			location=-separation*(self.num_arms-arm)

			upper=self.means[arm]+2*np.sqrt(self.variances[arm])
			lower=self.means[arm]-2*np.sqrt(self.variances[arm])

			self.plot_content.extend(true_ax.plot([location,location],[lower,upper],lw=2.5,c=self.bandit_env.COLOURS[arm]))

			self.plot_content.extend(true_ax.plot([-separation*(self.num_arms-arm-0.25),-separation*(self.num_arms-arm+0.25)],[lower,lower],'-',lw=3,c=self.bandit_env.COLOURS[arm]))

			self.plot_content.extend(true_ax.plot([-separation*(self.num_arms-arm-0.25),-separation*(self.num_arms-arm+0.25)],[upper,upper],'-',lw=3,c=self.bandit_env.COLOURS[arm]))

			self.plot_content.extend(true_ax.plot([location],[self.samples[arm]],'.',ms=15,c=self.bandit_env.COLOURS[arm]))
			
			if self.num_arms<=8:
				self.plot_content.append(true_ax.text(location, -3, r' Arm {}'.format(arm),ha='center',va='center',fontsize= fs))

				self.plot_content.append(true_ax.text(location, -4.6, r'$n_{}={}$'.format(arm,self.times_pulled[arm]),ha='center', va='center',fontsize= fs))


		if make_buttons:
			if self.show_dists:
				true_ax.set_ylim([-10,15])
			else:
				true_ax.set_ylim([-13,15])
		else:
			true_ax.set_ylim([-5.5,15])

		ax=self.bandit_env.get_regret_ax('TS')

		self.plot_content.extend(ax.plot(self.regret,'--',c=self.bandit_env.COLOURS2[1],label='TS'))
		return

	def _make_Plot(self,label, make_buttons=1):

		changefig=0
		if label=='Distributions':
			self.show_dists=1-self.show_dists
			self.true_ax=self.bandit_env.make_axes(self.show_dists,self.show_regret)
			changefig=1
		elif label=='Regret':
			self.show_regret=1-self.show_regret
			self.true_ax=self.bandit_env.make_axes(self.show_dists,self.show_regret)
			changefig=1
		elif label=='start':
			self.plot_Prior_Means()
			self.true_ax=self.bandit_env.make_axes(self.show_dists,self.show_regret)
			changefig=1
			
		self.viewBounds(self.true_ax, make_buttons)
		if changefig and make_buttons:
			self.remove_buttons()
			self._make_buttons(self.true_ax)
		plt.draw()
		return

if __name__ == "__main__":
	
	num_arms=10
	T=200
	num_runs=30
	show=1
	env=BanditEnv(num_arms, 'rand')

	alg1=ThompsonSampling(env)
	TS_regret=0
	for run in range(num_runs):
		print('UCB Run: '+str(run))
		TS_regret+=np.array(alg1.run_TS(200,1,1,show=1))


	plt.plot(TS_regret/num_runs)
		







