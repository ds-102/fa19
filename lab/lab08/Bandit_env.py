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

sns.set()

class BanditEnv():

	def __init__(self,means,standard_deviations=None):
		
		self.num_arms=len(means)
		assert len(means)==len(standard_deviations)

		self.means=np.array(means)
		self.standard_deviations=np.array(standard_deviations)
		self.best_arm=np.argmax(self.means)

		return

	def _interactive_pull_arm(self,event,pull):
		
		arm=pull
		mu=self.means[arm]
		sigma=self.standard_deviations[arm]
		reward=np.random.normal(mu,sigma**2)
		self.rewards[arm].append(reward)
		self.regret.append(self.regret[-1]+self.means[self.best_arm]-self.means[arm])
		self.times_pulled[arm]+=1
		self._make_Plot(0)
		self.t+=1
		return 

	def pull_arm(self,pull):

		arm=pull
		mu=self.means[arm]
		sigma=self.standard_deviations[arm]
		reward=np.random.normal(mu,sigma**2)
		self.rewards[arm].append(reward)
		self.regret.append(self.regret[-1]+self.means[self.best_arm]-self.means[arm])
		self.times_pulled[arm]+=1
		self.t+=1
		return

	def initialize(self,make_plot=1,alg_defined=0,alg=None):

		self.COLOURS=sns.color_palette("Set2",self.num_arms)
		self.COLOURS2=sns.color_palette("Paired")

		if make_plot:
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
			self.figure.canvas.set_window_title('Bandits Demo') 
			self._plot_dists()
			self.ax2.set_xlabel('Time (t)',fontsize=10)
			self.ax2.set_ylabel('Pseudo-regret',fontsize=10)
			self.plot_content=[]
		self.rewards=[[] for arm in range(self.num_arms)]
		self.regret=[0]
		self.times_pulled=[0 for arm in range(self.num_arms)]
		self.buttons=[]
		self.t=1
		self.alg_defined=alg_defined
		self.alg=alg

		return


	def _plot_dists(self):
		separation=1.0/self.num_arms
		dist_maxes=[]
		for ax in self.axes:
			for arm in range(self.num_arms):
				mu=self.means[arm]
				sigma=self.standard_deviations[arm]
				rect = patches.Rectangle((0,-10),1,25,facecolor='w',zorder=0)
				ax.add_patch(rect)
				a=sns.kdeplot(np.random.normal(mu,sigma**2,(100000)), color=self.COLOURS[arm],shade=False,vertical=True,alpha=1.0,ax=ax,kernel='gau',lw=1.2,ls='-')
				ax.plot([-separation*(self.num_arms+0.5),0],[mu,mu],'--',lw=1.0,c=self.COLOURS[arm])
				dist_maxes.append(np.max(a.get_lines()[0].get_data()[0]))

		self.dist_height=np.max(dist_maxes)
		return


	def _toggle_dists(self,show_dists):
		separation=1.0/self.num_arms

		for ax in self.axes:
			if show_dists:
				ax.set_xlim([-separation*(self.num_arms+0.5),0.35])
			else:
				ax.set_xlim([-separation*(self.num_arms+0.5),-0.05])
		return
		
	def _make_axes(self,show_dists,show_regret):
		
		self._toggle_dists(show_dists)

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

	def _get_arm_axes(self):
		return self.axes

	def _get_regret_ax(self,label):
		return self.ax2

	def _make_buttons(self,true_ax):

		self.buttons=[]
		width,height, locations=self._get_Button_Locations(true_ax)

		for arm in range(self.num_arms):
			self.buttons.append(Button(plt.axes(locations[arm]),str(arm),color=self.COLOURS[arm], hovercolor=self.COLOURS2[1]))

			self.buttons[-1].on_clicked(functools.partial(self._interactive_pull_arm, pull=arm))


		self.buttons.append(CheckButtons(plt.axes([0.2,0.88,0.12,0.12],facecolor='w'),('Distributions',),[self.show_dists]))
		self.buttons[-1].on_clicked(self._make_Plot)
		self.buttons.append(CheckButtons(plt.axes([0.5,0.88,0.12,0.12],facecolor='w'),('Regret',),[self.show_regret]))
		self.buttons[-1].on_clicked(self._make_Plot)

		if not self.alg_defined:
			self.buttons.append(Button(plt.axes([0.775,0.915,0.12,height],facecolor='w'),'Reset',color=self.COLOURS2[0], hovercolor=self.COLOURS2[1]))
			self.buttons[-1].on_clicked(functools.partial(self._reset))
		else:
			self.buttons.append(Button(plt.axes([0.775,0.915,0.12,height],facecolor='w'),'Reset',color=self.COLOURS2[0], hovercolor=self.COLOURS2[1]))
			self.buttons[-1].on_clicked(self.alg.reset)
			self.alg._make_buttons(true_ax)

		return

	def _reset(self,event):
		self.rewards=[[] for arm in range(self.num_arms)]
		self.regret=[0]
		self.times_pulled=[0 for arm in range(self.num_arms)]
		self._make_Plot(0)
		self.t=1
		return

	def _remove_buttons(self):

		for button in self.buttons:
			self.figure.delaxes(button.ax)
		return

	def _get_Button_Locations(self,true_ax):

		locations=[]
		separation=1.0/self.num_arms
		axes_height=2.5
		axes_width=0.5*separation

		width,h=self.figure.transFigure.inverted().transform(true_ax.transData.transform([axes_width,0]))-self.figure.transFigure.inverted().transform(true_ax.transData.transform([0,0]))

		w,height=self.figure.transFigure.inverted().transform(true_ax.transData.transform([0,axes_height]))-self.figure.transFigure.inverted().transform(true_ax.transData.transform([0,0]))

		for arm in range(self.num_arms):
			location=-separation*(self.num_arms-arm)
			t1,t2=self.figure.transFigure.inverted().transform(true_ax.transData.transform([location-axes_width/2.0,-8.5]))
			locations.append([t1,t2,width,height])

		if self.show_dists:
			t1,t2=self.figure.transFigure.inverted().transform(true_ax.transData.transform([0.1,-8.5]))
			locations.append([t1, t2, 2*width, height])
		elif not self.show_dists:
			t1,t2=self.figure.transFigure.inverted().transform(true_ax.transData.transform([-0.5-2*axes_width,-12.5]))
			locations.append([t1, t2, 2*width, height])

		return width, height, locations

	def _make_Plot(self,label, make_buttons=1):

		changefig=0
		if label=='Distributions':
			self.show_dists=1-self.show_dists
			self.true_ax=self._make_axes(self.show_dists,self.show_regret)
			changefig=1
		elif label=='Regret':
			self.show_regret=1-self.show_regret
			self.true_ax=self._make_axes(self.show_dists,self.show_regret)
			changefig=1
		elif label=='start':
			self.true_ax=self._make_axes(self.show_dists,self.show_regret)
			changefig=1
			
		self._viewBounds(self.true_ax, make_buttons)
		if changefig and make_buttons:
			self._remove_buttons()
			self._make_buttons(self.true_ax)
		plt.draw()
		return

	def _viewBounds(self,true_ax,make_buttons=1, fs=None):

		if fs is None:
			fs=min(60/(self.num_arms),10)

		if self.plot_content is not None:
			for line in self.plot_content:
				line.remove()

		self.plot_content=[]
		
		for arm in range(self.num_arms):
			separation=1.0/self.num_arms
			location=-separation*(self.num_arms-arm)

			if not self.alg_defined:
				if len(self.rewards[arm])>0:

					self.plot_content.extend(true_ax.plot([location for r in self.rewards[arm]],self.rewards[arm],'o',ms=5,mfc="None",mec=self.COLOURS[arm],mew=0.5))

					self.plot_content.extend(true_ax.plot([location],[np.mean(self.rewards[arm])],'.',ms=20,c=self.COLOURS[arm]))
			
			if self.num_arms<=8:
				self.plot_content.append(true_ax.text(location, -3, r' Arm {}'.format(arm),ha='center',va='center',fontsize= fs))

				self.plot_content.append(true_ax.text(location, -4.6, r'$n_{}={}$'.format(arm,self.times_pulled[arm]),ha='center', va='center',fontsize= fs))

		if self.alg_defined:
			self.alg._make_plots(true_ax)

		if make_buttons:
			if self.show_dists:
				true_ax.set_ylim([-10,15])
			else:
				true_ax.set_ylim([-13,15])
		else:
			true_ax.set_ylim([-5.5,15])

		ax=self._get_regret_ax('TS')

		self.plot_content.extend(ax.plot(self.regret,'-',c=self.COLOURS2[1]))
		return

	def run_Interactive(self,dists=1,reg=1):
		
		self.initialize()
		self.show_regret=reg
		self.show_dists=dists
		self._make_Plot('start')
		plt.show()
		return 

class Interactive_UCB_Algorithm:

	def __init__(self,bandit_env,alg,name):
		
		self.name=name
		self.bandit_env=bandit_env
		self.num_arms=self.bandit_env.num_arms

		self.alg=alg


		return

	def _make_plots(self,true_ax):

		self.recalculate_bounds()

		for arm in range(self.bandit_env.num_arms):
			separation=1.0/self.bandit_env.num_arms
			location=-separation*(self.bandit_env.num_arms-arm)

			if len(self.bandit_env.rewards[arm])>0:
				self.bandit_env.plot_content.extend(true_ax.plot([location,location],[np.mean(self.bandit_env.rewards[arm]),self.upper_confidence_bounds[arm]],lw=2.5,c=self.bandit_env.COLOURS[arm]))
		
				self.bandit_env.plot_content.extend(true_ax.plot([-separation*(self.num_arms-arm-0.25),-separation*(self.num_arms-arm+0.25)],[self.upper_confidence_bounds[arm],self.upper_confidence_bounds[arm]],'-',lw=3,c=self.bandit_env.COLOURS[arm]))

				self.bandit_env.plot_content.extend(true_ax.plot([location],[np.mean(self.bandit_env.rewards[arm])],'.',ms=15,c=self.bandit_env.COLOURS[arm]))
			else:
				self.bandit_env.plot_content.extend(true_ax.plot([location,location],[0,50],lw=2.5,c=self.bandit_env.COLOURS[arm]))
		
				self.bandit_env.plot_content.extend(true_ax.plot([-separation*(self.num_arms-arm-0.25),-separation*(self.num_arms-arm+0.25)],[self.upper_confidence_bounds[arm],self.upper_confidence_bounds[arm]],'-',lw=3,c=self.bandit_env.COLOURS[arm]))

				self.bandit_env.plot_content.extend(true_ax.plot([location],[50],'.',ms=15,c=self.bandit_env.COLOURS[arm]))
		return

	def _make_buttons(self,true_ax):
		width,height, locations=self.bandit_env._get_Button_Locations(true_ax)
		self.bandit_env.buttons.append(Button(plt.axes(locations[-1]),self.name,color=self.bandit_env.COLOURS2[0], hovercolor=self.bandit_env.COLOURS2[1]))

		self.bandit_env.buttons[-1].on_clicked(functools.partial(self._pull_arm_alg))
		return

	def recalculate_bounds(self):
		t=self.bandit_env.t
		var=self.bandit_env.standard_deviations[0]**2
		arm,self.upper_confidence_bounds=self.alg(t,var,self.bandit_env.times_pulled,self.bandit_env.rewards)
		return

	def _pull_arm_alg(self,event):

		t=self.bandit_env.t
		var=self.bandit_env.standard_deviations[0]**2

		arm,self.upper_confidence_bounds=self.alg(t,var,self.bandit_env.times_pulled,self.bandit_env.rewards)

		assert arm in range(self.num_arms) and len(self.upper_confidence_bounds)==self.num_arms
		self.bandit_env._interactive_pull_arm(event,arm)

		return


	def run_Interactive_Alg(self,show_dists=1,show_reg=1):

		self.upper_confidence_bounds=[np.inf for arm in range(self.num_arms)]
		self.bandit_env.initialize(make_plot=1,alg_defined=True,alg=self)
		self.bandit_env.show_regret=show_reg
		self.bandit_env.show_dists=show_dists
		self.bandit_env._make_Plot('start')
		plt.show()

	def reset(self,event):

		self.upper_confidence_bounds=[np.inf for arm in range(self.num_arms)]
		self.bandit_env._reset(0)
		return


class Interactive_TS_Algorithm:

	def __init__(self,bandit_env,alg,name,prior_means,prior_vars):
		
		self.name=name
		self.bandit_env=bandit_env
		self.num_arms=self.bandit_env.num_arms
		self.prior_means=np.array(prior_means, dtype=float)
		self.prior_vars=np.array(prior_vars, dtype=float)
		self.alg=alg


		return

	def _make_plots(self,true_ax):

		self.recalculate_bounds()

		for arm in range(self.bandit_env.num_arms):
			separation=1.0/self.bandit_env.num_arms
			location=-separation*(self.bandit_env.num_arms-arm)
			upper=self.means[arm]+2*np.sqrt(self.vars[arm])
			lower=self.means[arm]-2*np.sqrt(self.vars[arm])

			self.bandit_env.plot_content.extend(true_ax.plot([location,location],[lower,upper],lw=2.5,c=self.bandit_env.COLOURS[arm]))
	
			self.bandit_env.plot_content.extend(true_ax.plot([-separation*(self.num_arms-arm-0.25),-separation*(self.num_arms-arm+0.25)],[upper,upper],'-',lw=3,c=self.bandit_env.COLOURS[arm]))
			self.bandit_env.plot_content.extend(true_ax.plot([-separation*(self.num_arms-arm-0.25),-separation*(self.num_arms-arm+0.25)],[lower,lower],'-',lw=3,c=self.bandit_env.COLOURS[arm]))

			if len(self.bandit_env.rewards[arm])>0:
				self.bandit_env.plot_content.extend(true_ax.plot([location],[np.mean(self.bandit_env.rewards[arm])],'.',ms=15,c=self.bandit_env.COLOURS[arm]))

		return

	def _make_buttons(self,true_ax):
		width,height, locations=self.bandit_env._get_Button_Locations(true_ax)
		self.bandit_env.buttons.append(Button(plt.axes(locations[-1]),self.name,color=self.bandit_env.COLOURS2[0], hovercolor=self.bandit_env.COLOURS2[1]))

		self.bandit_env.buttons[-1].on_clicked(functools.partial(self._pull_arm_alg))
		return

	def recalculate_bounds(self,start=0):

		if start:
			t=0
		else:
			t=self.bandit_env.t
		var=self.bandit_env.standard_deviations[0]**2
		arm , self.samples , self.means,self.vars=self.alg(t,var,self.bandit_env.times_pulled,self.bandit_env.rewards,self.prior_means,self.prior_vars)
		return

	def _pull_arm_alg(self,event):

		t=self.bandit_env.t
		var=self.bandit_env.standard_deviations[0]**2
		arm , self.samples , self.means,self.vars=self.alg(t,var,self.bandit_env.times_pulled,self.bandit_env.rewards,self.prior_means,self.prior_vars)

		assert arm in range(self.num_arms) and len(self.samples)==self.num_arms
		self.bandit_env._interactive_pull_arm(event,arm)

		return

	def get_samples(self):

		samples=[]
		for i in range(self.num_arms):
			samples.append(np.random.normal(self.means[i],self.vars[i],1))

		return samples



	def run_Interactive_Alg(self,show_dists=1,show_reg=1):

		self.means=np.copy(self.prior_means)
		self.vars=np.copy(self.prior_vars)
		self.samples=self.get_samples()
		self.bandit_env.initialize(make_plot=1,alg_defined=True,alg=self)
		self.bandit_env.rewards=[[] for arm in range(self.num_arms)]
		self.bandit_env.show_regret=show_reg
		self.bandit_env.show_dists=show_dists
		self.bandit_env._make_Plot('start')
		plt.show()

	def reset(self,event):

		self.means=np.copy(self.prior_means)
		self.vars=np.copy(self.prior_vars)
		self.samples=self.get_samples()
		self.bandit_env._reset(0)
		return





if __name__ == "__main__":
	
	pass
		







