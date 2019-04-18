"""
__author__  = Willy Fitra Hendria
"""
import matplotlib.pyplot as plt
import numpy as np
import random

n_armed = 6 # six armed bandit
alpha = 0.01 # constant learning rate
n_random_steps = 10 # number of steps for random action
n_greedy_steps = 4000 # number of steps for greedy action
optimistic_initial_value = 5 # optimistic initial value for greedy action
percentages = [[] for i in range(n_armed)] # percentage for every 100 steps (each arm)
avg_rewards = [] # avg rewards
x_axis  = [] # number of steps (x axis)


class SixArmedBandit:

	def uniform_action(self):
		""" Uniformly Random Action Selection.
		After 10 actions chosen randomly, it will return the average reward.
		"""
		global avg_rewards;
		avg_rewards = [0]
		for i in range(n_random_steps):
			a = np.random.randint(n_armed)
			r = self.__pull_bandit(a)
			avg_rewards.append(avg_rewards[i] + ((r - avg_rewards[i])/(i+1)))
		print("Average reward for",n_random_steps,"uniformly chosen actions:")
		print(avg_rewards[len(avg_rewards)-1])
		
	def epsilon_greedy_action(self, is_stationary = True, epsilon = 0.1, is_optimistic_init = False):
		""" Epsilon Greedy Action Selection.
		Default value of epsilon is 0.1.
		"""
		
		q = [(optimistic_initial_value if is_optimistic_init else 0) for i in range(n_armed)] # estimation value
		k = [0 for i in range(n_armed)] # counter of chosen actions
		rewards = [0 for i in range(n_armed)]
		for  i in range(n_greedy_steps):
			a = self.__choose_epsilon_greedy_action(q, epsilon)
			r = self.__pull_bandit(a, is_stationary, i+1)
			k, q = self.__update_estimation(a, k, q, r, is_stationary)
			rewards[a] += r
			self.__showActionResult(i, q, k, rewards)
			
	def __choose_epsilon_greedy_action(self, q, epsilon):
		""" Random action with probability epsilon.
		Greedy with probability 1 - epsilon.
		"""
		p = np.random.uniform()
		if p < epsilon:
			return np.random.randint(n_armed)
		else:
			return np.argmax(q);

		
	def __pull_bandit(self, action, is_stationary = True, n_steps = 0):
		""" Return reward of bandits.
		For non-stationary case, the 4th bandit arm (index 3),
		will return different reward after 2000 steps.
		"""
		return {
			0 : np.random.uniform(1, 3),
			1 : np.random.uniform(-3, 8),
			2 : np.random.uniform(2, 5),
			3 : np.random.uniform(5, 7) if not is_stationary and n_steps > 2000 else np.random.uniform(-2, 6),
			4 : np.random.uniform(3, 4),
			5 : np.random.uniform(-2, 2)
		}[action]
		
	def __update_estimation(self, a, k, q, r, is_stationary):
		""" Update estimation for both stationary and non-stationary case.
		alpha = 0.01.
		"""
		k[a] += 1
		if (is_stationary):
			q[a] += ((r - q[a])/k[a])
		else:
			q[a] += (alpha*(r - q[a]))
		return k, q
		
	def __showActionResult(self, n_steps, q, k, rewards):
		""" Every 100 actions, print the percentage of choosen actions,
		and average reward so far.
		"""
		if ((n_steps+1) % 100 == 0):
			print()
			print("Result after",(n_steps+1),"actions:")
			global percentages, x_axis, avg_rewards
			x_axis.append(n_steps+1);
			for i in range(len(k)):
				percentage = k[i]/sum(k)*100
				percentages[i].append(percentage)
				# avg_rewards[i].append(0 if k[i] == 0 else rewards[i]/k[i])
				print("arm-",i+1,":", percentage,"%")
			avg_reward = sum(rewards)/(n_steps+1)
			avg_rewards.append(avg_reward)
			print("Average reward :",avg_reward)

	def drawGreedyPlot(self):
		plt.subplot(2, 1, 1)
		for i in range(n_armed):
			plt.plot(x_axis ,percentages[i], 'o', label='arm-'+str(i+1))
		plt.xlabel('steps')
		plt.ylabel('percentage')
		plt.legend(loc='center right')
		
		plt.subplot(2, 1, 2)
		plt.plot(x_axis ,avg_rewards, 'o')
		plt.xlabel('steps')
		plt.ylabel('average rewards')
		plt.legend(loc='center right')
		plt.show()

six_armed_bandit = SixArmedBandit()

print("Created by:")
print("Willy Fitra Hendria")
print("------------------------------------------")
print()
print("1. Uniformly Random")
print("2. Epsilon Greedy, Stationary");
print("3. Epsilon Greedy, Non-Stationary");
print("4. Optimistic Greedy");
print()
i = input("Input the implementation (1 - 4):")
print()
print("------------------------------------------")
if i == '1':
	six_armed_bandit.uniform_action()
	plt.plot(avg_rewards)
	plt.xlabel('steps')
	plt.ylabel('average rewards')
	plt.show()
elif i=='2':
	six_armed_bandit.epsilon_greedy_action()
	six_armed_bandit.drawGreedyPlot();
elif i=='3':
	#non-stationary
	six_armed_bandit.epsilon_greedy_action(False)
	six_armed_bandit.drawGreedyPlot();
elif i=='4':
	# non-stationary, epsilon=0 (greedy action), optimistic
	six_armed_bandit.epsilon_greedy_action(False, 0, True)
	six_armed_bandit.drawGreedyPlot();
else:
	print("Input not valid");