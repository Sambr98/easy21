import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def selectAction(d, p, Q, N):
	epsilon = 100 / (100 + np.sum(N[d-1, p-1, :]))

	if (np.random.random() < epsilon):
		action = np.random.choice([0,1])
	else:
		action = np.argmax([Q[d-1, p-1, a] for a in [0,1]])

	return action

def selectActionConstant(d, p, Q, theta):
	epsilon = 0.05

	if (np.random.random() < epsilon):
		action = np.random.choice([0,1])
	else:
		action = np.argmax([Q(d, p, a, theta) for a in [0,1]])

	return action

def plotValueFunction(Q, path):
	dealerCard = []
	playerSum = []
	value = []
	for i in range(Q.shape[0]):
		for j in range(Q.shape[1]):
			dealerCard.append(i+1)
			playerSum.append(j+1)
			value.append(max(Q[i, j, 0], Q[i, j, 1]))

	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.plot_trisurf(dealerCard, playerSum, value, cmap=plt.cm.viridis)
	plt.savefig(path + 'monte_carlo.png')

def plotMSEsPerLambda(MSEs, lambdas, path):
	plt.plot(lambdas, MSEs, marker='o')
	plt.title('Mean Squared Error per Lambda')
	plt.xlabel('Lambda')
	plt.ylabel('MSE')
	plt.savefig(path + 'td.png')
	plt.clf()

def plotMSEsPerEpisode(MSEs, lambdas, path):
	for i in range(len(lambdas)):
		plt.plot(MSEs[i], label=lambdas[i])
	plt.legend()
	plt.title('Mean Squared Error per Episode')
	plt.xlabel('Episode')
	plt.ylabel('MSE')
	plt.savefig(path + 'td_.png')
