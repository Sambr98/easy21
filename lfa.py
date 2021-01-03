from environment import Easy21
from utils import selectActionConstant, plotMSEsPerLambda, plotMSEsPerEpisode

import numpy as np

nbEpisodes = 10000
lambdas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

dealer_feat = [(1, 4), (4, 7), (7, 10)]
player_feat = [(1, 6), (4, 9), (7, 12), (10, 15), (13, 18), (16, 21)]
alpha = 0.01

def features(d, p, a):
	features = np.zeros((3, 6, 2))

	dealer = [x[0] <= d <= x[1] for x in dealer_feat]
	player = [x[0] <= p <= x[1] for x in player_feat]

	for i in range(len(dealer)):
		for j in range(len(player)):
			if (i and j):
				features[i, j, a] = 1

	return features.flatten()

def Q(d, p, a, theta):
	return np.dot(features(d, p, a), theta)

def all_Q(theta):
	Qarr = np.zeros((10, 21, 2))

	for d in range(1, 11):
		for p in range(1, 22):
			for a in [0, 1]:
				q = Q(d, p, a, theta)
				Qarr[d-1, p-1, a] = q

	return Qarr

env = Easy21()

with open('monte_carlo.npy', 'rb') as f:
	Q_star = np.load(f)

MSEs = []
MSEs_ = []

for lmbda in lambdas:
	theta = np.random.randn(36)

	MSE = []

	for episode in range(nbEpisodes):

		E = np.zeros(36)

		terminal = False
		dealer, player = env.startGame()

		action = selectActionConstant(dealer, player, Q, theta)

		while (not terminal):
			newDealer, newPlayer, terminal, reward = env.step(dealer, player, action)

			if (not terminal):
				newAction = selectActionConstant(newDealer, newPlayer, Q, theta)
				delta = reward + Q(newDealer, newPlayer, newAction, theta) - Q(dealer, player, action, theta)
			else:
				delta = reward - Q(dealer, player, action, theta)

			E = lmbda * E + features(dealer, player, action)
			grad = alpha * delta * E
			theta += grad

			if (not terminal):
				dealer, player, action = newDealer, newPlayer, newAction

		MSE.append((np.square(all_Q(theta) - Q_star)).mean())

	MSEs.append((np.square(all_Q(theta) - Q_star)).mean())
	MSEs_.append(MSE)

plotMSEsPerLambda(MSEs, lambdas, 'plots/')
plotMSEsPerEpisode(MSEs_, lambdas, 'plots/')
