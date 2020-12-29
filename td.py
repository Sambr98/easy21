from environment import Easy21
from utils import selectAction, plotMSEsPerLambda, plotMSEsPerEpisode

import numpy as np

nbEpisodes = 10000
lambdas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

env = Easy21()

with open('monte_carlo.npy', 'rb') as f:
	Q_star = np.load(f)

MSEs = []
MSEs_ = []

for lmbda in lambdas:
	Q = np.zeros((10, 21, 2))
	N = np.zeros((10, 21, 2))

	MSE = []

	for episode in range(nbEpisodes):

		E = np.zeros((10, 21, 2))

		terminal = False
		history = []
		dealer, player = env.startGame()

		action = selectAction(dealer, player, Q, N)

		while (not terminal):
			newDealer, newPlayer, terminal, reward = env.step(dealer, player, action)

			if (not terminal):
				newAction = selectAction(newDealer, newPlayer, Q, N)
				delta = reward + Q[newDealer-1, newPlayer-1, newAction] - Q[dealer-1, player-1, action]
			else:
				delta = reward - Q[dealer-1, player-1, action]

			E[dealer-1, player-1, action] += 1
			N[dealer-1, player-1, action] += 1
			history.append((dealer, player, action))

			for (d, p, a) in history:
				Q[d-1, p-1, a] += (1 / N[d-1, p-1, a]) * delta * E[d-1, p-1, a]
				E[d-1, p-1, a] *= lmbda

			if (not terminal):
				dealer, player, action = newDealer, newPlayer, newAction

		MSE.append((np.square(Q - Q_star)).mean())

	MSEs.append((np.square(Q - Q_star)).mean())
	MSEs_.append(MSE)

plotMSEsPerLambda(MSEs, lambdas, 'plots/')
plotMSEsPerEpisode(MSEs_, lambdas, 'plots/')
