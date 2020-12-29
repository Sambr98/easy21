from environment import Easy21
from utils import selectAction, plotValueFunction

import numpy as np

nbEpisodes = 200000

env = Easy21()

Q = np.zeros((10, 21, 2))
N = np.zeros((10, 21, 2))

for episode in range(nbEpisodes):

	terminal = False
	history = []
	dealer, player = env.startGame()

	while (not terminal):
		action = selectAction(dealer, player, Q, N)

		N[dealer-1, player-1, action] += 1
		history.append((dealer, player, action))

		dealer, player, terminal, reward = env.step(dealer, player, action)

	G = reward

	for (d, p, a) in history:
		Q[d-1, p-1, a] += (1 / N[d-1, p-1, a]) * (G - Q[d-1, p-1, a])

np.save('monte_carlo', Q)
plotValueFunction(Q, 'plots/')
