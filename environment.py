import numpy as np

class Easy21:
	def __init__(self):
		self.minCard, self.maxCard = 1, 10
		self.lowerBound, self.upperBound = 1, 21
		self.dealerUpperBound = 17

	def startGame(self):
		return self.drawCard(start=True), self.drawCard(start=True)

	def drawCard(self, start=False):
		card = np.random.randint(self.minCard, self.maxCard+1)

		isRed = np.random.random()

		if (isRed <= 1/3 and not start):
			return -card
		else:
			return card

	def step(self, dealerSum, playerSum, action):
		terminal =  False
		reward = 0

		if (action == 0):
			terminal = True

			while (self.lowerBound <= dealerSum < self.dealerUpperBound):
				dealerSum += self.drawCard()

			if ((not (self.lowerBound <= dealerSum <= self.upperBound)) or (dealerSum < playerSum)):
				reward = 1
			elif (playerSum < dealerSum):
				reward = -1

		else:
			playerSum += self.drawCard()

			if (not (self.lowerBound <= playerSum <= self.upperBound)):
				terminal = True
				reward = -1

		return dealerSum, playerSum, terminal, reward