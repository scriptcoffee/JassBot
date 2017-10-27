import logging
import random

from elbotto.basebot import BaseBot, DEFAULT_TRUMPF

logger = logging.getLogger(__name__)


class Bot(BaseBot):
    """
    Trivial bot using DEFAULT_TRUMPF and randomly returning a card available in the hand.
    This is a simple port of the original Java Script implementation
    """

    def __init__(self, server_address, name, chosen_team_index=0):
        super(Bot, self).__init__(server_address, name, chosen_team_index)
        self.game_strategy = PlayStrategy()

        self.start()

    def handle_request_trumpf(self):
        # CHALLENGE2017: Ask the brain which gameMode to choose
        return self.game_strategy.chooseTrumpf(self.handCards)

    def handle_stich(self, winner, round_points, total_points):
        won_stich = self.in_my_team(winner)
        logger.debug("Stich: Won:%s, Winner: %s, Round points: %s, Total points: %s", won_stich, winner, round_points, total_points)

    def handle_reject_card(self, card):
        # CHALLENGE2017: When server sends this, you send an invalid card... this should never happen!
        # Server will send "REQUEST_CARD" after this once. Make sure you choose a valid card or your bot will loose the game
        logger.debug(" ######   SERVER REJECTED CARD   #######")
        logger.debug("Rejected card: %s", card)
        logger.debug("Hand Cards: %s", self.handCards)
        logger.debug("cardsAtTable %s", self.game_strategy.cardsAtTable)
        logger.debug("Gametype: %s", self.game_type)

    def handle_request_card(self, tableCards):
        # CHALLENGE2017: Ask the brain which card to choose
        card = self.game_strategy.chooseCard(self.handCards, tableCards)
        return card


class PlayStrategy(object):

    def __init__(self):
        self.geschoben= False
        self.cardsAtTable = []

    def chooseTrumpf(self, handcards):
        inputs = [0] * 36

        for c in handcards:
            inputs[c.id] = 1

        #CHALLENGE2017: Implement logic to chose game mode which is best suited to your handcards or schiaebae.
        # Consider that this decision ist quite crucial for your bot to be competitive
        # Use hearts as TRUMPF for now

        # if self.gschobe: nüme schiebe
        return DEFAULT_TRUMPF

    def chooseCard(self, handcards, tableCards):
        # 36 Inputs (one per card).
        # Status: 0 - no info, 1 - in hand, 2 - first card on table, 3 - second card on table, 4 - third card on table

        inputs = [0] * 36

        for card in handcards:
            inputs[card.id] = 1

        for x in range(0, len(tableCards)):
            c = tableCards[x]
            c = card.create(c["number"], c["color"])
            inputs[c.id] = x + 2

        #CHALLENGE2017: Implement logic to choose card so your bot will beat all the others.
        # Keep in mind that your counterpart is another instance of your bot
        validCards = self.getPossibleCards(handcards, tableCards)

        idx = random.randint(0, len(validCards)-1)

        card = validCards[idx]
        logger.debug("Chosen card: %s", card)
        return card

    def getPossibleCards(self, handCards, tableCards):
        # validation = Validation.create(self.gameType.mode, self.gameType.trumpfColor)
        # possibleCards = handCards.filter(function (card) {
        #     if (validation.validate(tableCards, handCards, card)) {
        #         return true
        #     }
        # }, this)

        # return possibleCards
        return handCards

    # def setValidation(self, gameMode, trumpfColor):
    #     self.validation = Validation.create(gameMode, trumpfColor)