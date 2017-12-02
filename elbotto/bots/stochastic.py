import logging
import random

from elbotto.basebot import DEFAULT_TRUMPF

logger = logging.getLogger(__name__)


class PlayStrategy:

    def __init__(self):
        self.geschoben= False
        self.cardsAtTable = []

    def choose_trumpf(self, hand_cards, geschoben):
        # CHALLENGE2017: Implement logic to chose game mode which is best suited to your handcards or schiaebae.
        # Consider that this decision ist quite crucial for your bot to be competitive
        # Use hearts as TRUMPF for now
        return DEFAULT_TRUMPF

    def choose_card(self, hand_cards, table_cards, game_type, played_cards):
        # CHALLENGE2017: Implement logic to choose card so your bot will beat all the others.
        # Keep in mind that your counterpart is another instance of your bot
        valid_cards = self.get_possible_cards(hand_cards, table_cards)

        idx = random.randint(0, len(valid_cards)-1)

        card = valid_cards[idx]
        logger.debug("Chosen card: %s", card)
        return card

    def get_possible_cards(self, hand_cards, table_cards):
        # validation = Validation.create(self.gameType.mode, self.gameType.trumpfColor)
        # possibleCards = handCards.filter(function (card) {
        #     if (validation.validate(tableCards, handCards, card)) {
        #         return true
        #     }
        # }, this)

        # return possibleCards
        return hand_cards

    def stich_reward(self, round_points):
        pass

    def game_finished(self, match):
        pass

    def card_rejected(self):
        pass