import json
import logging
from enum import Enum

from elbotto import messages, card
from elbotto.connection import Connection
from elbotto.messages import MessageType, GameType

logger = logging.getLogger(__name__)


class SessionType(Enum):
    TOURNAMENT = "TOURNAMENT"
    SINGLE_GAME = "SINGLE_GAME"


DEFAULT_TRUMPF = GameType("TRUMPF", card.Color.HEARTS.name)


class BaseBot(object):
    """
    Base class for a Jass bot

    :param server_address: the web socket address of the server
    :param name: the name of the bot
    :param chosen_team_index: (optional) the index of the team to join
    """

    connection = None

    def __init__(self, server_address, name, chosen_team_index=0):
        self.name = name
        self.session_name = name
        self.server_address = server_address
        self.chosen_team_index = chosen_team_index

        self.teams = None
        self.hand_cards= []
        self.won_stich_in_game = []
        self.last_round_points = 0

    def start(self):
        logger.info("Connecting to %s", self.server_address)
        Connection.create(self.server_address, self)

    def handle_message(self, message):
        answer = None

        type = message["type"]
        if isinstance(type, MessageType):
            message_type = type
        else:
            message_type = MessageType[type]

        try:
            data = message["data"]
        except KeyError:
            data = {}

        if message_type == MessageType.REQUEST_PLAYER_NAME:
            # CHALLENGE2017: Respond with your BotName
            logger.info('MyName: ' + self.name)
            answer = messages.create(MessageType.CHOOSE_PLAYER_NAME, self.name)
            
        elif message_type == MessageType.REQUEST_SESSION_CHOICE:
            answer = messages.create(MessageType.CHOOSE_SESSION,
                                     "AUTOJOIN",
                                     self.session_name,
                                     SessionType.SINGLE_GAME.name,
                                     False,
                                     self.chosen_team_index)
            logger.info('session choice answer: %s', answer)
            
        elif message_type == MessageType.DEAL_CARDS:
            self.last_round_points = 0
            self.hand_cards = data

        elif message_type == MessageType.REQUEST_TRUMPF:
            game_type = self.handle_request_trumpf()
            answer = messages.create(MessageType.CHOOSE_TRUMPF, game_type)
            
        elif message_type == MessageType.REQUEST_CARD:
            card = self.handle_request_card(data)
            answer = messages.create(MessageType.CHOOSE_CARD, card)
            
        elif message_type == MessageType.PLAYED_CARDS:
            self.handle_played_cards(data)
            
        elif message_type == MessageType.REJECT_CARD:
            self.handle_reject_card(data)
            
        elif message_type == MessageType.BROADCAST_GAME_FINISHED:
            won_stich = self.won_stich_in_game
            self.won_stich_in_game = []
            self.handle_game_finished(data[0].current_game_points, won_stich)

        elif message_type == MessageType.BROADCAST_SESSION_JOINED:
            player = data["player"]
            if self.name == player.name:
                self.player = player

            self.players_in_session = data["playersInSession"]

        elif message_type == MessageType.BROADCAST_STICH:
            winner = data["winner"]
            won_stich = self.in_my_team(winner)
            self.won_stich_in_game.append(won_stich)
            total_points = self.total_points(data["score"])
            current_game_points = self.current_game_points(data["score"])

            if won_stich:
                round_points = current_game_points - self.last_round_points
            else:
                round_points = 0

            self.last_round_points = current_game_points
            self.handle_stich(winner, round_points, total_points)

        elif message_type == MessageType.BROADCAST_TOURNAMENT_STARTED:
            #Do nothing with that :-)
            pass
        elif message_type == MessageType.BROADCAST_TOURNAMENT_RANKING_TABLE:
            #Do nothing with that :-)
            pass
        elif message_type == MessageType.BROADCAST_TEAMS:
            self.teams = data
            for team in self.teams:
                if team.is_member(self.player):
                    self.my_team = team

        elif message_type == MessageType.BROADCAST_TRUMPF:
            self.handle_trumpf(data)

        elif message_type == MessageType.BROADCAST_WINNER_TEAM:
            #Do nothing with that :-)
            pass
        else:
            logger.warning("Sorry, i cannot handle this message: " + json.dumps(message))

        if answer:
            self.connection.send(answer)

    def handle_played_cards(self, played_cards):
        # CHALLENGE2017: This removes a handcard if the last played card on the table was one of yours.
        self.update_hand(played_cards)

    def handle_request_trumpf(self):
        # CHALLENGE2017: Ask the brain which gameMode to choose
        return DEFAULT_TRUMPF

    def handle_trumpf(self, game_type):
        self.geschoben = game_type.mode == "SCHIEBE"  # just remember if it's a geschoben match
        self.game_type = game_type

    def handle_stich(self, winner, round_points, total_points):
        # Do nothing with that :-)
        pass

    def handle_game_finished(self, current_game_points, won_stich_in_game):
        self.last_round_points = 0
        pass

    def handle_reject_card(self, data):
        # CHALLENGE2017: When server sends this, you send an invalid card... this should never happen!
        # Server will send "REQUEST_CARD" after this once.
        # Make sure you choose a valid card or your bot will loose the game
        logger.warning(" ######   SERVER REJECTED CARD   #######")
        logger.warning("Rejected card: %s", data)
        logger.warning("Hand Cards: %s", self.hand_cards)
        logger.warning("Gametype: %s", self.game_type)

    def handle_request_card(self, table_cards):
        # CHALLENGE2017: Ask the brain which card to choose
        card = self.hand_cards[0]
        return card

    def in_my_team(self, winner):
        return self.my_team.is_member(winner)
        # return self.player == winner

    def current_game_points(self, scores):
        for score in scores:
            if self.my_team.name == score.team_name:
                return score.current_game_points

        return 0

    def total_points(self, scores):
        for score in scores:
            if self.my_team.name == score.team_name:
                return score.total_points

        return 0

    def update_hand(self, played_cards):
        last_played_card = played_cards[-1]
        hand_cards = []
        for card in self.hand_cards:
            if card.number != last_played_card.number or card.color != last_played_card.color:
                hand_cards.append(card)
        self.hand_cards = hand_cards

