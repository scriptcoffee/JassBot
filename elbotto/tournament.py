import threading

from elbotto.bots import stochastic
from elbotto.bots import reinforcement_learning_strategy
from elbotto.basebot import BaseBot

DEFAULT_BOT_NAME = "El botto del jasso"

DEFAULT_SERVER_NAME = "ws://127.0.0.1:3000"


def launch(strategy, bot_name, server_address=DEFAULT_SERVER_NAME):
    BaseBot(server_address, bot_name, strategy)


def start_bots():
    threading.Thread(target=launch, kwargs={"strategy": stochastic.PlayStrategy, "bot_name": DEFAULT_BOT_NAME}).start()
    threading.Thread(target=launch, kwargs={"strategy": stochastic.PlayStrategy, "bot_name": DEFAULT_BOT_NAME}).start()
    threading.Thread(target=launch, kwargs={"strategy": stochastic.PlayStrategy, "bot_name": "NeuroBot"}).start()
    t = threading.Thread(target=launch, kwargs={"strategy": reinforcement_learning_strategy.PlayStrategy, "bot_name": "NeuroBot"})

    t.start()
    t.join()


if __name__ == '__main__':
    start_bots()