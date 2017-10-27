import threading

from elbotto import launcher
from elbotto.bots import stochastic
from elbotto.bots import neuro

DEFAULT_BOT_NAME = "El botto del jasso"

DEFAULT_SERVER_NAME = "ws://127.0.0.1:3000"


def launch(bot_class, bot_name, server_address=DEFAULT_SERVER_NAME):
    bot_class(server_address, bot_name)

def start_bots():
    threading.Thread(target=launch, kwargs={"bot_class": stochastic.Bot, "bot_name": DEFAULT_BOT_NAME}).start()
    threading.Thread(target=launch, kwargs={"bot_class": stochastic.Bot, "bot_name": DEFAULT_BOT_NAME}).start()
    threading.Thread(target=launch, kwargs={"bot_class": stochastic.Bot, "bot_name": "NeuroBot"}).start()
    t = threading.Thread(target=launch, kwargs={"bot_class": neuro.Bot, "bot_name": "NeuroBot"})

    t.start()
    t.join()


if __name__ == '__main__':
    start_bots()