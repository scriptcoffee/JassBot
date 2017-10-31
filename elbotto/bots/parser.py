import json
import glob

# Import and validate all dates
files = glob.glob('C:\Programming\BA\ExterneLogfiles\swisslos\Logs\MLAI_0-0_log.txt')
for file_path in files:
    print(file_path)

    lines = open(file_path).readlines()

    for line in lines:

        game = line[43:]
        rounds = json.loads(game)

        print(rounds)

        amount_rounds = len(rounds['rounds'])
        amount_players = len(rounds['rounds'][0]['player'])

        for i in range(amount_rounds):

            table = []

            print(rounds['rounds'][i])

            if rounds['rounds'][i] is None:
                # print("Type None isn't valid.")
                break

            if 'trump' not in rounds['rounds'][i]:
                # print("Round hasn't a trump, so we skip it")
                break


            for player in range(amount_players):
                if 'hand' not in rounds['rounds'][i]['player'][player]:
                    # print("Round has no hands, so we skip it")
                    break
                table.insert(player, rounds['rounds'][i]['player'][player]['hand'])

            trumpf = rounds['rounds'][i]['trump']
            amount_stich = len(rounds['rounds'][i]['tricks'])
            print(amount_stich)
            for stich in range(amount_stich):
                current_player = int(rounds['rounds'][i]['tricks'][stich]['first'])
                for player_seat in range(amount_players):
                    card = rounds['rounds'][i]['tricks'][stich]['cards'][player_seat]
                    table[current_player].append(card)
                    current_player = (current_player + 1) % amount_players

            # Round completet with all hand cards for all players and trump
            print("trump: " + str(trumpf))
            print("player0: " + str(table[0]))
            print("player1: " + str(table[1]))
            print("player2: " + str(table[2]))
            print("player3: " + str(table[3]))

            for learning_player in range(amount_players):
                for stich in range(amount_stich):

                    cards_on_table = []
                    hand = table[learning_player][stich:amount_stich-1]
                    print(rounds['rounds'][i]['tricks'][stich])
                    current_player = int(rounds['rounds'][i]['tricks'][stich]['first'])

                    print("aktueller Spieler: " + str(current_player))
                    player_seat = 0

                    while current_player != learning_player:
                        print("aktueller Spieler: " + str(current_player))
                        print("Spielersitz: " + str(player_seat))
                        print("lernender Spieler: " + str(learning_player))

                        card = rounds['rounds'][i]['tricks'][stich]['cards'][player_seat]

                        cards_on_table.insert(player_seat, card)

                        print("Change current player!!")

                        current_player = (current_player + (amount_players-1)) % amount_players
                        player_seat += 1

                    target_card = rounds['rounds'][i]['tricks'][stich]['cards'][learning_player]

                    print("Cards on table: " + str(cards_on_table))
                    print("handcards: " + str(hand))
                    print("Trumpf: " + str(trumpf))
                    print("Ziel: " + str(target_card))

                    ### call BotNetwork with hand, table cards and trump
                    # setTrump(trumpf)
                    # chooseCard(hand, cards_on_table)