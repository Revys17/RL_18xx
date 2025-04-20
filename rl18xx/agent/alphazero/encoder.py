from rl18xx.game.engine.game.base import BaseGame
from rl18xx.game.engine.round import WaterfallAuction
from torch import Tensor, from_numpy
import numpy as np


class Encoder_1830:
    GAME_SETTINGS = {
        "num_players": 4,
        "bank_cash": 12000,
        "cert_limit": 16,
        "starting_cash": 600,
        "num_companies": 6,
    }
    NUM_PLAYERS = 4  # Hardcoded for 4-player games

    def __init__(self):
        self.encoding_size = (
            self.GAME_SETTINGS["num_players"]
            + 2
            * self.GAME_SETTINGS["num_companies"]
            * self.GAME_SETTINGS["num_players"]
            + 3 * self.GAME_SETTINGS["num_companies"]
        )

    # This method takes a game state object and returns a tensor represention
    def encode(self, game: BaseGame) -> Tensor:
        # For the auction-only version, we need the following info:
        # - All players' cash
        # - Private company ownership status (1-hot encoded)
        # - Private company bid status (1 per player for having a bid from them, 1 float for the current bid amount)

        # Initialize the encoding vector
        encoding = np.zeros(self.encoding_size, dtype=np.float32)
        offset = 0

        assert (
            game.active_players()[0] == game.players[0]
        ), "game.players[0] is not the active player!"

        # 1. All players' cash (normalized)
        for i, player in enumerate(game.players):
            encoding[offset + i] = player.cash / self.GAME_SETTINGS["starting_cash"]
        offset += self.GAME_SETTINGS["num_players"]

        # TODO: Corporation cash once we support not just the initial auction

        # 2. Private company ownership status (1-hot encoded)
        for i, company in enumerate(game.companies):
            for j, player in enumerate(game.players):
                if company.owner == player:
                    encoding[offset + j] = 1.0
            offset += self.GAME_SETTINGS["num_players"]
            # TODO: When we move out of private auction only, we need to support corporation ownership

        # 3. Private company current bids from each player (1-hot encoded per private)
        if isinstance(game.round.active_step(), WaterfallAuction):
            for i, company in enumerate(game.companies):
                for bid in game.round.active_step().bids.get(company, []):
                    player = bid.entity
                    price = bid.price
                    player_offset = game.players.index(player)
                    encoding[offset + player_offset] = (
                        price / self.GAME_SETTINGS["starting_cash"]
                    )
                offset += self.GAME_SETTINGS["num_players"]

            # 4. Current bids/prices for each private company. -1.0 if already purchased.
            for i, company in enumerate(game.companies):
                if company.owner:
                    encoding[offset + i] = -1.0
                    continue

                encoding[offset + i] = (
                    game.round.active_step().min_bid(company)
                    / self.GAME_SETTINGS["starting_cash"]
                )
            offset += self.GAME_SETTINGS["num_companies"]

            # 5. Available for purchase (1-hot encoded) - always the first non-purchased company
            if not game.round.active_step().auctioning:
                for i, company in enumerate(game.companies):
                    if company == game.round.active_step().available()[0]:
                        encoding[offset + i] = 1.0
            offset += self.GAME_SETTINGS["num_companies"]

            # 6. Face value of each company
            for i, company in enumerate(game.companies):
                encoding[offset + i] = (
                    company.value / self.GAME_SETTINGS["starting_cash"]
                )
            offset += self.GAME_SETTINGS["num_companies"]
        else:
            # only add bids for private companies during the waterfall auction round
            offset += (
                self.GAME_SETTINGS["num_companies"] * self.GAME_SETTINGS["num_players"]
            ) + 3 * self.GAME_SETTINGS["num_companies"]

        # Final check on offset (optional sanity check)
        assert (
            offset == self.encoding_size
        ), f"Final offset {offset} != expected size {self.encoding_size}"

        # Convert numpy array to torch tensor
        return from_numpy(encoding)
