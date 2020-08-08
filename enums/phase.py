from enum import Enum


class Phase(Enum):
    PRIVATE_AUCTION = 1
    PHASE_2 = 2  # ends with the purchase of the first 3-train
    PHASE_3 = 3  # ends with the purchase of the first 4-train
    PHASE_4 = 4  # ends with the purchase of the first 5-train
    PHASE_5 = 5  # ends with the purchase of the first 6-train
    PHASE_6 = 6  # ends with the purchase of the first diesel train
    PHASE_7 = 7  # ends when the bank runs out of money and the next time a stock round would normally start