# Roadmap

This document contains a list of the known shortcomings and necessary changes that I want to make to this repo, in no particular order.


# Simplifications I want to eventually undo

* Currently, bidding on companies, buying companies, and buying trains have limited action choices. When you bid, you must make the minimum bid, when a corp buys a company it must pay either min or max, and when a corp buys a train it must select one of many pre-defined choices. I would like to redesign the action space and model architecture to allow all legal action parameters from the full game. 
* Currently, the game engine and model only support 4 players. I would like to support all legal player counts, 2-6.
* The game rules technically allow certain actions to be made "at any time", including during or between other players' turns. This is not allowable in the current engine, and I would like to support it.
* The game engine only supports 1830. I would like to support other titles in the same genre, such as 1867 or 1822.

# Model design

* The V1 model uses a GNN, which probably works. The V2 model uses transformers, which also probably work. I want to investigate various other possible model architectures to see which work best - ResNet for example.
* I want to revisit the value head/prediction to ensure that it works correctly with MCTS. I also am not sure whether the value should be 1 for winning or just be the cash value of the player. Ideally, we would do the former, but the latter gives better signal.
* I would like to fully implement the supervised learning version of this. I only have ~250 games of 4p 1830, but hopefully that's enough to get started.

# General improvements

* I think the dashboard/metrics could be improved
* Definitely more but I need to refresh my understanding of the app as-is.

# Plan

1. Walk through each step of the training processes and fix issues.
  a. Model initialization/saving/loading
    i. Starting a fresh training run should be as frictionless as possible.
    ii. Same for resuming.
  b. Pretraining
    i. Data cleaning.
    ii. Data loading.
    iii. Supervised learning steps: loss, backprop, etc.
    iv. Metrics, checkpointing, etc.
  c. Self-play
    i. Outer loop (opponent selection, game loop, engine)
    ii. Inner loop (MCTS)
    iii. Data saving
    iv. Training step
    v. Metrics, checkpointing, etc.
  d. Model design
    i. Encoder
    ii. Network architecture
    iii. Policy design
    iv. Value design
  e. Job monitoring
2. Run the current version of the model with the pretraining data. See how far supervised learning can get us. Graph performance over time.
3. If it works well, try doing MCTS afterward.
4. Investigate issues outlined above:
  a. Better action parameterization to reduce action space
  b. Better action parameterization to allow full action space from game rules
  c. Different model architectures
  d. Support any-time actions
  e. Improve map encoding
5. Stretch goals:
  a. Support different player counts
  b. Support multiple titles
