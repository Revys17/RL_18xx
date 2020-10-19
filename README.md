# RL_18xx
RL agent for 18xx (currently 1830)

Add the script directory to PYTHONPATH for the module to be found (Otherwise `No module named 'e30'` errors can occur on execution)
* e.g. export PYTHONPATH=$PYTHONPATH:~/workspace/RL_18xx/e30

`python3 e30/main.py` runs the game
* or `python3 -m main.py`

### Unit tests
* `python3 -m unittest` runs the full suite
* `python3 -m unittest test.test_bank` runs the test_bank.py file
* `python3 -m unittest test.test_bank.BankTest.test_bank_money` runs the test_bank_money method

* unit tests are under `test/`
* the directory must be named `test/`
* test files must be prefixed with `test_`
* test methods must be prefixed with `test_`