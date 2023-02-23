# Efficient Generation of Justifications for Collective Decision Making
This code has been implemented for the bachelor thesis: Efficient Generation of Justifications for Collective Decision Making.
Handed in on June 26th at the University of Amsterdam.

#### Currently implemented axioms:
1. Pareto principle
2. Condorcet principle
3. Faithulness
4. Cancellation
5. Neutrality
6. Anonymity

#### This code uses the packages numpy, collections and pylgl:
please run `pip install -r requirements.txt` or `pip3 install -r requirements.txt` before running the code

#### Two algorithms have been implemented:
1. Algorithm to generate justifications based on axioms that refer to one profile (axiom 1 until 4). <br>
   How to run: `python3 oneProfile.py` <br>
2. Algorithm to generate justifications based on axioms that refer to at most two profiles (axioms 1 until 6). <br>
   How to run: `python3 twoProfile.py`

In the files `oneProfile.py` and `twoProfile.py` the target profile and target outcome can be changed as well as the axioms in the corpus.
