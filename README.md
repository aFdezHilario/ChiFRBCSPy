# ChiFRBCSPy
Chi FRBCS implementation in Python. Scikit-learn compatible

Parameters needed:

* labels
* t-norm 
* rule weight heuristic
* fuzzy reasoning method

All parameters are taken by default:
* labels=3, 
* tnorm="product", #This is the only available right now
* rw="pcf", #Penalized Certainty Factor is the only available right now
* frm="wr", #Winning Rule. "ac" (Additive Combinaton) is also avaiable

A "main.py" is provided as a running example
