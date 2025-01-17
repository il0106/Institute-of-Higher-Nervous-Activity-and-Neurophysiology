## Russian Academy of Sciences
# Institute of Higher Nervous Activity and Neurophysiology
This repository is related to work at the Institute of Higher Nervous Activity and Neurophysiology. 
Key words: mice, mouse, PyMICE, IntelliCage, behavior, behavioral contagion, graphs, pyviz, networkx, permutation test, wilcoxon test

The main module (yet) is 'ratcontagion'. This module contains RBCA class (Rat behavioral contagion analyzer) that helps analyze visit data from the IntelliCage system to research behavioral contagion among rats.
### How can RBCA help you?
- Transforming the data from IntelliCage to comfortable pandas format
- Processing the data slicing as you wish
- Creating static (via pyvis) and dynamic (via plotly time slider) graphs describing behavioral contagion among rats in the IntelliCage
- Analyzing the number of rat visits, time intervals between visits in various slices making histograms
- Building your own modified classes and functions 
### Where should you start?
1) Learn more about behavioral contagion
2) Read about IntelliCage and its output files
3) See the tutorial.ipynb with the useful examples, also check cookbook.ipynb
4) Try to import ratcontagion.py module with RBCA class and work with it
5) Observe requirements.txt, also know that the python version of the lib is 3.9.7
