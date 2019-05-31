#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 18:32:21 2019
Modified on Mon May 27 18:50:14 2019


@author: Alberto Fernandez - University of Granada - (alberto@decsai.ugr.es)
"""

class FuzzyRule:
    """
        Simple fuzzy rule structure
        * antecedents: list of fuzzy/nominal variables
        * classLabel: consequent class
        * ruleWeight: consequent confidence degree
    """
    
    def __init__(self,antecedents,classLabel,ruleWeight):
        """
            Creates a rule from already established values
            * antecedents is a list of strings, each of each is simple a number (fuzzy label index)
            * classLabel is the index of the class
            * ruleWeight (between [0,1]) is the computed confidence (fuzzy membership class/fm all)
        """
        self.antecedents = list()
        for value in antecedents:
            self.antecedents.append(int(value))
        self.classLabel = classLabel
        self.ruleWeight = ruleWeight
        
    def getAntecedents(self):
        return self.antecedents
    
    def getAntecedent(self, pos):
        return self.antecedents[pos]
    
    def getRW(self):
        return self.ruleWeight
    
    def getClassLabel(self):
        return self.classLabel
        
        
    