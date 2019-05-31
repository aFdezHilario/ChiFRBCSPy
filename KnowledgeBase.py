#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 20:41:02 2019
Modified on Tue May 28 16:47:00 2019

@author: Alberto Fernandez - University of Granada - (alberto@decsai.ugr.es)
"""

import numpy as np
from FuzzyRule import FuzzyRule

class KnowledgeBase:
    """
    The whole Knowledge Base is represented here
    
    Parameters
    ----------
        * X: numpy 
            The training set input values
        * y: numpy 
            The training set output class labels
        * dataBase: DataBase 
            The definition of the Fuzzy Variables
    
    Attributes
    ----------
        * X: the training set input values
        * y: the trainign set output class labels
        * dataBase: the whole definition of the fuzzy variables
        * matchingDegrees: a numpy array of the matching degrees for each pair {rule, class} 
            this is needed to improve the RW computation
        * ruleBase: a list with all the rules (FuzzyRule class)
        * classLabels: a numpy array with the different class label indices
    
    """
    
    def __init__(self,X,y,dataBase):
        self.X = X
        self.y = y
        self.dataBase = dataBase
        self.matchingDegrees = np.zeros([1,1],dtype=float)
        #print(self.matchingDegrees)
        self.ruleBase = list()
        self.classLabels = 0

    def includeInitialRules(self, ruleBaseTmp):
        self.classLabels = np.unique(self.y)
        self.matchingDegrees = np.resize(self.matchingDegrees,(len(ruleBaseTmp),len(self.classLabels)))
        
        #Create rules
        for rule in ruleBaseTmp.keys():
            fuzzyRule = FuzzyRule(rule,0,0) #no class yet, no rule weight yet
            self.ruleBase.append(fuzzyRule)
        
    def computeMatchingDegreesAll(self):
        """
            It obtains the matching degrees of each rule with all the examples,
            and stores the accumulation value for each class label
        """
        for example,classLabel in zip(self.X,self.y):
            j=-1
            for rule in self.ruleBase:
                j+=1
                nu = self.dataBase.computeMatchingDegree(rule, example)
                self.matchingDegrees[j][classLabel] += nu

    def computeRuleWeight(self,rule,classLabels,i):
        """
            It computes the confidence of the rule by Penalized Certainty Factor 
            (no other is supported yet)
            
            * rule: the fuzzy rule
            * classLabels: the possible class labels of the rule (those with MF > 0)
            * i: the index of the rule (for pre-computed matchingDegrees array)
        """
        ruleWeight = 0.0
        classIndex = -1
        accum = 0.0
        accum = np.sum(self.matchingDegrees[i])

        for classLabel in classLabels:
            sumOthers = accum-self.matchingDegrees[i][classLabel];
            currentRW = (self.matchingDegrees[i][classLabel] - sumOthers) / accum #P-CF
            if (currentRW > ruleWeight):
                ruleWeight = currentRW
                classIndex = classLabel
        return classIndex,ruleWeight

        
    def generation(self):
        """
            The whole Rule Base generation by grid covering
        """
        ruleBaseTmp = dict() #initial hash table to avoid repetitions

        print("Rule Generation")
        #Get all possible pairs of <antecedents,consequents>
        for example,label in zip(self.X,self.y):
            
            antecedents = self.dataBase.getRuleFromExample(example)
            if antecedents in ruleBaseTmp:
                classes = ruleBaseTmp[antecedents]
                if label not in classes:
                    ruleBaseTmp[antecedents].append(label)
            else:
                classes = list()
                classes.append(label)
                ruleBaseTmp[antecedents] = classes   
            

        print("Computing Matching Degrees Rule")
        """
            Transform the rule base into arrays
        """
        self.includeInitialRules(ruleBaseTmp);

        print("Computing Matching Degrees All")
        """
            Compute the matching degree of all the examples with all the rules
        """
        self.computeMatchingDegreesAll();

        print("Computing Rule Weights")
        """
            Compute the rule weight of each rule and solve the conflicts
        """
        i=-1
        self.ruleBase = list() #remove all rules
        for rule,classLabels in ruleBaseTmp.items():
            i+=1
            classLabel,ruleWeight = self.computeRuleWeight(rule,classLabels,i)
            if ruleWeight > 0:
                self.ruleBase.append(FuzzyRule(rule,classLabel,ruleWeight))
                print("Rule found: IF ",rule,"THEN",classLabel,"RW:",ruleWeight)

        print("Rule Base: "+str(len(self.ruleBase)))
        
    def WR(self,example):
        """
            Winning rule inference
            
            Only the single best rule (that with the highest fuzzy matching) 
            determines the class output
        """
        class_degrees = np.zeros(len(self.classLabels))
        for fuzzyRule in self.ruleBase:
            degree = self.dataBase.computeMatchingDegree(fuzzyRule,example)
            degree *= fuzzyRule.getRW()
            class_label = fuzzyRule.getClassLabel()
            if class_degrees[class_label] < degree:
                class_degrees[class_label] = degree
        
        return class_degrees
    
    def AC(self,example):
        """
            Additive combination inference
            
            All rules take course in the decision of the class label
        """
        classDegrees = np.zeros(len(self.classLabels))
        for fuzzyRule in self.ruleBase:
            degree = self.dataBase.computeMatchingDegree(fuzzyRule,example)
            degree *= fuzzyRule.getRW()
            classDegrees[fuzzyRule.getClassLabel()] += degree
                
        return classDegrees
    
    def classification(self,example,frm):
        if frm == "wr":
            return self.WR(example)
        else:
            return self.AC(example)
        
    def predict(self,X,frm):
        prediction = np.zeros(X.shape[0],dtype=int)
        for i in range(X.shape[0]):
            prediction[i] = np.argmax(self.classification(X[i],frm))
        return prediction
    
    def predict_proba(self,X,frm):
        prediction = np.zeros((X.shape[0],len(self.classLabels)),dtype=float)
        for i in range(X.shape[0]):
            prediction[i] = self.classification(X[i],frm) #probabilities (unnormalized)
        return prediction