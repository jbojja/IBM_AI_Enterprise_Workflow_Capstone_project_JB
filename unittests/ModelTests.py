#!/usr/bin/env python
"""
model tests
"""

import sys, os
import unittest
localpath = os.path.abspath('')
data_dir = os.path.join(localpath,"data_dir")
sys.path.insert(1, localpath)

## import model specific functions and variables
from model2 import *




class ModelTest(unittest.TestCase):
    """
    test the essential functionality
    """
        
    def test_01_train(self):
        """
        test the train functionality
        """

        ## train the model
        model_train(data_dir,test=True)
        self.assertTrue(os.path.exists(os.path.join("models", "test-all-0_1.joblib")))

    def test_02_load(self):
        """
        test the train functionality
        """
                        
        ## train the model
        alldata,allmodels = model_load()
        
        self.assertTrue('predict' in dir(allmodels['all']))
        self.assertTrue('fit' in dir(allmodels['all']))

       
    def test_03_predict(self):
        """
        test the predict function input
        """

        ## load model first
        ##model = model_load(test=True)
    
        ## ensure that a list can be passed
        country='all'
        year='2018'
        month='01'
        day='05'
        result = model_predict(country,year,month,day)
        y_pred = result['y_pred']
        self.assertTrue(y_pred[0]>0)
### Run the tests
if __name__ == '__main__':
    unittest.main()
