import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

st.sidebar.write("# Machine Learning")

chapter = st.sidebar.radio("",
                    ['Introduction'
                    ])

if chapter=="Introduction":
    '''
    # Mini-Project 1 
    #### By: Siddharth Ashok Unnithan
    ## Introduction

    ### Goal
    To find patterns that can help in understanding demand variations of bike rental service.
