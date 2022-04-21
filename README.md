# CapstoneTeamJim

Welcome to the **Real Estate Investment Region Analysis Capstone Project** for our Master of
Applied Data Science from the University of Michigan School of Information!

The authors Jenny Ney, Ian Byrne, and Melanie Starr formed their Capstone Team JIM
with the intent of creating a framework to predict the direction of real estate
markets after discussing a common interest in real estate investment.



## **Data**:

In preparation for and during the early stages of the project, we identified open
source data that would help us potentially make predictions on county home values.
We ended up using the following resources in some form or another during the
project: 

*  Bureau of Labor Statistics Wage Data
*  American Community Survey Demographic Data
*  County Debt Ratio
*  State Personal and Corporate Income Tax Rates
*  Redfin Real Estate Data
*  FRED Unemployment Rates
*  World Bank GDP Data
  
These data sources were originally stored in a Postgres database, however depending
on when you are viewing this, that database may or may not be available. All data
used within the final models of the project presented here can be found in this github repository and our [Google drive](https://drive.google.com/file/d/1BwXfHoRVx37aIbWZynjwH4fLo7ue90nP/view?usp=sharing)


## **Project**:

We each selected a model type that interested us and proved to test well on our preliminary data. This led to a time series Vector Autoregression (VAR) model, a supervised ExtraTreesRegressor model, and neural network Transformer model being used in the project.  We looked at several targets for each region, honing in on median sale price. As you will see, some models did better or worse depending on the target metric and the time granularity. In preparation for and in the early stages
of the project, we identified open source data that would help us potentially make predictions on county home values.  

View the presentation of the project by visiting [Streamlit](https://share.streamlit.io/ian-byrne/capstoneteamjim/main/streamlit/app.py). 


## **Project Structure**:



```bash
├── Notebooks
├── README.md
├── data
│   └── processed
├── models
│   ├── Neural_Network
│   ├── Supervised
│   └── Time_Series
├── reports
│   ├── figures
│   │   ├── Supervised
│   │   └── Time_Series
│   │   └── Neural_Network
│   └── results
│       ├── Supervised
│       └── Time_Series
│       └── Neural_Network
├── requirements.txt
└── streamlit

```


* **Notebooks** 

   This is where each of the team members explored the data and examined different models.


* **Data** 

   The formatted data that is used in each of the models.

* **Models** 

   The final scripts for each of the different models that produces the results and figures.

* **Reports** 

   The output results and figures of each of the models.

* **Streamlit** 

   The presentation of the results of each of the models and conclusion of the project.

```bash
  __  __   _   ___  ___    ___   _   ___  ___ _____ ___  _  _ ___  
 |  \/  | /_\ |   \/ __|  / __| /_\ | _ \/ __|_   _/ _ \| \| | __| 
 | |\/| |/ _ \| |) \__ \ | (__ / _ \|  _/\__ \ | || (_) | .` | _|  
 |_|__|_/_/ \_\___/|___/  \___/_/_\_\_| _|___/ |_| \___/|_|\_|___| 
 |_   _| __| /_\ |  \/  |  _ | |_ _|  \/  |                        
   | | | _| / _ \| |\/| | | || || || |\/| |                        
   |_| |___/_/ \_\_|  |_|  \__/|___|_|  |_|    
