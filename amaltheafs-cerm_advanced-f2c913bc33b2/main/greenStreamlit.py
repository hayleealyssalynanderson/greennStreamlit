# Package Imports 
from locale import D_FMT
import os
import streamlit as st
st.set_page_config(layout="wide")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import plotly.figure_factory as ff
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from mpld3 import plugins
import mpld3
import streamlit.components.v1 as components
from plotly.subplots import make_subplots
from Portfolio import Portfolio, load_from_file, show
from Ratings import TEST_8_RATINGS
from parameters import *
from ScenarioGenerator import ScenarioGenerator
from LargeCERMEngine import LargeCERMEngine
import pickle as pkle
from utils import correlation_from_covariance
from PIL import Image

# Template portfolio
#dumpFilePath = 'amaltheafs-cerm_advanced-f2c913bc33b2/main/portfolio1000loans.dump'

#myCwd = os.getcwd();
#portfolio_path = os.path.join(myCwd, dumpFilePath)
#portfolio = load_from_file(portfolio_path)

## Styling 
button_style = """
        <style>
        .stButton > button {
            margin-left: 60px;
            width: 150px;
            height: 50px;
        }
        </style>
        """
theme = """ 
    base="light"
    primaryColor="#4cb744"
"""
image = 'amaltheafs-cerm_advanced-f2c913bc33b2/main/logos/fin_rwa.png'
myCwd = os.getcwd();
logo_path = os.path.join(myCwd, image)
st.sidebar.image(logo_path,width=130)
st.sidebar.markdown('**Upload Portfolio:**')        

uploaded_files = st.sidebar.file_uploader("Select file", accept_multiple_files=True)
for file in uploaded_files:
    if file.type == "text/csv":
        with open(file.name, "wb") as f:
                    bytes_data = file.read()
                    f.write(bytes_data)
                    portfolio = load_from_file(file.name)
    else:
        dumpFilePath = 'amaltheafs-cerm_advanced-f2c913bc33b2/main/portfolio1000loans.dump'
        myCwd = os.getcwd();
        portfolio_path = os.path.join(myCwd, dumpFilePath)
        portfolio = load_from_file(portfolio_path)

## App Heading 
def heading():

    st.markdown("""
        <h1 style='text-align: center; margin-bottom: -35px;'>
        Green RWA
        </h1>
    """, unsafe_allow_html=True
    )
    st.markdown("""
        <h6 style='text-align: center; margin-bottom: -35px;'>
        A Stochastic Climate Model - An approach to calibrate the Climate-Extended Risk Model (CERM) 
        </h6>
        <br>
        <br>

    """, unsafe_allow_html=True
    )
    
## SideBar Parameters 
def sideBar():

    ### Parameters
    st.sidebar.markdown('**Set Parameters:**')
    st.sidebar.markdown("""
            <p style= font-size:14px;color:#898A8B;'>
            Select Values assessed for climate evolution
            </p>
        """, unsafe_allow_html=True
        )
    #time horizon of the study
    horizon = st.sidebar.number_input( 'Horizon', 1, 50,value=10)
    st.sidebar.markdown( """ 
                            <p style= font-size:12px;color:#898A8B;margin-top:-86px;margin-left:50px;'>
                             time horizon of the study
                            </p>
                        """, unsafe_allow_html=True
                        )
    #transition efficiency coefficient (reduced)
    alpha = st.sidebar.slider( 'Alpha ', min_value=0.001, step=0.001, max_value=0.1, value=.02, format="%f")
    st.sidebar.markdown( """ 
                            <p style= font-size:12px;color:#898A8B;margin-top:-100px;margin-left:40px;'>
                             transition efficiency coefficient (reduced)
                            </p>
                        """, unsafe_allow_html=True
                        )
    #transition effort reactivity coefficient
    beta = st.sidebar.slider( 'Beta', min_value=0.001, step=0.001, max_value=5.00, value=1.5, format="%f")
    st.sidebar.markdown( """ 
                            <p style= font-size:12px;color:#898A8B;margin-top:-101px;margin-left:35px;'>
                             transition effort reactivity coefficient
                            </p>
                        """, unsafe_allow_html=True
                        )
    #climate change intensity of the economic activity (idiosyncratic)
    gamma = st.sidebar.slider( 'Gamma', min_value=0.001, step=0.001, max_value=0.1, value=0.005, format="%f", )
    st.sidebar.markdown( """ 
                        <p style= font-size:11px;color:#898A8B;margin-top:-99px;margin-left:55px;'>
                         climate change intensity of economic activity
                        </p>
                    """, unsafe_allow_html=True
                    )
    #hypothetical climate-free average growth rate of log GDP
    R = st.sidebar.slider( 'R', 1.00, 3.00, value=1.00)
    st.sidebar.markdown( """ 
                        <p style= font-size:12px;color:#898A8B;margin-top:-101px;margin-left:20px;'>
                         climate-free average growth rate of log GDP
                        </p>
                    """, unsafe_allow_html=True
                    )
    #idiosyncratic economic risk    
    e = st.sidebar.slider( 'e', min_value=0.001, step=0.001, max_value=1.000, value=0.10)
    st.sidebar.markdown( """ 
                    <p style= font-size:12px;color:#898A8B;margin-top:-100px;margin-left:20px;'>
                     idiosyncratic economic risk
                    </p>
                """, unsafe_allow_html=True
                )
    #idiosyncratic physical risk
    p = st.sidebar.slider( 'p', min_value=0.001,step=0.001, max_value=1.000,value=0.10)
    st.sidebar.markdown( """ 
                <p style= font-size:12px;color:#898A8B;margin-top:-100px;margin-left:20px;'>
                 idiosyncratic physical risk
                </p>
            """, unsafe_allow_html=True
            )
    #independent transition coefficient
    theta = st.sidebar.slider( 'theta', min_value=0.001, step=0.001, max_value=0.3,value=0.10)
    st.sidebar.markdown( """ 
            <p style= font-size:12px;color:#898A8B;margin-top:-100px;margin-left:40px;'>
             independent transition coefficient
            </p>
        """, unsafe_allow_html=True
        )

    #number of iterations for monte-carlo simulation
    N = st.sidebar.number_input('N', 1, value=10)  
    st.sidebar.markdown( """ 
            <p style= font-size:12px;color:#898A8B;margin-top:-87px;margin-left:20px;'>
             number of iterations for Monte-Carlo simulaton
            </p>
        """, unsafe_allow_html=True
        )
        
    ## Run Model Button Click
    if st.sidebar.button('Run Model'):
        st.write('')
        if not uploaded_files:
            st.sidebar.error('Please Upload file')
        else: 
            st.markdown("### LargeCERMEngine ")
            scenarioGenerator(horizon, alpha, beta, gamma, R, e , p , theta, N)
            loan_analysis()  

    st.markdown(button_style, unsafe_allow_html=True)
    st.sidebar.text("")
    st.sidebar.text("")
    st.sidebar.text("")
    st.sidebar.text("")

def loan_analysis():
    st.write("")   
    st.write("")
    st.markdown("***")

    st.markdown("""
        <p style= font-size:15px;color:#898A8B;'>
        The choice of ratings can be represented with a single square matrix, with as many indices as ratings. This particular regulatory ratings matrix has 8 ratings, from 'AAA' to 'D', 'D' being default.
        </p>
        """, unsafe_allow_html=True
        )

    with st.expander("See Ratings"): 
        #Ratings
        ratings_table = pd.DataFrame(TEST_8_RATINGS.reg_mat())
        ratings_table.index, ratings_table.columns = TEST_8_RATINGS.list(), TEST_8_RATINGS.list()
        ratings_table
           

    #Portfolio Loan List
    portfolio_dict = show(portfolio)
    st.markdown("""
        <p style= font-size:15px;color:#898A8B;'>
        Loan groupings for this portfolio:
        </p>
        """, unsafe_allow_html=True
    )
    st.code(f" \n{list(portfolio_dict.keys())}.")

    #Computing the total EAD of this protfolio at time 1
    st.markdown("""
        <p style= font-size:15px;color:#898A8B;'>
        Total EAD of 'portfolio1000loan' portfolio at time 1:
        </p>
        """, unsafe_allow_html=True
    )

    loan_profile = np.array([[4,21,60,72,34,20,2,0]])/213
    loan_profile_df = pd.DataFrame(loan_profile*100)
    loan_profile_df.columns = TEST_8_RATINGS.list()
    loan_profile_df
    

    #Total Principal of the Portfolio
    all_loans = np.vstack(portfolio_dict.values())
    all_principals = all_loans[:,0]
    total_principal = sum(np.array(all_principals, dtype=np.int32))
    # st.write(f"Total principal of portfolio is: $ {total_principal}")
    st.markdown("""
            <p style= font-size:15px;color:#898A8B;'>
            Total principal of portfolio is: $ 1000000000
            </p>
            """, unsafe_allow_html=True
        )

def scenarioGenerator(horizon, alpha, beta, gamma, R, e , p , theta, N):
 
    ## --  modeling the stochastic evolution of climate:
    scenario = ScenarioGenerator(horizon, alpha, beta, gamma, R, e, p, theta)
    scenario.compute()
    #logging of all macro-correlations evolutions
    macros = scenario.macro_correlation.T

    #plotting
    fig = plt.figure(figsize=(6,6)) 
    plt.plot(range(1,horizon), macros[1:,:], label=["economic","physical","transition"])
    plt.title("Evolution of macro-correlations under calibrated scenario over a "+str(horizon)+"-year horizon")
    plt.ylabel("value (in economic risk units)")
    plt.xlabel("time (years)")
    plt.legend()
    


    # we compute the climate scenario until 2 * horizon to get auto- and cross- correlations as delayed as the horizon

    scenario_extended = ScenarioGenerator(2 * horizon, alpha, beta, gamma, R, e, p, theta)
    scenario_extended.compute()

    # generation of the incremental matrix

    A = np.array(
        [[0, 0, 0], [-scenario.gamma, 1, -scenario.alpha], [0, scenario.beta, 0]])

    # initialization of autocorrelations

    autocorrelation = np.zeros(
        (horizon, horizon-1, scenario.nb_rf, scenario.nb_rf))
    autocorrelation_phy = np.zeros((horizon-1, horizon-1))
    autocorrelation_tra = np.zeros((horizon-1, horizon-1))
    autocorrelation_phy_tra = np.zeros((horizon-1, horizon))
    autocorrelation_tra_phy = np.zeros((horizon-1, horizon))

    # initialization of times and delays for which is drawn the graph

    times = range(1, horizon)
    taus = range(1, horizon)

    # execution

    for t in times:

        # logging of variance matrix at time t

        var_t = scenario_extended.var_at(t)
        corr = correlation_from_covariance(var_t)

        # logging of simultaneous cross-correlations, i.e. for delay tau=0

        autocorrelation_phy_tra[t-1, 0] = corr[2, 1]
        autocorrelation_tra_phy[t-1, 0] = corr[1, 2]

        # execution for each possible delay

        for tau in taus:

            # logging of variance matrix at time t+tau

            var_delay = scenario_extended.var_at(t+tau)

            # logging of inverse [standard deviations (macro-correlations)] at times t and t+tau

            at_time = np.reshape(1/np.sqrt(np.diag(var_t)), (3, 1))
            at_delay = np.reshape(1/np.sqrt(np.diag(var_delay)), (3, 1))

            # following the formula from the paper

            invsd = (at_delay@at_time.T)
            autocorrelation = invsd*(np.linalg.matrix_power(A, tau)@var_t)

            # logging all auto- and cross-correlations

            autocorrelation_phy[t-1, tau-1] = autocorrelation[1, 1]
            autocorrelation_tra[t-1, tau-1] = autocorrelation[2, 2]
            autocorrelation_phy_tra[t-1, tau] = autocorrelation[2, 1]
            autocorrelation_tra_phy[t-1, tau] = autocorrelation[1, 2]

    # plotting results

    figur, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, )
   # figur(figsize)
    plt.suptitle('Evolution of autocorrelations as functions of delay')

    ax1.plot(times, autocorrelation_phy, label=["tau = "+str(tau) for tau in taus])
    ax1.set_title('physical auto-correlation')
    ax1.legend(loc="upper left", prop={"size": 7})

    ax1.set_ylabel("correlation")

    ax2.plot(times, autocorrelation_tra, label=["tau = "+str(tau) for tau in taus])
    ax2.set_title('transition auto-correlation')

    taus = [0]+list(taus)

    ax3.plot(times, autocorrelation_tra_phy, label=[
            "tau = "+str(tau) for tau in taus])
    ax3.set_title('physical after transition correlation')
    ax3.legend(loc="upper left", prop={"size": 7})

    ax3.set_ylabel("correlation")
    ax3.set_xlabel("time (years)")

    ax4.plot(times, autocorrelation_phy_tra, label=[
            "tau = "+str(tau) for tau in taus])
    ax4.set_title('transition after physical correlation')
    ax4.set_xlabel("time (years)")


    # running the model: the model runs the ratings and the portfolio we have presented above. 
    st.markdown("""

            <p style= font-size:15px;color:#898A8B;text-align: left;'>
            The class for the Climate-Extended Risk Model, when the sample size is so large that idiosyncratic risks can be considered negligible.

            </p>
            """, unsafe_allow_html=True
        )
  ##  with st.expander("Histogram of loss distribution for given portfolio with given iterations:", expanded=False):
  
    from LargeCERMEngine import LargeCERMEngine
    
    # number of iterations for Monte-Carlo simulation

    #N = 10

    # risk values

    risk1 = .05
    risk2 = .01

    #computation of all losses through LCERM

    engine = LargeCERMEngine(portfolio, TEST_8_RATINGS, scenario)
    engine.compute(N)

    #definition of risk indices matching risk1 and risk2

    ind1 = int(np.floor(N*(1-risk1)))
    ind2 = int(np.floor(N*(1-risk2)))

    #logging of all losses

    losses = engine.loss_results
    #logging of final physical and transition cumulative risks (the - sign is simply so that loss distribution in the physical/transition plane is well-oriented)

    cumulative_growth_factors = -engine.cumulative_growth_factors[1:, :]

    #logging of final losses for plane distribution

    scenario_losses = losses.sum(axis=(1,2))

    #sorting of all losses, to assess the losses at risk1 and risk2

    sorted_losses = np.sort(losses, axis=0)

    #logging of non-cumulative expected loss, unexpected loss at risk risk1, unexpected loss at risk risk2 at each time

    el, ul1, ul2 = sorted_losses.sum(axis=(0,1))/N, (sorted_losses.sum(axis=1))[ind1], (sorted_losses.sum(axis=1))[ind2]

    #logging of all final losses or all iterations

    draws = np.sort(losses.sum(axis=(1,2)))

    #computation of cumulative expected loss, unexpected loss at risk risk1, unexpected loss at risk risk2 at each time

    for t in range(1,horizon):
        el[t] += el[t-1]
        ul1[t] += ul1[t-1]
        ul2[t] += ul2[t-1]

    #logging of final cumulative expected loss, unexpected loss at risk risk1, unexpected loss at risk risk2

    expected_loss = el[-1]
    unexpected_loss1 = ul1[-1]
    unexpected_loss2 = ul2[-1]


    #plotting of final loss distribution, along with expected loss and unexpected losses at risks risk1 and risk2

    figure = plt.figure(figsize=(6,5.12))
    px.histogram(draws, nbins=max(200, N//100))
    plt.hist(draws, bins=max(159, N//100), alpha=.7, label="histogram of loss distribution for given portfolio")

    plt.axvline(x = expected_loss, color='pink')
    plt.text(expected_loss+20, N/100,"expected loss",rotation=90)
    plt.axvline(x = unexpected_loss1, color='orange')
    plt.text(unexpected_loss1+20, N/100,"unexpected loss at risk "+str(int(100*risk1))+"%",rotation=90)
    plt.axvline(x = unexpected_loss2, color='red')
    plt.text(unexpected_loss2+20, N/100,"unexpected loss at risk "+str(int(100*risk2))+"%",rotation=90)

    plt.xlabel("loss")
    plt.ylabel("number of occurences")
    plt.title("histogram of loss distribution for given portfolio with "+str(N)+" iterations")

    plt.legend()
    plt.show()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.pyplot(figure)
        st.write('Loss Distribution')
        st.markdown("""
            <p style= font-size:15px;color:#898A8B;text-align: left;'>
            Computing the MC distribution of total losses at horizon.

            </p>
            """, unsafe_allow_html=True
        )


    fig1= plt.figure(figsize=(6,5.1))
    plt.plot(el, label="expected loss")
    plt.plot(ul1, label="unexpected loss at risk "+str(int(100*risk1))+"%")
    plt.plot(ul2, label="unexpected loss at risk "+str(int(100*risk2))+"%")

    plt.xlabel("time")
    plt.ylabel("loss")
    plt.title("evolution of relevant losses for given portfolio with "+str(N)+" iterations")

    plt.legend()
    plt.show()
    with col3:
        st.pyplot(fig1)
        st.write('Evolution of relevant losses')
        st.markdown("""
            <p style= font-size:15px;color:#898A8B;text-align: left;'>
            Plotting the evolution of the estimates of the expected loss, and the 5% and 1% unexpected losses.

            </p>
            """, unsafe_allow_html=True
        )
 

    figure2 = plt.figure()

    c = ["green", "greenyellow", "yellow", "orange", "darkorange", "red", "darkred"]
    v = [0,.05,.1,.4,.6,.9,1.]
    l = list(zip(v,c))
    cmap=LinearSegmentedColormap.from_list('rg',l, N=256)

    plt.scatter(cumulative_growth_factors[0], cumulative_growth_factors[1], c=scenario_losses, cmap=cmap, label="data point")

    plt.xlabel("cumulative centered physical risk")
    plt.ylabel("cumulative centered transition risk")
    plt.title("scenario-loss distribution")
    plt.colorbar()

    plt.show()

    
    with col2:
        st.pyplot(figure2)
        st.write('Loss-Scenario')
        st.markdown("""

            <p style= font-size:15px;color:#898A8B;text-align: left;'>
                Plotting of loss-scenario (physical and transition) distribution.
            </p>
            """, unsafe_allow_html=True
        )

    st.markdown("***")

    st.markdown('### Scenario Generator ')
    st.markdown("""

        <p style= font-size:15px;color:#898A8B;text-align: left;'>
        Modeling the stochastic evolution of climate.

        </p>
        """, unsafe_allow_html=True
    )
    col3, col4 = st.columns(2)
    ## Evolution of auto correlations graph 
    with col4:
        #st.plotly_chart(figur, use_container_width=False)
        st.pyplot(figur)
        st.write("Evolution of autocorrelations")
        st.markdown("""
            <p style= font-size:15px;color:#898A8B;text-align: left;'>
                This graph presents the auto- and cross-correlations of these same risks over the next decade.
            </p>
        """, unsafe_allow_html=True
            )
        
    with col3:
        ## Evolution of macro-correlations graph 
        #st.plotly_chart(fig, use_container_width=False)
        st.pyplot(fig)
        st.write('')
        st.write("Evolution of macro-correlations")
        st.markdown("""
            <p style= font-size:15px;color:#898A8B;text-align: left;'>
                This graph is the evolution of the macro-correlations of the risks considered, i.e. their relative importance, over the next decade.
            </p>
        """, unsafe_allow_html=True
            )
heading()
sideBar()

