# -*- coding: utf-8 -*-
"""
Quantitative Investment Final Project

@author: David Cecchini
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateFormatter, AutoDateLocator
import matplotlib.patches as mpatches
import seaborn as sns
from datetime import datetime
import pickle
import os
from tabulate import tabulate
import dcor
import networkx as nx
from pypfopt import discrete_allocation
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.expected_returns import mean_historical_return
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


TRAIN_RANGE = ('2013-01-02', '2016-12-29')
TEST_RANGE = ('2017-01-02', '2020-05-29')

#function to compute the distance correlation (dcor) matrix from a DataFrame and output a DataFrame
#of dcor values.
def df_distance_correlation(df, stocks):

    #initializes an empty DataFrame
    df_dcor = pd.DataFrame(index=stocks, columns=stocks)

    #initialzes a counter at zero
    k=0

    # iterates over the time series of eachstocks stock
    for i in stocks:

        # stores the ith time series as a vector
        v_i = df.loc[:, i].values

        # iterates over the time series of each stock subect to the counter k
        for j in stocks[k:]:

            # stores the jth time series as a vector
            v_j = df.loc[:, j].values

            # computes the dcor coefficient between the ith and jth vectors
            dcor_val = dcor.distance_correlation(v_i, v_j)

            # appends the dcor value at every ij entry of the empty DataFrame
            df_dcor.at[i,j] = dcor_val

            # appends the dcor value at every ji entry of the empty DataFrame
            df_dcor.at[j,i] = dcor_val

        # increments counter by 1
        k+=1

    # returns a DataFrame of dcor values for every pair of stocks
    return df_dcor


# takes in a pre-processed dataframe and returns a time-series correlation
# network with pairwise distance correlation values as the edges
def build_corr_nx(df, corr_threshold=0.4):

    # converts the distance correlation dataframe to a numpy matrix with dtype float
    cor_matrix = df.values.astype('float')

    # Since dcor ranges between 0 and 1, (0 corresponding to independence and 1
    # corresponding to dependence), 1 - cor_matrix results in values closer to 0
    # indicating a higher degree of dependence where values close to 1 indicate a lower degree of
    # dependence. This will result in a network with nodes in close proximity reflecting the similarity
    # of their respective time-series and vice versa.
    sim_matrix = 1 - cor_matrix

    # transforms the similarity matrix into a graph
    G = nx.from_numpy_matrix(sim_matrix)

    # extracts the indices (i.e., the stock names from the dataframe)
    stock_names = df.index.values

    # relabels the nodes of the network with the stock names
    G = nx.relabel_nodes(G, lambda x: stock_names[x])

    # assigns the edges of the network weights (i.e., the dcor values)
    G.edges(data=True)

    # copies G
    ## we need this to delete edges or othwerwise modify G
    H = G.copy()

    # iterates over the edges of H (the u-v pairs) and the weights (wt)
    for (u, v, wt) in G.edges.data('weight'):
        # selects edges with dcor values less than or equal to 0.33
        if wt >= 1 - corr_threshold:
            # removes the edges
            H.remove_edge(u, v)

        # selects self-edges
        if u == v:
            # removes the self-edges
            H.remove_edge(u, v)

    # returns the final stock correlation network
    return H


def is_irreducible(H):
    for node, weight in H.degree():
        if weight == 0:
            return False
    return True


def grid_search_threshold(df_dcor, threshold_list):
    for threshold in threshold_list:
        print("Testing for threshold {:,.4f}:".format(threshold))
        H = build_corr_nx(df_dcor, corr_threshold=threshold)
        print("Result: {}".format("Irreducible!" if is_irreducible(H) else "Not irreducible!"))
        print()


# function to display the network from the distance correlation matrix
def plt_corr_nx(H, title):

    # creates a set of tuples: the edges of G and their corresponding weights
    edges, weights = zip(*nx.get_edge_attributes(H, "weight").items())

    # This draws the network with the Kamada-Kawai path-length cost-function.
    # Nodes are positioned by treating the network as a physical ball-and-spring system. The locations
    # of the nodes are such that the total energy of the system is minimized.
    pos = nx.kamada_kawai_layout(H)

    with sns.axes_style('whitegrid'):
        # figure size and style
        plt.figure(figsize=(16, 9))
        plt.title(title, size=16)

        # computes the degree (number of connections) of each node
        deg = H.degree

        # list of node names
        nodelist = []
        # list of node sizes
        node_sizes = []

        # iterates over deg and appends the node names and degrees
        for n, d in deg:
            nodelist.append(n)
            node_sizes.append(d)

        # draw nodes
        nx.draw_networkx_nodes(
            H,
            pos,
            node_color= "blue", #"#DA70D6",
            nodelist=nodelist,
            node_size= [(x+1) * 100 for x in node_sizes], #np.power(node_sizes, 2.33),
            alpha=0.8,
            font_weight="bold",
        )

        # node label styles
        nx.draw_networkx_labels(H, pos, font_size=13, font_family="sans-serif", font_weight='bold')

        # color map
        cmap = sns.cubehelix_palette(n_colors=9, start=2.2, dark=0.1, rot=0.3, gamma=1.1, hue=1.0, light=0.6, as_cmap=True, reverse=True)

        # draw edges
        nx.draw_networkx_edges(
            H,
            pos,
            edge_list=edges,
            style="solid",
            edge_color=weights,
            edge_cmap=cmap,
            edge_vmin=min(weights),
            edge_vmax=max(weights),
        )

        # builds a colorbar
        sm = plt.cm.ScalarMappable(
            cmap=cmap,
            norm=plt.Normalize(vmin=min(weights),
            vmax=max(weights))
        )
        sm._A = []
        plt.colorbar(sm)

        # displays network without axes
        plt.axis("off")


# function to visualize the degree distribution
def hist_plot(network, title, bins, xticks):

    # extracts the degrees of each vertex and stores them as a list
    deg_list = list(dict(network.degree).values())

    # sets local style
    with plt.style.context('fivethirtyeight'):
        # initializes a figure
        plt.figure(figsize=(9,6))

        # plots a pretty degree histogram with a kernel density estimator
        sns.distplot(
            deg_list,
            kde=True,
            bins = bins,
            color='darksalmon',
            hist_kws={'alpha': 0.7}

        )

        # turns the grid off
        plt.grid(False)

        # controls the number and spacing of xticks and yticks
        #xticks = range()
        plt.xticks(xticks, size=11)
        plt.yticks(size=11)

        # removes the figure spines
        sns.despine(left=True, right=True, bottom=True, top=True)

        # labels the y and x axis
        plt.ylabel("Probability", size=15)
        plt.xlabel("Number of Connections", size=15)

        # sets the title
        plt.title(title, size=20);

        # draws a vertical line where the mean is
        plt.axvline(sum(deg_list)/len(deg_list),
                    color='darkorchid',
                    linewidth=3,
                    linestyle='--',
                    label='Mean = {:2.0f}'.format(sum(deg_list)/len(deg_list))
        )

        # turns the legend on
        plt.legend(loc=0, fontsize=12)


# a function to generate a random approximate MIS
### WARNING: rerunning kernel will produce different MISs
def generate_mis(G, sample_size, nodes=None):

    """Returns a random approximate maximum independent set.

    Parameters
    ----------
    G: NetworkX graph
       Undirected graph

    nodes: list, optional
        a list of nodes the approximate maximum independent set must contain.

    sample_size: int
        number of maximal independent sets sampled from

    Returns
    -------
    max_ind_set: list or None
        list of nodes in the apx-maximum independent set
        NoneType object if any two specified nodes share an edge

    """

    # list of maximal independent sets
    max_ind_set_list=[]

    # iterates from 0 to the number of samples chosen
    for i in range(sample_size):

        # for each iteration generates a random maximal independent set that contains
        # UnitedHealth and Amazon
        max_ind_set = nx.maximal_independent_set(G, nodes=nodes, seed=i)

        # if set is not a duplicate
        if max_ind_set not in max_ind_set_list:

            # appends set to the above list
            max_ind_set_list.append(max_ind_set)

        # otherwise pass duplicate set
        else:
            pass

    # list of the lengths of the maximal independent sets
    mis_len_list=[]

    # iterates over the above list
    for i in max_ind_set_list:

        # appends the lengths of each set to the above list
        mis_len_list.append(len(i))

    # extracts the largest maximal independent set, i.e., the maximum independent set (MIS)
    ## Note: this MIS may not be unique as it is possible there are many MISs of the same length
    max_ind_set = max_ind_set_list[mis_len_list.index(max(mis_len_list))]

    return max_ind_set


# a function to convert centrality scores to portfolio weights
def centrality_to_portfolio_weights(weights):

    """Returns a dictionary of portfolio weights.

    Parameters
    ----------
    weights: dictionary
        NetworkX centrality scores

    Returns
    -------
    portfolio weights: dictionary
        normalized inverse of chosen centrality measure

    """

    # iterates over the key, value pairs in the weights dict
    for key, value in weights.items():

        # takes the inverse of the communicability betweeness centrality of each asset
        weights[key] = 1/value

    # normalization parameter for all weights to add to 1
    norm = 1.0 / sum(weights.values())

    # iterates over the keys (stocks) in the weights dict
    for key in weights:

        # updates each key value to the normalized value and rounds to 3 decimal places
        weights[key] = round(weights[key] * norm, 3)

    return weights


# function to compute the cumulative returns of a portfolio
def cumulative_returns(shares_allocation, capital, test_data):

    """Returns the cumulative returns of a portfolio.

    Parameters
    ----------
    shares_allocation: DataFrame
        number of shares allocated to each asset in the portfolio

    capital: float
        total amount of money invested in the portfolio

    test_data: DataFrame
        daily closing prices of portfolio assets

    Returns
    -------
    cumulative_daily_returns: Series
        cumulative daily returns of the portfolio

    """

    # list of DataFrames of cumulative returns for each stock
    daily_returns = []

    # iterates over every stock in the portfolio
    for stock in shares_allocation.index:

        # multiples shares by share prices in the validation dataset
        daily_returns.append(shares_allocation.loc[stock].values * test_data[stock])

    # concatenates every DataFrame in the above list to a single DataFrame
    daily_returns_df = pd.concat(daily_returns, axis=1).reset_index()

    # sets the index as the date
    daily_returns_df.set_index("Day", inplace=True)

    # adds the cumulative returns for every stock
    cumulative_daily_returns = daily_returns_df.sum(axis=1)

    # returns the cumulative daily returns of the portfolio
    return cumulative_daily_returns


# function to compute daily return on investment (roi)
def portfolio_daily_roi(shares_allocation, capital, test_data):

    """Returns the daily return on investment.

    Parameters
    ----------
    shares_allocation: DataFrame
        number of shares allocated to each asset

    capital: float
        total amount of money invested in the portfolio

    test_data: DataFrame
        daily closing prices of each asset

    Returns
    -------
    daily_roi: Series
        daily return on investment of the portfolio

    """

    # computes the cumulative returns
    cumulative_daily_returns = cumulative_returns(
        shares_allocation,
        capital,
        test_data
    )

    # calculates daily return on investment
    daily_roi = cumulative_daily_returns.apply(
        lambda returns: ((returns - capital) / capital)*100
    )

    # returns the daily return on investment
    return daily_roi


# function to extract the end of year returns
def end_of_year_returns(model_roi, return_type, start, end):

    """Returns the end of year returns of a portfolio.

    Parameters
    ----------
    model_roi: Series
        portoflio returns on investment

    return_type: string
        'returns': returns roi
        'returns_rate': returns rate of returns

    start: int
        starting year to extract last trading day from

    end: int
        ending year to extract last trading day from

    Returns
    -------
    end_of_year_returns: dictionary
        each year's returns or rate of returns

    """

    # converts index of datetimes to a list of strings
    dates = model_roi.index.astype('str').tolist()

    # list comprehension of a string of dates between the
    # start and end dates
    years = [str(x) for x in range(start, end + 1)]

    # generates an empty list of lists for each year
    end_year_dates = [[] for _ in range(len(years))]

    # iterates over every date in the roi series
    for date in dates:

        # iterates over every year in the years list
        for year in years:

            # iterates over every year in each date
            if year in date:

                # converts each date string to a datime object
                datetime_object = datetime.strptime(date, '%Y-%m-%d')

                # appends each date to its corresponding year in the years list
                (end_year_dates[years.index(year)]
                    .append(datetime.strftime(datetime_object, '%Y-%m-%d')))

    # finds the last date in each year
    end_year_dates = [max(x) for x in end_year_dates]

    # gets the rounded end of year returns
    returns = [round(model_roi[date], 1) for date in end_year_dates]

    # shifts the returns list by 1 and appends 0 to the beginning of the list
    return_rates = [0] + returns[:len(returns)-1]
    """Example: [a, b, c] -> [0, a, b]"""

    # converts returns list to an array
    returns_arr = np.array(returns)

    # converts the return_rates list to an array
    return_rates_arr = np.array(return_rates)

    # calculates the rounded rate of returns
    return_rates = [round(x, 1) for x in list(returns_arr - return_rates_arr)]
    """Example: [a, b, c] - [0, a, b] = [a, b-a, c-b]"""

    # dictionary with the years as keys and returns as values
    returns = dict(zip(years, returns))

    # dictionary with the years as keys and return rates as values
    return_rates = dict(zip(years, return_rates))

    if return_type == 'returns':
        return returns

    if return_type == 'return_rates':
        return return_rates


# function to calculate avg annual portfolio returns
def avg_annual_returns(end_of_year_returns, mstat):

    """Returns average annual returns.

    Parameters
    ----------
    end_of_year_returns: dictionary
        annual returns

    mstat: string
        'arithmetic': returns the arithmetic mean
        'geometric': returns the geometric mean

    Returns
    -------
    average annual returns: float

    """

    # imports mean stats
    from scipy.stats import mstats

    # converts returns dict to an array (in decimal fmt)
    returns_arr = np.array(list(end_of_year_returns.values()))/100

    if mstat == 'geometric':

        # calculates the geometric mean
        gmean_returns = (mstats.gmean(1 + returns_arr) - 1)*100

        return round(gmean_returns, 2)

    if mstat == 'arithmetic':

        # calculates the arithmetic mean
        mean_returns = np.mean(returns_arr)

        return round(mean_returns, 2)


# function to calculate annualized portoflio standard deviation
def portfolio_std(weights, test_data):

    """Returns annualized portfolio volatility.

    Parameters
    ----------
    weights: dictionary
        portfolio weights

    test_data: DataFrame
        validation data set

    Returns
    -------
    portfolio_std_dev: float
        annualized portfolio standard deviaion

    """

    # computes daily change in returns from 2015-2017
    daily_ret_delta = test_data.pct_change()

    # computes the covariance matrix of the above
    cov_matrix = daily_ret_delta.cov()

    # initializes weights
    weights_list = []

    # iterates over weights dict and appends above list
    for key, value in weights.items():
        weights_list.append(value)

    # converts weights list to numpy array
    weights_arr = np.array(weights_list)

    # calculates the annualized portfolio standard deviation from 2015-2017 in pct format
    portfolio_std_dev = np.sqrt(
        np.dot(
            weights_arr.T,
            np.dot(
                cov_matrix,
                weights_arr
            )
        )
    )*np.sqrt(252)*100

    return round(portfolio_std_dev, 2)


# function to calculate annualized portfolio standard deviation with a
# maximum independent set parameter
def mis_portfolio_std(weights, test_data, maximum_independent_set):

    """Returns annualized portfolio volatility.

    Parameters
    ----------
    weights: dictionary
        portfolio weights

    test_data: DataFrame
        validation data set

    maximum_independent_set: list
        largest list of assets such that no two are adjacent

    Returns
    -------
    portfolio_std_dev: float
        annualized portfolio standard deviation

    """

    # computes daily change in returns from 2015-2017
    daily_ret_delta = test_data[maximum_independent_set].pct_change()

    # computes the covariance matrix
    cov_matrix = daily_ret_delta.cov()

    # initializes weights list
    weights_list = []

    # iterates over weights dict and appends above list
    for key, value in weights.items():
        weights_list.append(value)

    # converts weights list to numpy array
    weights_arr = np.array(weights_list)

    # calculates portfolio standard deviation from 2015-2017
    portfolio_std_dev = np.sqrt(
        np.dot(
            weights_arr.T,
            np.dot(
                cov_matrix,
                weights_arr
            )
        )
    )*np.sqrt(252)*100

    return round(portfolio_std_dev, 2)


# function to compute the Sharpe ratio
def portfolio_sharpe_ratio(avg_annual_returns, portfolio_std, risk_free_rate):

    """Returns Sharpe ratio.

    Parameters
    ----------
    avg_annual_returns: float
        portoflio avg annual returns

    portfolio_std: float
        annualized portfolio volatility

    risk_free_rate: float
        usually taken as the avg 10-year treasury rate over investment period

    Returns
    -------
    portfolio_std_dev: float
        annualized portfolio standard deviaion

    """

    # calculates the Sharpe ratio
    sharpe_ratio = (avg_annual_returns - risk_free_rate) / portfolio_std

    return round(sharpe_ratio, 2)


# function to compute the 252-day daily rolling maximum
def daily_rolling_max(cumulative_returns, window=252, min_periods=1):

    """Returns the daily running 252-day maximum.

    Parameters
    ----------
    cumulative_returns: Series
        portoflio's cumulative returns

    window: int, default 252
        size of the moving window.

    min_periods: int
        minimum number of observations in window required to have a value

    Returns
    -------
    daily_rolling_max: Series

    """

    return cumulative_returns.rolling(
        window=window,
        min_periods=min_periods
    ).max()


# function to compute the 252-day rolling drawdown
def daily_rolling_drawdown(cumulative_returns, rolling_max):

    """Returns the daily running 252-day drawdown.

    Parameters
    ----------
    cumulative_returns: Series
        portoflio's cumulative returns

    rolling_max: Series
        rolling 252-day maximum

    Returns
    -------
    daily_rolling_drawdown: Series

    """

    return (cumulative_returns / rolling_max) - 1


# function to compute the 252-day maximum daily drawdown
def max_daily_rolling_drawdown(daily_drawdown, window=252, min_periods=1):

    """Returns the daily running 252-day maximum daily drawdown.

    Parameters
    ----------
    daily_drawdown: Series
       daily rolling 252-day drawdown

    window: int, default 252
        size of the moving window.

    min_periods: int
        minimum number of observations in window required to have a value

    Returns
    -------
    max_daily_rolling_drawdown: Series

    """

    return daily_drawdown.rolling(
        window=window,
        min_periods=min_periods
    ).min()


# function to compute the lifetime maximum drawdown
def lifetime_max_drawdown(daily_drawdown):

    """Returns the lifetime maximum drawdown.

    Parameters
    ----------
    daily_drawdown: Series
       daily rolling 252-day drawdown

    Returns
    -------
    lifetime_max_drawdown: float
        largest amount of money lost

    """

    return round(daily_drawdown.min()*100, 2)


# calculates returns over lifetime maximum drawdown
def returns_over_max_drawdown(tot_returns_dict, year, lifetime_maximum_drawdown):

    """Returns the lifetime maximum drawdown.

    Parameters
    ----------
    tot_returns_dict: dictionary
       cumulative annual portfolio returns

    year: int

    lifetime_maximum_drawdown: float
        largest amount of money lost

    Returns
    -------
    returns_over_max_drawdown: float
        cumulative returns divded by largest sum of money lost

    """

    return round(tot_returns_dict[year] / abs(lifetime_maximum_drawdown), 2)


# function to calculate the growth-risk ratio
def portfolio_growth_risk(avg_annual_returns, max_daily_rolling_drawdown):

    """Returns the growth-risk ratio.

        Parameters
        ----------
        avg_annual_returns: float
           average annual returns

        max_daily_rolling_drawdown: Series
            252-day rolling maximum daily drawdown

        Returns
        -------
        portfolio_growth_risk: float
            average annual returns divided by average rolling max daily drawdown

    """

    return round(avg_annual_returns / abs(max_daily_rolling_drawdown.mean()*100), 2)


def collect_results(model):

    if model == 'MRP':

        collection = [[], [], []]

        collection[0].append([str(x) + '%' for x in list(returns_dict.values())])
        collection[1].append([str(x) + '%' for x in list(tot_returns_dict.values())])
        collection[2].append([str(x) + '%' for x in [gmean_returns, portfolio_std_dev, max_drawdown]])
        collection[2].append([sharpe_ratio, risk_return_ratio, growth_risk_ratio])

        return collection

    if model == 'efficient_frontier':

        collection = [[], [], []]

        collection[0].append([str(x) + '%' for x in list(ef_returns_dict.values())])
        collection[1].append([str(x) + '%' for x in list(ef_tot_returns_dict.values())])

        collection[2].append([str(x) + '%' for x in [
            ef_gmean_returns,
            ef_portfolio_std_dev,
            ef_max_drawdown]
                             ]
                            )

        collection[2].append([ef_sharpe_ratio, ef_risk_return_ratio, ef_growth_risk_ratio])

        return collection

    if model == 'MIS':

        collection = [[], [], []]

        collection[0].append([str(x) + '%' for x in list(mis_returns_dict.values())])
        collection[1].append([str(x) + '%' for x in list(mis_tot_returns_dict.values())])

        collection[2].append([str(x) + '%' for x in [
            mis_gmean_returns,
            mis_portfolio_std_dev,
            mis_max_drawdown]
                             ]
                            )

        collection[2].append([mis_sharpe_ratio, mis_risk_return_ratio, mis_growth_risk_ratio])

        return collection


# function to plot many overlaping kde plots
def multi_distplot(rdist1, rdist2, rdist3, kde=True):

    # initializes figure and axis
    fig = plt.figure(figsize=(12,5))
    ax = fig.add_subplot(111)

    # pretty seaborn kde plots for each model
    sns.distplot(rdist1, bins=12, kde=bool)
    sns.distplot(rdist2, bins=10, kde=bool)
    sns.distplot(rdist3, bins=12, kde=bool)

    # gets xticks
    vals1 = ax.get_xticks()

    # reformats xticks to pcts
    ax.set_xticklabels(['{:.0f}%'.format(x) for x in vals1])

    # plot labels and title
    ax.set_ylabel('Probability')
    ax.set_xlabel('Returns')
    plt.title('Distribution of Returns')

    # removes spines
    sns.despine(top=True, right=True)

    # sets legend patches color and labels
    ef_patch = mpatches.Patch(color='darksalmon', label='Efficient Frontier', alpha=0.5)
    patch = mpatches.Patch(color='royalblue', label='Hedgecraft', alpha=0.5)
    mis_patch = mpatches.Patch(color='seagreen', label='Hedgecraft MIS', alpha=0.5)

    # turns legend on with patches
    plt.legend(handles=[ef_patch, patch, mis_patch])

# Avoid running any code here when importing
if __name__ == "__main__":
    pass

