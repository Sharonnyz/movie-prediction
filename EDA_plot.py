import seaborn as sns
import pandas as pd
import numpy as np
# import plotly.offline as py
# import plotly.graph_objs as go
import matplotlib.pyplot as plt
def revenue_and_budget():
    df = pd.read_csv('D:/Gra1/computing/archive/new_m.csv')
    df_budget=df[df['originalBudget']>0]
    print(df_budget.info())
    x = df_budget['originalBudget']
    y = df_budget['revenue']
    z1 = np.polyfit(x, y, 1) # linear
    p1 = np.poly1d(z1)
    # print(p1) 
    yvals=p1(x) # also yvals=np.polyval(z1,x)
    plot1=plt.plot(x, y, '.',label='original values')
    plot2=plt.plot(x, yvals, 'r',label='polyfit values')
    plt.xlabel('budget')
    plt.ylabel('revenue')
    plt.title('Link between revenue and budget')
    plt.savefig('D:/Gra1/computing/data/Link between revenue and budget.png')
    plt.show()
    plt.close()

def release_year():
    df = pd.read_csv('D:/Gra1/computing/archive/new_m.csv')
    ax = df.groupby('release_year')['title'].count().plot.bar(title="Number of movies released by year",figsize=(14, 8))
    _ = ax.set_xlabel('Release year')
    _ = ax.set_ylabel('Number of movies')
    plt.savefig('D:/Gra1/computing/data/Number of movies released by year.png')
    plt.show()
    plt.close()

def mean_year():
    df = pd.read_csv('D:/Gra1/computing/archive/new_m.csv')
    grouped_1 = df.groupby('release_year')['originalBudget']
    # grouped_1.mean()
    plt.plot(grouped_1.mean())
    plt.grid()
    plt.xlabel('Release year')
    plt.ylabel('Mean budget')
    plt.title('Mean budget by year')
    plt.savefig('D:/Gra1/computing/data/Mean budget by year.png')
    plt.show()
    plt.close()

    grouped_2 = df.groupby('release_year')['revenue']
    plt.plot(grouped_2.mean())
    plt.grid()
    plt.xlabel('Release year')
    plt.ylabel('Mean revenue')
    plt.title('Mean revenue by year')
    plt.savefig('D:/Gra1/computing/data/Mean revenue by year.png')
    plt.show()
    plt.close()

    # d1 = df['release_year'].value_counts().sort_index()
    # d2 = df.groupby(['release_year'])['revenue'].sum()
    # data = [go.Scatter(x=d1.index, y=d1.values, name='Count'),go.Scatter(x=d2.index, y=d2.values, name='overall_revenue', yaxis='y2')]
    # layout = go.Layout(
    #     dict(title="Number of films and total revenue per year", xaxis=dict(title='year'), yaxis=dict(title='Count'),
    #          yaxis2=dict(title='Total revenue', overlaying='y', side='right')),
    #     legend=dict(orientation='v'))
    # py.iplot(dict(data=data, layout=layout))
    # plt.show()
    #
    # d1=df['release_year'].value_counts().sort_index()
    # d2=df.groupby(['release_year'])['revenue'].mean()
    # data=[go.Scatter(x=d1.index,y=d1.values,name='Count'),go.Scatter(x=d2.index,y=d2.values,name='Average revenue',yaxis='y2')]
    # layout=go.Layout(dict(title="Number of films and average revenue per year",xaxis=dict(title='year'),yaxis=dict(title='Count'),
    #                      yaxis2=dict(title='Average revenue',overlaying='y',side='right')),legend=dict(orientation='v'))
    # py.iplot(dict(data=data,layout=layout))
    # plt.show()


def distribution_revenue():
    df = pd.read_csv('D:/Gra1/computing/archive/new_m.csv')
    sns.distplot(df['revenue'])
    plt.ylabel('distribution')
    plt.title('Distribution of revenue')
    plt.savefig('D:/Gra1/computing/data/Distribution of revenue.png')
    plt.show()
    plt.close()

    sns.distplot(np.log1p(df['revenue']))
    plt.ylabel('distribution')
    plt.xlabel('log-revenue')
    plt.title('Distribution of log-revenue')
    plt.savefig('D:/Gra1/computing/data/Distribution of log-revenue.png')
    plt.show()
    plt.close()

def lan_revenue():
    df = pd.read_csv('D:/Gra1/computing/archive/new_m.csv')
    sns.boxplot(x='original_language',y=np.log1p(df['revenue']),
                data=df.loc[df['original_language'].isin(df['original_language'].value_counts().head(10).index)])
    plt.title("Log_Revenue VS Original_language")
    plt.savefig('D:/Gra1/computing/data/Log_Revenue VS Original_language.png')
    plt.show()
    plt.close()


def heat():
    df = pd.read_csv('D:/Gra1/computing/archive/new_m.csv')
    x = df[['originalBudget','vote_average','vote_count','popularity','runtime','release_year','release_month','release_dayofweek','revenue']]
    x_coor = x.corr()
    # plt.subplots(figsize=(9, 9), dpi=1080, facecolor='w')  
    sns.heatmap(x_coor, annot=True, square=True, cmap="Blues",fmt='.2g')  
    plt.savefig('D:/Gra1/computing/data/df_cor.png', bbox_inches='tight')
    plt.show()
    plt.close()

def genres():
    df = pd.read_csv('D:/Gra1/computing/archive/new_m.csv')
    df['num_genres'].value_counts()
    sns.catplot(x='num_genres',y='revenue',data=df)
    plt.savefig('D:/Gra1/computing/data/num_genres.png', bbox_inches='tight')
    plt.show()
    plt.close()

def overview():
    df = pd.read_csv('D:/Gra1/computing/archive/test_df_1.csv')
    df['overview_word_count'].value_counts()
    sns.catplot(x='overview_word_count',y='revenue',data=df)
    plt.xticks(rotation=90)
    plt.savefig('D:/Gra1/computing/data/overview.png', bbox_inches='tight')
    plt.show()
    plt.close()

def data_view():
    df = pd.read_csv('D:/Gra1/computing/archive/new_m.csv')
    df = df[df['budget']>0]
    df['log_revenue'] = np.log1p(df['revenue'])
    df['log_budget'] = df['budget']
    df_view = df[['revenue','log_revenue','log_budget','originalBudget','popularity','vote_average','vote_count','release_year','release_month','release_dayofweek',\
                  'release_quarter','runtime','isbelongs_to_collectionNA','num_genres','genre_Action','tagline_word_count']]
    df_view.to_csv('D:/Gra1/computing/archive/df_view.csv', index=False)

def data_stat():
    df = pd.read_csv('D:/Gra1/computing/archive/test_df_1.csv')
    print(df.info())
    df_stat = df[['revenue','originalBudget','popularity','vote_average','vote_count']]
    print(df_stat.describe())
    
revenue_and_budget()
release_year()
mean_year()
distribution_revenue()
lan_revenue()
heat()
genres()
data_view()
data_stat()
overview()
