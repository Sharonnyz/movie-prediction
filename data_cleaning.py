import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from math import sqrt



def feature_engineer(df):
    global json_cols
    global train_dict
    df['vote_average'] = df['vote_average'].fillna(1.5)
    df['vote_count'] = df['vote_count'].fillna(6)
    # df['weightedRating'] = (df['vote_average'] * df['vote_count'] + 6.367 * 300) / (df['vote_count'] + 300)

    releaseDate = pd.to_datetime(df['release_date'])
    df['release_dayofweek'] = releaseDate.dt.dayofweek
    df['release_quarter'] = releaseDate.dt.quarter

    df['originalBudget'] = df['budget']
    df['inflationBudget'] = df['budget'] + df['budget'] * 1.8 / 100 * (
            2019 - df['release_year'])  # Inflation simple formula
    df['budget'] = np.log1p(df['budget'])

    df['genders_0_crew'] = df['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 0]))
    df['genders_1_crew'] = df['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 1]))
    df['genders_2_crew'] = df['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 2]))
    # df['_collection_name'] = df['belongs_to_collection'].apply(lambda x: x[0]['name'] if x != {} else '').fillna('')
    # le = LabelEncoder()
    # df['_collection_name'] = le.fit_transform(df['_collection_name'])
    df['_num_Keywords'] = df['Keywords'].apply(lambda x: len(x) if x != {} else 0)
    df['_num_cast'] = df['cast'].apply(lambda x: len(x) if x != {} else 0)

    df['_popularity_mean_year'] = df['popularity'] / df.groupby("release_year")["popularity"].transform('mean')
    df['_budget_runtime_ratio'] = df['budget'] / df['runtime']
    df['_budget_popularity_ratio'] = df['budget'] / df['popularity']
    df['_budget_year_ratio'] = df['budget'] / (df['release_year'] * df['release_year'])
    df['_releaseYear_popularity_ratio'] = df['release_year'] / df['popularity']
    df['_releaseYear_popularity_ratio2'] = df['popularity'] / df['release_year']

    df['_popularity_vote_count_ratio'] = df['vote_count'] / df['popularity']
    df['_vote_average_popularity_ratio'] = df['vote_average'] / df['popularity']
    df['_vote_average_vote_count_ratio'] = df['vote_count'] / df['vote_average']
    df['_vote_count_releaseYear_ratio'] = df['vote_count'] / df['release_year']
    df['_budget_vote_average_ratio'] = df['budget'] / df['vote_average']
    df['_runtime_vote_average_ratio'] = df['runtime'] / df['vote_average']
    df['_budget_vote_count_ratio'] = df['budget'] / df['vote_count']

    df['isbelongs_to_collectionNA'] = 0
    df.loc[pd.isnull(df['belongs_to_collection']), "isbelongs_to_collectionNA"] = 1

    df['isTaglineNA'] = 0
    df.loc[df['tagline'] == 0, "isTaglineNA"] = 1

    df['isOriginalLanguageEng'] = 0
    df.loc[df['original_language'] == "en", "isOriginalLanguageEng"] = 1

    df['isTitleDifferent'] = 1
    df.loc[df['original_title'] == df['title'], "isTitleDifferent"] = 0

    df['isMovieReleased'] = 1
    df.loc[df['status'] != "Released", "isMovieReleased"] = 0

    # # collections
    # df['collection_name'] = df['belongs_to_collection'].apply(lambda x: x[0]['name'] if x != {} else 0)


    df['original_title_letter_count'] = df['original_title'].str.len()
    df['original_title_word_count'] = df['original_title'].str.split().str.len()

    df['title_word_count'] = df['title'].str.split().str.len()
    df['overview_word_count'] = df['overview'].str.split().str.len()
    df['tagline_word_count'] = df['tagline'].str.split().str.len()

    df['production_countries_count'] = df['production_countries'].apply(lambda x: len(x))
    df['production_companies_count'] = df['production_companies'].apply(lambda x: len(x))
    df['crew_count'] = df['crew'].apply(lambda x: len(x) if x != {} else 0)

    # genres
    list_of_genres = list(df['genres'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)
    df['num_genres'] = df['genres'].apply(lambda x: len(x) if x != {} else 0)
    df['all_genres'] = df['genres'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')
    top_genres = [m[0] for m in Counter([i for j in list_of_genres for i in j]).most_common(15)]
    for g in top_genres:
        df['genre_' + g] = df['all_genres'].apply(lambda x: 1 if g in x else 0)


    # df['meanruntimeByYear'] = df.groupby("release_year")["runtime"].aggregate('mean')
    # df['meanPopularityByYear'] = df.groupby("release_year")["popularity"].aggregate('mean')
    # df['meanBudgetByYear'] = df.groupby("release_year")["budget"].aggregate('mean')
    # df['meantotalVotesByYear'] = df.groupby("release_year")["totalVotes"].aggregate('mean')
    # df['meanTotalVotesByRating'] = df.groupby("rating")["totalVotes"].aggregate('mean')
    # df['medianBudgetByYear'] = df.groupby("release_year")["budget"].aggregate('median')

    # for col in ['genres', 'production_countries', 'spoken_languages', 'production_companies']:
    #     df[col] = df[col].map(lambda x: sorted(
    #         list(set([n if n in train_dict[col] else col + '_etc' for n in [d['name'] for d in x]])))) \
    #         .map(lambda x: ','.join(map(str, x)))
    #     temp = df[col].str.get_dummies(sep=',')
    #     df = pd.concat([df, temp], axis=1, sort=False)
    # df.drop(['genres_etc'], axis=1, inplace=True)

    # df = df.drop(['id', 'revenue', 'belongs_to_collection', 'genres', 'homepage', 'imdb_id', 'overview', 'runtime'
    #                  , 'poster_path', 'production_companies', 'production_countries', 'release_date', 'spoken_languages'
    #                  , 'status', 'title', 'Keywords', 'cast', 'crew', 'original_language', 'original_title', 'tagline',
    #               'collection_id'
    #               ], axis=1)

    df = df.drop(['belongs_to_collection', 'homepage','poster_path','genres', 'all_genres', 'imdb_id', 'overview'
                     , 'production_companies', 'production_countries', 'release_date', 'spoken_languages'
                     , 'status', 'title', 'Keywords', 'cast', 'crew', 'original_language', 'original_title', 'tagline'], axis=1)

    df[np.isinf(df)] = np.nan
    df.fillna(value=0.0, inplace=True)
    df.dropna(inplace=True)
    return df


def get_dictionary(s):
    try:
        d = eval(s)
    except:
        d = {}
    return d


def get_json_dict(df) :
    global json_cols
    result = dict()
    for e_col in json_cols :
        d = dict()
        rows = df[e_col].values
        for row in rows :
            if row is None : continue
            for i in row :
                if i['name'] not in d :
                    d[i['name']] = 0
                d[i['name']] += 1
        result[e_col] = d
    return result


def data_prepare(df,train,test):
    global json_cols
    global train_dict
    # for train

    train[['release_month', 'release_day', 'release_year']] = train['release_date'].str.split('/', expand=True).replace(
        np.nan, 0).astype(int)
    train['release_year'] = train['release_year']
    train.loc[(train['release_year'] <= 19) & (train['release_year'] < 100), "release_year"] += 2000
    train.loc[(train['release_year'] > 19) & (train['release_year'] < 100), "release_year"] += 1900

    # for test
    test[['release_month', 'release_day', 'release_year']] = test['release_date'].str.split('/', expand=True).replace(
        np.nan, 0).astype(int)
    test['release_year'] = test['release_year']
    test.loc[(test['release_year'] <= 19) & (test['release_year'] < 100), "release_year"] += 2000
    test.loc[(test['release_year'] > 19) & (test['release_year'] < 100), "release_year"] += 1900

    # for df
    df[['release_year','release_month', 'release_day']] = df['release_date'].str.split('-', expand=True).replace(
        np.nan, 0).astype(int)



    new_m_1 = df[df['title']=='Inhumans: The First Chapter']

    df = pd.concat([df,train])
    # print('new_m1',df.info())
    df.drop_duplicates(subset=['imdb_id'], keep='first', inplace=True)
    df = pd.concat([df,new_m_1])
    df_1 = feature_engineer(df)


    # new_m.drop(new_m.columns[-1],axis=1, inplace=True)
    df_1.to_csv('D:/Gra1/computing/archive/new_m_2.csv', index=False)
    print('new_m2',df_1.info())




    test_df = pd.concat([df, test])
    # print('test_df1',test_df.info())
    test_df.drop_duplicates(subset=['imdb_id'], keep='first', inplace=True)
    test_df_1 = feature_engineer(test_df)
    test_df_1.dropna(inplace=True)
    print('test_df_1',test_df_1.info())
    test_df_1.to_csv('D:/Gra1/computing/archive/test_df_1.csv', index=False)
    test_df_1['log_revenue'] = np.log1p(test_df_1['revenue'])
    y = test_df_1['log_revenue']
    test_df_1 = test_df_1.drop(['log_revenue'], axis=1)

    test_df_1.fillna(value=0.0, inplace=True)

    # X, y = my_matrix[:, :-1], my_matrix[:, -1]
    X_train_1, X_test_1, y_train, y_test = train_test_split(test_df_1, y, test_size=0.3, random_state=42)
    # train = test_df_1.loc[:df_1.shape[0] - 1, :]
    # test = test_df_1.loc[df_1.shape[0]:, :]
    X_train_1.to_csv('D:/Gra1/computing/archive/X_train_1.csv', index=False)
    X_test_1.to_csv('D:/Gra1/computing/archive/X_test_1.csv', index=False)
    X_train = X_train_1.drop(['revenue','id'], axis=1)
    X_train.fillna(value=0.0, inplace=True)
    X_test = X_test_1.drop(['revenue','id'], axis=1)
    X_test.fillna(value=0.0, inplace=True)
    print(train.columns)
    print(train.shape)
    X_train.to_csv('D:/Gra1/computing/archive/X_train.csv', index=False)
    X_test.to_csv('D:/Gra1/computing/archive/X_test.csv', index=False)
    y_train.to_csv('D:/Gra1/computing/archive/y_train.csv', header=True, index=False)
    y_test.to_csv('D:/Gra1/computing/archive/y_test.csv', header=True, index=False)

# def stat_plot():

if __name__ == '__main__':
    # tmdb = pd.read_csv('D:/Gra1/computing/archive/movies.csv')
    # test = pd.read_csv('D:/Gra1/computing/archive/test.csv')
    # train = pd.read_csv('D:/Gra1/computing/archive/train.csv')

    # print(tmdb.info())


    # print(test.info())
    # movies = pd.merge(tmdb, train, on="imdb_id", how="left")
    # movies.to_csv('D:/Gra1/computing/archive/concat.csv')
    # print(movies.info())
    # print(tmdb.dtypes.sort_values())
    # df = tmdb[tmdb['revenue']>0]
    # df.to_csv('D:/Gra1/computing/archive/df.csv', index=False)
    df=pd.read_csv('D:/Gra1/computing/archive/df.csv')
    # list1=[column for column in train]
    # list2=[column for column in df]
    # list = [item for item in list1 if item not in set(list2)]
    # print(list)
    # list3 = df['imdb_id'].to_list
    # list4 = train['imdb_id'].to_list
    # print(list3,list4)
    # #
    test1 = pd.read_csv('D:/Gra1/computing/archive/test1.csv')
    train1 = pd.read_csv('D:/Gra1/computing/archive/train1.csv')
    # clean train
    # train = pd.merge(train, pd.read_csv('D:/Gra1/computing/archive/TrainAdditionalFeatures.csv'),
    #                  how='left', on=['imdb_id'])
    # test = pd.merge(test, pd.read_csv('D:/Gra1/computing/archive/TestAdditionalFeatures.csv'),
    #                 how='left', on=['imdb_id'])
    # train.rename(columns={'rating':'vote_average','totalVotes':'vote_count'},inplace=True)
    # test.rename(columns={'rating':'vote_average','totalVotes':'vote_count'},inplace=True)
    # train.loc[train['id'] == 16, 'revenue'] = 192864  # Skinning
    # train.loc[train['id'] == 90, 'budget'] = 30000000  # Sommersby
    # train.loc[train['id'] == 118, 'budget'] = 60000000  # Wild Hogs
    # train.loc[train['id'] == 149, 'budget'] = 18000000  # Beethoven
    # train.loc[train['id'] == 313, 'revenue'] = 12000000  # The Cookout
    # train.loc[train['id'] == 451, 'revenue'] = 12000000  # Chasing Liberty
    # train.loc[train['id'] == 464, 'budget'] = 20000000  # Parenthood
    # train.loc[train['id'] == 470, 'budget'] = 13000000  # The Karate Kid, Part II
    # train.loc[train['id'] == 513, 'budget'] = 930000  # From Prada to Nada
    # train.loc[train['id'] == 797, 'budget'] = 8000000  # Welcome to Dongmakgol
    # train.loc[train['id'] == 819, 'budget'] = 90000000  # Alvin and the Chipmunks: The Road Chip
    # train.loc[train['id'] == 850, 'budget'] = 90000000  # Modern Times
    # train.loc[train['id'] == 1007, 'budget'] = 2  # Zyzzyx Road
    # train.loc[train['id'] == 1112, 'budget'] = 7500000  # An Officer and a Gentleman
    # train.loc[train['id'] == 1131, 'budget'] = 4300000  # Smokey and the Bandit
    # train.loc[train['id'] == 1359, 'budget'] = 10000000  # Stir Crazy
    # train.loc[train['id'] == 1542, 'budget'] = 1  # All at Once
    # train.loc[train['id'] == 1570, 'budget'] = 15800000  # Crocodile Dundee II
    # train.loc[train['id'] == 1571, 'budget'] = 4000000  # Lady and the Tramp
    # train.loc[train['id'] == 1714, 'budget'] = 46000000  # The Recruit
    # train.loc[train['id'] == 1721, 'budget'] = 17500000  # Cocoon
    # train.loc[train['id'] == 1865, 'revenue'] = 25000000  # Scooby-Doo 2: Monsters Unleashed
    # train.loc[train['id'] == 1885, 'budget'] = 12  # In the Cut
    # train.loc[train['id'] == 2091, 'budget'] = 10  # Deadfall
    # train.loc[train['id'] == 2268, 'budget'] = 17500000  # Madea Goes to Jail budget
    # train.loc[train['id'] == 2491, 'budget'] = 6  # Never Talk to Strangers
    # train.loc[train['id'] == 2602, 'budget'] = 31000000  # Mr. Holland's Opus
    # train.loc[train['id'] == 2612, 'budget'] = 15000000  # Field of Dreams
    # train.loc[train['id'] == 2696, 'budget'] = 10000000  # Nurse 3-D
    # train.loc[train['id'] == 2801, 'budget'] = 10000000  # Fracture
    # train.loc[train['id'] == 335, 'budget'] = 2
    # train.loc[train['id'] == 348, 'budget'] = 12
    # train.loc[train['id'] == 470, 'budget'] = 13000000
    # train.loc[train['id'] == 513, 'budget'] = 1100000
    # train.loc[train['id'] == 640, 'budget'] = 6
    # train.loc[train['id'] == 696, 'budget'] = 1
    # train.loc[train['id'] == 797, 'budget'] = 8000000
    # train.loc[train['id'] == 850, 'budget'] = 1500000
    # train.loc[train['id'] == 1199, 'budget'] = 5
    # train.loc[train['id'] == 1282, 'budget'] = 9  # Death at a Funeral
    # train.loc[train['id'] == 1347, 'budget'] = 1
    # train.loc[train['id'] == 1755, 'budget'] = 2
    # train.loc[train['id'] == 1801, 'budget'] = 5
    # train.loc[train['id'] == 1918, 'budget'] = 592
    # train.loc[train['id'] == 2033, 'budget'] = 4
    # train.loc[train['id'] == 2118, 'budget'] = 344
    # train.loc[train['id'] == 2252, 'budget'] = 130
    # train.loc[train['id'] == 2256, 'budget'] = 1
    # train.loc[train['id'] == 2696, 'budget'] = 10000000
    # train.to_csv('D:/Gra1/computing/archive/train1.csv', index=False)
    #
    # test.loc[test['id'] == 6733, 'budget'] = 5000000
    # test.loc[test['id'] == 3889, 'budget'] = 15000000
    # test.loc[test['id'] == 6683, 'budget'] = 50000000
    # test.loc[test['id'] == 5704, 'budget'] = 4300000
    # test.loc[test['id'] == 6109, 'budget'] = 281756
    # test.loc[test['id'] == 7242, 'budget'] = 10000000
    # test.loc[test['id'] == 7021, 'budget'] = 17540562  # Two Is a Family
    # test.loc[test['id'] == 5591, 'budget'] = 4000000  # The Orphanage
    # test.loc[test['id'] == 4282, 'budget'] = 20000000  # Big Top Pee-wee
    # test.loc[test['id'] == 3033, 'budget'] = 250
    # test.loc[test['id'] == 3051, 'budget'] = 50
    # test.loc[test['id'] == 3084, 'budget'] = 337
    # test.loc[test['id'] == 3224, 'budget'] = 4
    # test.loc[test['id'] == 3594, 'budget'] = 25
    # test.loc[test['id'] == 3619, 'budget'] = 500
    # test.loc[test['id'] == 3831, 'budget'] = 3
    # test.loc[test['id'] == 3935, 'budget'] = 500
    # test.loc[test['id'] == 4049, 'budget'] = 995946
    # test.loc[test['id'] == 4424, 'budget'] = 3
    # test.loc[test['id'] == 4460, 'budget'] = 8
    # test.loc[test['id'] == 4555, 'budget'] = 1200000
    # test.loc[test['id'] == 4624, 'budget'] = 30
    # test.loc[test['id'] == 4645, 'budget'] = 500
    # test.loc[test['id'] == 4709, 'budget'] = 450
    # test.loc[test['id'] == 4839, 'budget'] = 7
    # test.loc[test['id'] == 3125, 'budget'] = 25
    # test.loc[test['id'] == 3142, 'budget'] = 1
    # test.loc[test['id'] == 3201, 'budget'] = 450
    # test.loc[test['id'] == 3222, 'budget'] = 6
    # test.loc[test['id'] == 3545, 'budget'] = 38
    # test.loc[test['id'] == 3670, 'budget'] = 18
    # test.loc[test['id'] == 3792, 'budget'] = 19
    # test.loc[test['id'] == 3881, 'budget'] = 7
    # test.loc[test['id'] == 3969, 'budget'] = 400
    # test.loc[test['id'] == 4196, 'budget'] = 6
    # test.loc[test['id'] == 4221, 'budget'] = 11
    # test.loc[test['id'] == 4222, 'budget'] = 500
    # test.loc[test['id'] == 4285, 'budget'] = 11
    # test.loc[test['id'] == 4319, 'budget'] = 1
    # test.loc[test['id'] == 4639, 'budget'] = 10
    # test.loc[test['id'] == 4719, 'budget'] = 45
    # test.loc[test['id'] == 4822, 'budget'] = 22
    # test.loc[test['id'] == 4829, 'budget'] = 20
    # test.loc[test['id'] == 4969, 'budget'] = 20
    # test.loc[test['id'] == 5021, 'budget'] = 40
    # test.loc[test['id'] == 5035, 'budget'] = 1
    # test.loc[test['id'] == 5063, 'budget'] = 14
    # test.loc[test['id'] == 5119, 'budget'] = 2
    # test.loc[test['id'] == 5214, 'budget'] = 30
    # test.loc[test['id'] == 5221, 'budget'] = 50
    # test.loc[test['id'] == 4903, 'budget'] = 15
    # test.loc[test['id'] == 4983, 'budget'] = 3
    # test.loc[test['id'] == 5102, 'budget'] = 28
    # test.loc[test['id'] == 5217, 'budget'] = 75
    # test.loc[test['id'] == 5224, 'budget'] = 3
    # test.loc[test['id'] == 5469, 'budget'] = 20
    # test.loc[test['id'] == 5840, 'budget'] = 1
    # test.loc[test['id'] == 5960, 'budget'] = 30
    # test.loc[test['id'] == 6506, 'budget'] = 11
    # test.loc[test['id'] == 6553, 'budget'] = 280
    # test.loc[test['id'] == 6561, 'budget'] = 7
    # test.loc[test['id'] == 6582, 'budget'] = 218
    # test.loc[test['id'] == 6638, 'budget'] = 5
    # test.loc[test['id'] == 6749, 'budget'] = 8
    # test.loc[test['id'] == 6759, 'budget'] = 50
    # test.loc[test['id'] == 6856, 'budget'] = 10
    # test.loc[test['id'] == 6858, 'budget'] = 100
    # test.loc[test['id'] == 6876, 'budget'] = 250
    # test.loc[test['id'] == 6972, 'budget'] = 1
    # test.loc[test['id'] == 7079, 'budget'] = 8000000
    # test.loc[test['id'] == 7150, 'budget'] = 118
    # test.loc[test['id'] == 6506, 'budget'] = 118
    # test.loc[test['id'] == 7225, 'budget'] = 6
    # test.loc[test['id'] == 7231, 'budget'] = 85
    # test.loc[test['id'] == 5222, 'budget'] = 5
    # test.loc[test['id'] == 5322, 'budget'] = 90
    # test.loc[test['id'] == 5350, 'budget'] = 70
    # test.loc[test['id'] == 5378, 'budget'] = 10
    # test.loc[test['id'] == 5545, 'budget'] = 80
    # test.loc[test['id'] == 5810, 'budget'] = 8
    # test.loc[test['id'] == 5926, 'budget'] = 300
    # test.loc[test['id'] == 5927, 'budget'] = 4
    # test.loc[test['id'] == 5986, 'budget'] = 1
    # test.loc[test['id'] == 6053, 'budget'] = 20
    # test.loc[test['id'] == 6104, 'budget'] = 1
    # test.loc[test['id'] == 6130, 'budget'] = 30
    # test.loc[test['id'] == 6301, 'budget'] = 150
    # test.loc[test['id'] == 6276, 'budget'] = 100
    # test.loc[test['id'] == 6473, 'budget'] = 100
    # test.loc[test['id'] == 6842, 'budget'] = 30
    # test.to_csv('D:/Gra1/computing/archive/test1.csv', index=False)

    # trate = pd.concat([train1, test1])
    # trate_1 = trate[['imdb_id','Keywords','cast','crew']]
    # df = pd.merge(df, trate_1,how='left', on=['imdb_id'])
    # df.to_csv('D:/Gra1/computing/archive/df.csv', index=False)

    json_cols = ['genres', 'production_companies', 'production_countries', 'spoken_languages', 'Keywords', 'cast',
                     'crew']

    for col in tqdm(json_cols + ['belongs_to_collection']):
        train1[col] = train1[col].apply(lambda x: get_dictionary(x))
        test1[col] = test1[col].apply(lambda x: get_dictionary(x))
        df[col] = df[col].apply(lambda x: get_dictionary(x))

    train_dict = get_json_dict(train1)
    test_dict = get_json_dict(test1)
    df_dict = get_json_dict(df)

    # remove cateogry with bias and low frequency
    for col in json_cols:
        remove = []
        train_id = set(list(train_dict[col].keys()))
        test_id = set(list(test_dict[col].keys()))
        # df_id = set(list(df_dict[col].keys()))

        remove += list(train_id - test_id) + list(test_id - train_id)
        for i in train_id.union(test_id) - set(remove):
            if train_dict[col][i] < 10 or i == '':
                remove += [i]

        for i in remove:
            if i in train_dict[col]:
                del train_dict[col][i]
            if i in test_dict[col]:
                del test_dict[col][i]

    data_prepare(df,train1,test1)
