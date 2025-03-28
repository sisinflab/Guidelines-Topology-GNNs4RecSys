import pandas as pd

dataset = 'yelp-2018'
train_filename = 'train_sparse.csv'
test_filename = 'test_sparse.csv'

if dataset == 'yelp-2018':
    train = pd.read_csv('./data/{0}/{1}'.format(dataset, train_filename), sep=',', header=[0, 1])
    test = pd.read_csv('./data/{0}/{1}'.format(dataset, test_filename), sep=',', header=[0, 1])
    train.columns = [0, 1]
    test.columns = [0, 1]
else:
    rows, cols = [], []

    with open('./data/{0}/{1}'.format(dataset, train_filename), 'r') as f:
        for line in f:
            all_elements = line.split(' ')
            if '\n' not in all_elements:
                for el in all_elements[1:]:
                    rows.append(int(all_elements[0]))
                    cols.append(int(el))

    train = pd.concat([pd.Series(rows), pd.Series(cols)], axis=1)

    rows, cols = [], []

    with open('./data/{0}/{1}'.format(dataset, test_filename), 'r') as f:
        for line in f:
            all_elements = line.split(' ')
            if '\n' not in all_elements:
                for el in all_elements[1:]:
                    rows.append(int(all_elements[0]))
                    cols.append(int(el))

    test = pd.concat([pd.Series(rows), pd.Series(cols)], axis=1)

df = pd.concat([train, test], axis=0).sort_values(0).reset_index(drop=True)
df.to_csv('./data/{0}/dataset.tsv'.format(dataset), sep='\t', header=None, index=None)
