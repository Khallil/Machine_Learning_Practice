from csv import reader

def load_csv(filename):
    file = open(filename,"r")
    lines = reader(file)
    dataset = list(lines)
    return dataset

filename = 'pima-indians-diabetes.csv'
dataset = load_csv(filename)
print('Loaded data file {0} with {1} rows and {2} columns'.format(filename,len(dataset),len(dataset[0])))

