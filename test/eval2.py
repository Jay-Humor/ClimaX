
import pickle
def to_pickle(obj, fn):
    with open(fn, 'wb') as f:
        pickle.dump(obj, f)
def read_pickle(fn):
    with open(fn, 'rb') as f:
        return pickle.load(f)

DATADIR = f'/home/humor/sugon/ClimaX/test/weatherbench/'
PREDDIR = '/home/humor/sugon/ClimaX/test/preds/'
OTHERDIR = '/home/humor/sugon/ClimaX/temp/'

rmse = read_pickle(f'{OTHERDIR}rmse.pkl')
acc = read_pickle(f'{OTHERDIR}acc.pkl')
mae = read_pickle(f'{OTHERDIR}mae.pkl')