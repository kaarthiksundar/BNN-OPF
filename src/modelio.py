import pickle 


def write_to_file(file, rng_key, sample_counts, params):     
    with open(file, 'wb') as f:
        pickle.dump((rng_key, sample_counts, params), f)


def read_from_file(file):
    with open(file, 'rb') as g:
        rng_key, sample_counts, params = pickle.load(g)
    return rng_key, sample_counts, params
