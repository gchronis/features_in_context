import pickle
from clustering import obtain_clusterings, plot_usage_distribution


# options: 'bert', 'binder', 'buchanan'
vector_type = 'binder'


with open('data/cwr4lsc/{}/matrix_usages_16_len128.dict'.format(vector_type), 'rb') as f:
    usages = pickle.load(f)

clusterings = obtain_clusterings(
    usages,
    out_path='data/cwr4lsc/{}/usages_len128.clustering.2.dict'.format(vector_type),
    method='kmeans',
    criterion='silhouette'
)

with open('data/cwr4lsc/{}/usages_len128.clustering.2.dict'.format(vector_type), 'rb') as f:
    clusterings = pickle.load(f)
    plot_usage_distribution(usages, clusterings, '/home/gsc685/features_in_context/data/cwr4lsc/plots/{}'.format(vector_type), normalized=True)
