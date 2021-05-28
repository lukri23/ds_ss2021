import streamlit as st
from ipynb.fs.full.group_4 import *



def make_pictur_st(n_clusters, labels, data, reduction_method, metric, dataset_name):
    #alpha passt sich dynamisch an, je größer datenset desto niedriger alpha
    alpha = 1.2**(-1*(len(data)/1000))

    if reduction_method == 'TSNE':
        perplexity=n_clusters
        reduced_data = TSNE(n_components=2,perplexity=n_clusters, random_state=42).fit_transform(data['X'])

    elif reduction_method == 'PCA':
        reduced_data = PCA(n_components=2).fit_transform(data['X'])

    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(15,7.5))
    fig.suptitle('Output Clustering projected with {} for dataset {}'.format(reduction_method,dataset_name))
    ax1.scatter(reduced_data[:,0],reduced_data[:,1], c=labels,alpha=alpha)
    ax2.scatter(reduced_data[:,0],reduced_data[:,1], c=data['y'],alpha=alpha)

    ax1.title.set_text('Clusters from KMedoids with metric {}'.format(metric))
    ax2.title.set_text('Real Clusters')

    return fig
    
def change_dataset(name):
    index_data = {'penguins': 0,
                  'winequality': 3,
                  'iris': 2,
                  'banknote_authentication': 1}[name]
    
    
    dataset = prepare_the_dataset(df[index_data], dataset_name[index_data], categorical_variables)
    labels = make_the_clusters(n_clusters,dataset['X'],None,metric)
    return make_pictur_st(n_clusters, labels, dataset, reduction_method, metric, dataset_name[index_data])



st.write("""
# Überschrift 2

Hallo hier ist meine erste App!!!
""")

select_data = st.selectbox('select dataset', dataset_name)
n_clusters = st.slider(label='Select k', min_value=2, max_value=5, key=4)

st.write(change_dataset(select_data))

