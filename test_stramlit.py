import streamlit as st
from sklearn import metrics
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

def change_dataset(name, metric):
    index_data = {'penguins': 0,
                  'winequality': 3,
                  'iris': 2,
                  'banknote_authentication': 1}[name]
    
    
    dataset = prepare_the_dataset(df[index_data], dataset_name[index_data], categorical_variables)
    labels = make_the_clusters(n_clusters,dataset['X'],None,metric)
    metric_list_result = []
    for i in metric_list_web:
        dataset_temp = prepare_the_dataset(df[index_data], dataset_name[index_data], categorical_variables)
        labels_temp = make_the_clusters(n_clusters,dataset['X'],None,i)
        metric_list_result.append(metrics.adjusted_rand_score(labels_temp, dataset_temp['y']))
    return make_pictur_st(n_clusters, labels, dataset, reduction_method, metric, dataset_name[index_data]), metric_list_result


dataset_name_emoji = {'penguins': "\U0001F427\tPenguins (3)", 
                      'banknote_authentication': '\U0001F4B8\tBanknote (2)', 
                      'iris': "\U0001F33A\tIris (3)", 
                      'winequality': "\U0001F377\tWinequality (7)"}

metric_name = {"euclidean": "Euclidean", "cosine": "Cosine", "manhattan": "Manhattan", "minkowskinorm": "Minkowski"}
metric_list_web = ["euclidean", "cosine", "manhattan", "minkowskinorm"]
st.write("""
# Topic T4
""")

col1, col2 = st.beta_columns([3, 1])

select_data = col2.selectbox('Select dataset', dataset_name, format_func=lambda x: dataset_name_emoji[x])
n_clusters = col2.number_input(label='Number of Clusters (k)', min_value=2, max_value=20, key=4)
celect_metric = col2.selectbox('Select metric', metric_list_web, format_func=lambda x: metric_name[x])
reduction_method = col2.selectbox('Select Reduction Method', ['TSNE', 'PCA']) # ['TSNE (distributed Stochastic Neighbor Embedding)', 'PCA (Principal component analysis)']
tem_dataset = change_dataset(select_data, celect_metric)
col1.write(tem_dataset[0])


st.write(""" # Quality of Clustering 

## Rand Index

""")

# multiselect_datset = st.multiselect('Welche Datasets select:', dataset_name)
df_evaluation = pd.DataFrame()
df_evaluation['metric'] = ["Euclidean", "Cosine", "Manhattan", "Minkowski"]
df_evaluation['value'] = tem_dataset[1]
st.write(df_evaluation)

st.write('similarity measure between clustering algorithms and real world cluster (between -1.0 and 1.0)')
