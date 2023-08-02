num_epochs = 2
num_batches_show_loss = 100
num_batches_validate = 1000
batch_size = 128
learning_rate = 0.0001
num_workers = 4
num_clicked_news_a_user = 50
num_words_title = 20
num_words_abstract = 50
word_freq_threshold = 1
entity_freq_threshold = 2
entity_confidence_threshold = 0.5
negative_sampling_ratio = 2  # K
dropout_probability = 0.2
num_words = 1 + 70974
num_categories = 1 + 274
num_entities = 1 + 12957
num_users = 1 + 50000
word_embedding_dim = 300
category_embedding_dim = 100
entity_embedding_dim = 100
query_vector_dim = 200
dataset_attributes = {
    "news": ['category', 'subcategory', 'title'],
    "record": ['user', 'clicked_news_length']
}
# CNN
num_filters = 300
window_size = 3
masking_probability = 0.5


