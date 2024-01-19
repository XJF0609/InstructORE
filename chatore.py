from text_classifier import TextClassifier
from clustering_algorithm import DeepClustering
from sklearn.cluster import KMeans
from tqdm import tqdm

class ChatORE:
    def __init__(self, settings):
        self.cluster_size = settings['k']
        self.iterations = settings['loop']
        algorithm = settings['cluster']
        
        self.clustering_model = None
        self.clustering_model = DeepClustering(n_clusters=self.cluster_size, input_dim=settings['bert_max_len']*768)

        self.text_classifier = TextClassifier(
            num_classes=self.cluster_size,
            sentence_file_path=settings['sentence_path'],
            max_sequence_length=settings['bert_max_len'],
            batch_size=settings['batch_size'],
            num_epochs=settings['epoch']
        )

    def process_cycle(self):
        bert_embeddings = self.text_classifier.get_hidden_states()
        labels = self.clustering_model.fit(bert_embeddings).labels_
        self.text_classifier.train_model(labels)
        

    def execute(self):
        for _ in tqdm(range(self.iterations)):
            self.process_cycle()
