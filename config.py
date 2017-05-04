class Config(object):
    def __init__(self):
        ## graph data
        self.file_path = "./data/records.pickle"
        ## embedding data
        self.embedding_filename = "ca-Grac"
        ## hyperparameter
        self.struct = {'input_dim': None,
                       'text_dim': None,
                       'output_dim': 500,
                       'layers': [None, 500, 100]}
        self.mode = 0
        ## para for training
        self.batch_size = 1024
        self.epochs_limit = 1000
        self.learning_rate = 0.001



