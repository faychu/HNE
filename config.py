class Config(object):
    def __init__(self):
        ## graph data
        self.file_path = "preprocess/data/final_records.pickle"
        ## embedding data
        self.embedding_filename = "dblp"
        ## hyperparameter
        self.struct = {'input_dim': 2085231, # nodes_num
                       'text_dim': None,
                       'output_dim': 300,
                       'layers': []}
        ## MODE: 0 -- FM, 1--NFM, 2--MFM,  3--NMFM
        self.mode = 0
        ## para for training
        self.batch_size = 2000
        self.epochs_limit = 2
        self.learning_rate = 0.01



