import os
import deeptrack as dt
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output

class MAGIK():
    """

    Parameters
    ----------
    weights_path : str
        Path to model weights.
    spat_temp_res: int
        Maximum frames between two nodes with an edge for the input graph.
    th: int
        Minimum number of frames a trajectory should have in order to be created.
    dataframe: None or str
        Path to the file with detections.
    data_shape: tuple
        Shape of the data.
    """
    def __init__(self, 
                 weights_path = os.getcwd()+"/MAGIK.h5",
                 spat_temp_res = 3,
                 th = 1,
                 dataframe = None,
                 data_shape = (1,800,750)):

        self.weigths_path = weights_path
        self.spat_temp_res = spat_temp_res
        self.th = th
        self.dataframe = dataframe
        self.data_shape = data_shape

    def load_dataframe(self, path):
        """Load a dataframe from path.
        centroid-0 is assumed to be x position and centroid-1 to be y position."""

        self.dataframe = pd.read_csv(path)
        self.dataframe['centroid-0'] /= self.data_shape[2]
        self.dataframe['centroid-1'] /= self.data_shape[1]

    def set_dataset(self, dataset):
        """Set dataset that is used for showing detections."""

        self.dataset = dataset

    def load_network(self, weigths_path):
        """Load pre-trained MAGIK network from weigths_path."""

        self._OUTPUT_TYPE = 'edges'
        self.radius = 0.03
        self.nofframes = 4
        
        self.model = dt.models.gnns.MAGIK(
            dense_layer_dimensions=(64, 96,),      # number of features in each dense encoder layer
            base_layer_dimensions=(96, 96, 96),    # Latent dimension throughout the message passing layers
            number_of_node_features=2,             # Number of node features in the graphs
            number_of_edge_features=1,             # Number of edge features in the graphs
            number_of_edge_outputs=1,              # Number of predicted features
            edge_output_activation="sigmoid",      # Activation function for the output layer
            output_type=self._OUTPUT_TYPE,              # Output type. Either "edges", "nodes", or "graph"
        )
        
        # Compile model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss = 'binary_crossentropy',
            metrics=['accuracy'],
        )
        self.model.load_weights(weigths_path)
    
    def create_nodes(self, spat_temp_res):
        """Create nodes with a spatio temporal resolution of spat_temp_res."""

        variables = dt.DummyFeature(
            radius=self.radius,
            output_type=self._OUTPUT_TYPE,
            nofframes=spat_temp_res, # time window to associate nodes (in frames) 
        )
        
        
        pred, gt, scores, graph = dt.models.gnns.get_predictions(
            self.dataframe, ["centroid"], self.model, **variables.properties()
        )

        self.edges_df, self.nodes, _ = dt.models.gnns.df_from_results(pred, gt, scores, graph)

    def create_trajectories(self, th):
        """Create trajectories with a threshold of th."""

        self.traj = dt.models.gnns.get_traj(self.edges_df, th=th)

    def setup(self, **kwargs):
        """Setup the nodes, edges and trajectories."""

        self.load_network(kwargs.get('weigths_path', self.weigths_path))
        self.create_nodes(kwargs.get('spat_time_res', self.spat_temp_res))
        self.create_trajectories(kwargs.get('th', self.th))
        if kwargs.get('clear', True):
            clear_output()

    def detect(self, dataset_frames=(0,10), setup=False, **kwargs):
        """Plot the detections and trajectories from MAGIK.
        Note again that centroid-0 is assumed to be the x position and centroid-1 the y position."""
        # Note again that centroid-0 is assumed to be the x position and centroid-1 the y position 

        if not hasattr(self, 'traj') or setup:
            self.setup(**kwargs)
        f=dataset_frames[0]
        for frame in self.dataset[dataset_frames[0]:dataset_frames[1]]:
        
            fig = plt.figure()
            plt.imshow(frame.squeeze(), cmap='gray')
            plt.text(10, 40, "Bild: " + str(f), fontsize=20, c="white")
            plt.axis("off")

            for i, (t, c) in enumerate(self.traj):
                detections = self.nodes[t][(self.nodes[t, 0] <= f) & (self.nodes[t, 0] >= f - 20), :]

                if (len(detections) == 0) or (np.max(self.nodes[t, 0]) < f):
                    continue
                
                plt.plot(detections[:, 1] * self.data_shape[2], detections[:, 2] * self.data_shape[1], color = c, linewidth=2)
                plt.scatter(detections[-1, 1] * self.data_shape[2], detections[-1, 2] * self.data_shape[1], linewidths=1.5, c = c)
                
            f += 1
            plt.show()