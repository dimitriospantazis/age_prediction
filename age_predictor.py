from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD, PCA, NMF
from sklearn.manifold import TSNE, Isomap
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from umap import UMAP
import matplotlib.pyplot as plt

import numpy as np 
import networkx as nx
from typing import List
from typing import Tuple    
import time
from tqdm import tqdm
import numpy as np
import os 
import pickle

NUM_ROIS = 360
date = "2023_6_8"
log_num = '9'
log_path = os.path.join("logs", "lp", date, log_num)

age_labels = np.load(os.path.join("data", "cam_can_multiple", "age_labels_592_sbj_filtered.npy"))
embeddings_dir = os.path.join(log_path, 'embeddings')

class Age_Predictor:
    def __init__(self, type_of_regression: str, projection_type: str="SQ-R", architecture: str="FHNN", dataset: str="Cam-CAN"):
        type_of_regression = type_of_regression.lower()
        projection_type = projection_type.replace("-", "").upper()
        self.architecture = architecture
        self.dataset = dataset
        if type_of_regression == "linear":
            self.regressor_model = LinearRegression()
        elif type_of_regression == "ridge":
            self.regressor_model = Ridge(alpha = 100.0)

        elif type_of_regression == "polynomial":
            raise AssertionError("Polynomial Regression not implemented yet!")
            poly = PolynomialFeatures(degree=2)
            embeddings_poly = poly.fit_transform(embeddings)
            self.regressor_model = LinearRegression()
        elif type_of_regression == "hyperbolic":
            raise AssertionError("Hyperbolic Regression not implemented yet!")
            self.regressor_model = HyperbolicCentroidRegression()
        else:
            raise AssertionError(f"Invalid Regression type : {type_of_regression}!")
        train_indices = []
        for train_embeddings_dir in os.listdir(embeddings_dir):
            if 'train' not in train_embeddings_dir: continue
            _, _, train_index_str = train_embeddings_dir.split("_")
            train_index, _ = train_index_str.split(".")
            train_index = int(train_index)
            train_indices.append(train_index)
        
        self.train_indices = train_indices
        self.train_age_labels = [age_labels[train_index] for train_index in train_indices]
        
        test_indices = []
        for test_embeddings_dir in os.listdir(embeddings_dir):
            if 'test' not in test_embeddings_dir: continue
            _, _, test_index_str = test_embeddings_dir.split("_")
            test_index, _ = test_index_str.split(".")
            test_index = int(test_index)
            test_indices.append(test_index)
        
        self.test_indices = test_indices
        self.test_age_labels = [age_labels[test_index] for test_index in test_indices]
        self.model_str = "Linear" if type(self.regressor_model) == LinearRegression else "Ridge"
        self.projection_type = projection_type
    # 1. Data MEG MRI fMRI
    #   1.1. Access PLV Subject Matrices 
    #
    # 2. Get Adjacency Matrix
    #    Define Threshold
    #    Binarize Matrix
    #   
    # 3. Create Brain Graph
    #    Training Set    
    #    Validation Set 
    #    Test Set
    #
    # 4. Create HGCNN Embeddings
    #    
    #    Visualize Embeddings by plotting in Poincare Disk --> Drew Wilimitis Code
    # 
    # 5. Ridge Regression
    #    Hyperbolic Manifold Regression ???
    # 
    # 6. Evaluate Regression Model: MSE
    #
    # 7. Visualize Predicted Age vs. Actual Age
    # 
    def regression(self) -> float:
        # self.get_embeddings()
        self.train()
        predicted_ages, mse_score = self.test()
        self.plot_age_labels_vs_predicted_ages(predicted_ages)
        self.visualize_model_parameters()
        print(f"{self.model_str} Model with Projection {self.projection_type} Mean Squared Error (MSE):", mse_score)
        return mse_score
    def visualize_model_parameters(self):
        
        plt.title(f"{self.model_str} Model with Projection {self.projection_type} Trained Parameters")
        plt.ylabel('Parameter Value')
        plt.xlabel('Region Of Interest (ROI) Index')
        plt.bar(range(NUM_ROIS), self.regressor_model.coef_)

    def plot_age_labels_vs_predicted_ages(self, predicted_ages):
        # Generate x-axis values
        x = np.arange(len(self.test_age_labels))
        # Set width of each bar
        bar_width = 0.35

        # Plotting the barplots
        fig, ax = plt.subplots()
        # print("WE ARE PLOTTING THESE PREDICTED AGES", predicted_ages)
        ax.bar(x - bar_width/2, self.test_age_labels, bar_width, label='Age Label')
        ax.bar(x + bar_width/2, predicted_ages, bar_width, label='Predicted Age')

        # Set labels, title, and legend
        ax.set_xlabel('Subject Index')
        ax.set_ylabel('Age')
        ax.set_title(f'{self.model_str} Model with Projection {self.projection_type} Predicted Ages CamCAN 592 Filtered')
        ax.legend()

        # Show the plot
        plt.show()
        
    def project_embeddings(self, embeddings_list) -> np.ndarray:
        projection_function = self.get_projection_function()
        # scaled_embeddings = self.scale_embeddings(embeddings_list)
        # projected_embeddings = [projection_function(embeddings) for embeddings in tqdm(scaled_embeddings)]
        projected_embeddings = [projection_function(embeddings) for embeddings in tqdm(embeddings_list)]
        np_projected_embeddings = np.array(projected_embeddings)
        if self.projection_type != "SQR": 
            projected_embeddings = np_projected_embeddings.reshape((len(np_projected_embeddings), NUM_ROIS))
        return projected_embeddings
    # TODO: FIGURE OUT IF SHOULD SCALE BEFORE OR AFTER PROJECTION!!!?
    def scale_embeddings(self, embeddings_list) -> np.ndarray:
        print("Scaling Embeddings :")
        scaler = StandardScaler()
        scaled_embeddings = [scaler.fit_transform(embeddings) for embeddings in tqdm(embeddings_list)]
        np_scaled_embeddings = np.array(scaled_embeddings)
        scaled_embeddings = np_scaled_embeddings.reshape((len(np_scaled_embeddings), NUM_ROIS))
        return scaled_embeddings
    def get_projection_function(self):
        projection_function = lambda x : x
        if self.projection_type == "SQR": 
            def get_squared_radius(embeddings):
                return [coord[0] ** 2 + coord[1] ** 2 + coord[2] ** 2 for coord in embeddings]
            projection_function = get_squared_radius
        elif self.projection_type == "TSNE": projection_function = TSNE(n_components=1, init='random', perplexity=3).fit_transform
        elif self.projection_type == "SVD": projection_function = TruncatedSVD(n_components=1).fit_transform
        elif self.projection_type == "PCA": projection_function = PCA(n_components=1).fit_transform
        elif self.projection_type == "ISOMAP": projection_function = Isomap(n_components=1).fit_transform
        elif self.projection_type == "UMAP": projection_function = UMAP(n_components=1).fit_transform
        else: raise AssertionError(f"Invalid Projection Type : {self.projection_type}!")
        # Other possibilities: MDS, LLE, Laplacian Eigenmaps, etc.
        return projection_function
    
    def test(self) -> float:
        """
        Must make sure training has been done beforehand
        Test Predicted Ages from embeddings
        """
        
        test_embeddings_list = []
        for test_index in self.test_indices:
            test_embeddings = np.load(os.path.join(embeddings_dir, f'embeddings_test_{test_index}.npy'))
            test_embeddings_list.append(test_embeddings)
        if type(self.regressor_model) == LinearRegression or type(self.regressor_model) == Ridge:
            print("Projecting Test Embeddings :")
            projected_embeddings = self.project_embeddings(test_embeddings_list)
            if self.projection_type == "SQR": 
                scaler = StandardScaler()
                projected_embeddings = scaler.fit_transform(projected_embeddings)
            predicted_ages = self.regressor_model.predict(projected_embeddings)
        print("PROJECTED TEST EMBEDDINGS", projected_embeddings)
        print("TEST LEN [0]", len(projected_embeddings))
        print("Age Labels: ", self.test_age_labels)
        print("Predicted Ages: ", predicted_ages)
        return predicted_ages, mean_squared_error(predicted_ages, self.test_age_labels)

    def get_embeddings_to_labels(self):
        embeddings = []
        embeddings_directory = 'embeddings'
        embeddings_to_labels = dict()
        for embeddings_filename in os.listdir(embeddings_directory):
            if os.path.isfile(os.path.join(embeddings_directory, embeddings_filename)):
                _, _, train_index = embeddings_filename.split()
                train_index = int(train_index)
                
                age_label = age_labels[train_index]
                
                embeddings = np.load(os.path.join(embeddings_directory, embeddings_filename))
                # TODO: Matrix to Label seems inefficient
                embeddings_to_labels[tuple(embeddings)] = age_label
        return embeddings_to_labels
    
    def train(self) -> List[Tuple[float]]:
        """
        1. Get embeddings
        2. Get age labels
        3. Perform regression
        4. Return predicted ages with age labels

        """
        
        train_embeddings_list = []
        
        for train_index in self.train_indices:
            train_embeddings = np.load(os.path.join(embeddings_dir, f'embeddings_train_{train_index}.npy'))
            train_embeddings_list.append(train_embeddings)
        if type(self.regressor_model) == LinearRegression or type(self.regressor_model) == Ridge:
            # Projection Mapping from 3D to 1D
            print("Projecting Train Embeddings :")
            projected_embeddings = self.project_embeddings(train_embeddings_list)
            if self.projection_type == "SQR": 
                scaler = StandardScaler()
                projected_embeddings = scaler.fit_transform(projected_embeddings)
            self.regressor_model.fit(projected_embeddings, self.train_age_labels)

    def predict_age(self, embeddings) -> float:
        """
        Predict age from embeddings, no label difference calculation
        """
        if type(self.regressor_model) == LinearRegression:
            radii_sqs = [coord[0] ** 2 + coord[1] ** 2 + coord[2] ** 2 for coord in embeddings]
            predicted_age = self.regressor_model.predict(radii_sqs)
        return predicted_age

    def mse_loss(self, predicted_ages_with_age_labels) -> float:    
        """
        Return mean-squared errors between predicted and actual ages
        
        """
        # mse_loss = sum((predicted_age - age_label) ** 2 for predicted_age, age_label 
        #     in predicted_ages_with_age_labels) / len(predicted_ages_with_age_labels)
        predicted_ages = [pred_label[0] for pred_label in predicted_ages_with_age_labels]
        respective_age_labels = [pred_label[1] for pred_label in predicted_ages_with_age_labels]
        mse_loss = mean_squared_error(predicted_ages, respective_age_labels)
        return mse_loss