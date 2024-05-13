from train.additional_transforms import *
from deeplay.applications.detection.lodestar.transforms import *
import deeplay as dl
import torch
import matplotlib.pyplot as plt

class LodeSTAR3DTrainer:

    def __init__(self,
                 epochs=1000,
                 batch_size=10,
                 num_transforms=8,
                 learning_rate= 1e-4,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 model=None
                 ):
    
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_transforms = num_transforms
        self.learning_rate = learning_rate
        self.device = device
        self.model = model.to(self.device)
        self.num_outputs = 3
        self.losses = []

        self.lodestar = dl.LodeSTAR(model = self.model,
                                    num_outputs = self.num_outputs,
                                    n_transforms = self.num_transforms,
                                    transforms = Transforms([
                                                            RandomTranslationZ(),
                                                            RandomTranslation2d(),
                                                            RandomRotation2d(),
                                                            RandomScaleImage(),
                                                            ]))
        self.model = self.lodestar._get_default_model() if model == None else model
    
    def train(self, image_size = 64, lower_loss_bound = None):
        print(f"Using device: {self.device}")
        self.losses = []
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        dataloader = torch.utils.data.DataLoader(dataset = 
                                                 torch.empty((self.batch_size, image_size, image_size)).to(self.device))

        for epoch in range(1, self.epochs+1):

            for sample in dataloader:
                transforms, inverses = self.lodestar.transform_data(sample)
                prediction = self.lodestar.forward(transforms)
                loss_dict = self.lodestar.compute_loss(prediction, inverses)
                loss1, loss2 = loss_dict['between_image_disagreement'], loss_dict['within_image_disagreement']
                loss = loss1 + loss2
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                self.losses.append(loss.item())

            if lower_loss_bound != None:
                if self.losses[epoch] < lower_loss_bound and self.losses[epoch-1] < lower_loss_bound:
                    print(f"Epoch {epoch}/{self.epochs} \tLoss: {self.losses[-1]}")
                    print(f"Lower loss bound of {lower_loss_bound} reached. Stopping training.")
                    break
        
            if epoch % 2 == 0:
                print(f"Epoch {epoch}/{self.epochs} \tLoss: {self.losses[-1]}", end='\r')
        
    def plot_losses(self, yscale='linear'):
        plt.scatter([i for i in list(range(1, len(self.losses)+1))], self.losses, s=1, color='black')
        plt.grid()
        plt.yscale(yscale)
        plt.show()