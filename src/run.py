import numpy as np

import torch
from torch import optim, nn

from .autoencoder import Autoencoder, AETrainer
from .utils import load_data, tfidf_generator


def run():
    documents = load_data("./pitchfork_reviews.csv")
    num_words = 1000
    data_tfidf, tfidf_tokens = tfidf_generator(documents, num_words)
    dataset = data_tfidf.toarray()
    device = torch.device("cpu")
    
    num_epochs = 200
    batch_size = 100
    model = Autoencoder(num_words).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()


    trainer = AETrainer(model, num_epochs, batch_size, criterion, optimizer, device, model_save_path='model.pth')
    trainer.train(dataset)
    encoded_reviews = model.encoder(torch.from_numpy(dataset).float()).detach().numpy()
    
    np.save('encoded_reviews.npy', encoded_reviews)


run()