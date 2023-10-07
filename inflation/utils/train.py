import torch
import sys
import numpy as np
from utils.evaluate import evaluate

def train(model, trainloader, validloader, epochs, optimizer, criterion, device, random_state):

    # Fijamos semilla 
    if random_state:
        torch.manual_seed(random_state)

    # número de batches
    n_batches = len(trainloader)

    MAE = torch.nn.L1Loss()

    for epoch in range(epochs):
        running_loss = 0.0
        valid_limit = np.inf
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Fijamos a cero los gradientes
            optimizer.zero_grad()
            # Pasada hacia adelante
            outputs = model(inputs)
            # Calculamos la funcion de perdida
            loss = criterion(outputs, labels)
            # Backpropagation
            loss.backward()
            # Actualizamos los parametros
            optimizer.step()
            # Agregamos la loss de este batch
            running_loss += loss.item()
            
            sys.stdout.write(f'\rEpoch: {epoch+1:03d} \t Avg Train Loss: {100 * running_loss/n_batches:.3f} \t Train MAE: {100 * MAE(outputs, labels):.2f}') # print de loss promedio x epoca
            
        # evaluate on valid set
        valid_loss, valid_MAE, valid_total_loss = evaluate(model, validloader, criterion)
        
        if epoch % (epochs * 0.1) == 0:
            print("\t" + f"Avg Val Loss: {100 * valid_loss:.3f} \t Val MAE: {100 * valid_MAE:.2f} \t Total Val Loss: {100 * valid_total_loss:.2f}")
        
        if valid_MAE < valid_limit:
            valid_limit = valid_MAE
            torch.save(model, f'inflation_model.pt')