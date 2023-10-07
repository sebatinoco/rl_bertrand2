import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAE = torch.nn.L1Loss()

def evaluate(model, validloader, criterion):

    '''
    Función que recibe un modelo y dataloader, retorna accuracy, f1, loss promedio y loss total.
    model: clasificador (torch.nn.Module)
    dataloader: datos a ser trabajados (DataLoader)
    '''
  
    # evaluate on valid set
    model.eval()
    with torch.no_grad():
        valid_running_loss = 0.0
        valid_MAE = 0.0
        for valid_inputs, valid_labels in validloader:
            valid_inputs, valid_labels = valid_inputs.to(device), valid_labels.to(device)
            outputs = model(valid_inputs) # generamos predicción
            
            loss = criterion(outputs, valid_labels)
            valid_running_loss += loss.item()
            valid_MAE += MAE(outputs, valid_labels) * len(valid_inputs)
        
    model.train()

    return valid_running_loss/len(validloader), valid_MAE/len(validloader.dataset), valid_running_loss