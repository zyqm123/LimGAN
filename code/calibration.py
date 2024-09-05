import torch
import torch.nn as nn

# Identity module that returns input as-is, can be used as a placeholder
class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

    def predict(self, x):
        return x


# Simple logistic regression model with one input and one output feature
class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)  # One input and one output feature

    def forward(self, x):
        return self.linear(x)


# Function to calibrate a logistic regression model using generator (netG) and multiple discriminators (netD_lis)
def LRcalibrator(netG, netD_lis, data_loader, device, nz=100, calib_frac=0.1):
    n_batches = int(calib_frac * len(data_loader))  # Calculate the number of batches to use for calibration
    batch_size = data_loader.batch_size  # Get batch size from data loader

    # Define a shortcut function to generate fake scores
    def gen_scores(batch_size):
        noise = torch.randn(batch_size, nz, 1, 1, device=device)  # Generate random noise as input for netG
        x = netG(noise)  # Generate fake data using the generator (netG)

        softmax_1 = nn.Softmax(dim=0)  # Define a softmax function to normalize discriminator outputs

        D_lis1 = []
        for netD_tmp in netD_lis:  # Collect discriminator outputs for fake data
            D_lis1.append(netD_tmp(x))
        D_lis = torch.stack(D_lis1, 0)  # Stack discriminator outputs into a tensor

        output_weight = softmax_1(D_lis)  # Apply softmax to discriminator outputs
        output_tmp = torch.mul(D_lis, output_weight)  # Weight discriminator outputs by softmax values
        D_tmp = output_tmp.mean(dim=0)  # Average the weighted outputs across discriminators
        return D_tmp

    print('prepare real scores ...')
    scores_real = []
    for i, (data, _) in enumerate(data_loader):  # Loop through real data from data loader
        softmax_1 = nn.Softmax(dim=0)  # Define a softmax function to normalize discriminator outputs

        real_score1 = []
        for netD_tmp in netD_lis:  # Collect discriminator outputs for real data
            real_score1.append(netD_tmp(data.to(device)))
        real_score = torch.stack(real_score1, 0)  # Stack discriminator outputs into a tensor

        output_weight = softmax_1(real_score)  # Apply softmax to discriminator outputs
        output_tmp = torch.mul(real_score, output_weight)  # Weight discriminator outputs by softmax values
        score_tmp = output_tmp.mean(dim=0)  # Average the weighted outputs across discriminators

        scores_real.append(score_tmp)  # Store the average score for the current batch
        if i > n_batches:  # Stop after processing the specified number of batches
            break
    scores_real = torch.cat(scores_real, dim=0)  # Concatenate scores across batches
    print('prepare fake scores ...')
    scores_fake = gen_scores(batch_size)  # Generate scores for fake data

    print('training LR calibrator ...')
    model = LogisticRegressionModel().to(device)  # Initialize logistic regression model
    x = torch.cat([scores_real, scores_fake], dim=0)  # Combine real and fake scores
    y = torch.cat([torch.ones_like(scores_real),
                   torch.zeros_like(scores_fake)], dim=0)  # Create labels: 1 for real, 0 for fake

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  # Define optimizer with learning rate 0.1

    for epoch in range(5000):  # Train for 5000 epochs
        optimizer.zero_grad()  # Clear gradients
        with torch.enable_grad():
            pred_y = model(x)  # Predict using the logistic regression model
            loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_y, y)  # Compute binary cross-entropy loss
            loss.backward(retain_graph=True)  # Backpropagate loss
        optimizer.step()  # Update model weights
        if epoch % 1000 == 0:  # Print loss every 1000 epochs
            print(f'Epoch: {epoch}; Loss: {loss.item():.3f}')

    return model  # Return the trained logistic regression model
