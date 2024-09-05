import torch.nn as nn
import torch, gc

def batch_l2_norm_squared(z):
    """
    Calculate the squared L2 norm for each batch.

    Args:
        z (Tensor): Input tensor.

    Returns:
        Tensor: Sum of squared L2 norms across all dimensions except the first.
    """
    return z.pow(2).sum(dim=tuple(range(1, len(z.shape))))

def repgan(netG, netD_lis, calibrator, device, nz=100, batch_size=10, clen=640, tau=0.01, eta=0.1):
    """
    Perform RepGAN sampling using Langevin dynamics and Metropolis-Hastings.

    Args:
        netG (nn.Module): Generator network. Input: latent (B x latent_dim x 1 x 1). Output: images (B x C x H x W).
        netD_lis (list of nn.Module): List of discriminator networks. Input: images (B x C x H x W). Output: raw score (B x 1).
        calibrator (nn.Module): Calibrator network for adjusting the discriminator scores.
        device (torch.device): Device on which to run the computation.
        nz (int, optional): Dimension of the latent vector z of the generator. Default is 100.
        batch_size (int, optional): Number of samples per batch. Default is 10.
        clen (int, optional): Length of the Markov chain (only the last sample is retained). Default is 640.
        tau (float, optional): Step size in Langevin dynamics. Default is 0.01.
        eta (float, optional): Scale of the white noise in Langevin dynamics. Default is 0.1.

    Returns:
        Tensor: Generated fake samples from the generator.
    """

    # Shortcut for getting score and updated means (latent + tau/2 * grads) from latent
    def latent_grad_step(latent):
        """
        Perform a gradient step in the latent space.

        Args:
            latent (Tensor): Latent vector.

        Returns:
            score (Tensor): Calibrated score.
            mean (Tensor): Updated mean after the gradient step.
        """
        with torch.autograd.enable_grad():
            latent.requires_grad = True
            score1 = []
            for netD_tmp in netD_lis:
                score1.append(netD_tmp(netG(latent)))

            softmax_1 = nn.Softmax(dim=0)
            score_lis = torch.stack(score1, 0)
            output_weight = softmax_1(score_lis)
            output_tmp = torch.mul(score_lis, output_weight)
            score_tmp = output_tmp.mean(dim=0)

            score = calibrator(score_tmp).squeeze()
            gc.collect()
            torch.cuda.empty_cache()
            obj = torch.sum(score - batch_l2_norm_squared(latent) / 2)  # Calculate L2MC gradients
            grads = torch.autograd.grad(obj, latent)[0]
        mean = latent.data + tau / 2 * grads  # Update mean using gradients
        return score, mean

    # Initialize the Markov chain with a random latent vector
    old_latent = torch.randn(batch_size, nz, 1, 1, device=device, requires_grad=True)  # Gaussian prior
    gc.collect()
    torch.cuda.empty_cache()
    old_score, old_mean = latent_grad_step(old_latent)  # Get current score and next-step mean
    one = old_score.new_tensor([1.0])

    data_num = 0

    # MCMC transitions using Langevin dynamics and Metropolis-Hastings
    for _ in range(clen):
        # 1) Proposal step: generate a new latent vector
        prop_noise = torch.randn_like(old_latent)  # Draw noise for proposal
        prop_latent = old_mean + eta * prop_noise  # Proposed latent vector
        prop_score, prop_mean = latent_grad_step(prop_latent)  # Get proposed score and next-step mean

        # 2a) Calculate the Metropolis-Hastings acceptance ratio (in log space for stability)
        score_diff = prop_score - old_score  # Difference in scores between proposal and current
        latent_diff = batch_l2_norm_squared(old_latent) / 2 - batch_l2_norm_squared(prop_latent) / 2
        noise_diff = batch_l2_norm_squared(prop_noise) / 2 - batch_l2_norm_squared(old_latent - prop_mean) / (2 * eta ** 2)
        alpha = torch.min((score_diff + latent_diff + noise_diff).exp(), one)

        # 2b) Accept or reject the proposal based on the calculated alpha
        accept = torch.rand_like(alpha) <= alpha

        # 2c) Update the latent vector and scores with the accepted proposal
        old_latent.data[accept] = prop_latent.data[accept]
        old_score.data[accept] = prop_score.data[accept]
        old_mean.data[accept] = prop_mean.data[accept]

    # Generate fake samples using the final latent vector after MCMC transitions
    gc.collect()
    torch.cuda.empty_cache()
    fake_samples = netG(old_latent)
    return fake_samples
