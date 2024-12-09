import torch

from model import UNet


def classifier_free_guidance(model, x_t, t, condition, guidance_weight):
    # Prédiction conditionnelle
    epsilon_cond = model(x_t, condition)

    # Prédiction non conditionnelle
    epsilon_uncond = model(x_t, None)

    # Combinaison des deux
    epsilon_guided = (
        1 + guidance_weight
    ) * epsilon_cond - guidance_weight * epsilon_uncond
    return epsilon_guided


def generate(model, shape, noise_schedule, guidance_weight, condition=None):
    device = next(model.parameters()).device
    x_t = torch.randn(shape).to(device)  # Échantillon initial bruité

    for t in reversed(range(len(noise_schedule))):
        # Prédiction guidée
        epsilon = classifier_free_guidance(model, x_t, t, condition, guidance_weight)

        # Mise à jour de x_t
        alpha_t = noise_schedule[t]
        x_t = (x_t - (1 - alpha_t) * epsilon) / alpha_t

    return x_t


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(input_dim=1, condition_dim=10)
model.load_state_dict(torch.load("model.pth"))
model.to(device)

noise_schedule = torch.linspace(1.0, 0.1, steps=100).to(device)


condition = torch.eye(10)[9].to(device)  # Par exemple, générer un "1"
generated_image = generate(
    model,
    shape=(1, 1, 28, 28),
    noise_schedule=noise_schedule,
    guidance_weight=5.0,
    condition=condition,
)

# Visualisation
import matplotlib.pyplot as plt

plt.imshow(generated_image.squeeze().cpu().detach().numpy(), cmap="gray")
plt.show()
