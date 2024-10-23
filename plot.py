# Re-importing necessary library
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Data for the subjects
data = {
    "Subject": list(range(1, 43)),
    "Vision": [55.20, 70.03, 76.43, 77.43, 62.03, 83.83, 74.77, 66.60, 62.13, 66.43, 59.20, 51.30, 73.43, 58.33, 73.80, 54.97, 83.77, 67.10, 61.50, 76.37, 71.47, 64.57, 58.63, 70.13, 60.50, 66.33, 82.57, 71.97, 61.87, 66.70, 67.73, 57.93, 76.00, 63.60, 57.23, 62.23, 57.20, 75.93, 67.43, 57.07, 78.23, 73.23],
    "Audio": [58.33, 72.50, 52.50, 60.00, 50.00, 80.00, 60.00, 48.33, 65.83, 53.33, 45.00, 48.33, 66.67, 52.50, 67.50, 55.00, 80.83, 56.67, 60.00, 67.50, 50.83, 74.17, 66.67, 67.50, 48.33, 58.33, 54.17, 55.83, 45.83, 55.00, 70.83, 50.83, 76.67, 40.83, 47.50, 60.83, 43.33, 44.17, 62.50, 60.00, 53.33, 55.00],
    "EEG": [59.17, 64.17, 54.17, 66.67, 40.00, 48.33, 59.17, 54.17, 45.00, 47.50, 44.17, 45.00, 55.00, 33.33, 51.67, 42.50, 67.50, 64.17, 51.67, 73.33, 60.00, 70.83, 50.00, 76.67, 50.00, 49.17, 65.00, 67.50, 42.50, 50.00, 44.17, 55.83, 62.50, 46.67, 28.33, 60.83, 55.83, 45.83, 50.83, 40.00, 43.33, 65.00],
    "Multimodal": [66.60, 76.27, 75.43, 81.83, 59.47, 69.73, 80.43, 68.97, 76.43, 69.37, 57.03, 55.17, 75.27, 57.97, 65.53, 57.83, 89.63, 74.97, 68.10, 86.50, 78.10, 76.37, 64.63, 85.13, 67.73, 68.57, 83.87, 81.17, 62.53, 56.10, 64.30, 63.83, 80.10, 68.20, 62.97, 74.23, 61.83, 76.27, 71.50, 66.90, 72.33, 77.10]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Calculate the average for each modality
averages = df[["Vision", "Audio", "EEG", "Multimodal"]].mean()

# Create a figure for the bar plot
plt.figure(figsize=(12, 8))

# Subjects and modalities
subjects = df["Subject"]
modalities = ["Vision", "Audio", "EEG", "Multimodal"]

# Creating the bar plot
x = np.arange(len(subjects) + 1)  # Including space for the average
width = 0.2  # Width of each bar

# Creating the bar plot
colors = ['blue', 'orange', 'green', 'red']  # Distinctive colors for each modality

# Plotting each modality for all subjects and the average
for i, (modality, color) in enumerate(zip(modalities, colors)):
    values = df[modality].tolist() + [averages[modality]]  # Adding average to the end
    plt.bar(x + (i - 1.5) * width, values, width, label=modality, color=color)

# Adding average accuracy line for each modality
for i, (modality, color) in enumerate(zip(modalities, colors)):
    plt.axhline(y=averages[modality], color=color, linestyle='--', linewidth=2, label=f"Avg. {modality}")

# Adding labels and title
plt.title("Comparative Results Across Subjects for Different Modalities")
plt.xlabel("Subjects")
plt.ylabel("Accuracy (%)")
plt.xticks(x, subjects.tolist() + ["Avg."], rotation=45)
plt.legend()
plt.tight_layout()

# Show the plot
# plt.show()
plt.savefig('plot.pdf', format='pdf')