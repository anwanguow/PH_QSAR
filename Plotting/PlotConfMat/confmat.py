import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

confusion_matrix = np.array([[53, 0, 0 ],
                             [0, 26, 0 ],
                             [1, 0, 49]])

confusion_df = pd.DataFrame(confusion_matrix, index=["Class I", "Class II", "Class III"], columns=["Class I", "Class II", "Class III"])

plt.figure(figsize=(6, 6))
ax = sns.heatmap(confusion_df, annot=False, fmt='d', cmap='Pastel2', cbar=False, linewidths=1, linecolor='black')

total = confusion_matrix.sum()
for i in range(confusion_matrix.shape[0]):
    for j in range(confusion_matrix.shape[1]):
        absolute_value = confusion_matrix[i, j]
        percentage = absolute_value / total * 100
        ax.text(j + 0.5, i + 0.4, f'{absolute_value}', ha='center', va='center', color='black', fontsize=22, weight='bold')
        ax.text(j + 0.5, i + 0.6, f'{percentage:.2f}%', ha='center', va='center', color='black', fontsize=22, weight='bold')

plt.xlabel('Predicted Classes', fontsize=20)
plt.ylabel('Actual Classes', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Laplacian, testing', fontsize=20)

plt.savefig("/Users/dmr/Desktop/cm_8.png", dpi=300)
plt.show()
