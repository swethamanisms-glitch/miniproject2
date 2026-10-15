import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

# Sample data (replace with your real dataset)
data = pd.DataFrame({
    'Cibil':      [520, 580, 610, 650, 650, 690, 720, 750, 780, 800, 560, 700, 740, 620],
    'LoanAmount': [5,   10,  8,   15,  5,   12,  25,  30,  20,  40,  18,  28,  22,  12],   # in lakhs
    'Eligible':   [0,   0,   0,   0,   1,   1,   1,   1,   1,   1,   0,   1,   1,   0]      # 1 = eligible, 0 = not
})

x = data[['Cibil', 'LoanAmount']]
y = data['Eligible']

# Scaling is needed because CIBIL (500-900) and loan amount (lakhs) are on very different ranges
model = make_pipeline(StandardScaler(), LogisticRegression())
model.fit(x, y)

print("accuracy on training data:", round(accuracy_score(y, model.predict(x)), 2))

# Predict eligibility for new applicants
new = pd.DataFrame({
    'Cibil':      [580, 700, 790],
    'LoanAmount': [8,   20,  35]
})
new["eligible_pred"] = model.predict(new[['Cibil', 'LoanAmount']])
new["eligible_prob"] = model.predict_proba(new[['Cibil', 'LoanAmount']])[:, 1].round(2)
print(new)

# Plot: CIBIL on x-axis, loan amount on y-axis
xx, yy = np.meshgrid(np.linspace(500, 850, 300), np.linspace(0, 45, 300))
grid = pd.DataFrame({'Cibil': xx.ravel(), 'LoanAmount': yy.ravel()})
Z = model.predict(grid).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.2, cmap='RdYlGn')
plt.scatter(data.loc[data['Eligible'] == 1, 'Cibil'],
            data.loc[data['Eligible'] == 1, 'LoanAmount'],
            color='green', label='Eligible')
plt.scatter(data.loc[data['Eligible'] == 0, 'Cibil'],
            data.loc[data['Eligible'] == 0, 'LoanAmount'],
            color='red', label='Not eligible')
plt.xlabel('CIBIL score')
plt.ylabel('Loan amount (lakhs)')
plt.title('Loan eligibility')
plt.legend()
plt.show()