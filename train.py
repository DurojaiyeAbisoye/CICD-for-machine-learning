import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import skops.io as sio
from skops.io import get_untrusted_types

drug_df = pd.read_csv("data/drug.csv").sample(frac=1).reset_index(drop=True)
X = drug_df.drop("Drug", axis=1)
y = drug_df["Drug"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=125
)


cat_cols = ["Sex", "BP", "Cholesterol"]
num_cols = ["Age", "Na_to_K"]

transform = ColumnTransformer(
    transformers=[
        ("encoder", OneHotEncoder(), cat_cols),
        ("num_imputer", SimpleImputer(strategy="mean"), num_cols),
        ("num_scaler", StandardScaler(), num_cols),
    ]
)

pipe = Pipeline(
    steps=[
        ("preprocessing", transform),
        ("model", RandomForestClassifier(random_state=125)),
    ]
)

pipe.fit(X_train, y_train)

predictions = pipe.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average="macro")

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")

with open("results/metrics.txt", "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")

cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipe.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.savefig("results/confusion_matrix.png", dpi=120, bbox_inches="tight")


sio.dump(pipe, "model/drug_pipeline.skops")
trusted_types = get_untrusted_types(file="model/drug_pipeline.skops")
