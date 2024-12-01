from perceptron import predict_obesity, load_model, train_model

print("Obesity Prediction Program")
print("-" * 25)

# Try to load the trained model
if not load_model():
    print("\nNo saved model found. Training new model...")
    if train_model():
        print("Model trained and saved successfully!")
    else:
        print("Error training model!")
        exit(1)
else:
    print("Loaded saved model successfully!")

# Now let's make some predictions with different cases
test_cases = [
    (70, 160),    # 5'10", 160 lbs
    (65, 190),    # 5'5", 190 lbs
    (72, 200),    # 6'0", 200 lbs
    (63, 120),    # 5'3", 120 lbs
    (68, 210),    # 5'8", 210 lbs
]

print("\nMaking predictions:")
print("Height(in) Weight(lbs)  Prediction")
print("-" * 35)

for height, weight in test_cases:
    prediction = predict_obesity(height, weight)
    status = "Obese" if prediction == 1 else "Not Obese"
    print(f"{height:^8} {weight:^10}  {status}")
