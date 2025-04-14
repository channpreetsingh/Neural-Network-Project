# Neural Network on MNIST Dataset 🧠✨

This project implements a simple neural network using **PyTorch** to classify handwritten digits from the **MNIST dataset**. It includes model training, evaluation, and inference with saved model weights.

---

## 📁 Project Structure

Neural network/ ├── torchnn.py # Main training and evaluation script
├── model.pth # Saved PyTorch model 
├── 1.jpg - 4.jpg # Sample images for testing inference 
├── data/ # MNIST dataset (raw format)

---

## 🚀 Features

- 🧠 Fully connected neural network (MLP)
- 🔁 Training and evaluation loops
- 💾 Model saving and loading (`model.pth`)
- 🖼️ Custom image inference support
- 🧪 MNIST dataset support via local files or download

---

## 🛠️ Requirements

- Python 3.8+
- PyTorch
- torchvision
- numpy
- matplotlib (optional, for visualizations)

You can install dependencies via:

```bash
pip install torch torchvision numpy matplotlib
              
▶️ How to Run
1. Train the Model
bash
Copy
Edit
python torchnn.py
This will train the model on MNIST and save it as model.pth.

2. Test or Inference
You can modify the script to test on the sample images (1.jpg, 2.jpg, etc.) using the saved model.

🧠 Model Architecture
Input: 28x28 grayscale images (flattened to 784)

Hidden Layers: Customizable (default: 1 or 2 hidden layers)

Activation: ReLU

Output: 10-class softmax for digits 0–9

📊 Sample Results
Coming soon – Add sample output accuracy or image predictions here

📸 Example Inference

Predicted: 1

📌 Notes
The MNIST dataset is auto-downloaded if not found locally.

Make sure your working directory is the same level as torchnn.py.

👤 Author
Channpreet Singh
GitHub Profile
