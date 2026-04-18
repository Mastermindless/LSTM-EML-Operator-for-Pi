import torch
import torch.nn as nn
import torch.optim as optim
import mpmath
import numpy as np
import argparse
import sys

# ---------------------------------------------------------
# 1. EML Symbolic Node (Integrated with mpmath)
# ---------------------------------------------------------
class EMLSymbolicPi:
    """
    Calculates Pi using the EML property: Im(ln(-1)) = pi.
    This acts as our symbolic 'truth' generator.
    """
    @staticmethod
    def calculate(precision):
        mpmath.mp.dps = int(precision)
        # EML formula for pi: Im(ln(-1))
        # Mathematically equivalent to nested EML(1, 1) structures
        pi_val = mpmath.im(mpmath.log(-1))
        return pi_val

    @staticmethod
    def get_digits(precision):
        pi_val = EMLSymbolicPi.calculate(precision)
        s = str(pi_val).replace('.', '')
        # Return as list of integers
        return [int(d) for d in s[:precision]]

# ---------------------------------------------------------
# 2. LSTM Controller Model
# ---------------------------------------------------------
class LSTMPiController(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(LSTMPiController, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Precision regression head
        self.fc_precision = nn.Linear(hidden_size, output_size)
        # Digit sequence head (outputs 10 digits for the next chunk)
        self.fc_digits = nn.Linear(hidden_size, 10 * 10) # 10 digits, 10 classes each (0-9)

    def forward(self, x):
        # x shape: (batch_size, seq_len, 1)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        # Last hidden state
        last_out = out[:, -1, :]
        
        pred_precision = self.fc_precision(last_out)
        pred_digits = self.fc_digits(last_out).view(-1, 10, 10)
        
        return pred_precision, pred_digits

# ---------------------------------------------------------
# 3. Training Logic & Loss Function
# ---------------------------------------------------------
def train_model(model, epochs=100, max_precision=100):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion_prec = nn.MSELoss()
    criterion_digits = nn.CrossEntropyLoss()
    
    print(f"Starting training for {epochs} epochs (Max Precision: {max_precision})...")
    
    for epoch in range(epochs):
        # Generate random target precision
        target_prec_val = np.random.randint(1, max_precision + 1)
        target_prec_tensor = torch.tensor([[float(target_prec_val)]]).view(1, 1, 1)
        
        # Get true digits from EML
        true_digits = EMLSymbolicPi.get_digits(target_prec_val)
        # For simplicity, we train the model to predict the first 10 digits
        true_digits_chunk = torch.tensor(true_digits[:10] if len(true_digits) >= 10 else true_digits + [0]*(10-len(true_digits)))
        true_digits_chunk = true_digits_chunk.unsqueeze(0) # Batch size 1
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        pred_prec, pred_digits = model(target_prec_tensor)
        
        # Loss 1: Precision mapping
        loss_p = criterion_prec(pred_prec, torch.tensor([[float(target_prec_val)]]))
        
        # Loss 2: Digit alignment (CrossEntropy over the first 10 digits)
        loss_d = criterion_digits(pred_digits.transpose(1, 2), true_digits_chunk)
        
        total_loss = loss_p + loss_d
        total_loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Total Loss: {total_loss.item():.4f}, Pred Prec: {pred_prec.item():.2f}")

# ---------------------------------------------------------
# 4. Inference & Comparison
# ---------------------------------------------------------
def run_inference(model, requested_precision):
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor([[float(requested_precision)]]).view(1, 1, 1)
        pred_prec, pred_digits = model(input_tensor)
        
        # Trigger EML with predicted precision (rounded)
        final_prec = int(round(pred_prec.item()))
        if final_prec < 1: final_prec = 1
        
        pi_eml = EMLSymbolicPi.calculate(final_prec)
        pi_true = EMLSymbolicPi.calculate(requested_precision)
        
        print("\n--- INFERENCE RESULT ---")
        print(f"Requested Precision: {requested_precision}")
        print(f"LSTM Predicted Precision: {pred_prec.item():.4f} (Used: {final_prec})")
        print(f"EML Output (First 50 chars): {str(pi_eml)[:50]}...")
        print(f"Standard Pi (First 50 chars): {str(pi_true)[:50]}...")
        
        # Error check
        error = abs(pi_true - pi_eml)
        print(f"Precision Match Error: {error}")
        
        if error == 0:
            print("SUCCESS: Perfect Precision Match!")
        else:
            print("INFO: Approximate Match reached.")

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSTM-EML Hybrid Pi Calculator")
    parser.add_argument("--precision", type=int, default=100, help="Target precision for inference")
    parser.add_argument("--train", action="store_true", help="Train the model before inference")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    args = parser.parse_args()

    # Initialize Model
    model = LSTMPiController()
    
    if args.train:
        train_model(model, epochs=args.epochs, max_precision=args.precision)
    
    # Run Inference
    run_inference(model, args.precision)
