package mlp;

public class MLP {
	private int numberInputs, numberHidden, numberOutputs;
	private double learningRate = 0.5;
	private double[] activations1, activations, input, hidden, output, dw1, dw2;
	private double[][] w1, w2;
	
	public MLP(int numberInputs, int numberHidden, int numberOutputs) {
		this.numberInputs = numberInputs;
		this.numberHidden = numberHidden;
		this.numberOutputs = numberOutputs;
		
		input = new double[numberInputs + 1];
		hidden = new double[numberHidden + 1];
		output = new double[numberOutputs + 1];
		
		dw1 = new double[numberHidden + 1];
		dw2 = new double[numberOutputs + 1];
		
		w1 = new double[numberHidden + 1][numberInputs + 1];
		w2 = new double[numberOutputs + 1][numberHidden + 1];
		
		randomise();
	}
	
	/**
	 * Initialise w1 and w2 to small random values. Set dw1 and dw2 to all zeroes
	 */
	public void randomise() {
		for (int j = 1; j <= numberHidden; j++) {
			for (int i = 0; i <= numberInputs; i++) {
				w1[j][i] = Math.random() - 0.5;
			}
		}
		
		for (int j = 1; j <= numberOutputs; j++) {
			for (int i = 0; i <= numberHidden; i++) {
				w2[j][i] = Math.random() - 0.5;
			}
		}
		
		initDw();
	}
	
	/**
	 * Forward pass. Input vector processed to produce an output, stored in O[]
	 * @param inputVector
	 */
	public void forward(double[] inputVector) {
		input[0] = 1.0;
		hidden[0] = 1.0;
		
		for (int i = 0; i < numberInputs; i++) {
			input[i + 1] = inputVector[i];
		}
		
		//Pass through hidden layer
		for (int j = 1; j <= numberHidden; j++) {
			hidden[j] = 0.0;
			for (int i = 0; i <= numberInputs; i++) {
				hidden[j] += w1[j][i] * input[i];
			}
			hidden[j] = 1.0 / (1.0 + Math.exp(-hidden[j]));
		}
		
		//Passing through output layer
		for (int j = 1; j <= numberOutputs; j++) {
			output[j] = 0.0;
			for (int i = 0; i <= numberHidden; i++) {
				output[j] += w2[j][i] * hidden[i];
			}
			output[j] = 1.0 / (1.0 + Math.exp(-output[j]));
		}
	}
	
	/**
	 * Compare target with output O
	 * @param target
	 */
	public void backwards(double[] target) {
		double error = 0.0;
		
		for (int i = 1; i <= numberOutputs; i++) {
			dw2[i] = output[i] * (1.0 - output[i]) * (target[i - 1] - output[i]);
		}
		
		for (int i = 0; i <= numberHidden; i++) {
			for (int j = 1; j < numberOutputs; j++) {
				error += w2[j][i] * dw2[j];
			}
			
			dw1[i] = hidden[i] * (1.0 - hidden[i]) * error;
			System.out.println("error : " + error);
			error = 0.0;
		}
		
		updateWeights(learningRate);
	}
	
	/**
	 * this simply does (component by component, i.e. within for loops):
	 * W1 += learningRate*dW1;
	 * W2 += learningRate*dW2;
	 * reset dw1,dw2
	 * @param learningRate
	 */
	public void updateWeights(double learningRate) {
		for (int j = 1; j <= numberOutputs; j++) {
			for (int i = 0; i <= numberHidden; i++) {
				w2[j][i] += learningRate * dw2[j] * hidden[i];
			}
		}
		
		for (int j = 1; j <= numberHidden; j++) {
			for (int i = 0; i <= numberInputs; i++) {
				w1[j][i] += learningRate * dw1[j] * input[i];
			}
		}
		
		initDw();
	}
	
	private void initDw() {
		for (int i = 0; i <= numberHidden; i++) {
			dw1[i] = 0;
		}
		
		for (int i = 0; i <= numberOutputs; i++) {
			dw2[i] = 0;
		}
	}
}
