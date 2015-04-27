package mlp;

public class MLP {
	private int numberInputs, numberHidden, numberOutputs;
	private double learningRate, input[], hidden[], output[], w1[][], w2[][];

	public MLP(int numberInputs, int numberHidden, int numberOutputs, double learningRate) {
		this.numberInputs = numberInputs;
		this.numberHidden = numberHidden;
		this.numberOutputs = numberOutputs;
		this.learningRate = learningRate;

		// Add an extra neuron/index to each layer as a placeholder for the bias
		input = new double[numberInputs + 1];
		hidden = new double[numberHidden + 1];
		output = new double[numberOutputs + 1];

		w1 = new double[numberHidden + 1][numberInputs + 1]; // hidden - input weights
		w2 = new double[numberOutputs + 1][numberHidden + 1]; // output - hidden weights

		randomise();
	}

	private void randomise() {
		// Do not include index [0][0] in loops as this will be the bias
		for (int j = 1; j <= numberHidden; j++)
			for (int i = 0; i <= numberInputs; i++) {
				w1[j][i] = Math.random() - 0.5; // -0.5 -> 0.5
			}

		for (int j = 1; j <= numberOutputs; j++)
			for (int i = 0; i <= numberHidden; i++) {
				w2[j][i] = Math.random() - 0.5; // -0.5 -> 0.5
			}
	}

	public double[] fForward(double[] inputVector) {
		if (inputVector.length != numberInputs) {
			System.out.println("Input length = " + inputVector.length + "  Expected = " + numberInputs);
			System.out.println("Input Vector Error, Incorrect Number of Inputs.");
			System.exit(1);
		}

		// Leave room for bias at 0 index
		for (int i = 0; i < numberInputs; i++) {
			input[i + 1] = inputVector[i];
		}

		// Set bias
		input[0] = 1.0;
		hidden[0] = 1.0;

		// Pass through hidden layer
		pass(numberHidden, numberInputs, hidden, w1, input);

		// Pass through output layer
		pass(numberOutputs, numberHidden, output, w2, hidden);

		return output;
	}

	public double bPropagate(double[] target) {

		double error = 0.0;

		double[] errorL2 = new double[numberOutputs + 1];
		double[] errorL1 = new double[numberHidden + 1];

		for (int i = 1; i <= numberOutputs; i++) { // Layer 2 error gradient
			double delta = target[i - 1] - output[i];
			error += Math.pow(delta, 2); // Always a positive error
			// If delta == 0, disregard. Else the error is the delta * output[i] * (1.0 - output[i])
			errorL2[i] = output[i] * (1.0 - output[i]) * delta;
		}

		double delta = 0.0; // Summation of all hidden -> output weights (per hidden neuron) plus their calculated errors.
		for (int i = 0; i <= numberHidden; i++) {  // Layer 1 error gradient
			for (int j = 1; j <= numberOutputs; j++)
				delta += w2[j][i] * errorL2[j];

			errorL1[i] = hidden[i] * (1.0 - hidden[i]) * delta;
			delta = 0.0;
		}

		// Update the weights (learningRate * error in this neuron * selected input to this neuron.)
		for (int j = 1; j <= numberHidden; j++)
			for (int i = 0; i <= numberInputs; i++)
				w1[j][i] += learningRate * errorL1[j] * input[i];

		for (int j = 1; j <= numberOutputs; j++)
			for (int i = 0; i <= numberHidden; i++)
				w2[j][i] += learningRate * errorL2[j] * hidden[i];

		return error / (numberOutputs + 1); // The more outputs, the more the error increments.
	}

	public void setLearningRate(double learningRate) {
		this.learningRate = learningRate;
	}

	private double sigmoid(double x) {
		// Sigmoid using logistic function
		return 1.0 / (1.0 + Math.exp(-x));
	}

	// Pass from one layer to next in ff method
	private void pass(int jLimit, int iLimit, double[] outputLayer, double[][] weights, double[] inputs) {
		for (int j = 1; j <= jLimit; j++) {
			outputLayer[j] = 0.0;
			// For all the inputs to neuron add them * their weights
			for (int i = 0; i <= iLimit; i++) {
				outputLayer[j] += inputs[i] * weights[j][i];
			}
			// Activation function
			outputLayer[j] = sigmoid(outputLayer[j]);
		}
	}
}
