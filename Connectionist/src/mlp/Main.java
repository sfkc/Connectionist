package mlp;

public class Main {
	private static final int MAX_EPOCHS = 500000;
	private static final double ERR_THRESHOLD = 0.01;

	public static void main(String[] args) {
		double error = 0.0;

		// QUESTION 1
		MLP q1 = new MLP(2, 2, 1, 0.5);
		double[][] trainingSet = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
		double[][] target = { { 0 }, { 1 }, { 1 }, { 0 } };

		for (int i = 0; i < MAX_EPOCHS; i++) {
			error = 0.0;
			for (int j = 0; j < trainingSet.length; j++) {
				q1.fForward(trainingSet[j]);
				error += q1.bPropagate(target[j]);
			}
			if (i % 1000 == 0)
				System.out.println("Epoch: " + i + "  Error: " + error);
			if (error < ERR_THRESHOLD) {
				System.out.println("END, Error < " + ERR_THRESHOLD + ": " + error);
				break;
			}
		}

		// QUESTION 2 Check if examples are predicted correctly
		error = 0.0;
		for (int i = 0; i < trainingSet.length; i++) {
			if (Math.round(q1.fForward(trainingSet[i])[1]) == target[i][0])
				System.out.println("CORRECT");
			else
				System.out.println("INCORRECT");
			error = q1.bPropagate(target[i]);
		}
		System.out.println("Test set error: " + error + "\n\n");

		// QUESTION 3
		double[][] trainingSet2 = new double[50][4];
		double[][] target2 = new double[50][1];
		for (int j = 0; j < 50; j++) {
			double sum = 0.0;
			for (int i = 0; i < 4; i++) {
				trainingSet2[j][i] = (Math.random() * 2.0) - 1.0;
				sum += trainingSet2[j][i];
			}
			target2[j][0] = Math.sin(sum);
		}
		MLP q3 = new MLP(4, 5, 1, 0.01);
		for (int i = 0; i < MAX_EPOCHS; i++) {
			error = 0.0;
			for (int j = 0; j < 40; j++) {
				q3.fForward(trainingSet2[j]);
				error += q3.bPropagate(target2[j]);
			}
			if (i % 50000 == 0)
				System.out.println("Epoch = " + i + ", Error = " + error);
			if (error < ERR_THRESHOLD) {
				System.out.println("END, Error < " + ERR_THRESHOLD + ": " + error);
				break;
			}
		}

		// QUESTION 4 Testing
		error = 0.0;
		for (int j = 40; j < 50; j++) {
			q3.fForward(trainingSet2[j]);
			error = q3.bPropagate(target2[j]);
		}
		System.out.println("Test set error: " + error);
	}
}
