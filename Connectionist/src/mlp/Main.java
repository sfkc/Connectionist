package mlp;

public class Main {
	private static final int maxEpochs = 10;
	
	public static void main(String[] args) {
		double[][] example = {{0, 0, 0}, {0, 1, 1}, {1, 0, 1}, {1, 1, 0}};
		MLP NN = new MLP(2, 2, 1);
		for (int e = 0; e < maxEpochs; e++) {
			for (int p = 0; p < 4; p++) {
				NN.forward(new double[]{example[p][0], example[p][1]});
				NN.backwards(new double[]{example[p][2]});
			}
		}
	}
}
