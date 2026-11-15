public class FractionalKnapsack {
    public static void main(String[] args) {
        int[] weight = {10, 20, 30};
        int[] profit = {60, 100, 120};
        int capacity = 50;

        double[] ratio = new double[weight.length];

        for (int i = 0; i < weight.length; i++) {
            ratio[i] = (double) profit[i] / weight[i];
        }

        // Sort items by profit/weight ratio in descending order
        for (int i = 0; i < weight.length - 1; i++) {
            for (int j = i + 1; j < weight.length; j++) {
                if (ratio[i] < ratio[j]) {
                    double temp = ratio[i];
                    ratio[i] = ratio[j];
                    ratio[j] = temp;

                    int t = weight[i];
                    weight[i] = weight[j];
                    weight[j] = t;

                    t = profit[i];
                    profit[i] = profit[j];
                    profit[j] = t;
                }
            }
        }

        double totalProfit = 0;

        for (int i = 0; i < weight.length; i++) {
            if (capacity >= weight[i]) {
                capacity -= weight[i];
                totalProfit += profit[i];
            } else {
                totalProfit += ratio[i] * capacity;
                capacity = 0;
                break;
            }
        }

        System.out.println("Maximum Profit = " + totalProfit);
    }
}
