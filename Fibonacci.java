
import java.util.Scanner;

public class Fibonacci {

    // Non-recursive method
    static void nonRecursive(int n) {
        int a = 0, b = 1;

        for (int i = 0; i < n; i++) {
            System.out.print(a + " ");
            int c = a + b;
            a = b;
            b = c;
        }
    }

    // Recursive method
    static int recursive(int n) {
        if (n <= 1)
            return n;

        return recursive(n - 1) + recursive(n - 2);
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        System.out.print("Enter number of terms: ");
        int n = sc.nextInt();

        System.out.println("Non-Recursive Fibonacci:");
        nonRecursive(n);

        System.out.println("\nRecursive Fibonacci:");
        for (int i = 0; i < n; i++) {
            System.out.print(recursive(i) + " ");
        }

        sc.close();
    }
}
