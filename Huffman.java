
import java.util.*;

public class Huffman {

    static class Node {
        char ch;
        int freq;
        Node left, right;

        Node(char ch, int freq) {
            this.ch = ch;
            this.freq = freq;
        }
    }

    public static void printCode(Node root, String code) {
        if (root == null)
            return;

        if (root.left == null && root.right == null) {
            System.out.println(root.ch + " : " + code);
            return;
        }

        printCode(root.left, code + "0");
        printCode(root.right, code + "1");
    }

    public static void main(String[] args) {
        char[] ch = {'A', 'B', 'C', 'D', 'E', 'F'};
        int[] freq = {5, 9, 12, 13, 16, 45};

        PriorityQueue<Node> pq = new PriorityQueue<>(
            Comparator.comparingInt(n -> n.freq)
        );

        for (int i = 0; i < ch.length; i++) {
            pq.add(new Node(ch[i], freq[i]));
        }

        while (pq.size() > 1) {
            Node left = pq.poll();
            Node right = pq.poll();

            Node parent = new Node('-', left.freq + right.freq);
            parent.left = left;
            parent.right = right;

            pq.add(parent);
        }

        System.out.println("Huffman Codes:");
        printCode(pq.poll(), "");
    }
}

