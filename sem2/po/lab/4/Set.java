import java.util.Arrays;
public class Set {
  private int[] arr;
  
  /* return size of the set */
  public int size() {
    return this.arr.length;
  }
  /* return nth element of the set */
  public int nth(int n) {
    assert (0 <= n && n < this.size()): "n not in range";
    return this.arr[n];
  }

  Set(int... args) {
    if (args.length == 0)
      return;
    Arrays.sort(args);
    this.arr = new int[countDistinctIntegersInSortedArray(args)];
    this.arr[0] = args[0];
    for (int i = 0, j = 0; i < args.length; i++)
      if (args[i] != arr[j])
        arr[++j] = args[i];
  }

  Set sum(Set a) {
    if (this.size() == 0)
      return a;
    if (a.size() == 0)
      return this;

    int n = 1; // number of distinct elements in 2 sorted arrays
    int i = 0, j = 0;
    int prev = min(this.nth(i), a.nth(j));
    while (i < this.size() || j < a.size()) {
      if (i < this.size() && (j == a.size() || this.nth(i) < a.nth(j))) {
        if (prev != this.nth(i))
          n++;
        prev = this.nth(i++);
      } else {
        if (prev != a.nth(j))
          n++;
        prev = a.nth(j++);
      }
    }

    int[] arr = new int[n];

    int idx = 0;
    i = 0;
    j = 0;

    arr[idx] = min(this.nth(i), a.nth(j));
    prev = arr[idx];
    while (i < this.size() || j < a.size()) {
      if (i < this.size() && (j == a.size() || this.nth(i) < a.nth(j))) {
        if (prev != this.nth(i))
          arr[++idx] = this.nth(i);
        prev = this.nth(i++);
      } else {
        if (prev != a.nth(j))
          arr[++idx] = a.nth(j);
        prev = a.nth(j++);
      }
    }

    return new Set(arr);
  }

  public void printSet(String name) {
    System.out.printf("%s : [", name);
    for (int i = 0; i < this.size() - 1; i++) {
      System.out.printf("%d, ", this.arr[i]);
    }
    System.out.printf("%d]\n", this.arr[this.size() - 1]);
  }

  public static int min(int a, int b) {
    return (a < b ? a : b);
  }

  public static int countDistinctIntegersInSortedArray(int[] arr) {
    if (arr.length == 0)
      return 0;
    int res = 1, prev = arr[0];
    for (int x : arr) {
      if (x != prev)
        res++;
      prev = x;
    }
    return res;
  }

  public static void main(String[] args) {
    Set a = new Set(2, 1, 2, 1, 2, 2, 2, 3, 2, 1, 1, 1);
    Set b = new Set(2, 2, 2, 4, 1, 1, 2, 3);
    Set c = a.sum(b);
    a.printSet("set A");
    b.printSet("set B");
    c.printSet("set AuB");

    a = new Set(2, 3, 5, 5, 2, 3);
    b = new Set(3, 1, 1, 2);
    c = a.sum(b);
    a.printSet("set A");
    b.printSet("set B");
    c.printSet("set AuB");
  }

};
