import java.util.*;

public class BST<T> {

 public static void main(String[] args) {
   System.out.println("===================== Integer ====================");
   BST<Integer> bstInteger = new BST<Integer>(Integer::compareTo);

   bstInteger.add(4);
   bstInteger.add(2);
   bstInteger.add(1);
   bstInteger.add(4);
   bstInteger.add(5);

   System.out.printf("find(3): %s\n", bstInteger.find(3) ? "true" : "false");
   System.out.printf("find(1): %s\n", bstInteger.find(1) ? "true" : "false");
   System.out.println(bstInteger.collect());

   System.out.println("===================== String ====================");
   BST<String> bstString = new BST<String>(String::compareTo);

   bstString.add("1");
   bstString.add("3");
   bstString.add("031");

   System.out.printf("find(\"3\"): %s\n", bstString.find("3") ? "true" : "false");
   System.out.printf("find(\"03\"): %s\n", bstString.find("03") ? "true" : "false");
   System.out.printf("find(\"1\"): %s\n", bstString.find("1") ? "true" : "false");
   System.out.println(bstString.collect());
 } 

  private class Node {
    public T    value;
    public Node l, r;
    public Node(T value) {
      this.value = value;
      this.l     = null;
      this.r     = null;
    }

    public void add(T value) {
      if (cmp.compare(this.value, value) == 0) { /* there is already this value */
        return;
      } else if (cmp.compare(this.value, value) < 0) {
        if (this.l == null)
          this.l = new Node(value);
        this.l.add(value);
      } else if (cmp.compare(this.value, value) > 0) {
        if (this.r == null)
          this.r = new Node(value);
        this.r.add(value);
      }
    }

    public boolean find(T value) {
      if (cmp.compare(this.value, value) == 0)
        return true;
      if (this.l == null && this.r == null)
        return false;
      if (this.l == null)
        return this.r.find(value);
      if (this.r == null)
        return this.l.find(value);
      return this.l.find(value) || this.r.find(value);
    }
  }

  private Node root;
  private Comparator<T> cmp;
  private ArrayList<T> lst;

  public BST(Comparator<T> cmp) {
    this.root = null;
    this.cmp  = cmp;
  }

  public void add(T value) {
    if (this.root == null)
      this.root = new Node(value);
    this.root.add(value);
  }

  public boolean find(T value) {
    return this.root.find(value);
  }


  private void collectWrapper(Node v) {
    if (v == null)
      return;
    collectWrapper(v.r);
    this.lst.add(v.value);
    collectWrapper(v.l);
  }

  public ArrayList<T> collect() {
    this.lst = new ArrayList<T>();
    collectWrapper(this.root);
    return this.lst;
  }
}
