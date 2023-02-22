public abstract class Napis {
  protected int length;
  public int length() { return length; };
  public abstract char getChar(int idx); // podaje znak na i-tej pozycji
  public abstract void putChar(int idx, char c); // wkłada znak na i-tą pozycję
  public abstract void add(Napis drugi); // dołącza na koniec
  @Override
  public abstract String toString();

  public static void main(String[] args) {
    Napis s;

    s = new NapisNaTablicy("tablica");
    System.out.printf("toString tablica: %s\n", s.toString());

    s = new NapisLinkedList("lista");
    System.out.printf("toString   lista: %s\n", s.toString());

    s = new NapisNaTablicy("tablica1");
    s.add(new NapisNaTablicy("tablica2"));
    System.out.printf("%s\n", s.toString());

    s = new NapisNaTablicy("tablica");
    s.add(new NapisLinkedList("lista"));
    System.out.printf("%s\n", s.toString());

    s = new NapisLinkedList("lista1");
    s.add(new NapisLinkedList("lista2"));
    System.out.printf("%s\n", s.toString());

    s = new NapisLinkedList("lista");
    s.add(new NapisNaTablicy("tablica"));
    System.out.printf("%s\n", s.toString());

    s = new NapisLinkedList("lista");
    for (int i = 0; i < s.length(); i++)
      System.out.printf("%c", s.getChar(i));
    System.out.printf("\n");
    s.putChar(5, 'X');
    for (int i = 0; i < s.length(); i++)
      System.out.printf("%c", s.getChar(i));
    System.out.printf("\n");

    s.add(s);
    System.out.printf("%s\n", s);

    s = new NapisNaTablicy("tablica");
    for (int i = 0; i < s.length(); i++)
      System.out.printf("%c", s.getChar(i));
    System.out.printf("\n");
    s.putChar(7, 'X');
    for (int i = 0; i < s.length(); i++)
      System.out.printf("%c", s.getChar(i));
    System.out.printf("\n");

    s.add(s);
    System.out.printf("%s\n", s);
  }

}

class NapisLinkedList extends Napis {
  private Node head, tail;

  class Node {
    public char c;
    public Node next;

    public Node(char c) { this.c = c; this.next = null; }
  }
  
  public NapisLinkedList(String s) {
    length = s.length();
    head = new Node(s.charAt(0));
    tail = head;
    for (int i = 1; i < length; i++) {
      Node tmp = new Node(s.charAt(i));
      tail.next = tmp;
      tail = tmp;
    }
  }
  
  public char getChar(int idx) {
    assert idx < length : "Index poza zakresem.";
    Node v = head;
    for (int i = 0; i < idx; i++)
      v = v.next;
    return v.c;
  }

  public void putChar(int idx, char c) {
    assert idx <= length : "Index za duży.";
    if (idx < length) {
      Node v = head;
      for (int i = 0; i < idx; i++)
        v = v.next;
      Node tmp = new Node(c);
      tmp.next = v.next;
      v.next = tmp;
    } else {
      tail.next = new Node(c);
      tail = tail.next;
    }
    length++;
  }

  public void add(Napis drugi) {
    String s = drugi.toString();
    for (int i = 0; i < s.length(); i++) {
      tail.next = new Node(s.charAt(i));
      tail = tail.next;
    }
    length += s.length();
  }

  @Override
  public String toString() {
    Node v = head;
    char[] arr = new char[length];
    for (int i = 0; i < length; i++, v = v.next)
      arr[i] = v.c;
    return new String(arr);
  }
}

class NapisNaTablicy extends Napis {
  private char[] arr;

  public NapisNaTablicy(String s) {
    arr = new char[s.length()];
    for (length = 0; length < s.length(); length++)
      arr[length] = s.charAt(length);
  }

  public char getChar(int idx) {
    assert idx < length : "Index spoza zakresu.";
    return arr[idx];
  }

  public void putChar(int idx, char c) {
    assert idx <= length : "Index za duży.";
    char[] old = arr.clone();
    arr = new char[++length];
    for (int i = 0; i < idx; i++)
      arr[i] = old[i];
    arr[idx] = c;
    for (int i = idx; i < length - 1; i++)
      arr[i + 1] = old[i];
  }

  public void add(Napis drugi) {
    String s = drugi.toString();
    char[] old = arr.clone();
    arr = new char[length + s.length()];
    for (int i = 0; i < length; i++)
      arr[i] = old[i];
    for (int i = 0; i < s.length(); i++)
      arr[i + length] = s.charAt(i);
    length += s.length();
  }

  @Override
  public String toString() {
    return new String(arr);
  }
}
